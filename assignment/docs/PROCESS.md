# EE290/194 Part B — process log

Nemotron-Nano-V3 SFT on 8×H100: measure where training time goes, explain it with
evidence, fix it, measure again.

---

## 1. What we are doing, and the setup

### The task

Take a training stack, run a real model on it, find out why it is slow, and prove it.
The grade is the chain **symptom → evidence → intervention → measured effect**, not the
size of the speedup. Required metrics: **MFU** and end-to-end latency.

**MFU** (Model FLOPs Utilization) = the arithmetic the model actually needed, divided by
what the GPUs could theoretically have done in that time. An H100 does 989.5 TFLOP/s in
bfloat16. Well-tuned large-scale training reaches 40–50%.

### The stack, all pulled from source and unmodified

| | source | version |
|---|---|---|
| Model | HuggingFace `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | 60 GB, as downloaded |
| Framework | `github.com/NVIDIA-NeMo/RL` | `main`, branch `assignment1` |
| Backend | Automodel submodule `NVIDIA-NeMo/Automodel` | `r0.6.0` pin (`72daceff`) |
| Hardware | 1 node × 8 H100 80GB HBM3, driver 550 | CUDA 13 via forward-compat |

Nothing in the training path has been modified. Our commits touch only instrumentation.

### The model

52 layers, decoded from `hybrid_override_pattern`: **23 Mamba2 + 23 mixture-of-experts +
6 attention**. Hidden size 2688, 128 routed experts with top-6 routing, 1 shared expert,
`relu2` activation (squared ReLU, not gated), vocab 131,072. ~30B total parameters,
~3B active per token.

### Parallelism

FSDP2 (**Fully Sharded Data Parallel**: each GPU stores 1/8 of every weight and
all-gathers the full weight over the network before using it), expert-parallel size 8,
tensor/context/pipeline parallel all 1.

### Baseline workload

The upstream recipe `sft-nanov3-30BA3B-2n8g-fsdp2.yaml` on SQuAD, global batch 16,
2048-token cap, run at 1 node. Note `2n8g` in the filename: **it is a 16-GPU recipe run
on 8 GPUs**, so batch sizes were never tuned for this configuration.

### Reproduce

```bash
EXP_NAME=<name> bash ~/assn1-f/run_sft_nanov3_baseline.sh \
    sft.max_num_steps=50 sft.val_at_start=false sft.val_period=1000
```

50 steps, warmup step discarded, median and p90 reported (not mean). 50 is chosen so the
median is stable: resampling the measured data, 10 steps gives ±8.7% and 49 gives ±0%.
A measurement must be sharper than the effect it is hunting.

---

## 2. Bottlenecks, and how each was identified

### 2.0 First, MFU did not work at all — fixed

**Symptom.** No MFU line in any run, ever.
**Found by** reading `run.log` line 59:

```
FLOPS tracker not supported for model ...: Unsupported config type: NemotronHConfig
```

`convert_config_to_flops_config` is an if/elif chain over Qwen2/Qwen3/Llama/DeepSeek/GLM;
NemotronH hits the `else`, and `lm_policy.py:435` silently sets `flops_tracker = None`.

Underneath, the `nemotronh` formula **existed but was dead code** — nothing dispatched to
it, so it had never run, and it did not work:

1. It counted layer types `M`, `-`, `*` and skipped **`E`**, which is what Nemotron-V3
   calls its MoE blocks. **23 of 52 layers contributed zero FLOPs — 57% of the model.**
   Adding only the dispatch branch would have printed 1.9% instead of 4.4%, and looked
   entirely plausible.
2. `_mlp_layer_flops` read `config.gated_linear_unit`, which was not a field.
3. No MoE formula existed at all.

**Verified** against `6 × active_params × tokens`: 6.32e14 vs 5.90e14, a 1.07× match.
Fixed in commit `9998e3ebd`. This is an upstream defect.

### The tools, and why four were needed

| tool | watches | answers |
|---|---|---|
| **nsys** (Nsight Systems) | the GPU | what ran on the GPU, when, how long |
| **Kineto** (PyTorch profiler) | PyTorch | which PyTorch call caused which GPU kernel |
| **py-spy** | the CPU | what Python is actually doing |
| **memory snapshot** | GPU memory | where memory went, and whether it fragments |

The point of four: nsys and Kineto showed the GPU was idle, but **a GPU tool cannot say
what the CPU is doing**. Only py-spy could finish the chain.

### 2.1 Batches are 9.5× under-filled

**Type:** data shape. **Found by:** arithmetic, before any profiler.

Predicted MFU was ~4%; measured was 0.43%. Rather than assume the code was wrong, checked
what the tracker measures: `track_batch` feeds **actual** `input_lengths`, not the cap.
Inverting the FLOPs formula against observed work:

```
observed  5.711e13 FLOP/step  ->  implied mean sequence length 187 tokens (cap 2048)
```

Later confirmed by direct tokenization: mean **214**, median 196, p99 499, max 545.

The global batch of 16 splits across 8 GPUs (`local_gbs = gbs // dp_size`), so **each GPU
processes 2 sequences ≈ 428 tokens per step**, in two separate passes. Overhead — FSDP
all-gathers, ~17,500 kernel launches, the MoE all-to-all — is paid **per microbatch, not
per token**. The overhead is constant; the useful work is what is tiny.

### 2.2 GPU idle 45–65%, waiting on the CPU

**Type:** host-bound. **Found by:** Kineto traces, all 8 ranks.

```
idle 45-65% of wall per rank (rank 0: 64.7%)
idle breakdown: host_wait 97-100%
```

`host_wait` means nothing is queued because Python has not submitted it. Not peer wait,
not kernel wait.

### 2.3 `empty_cache()` every step

**Type:** host-bound, root cause of 2.2. **Found by:** py-spy, then grep.

```
empty_cache  torch/cuda/memory.py   30.3% of host self-time
```

Call site is `dtensor_policy_worker_v2.py:571`, unconditional, once per step:

```python
# dynamic batch and sequence dims causes alot of fragmentation, so clear
# the memory allocator before moving on
torch.cuda.empty_cache()
```

`git blame`: commit `e861b94dd0`, 2025-08-24, the commit that **created** the DTensor V2
worker. **Upstream NVIDIA code**, not ours, shipping to everyone on this backend. It
synchronizes the device and frees every cached block.

Its own comment names 2.1 as the cause — dynamic dims come from variable-length sequences.

### 2.4 Communication is 88% of GPU work

**Type:** communication-bound. **Found by:** nsys, all 8 ranks, kernels bucketed by name.

```
NCCL               56.3%        real math          2.1%
DeepEP             31.4%        elementwise        9.4%
-> communication   87.7%        overlap w/ compute 1.3%
```

Communication is essentially **never** hidden behind compute.

### 2.5 Communication imbalance, not compute imbalance

**Type:** load imbalance. **Found by:** nsys per-rank comparison.

Compute 769–776 ms across ranks (0.9% spread — effectively identical). Communication
3723–4989 ms (1.34×). So it is **not** a compute straggler.

### 2.6 Intermittent stalls

**Type:** stall. **Found by:** run-log arithmetic (TFLOPS × step time = work per step).

Work per step varies **1.59×** while step time varies **4.41×**, and the six steps slower
than 2.2 s did *less* work (56.4 vs 59.2 TFLOP median). Not explained by compute.

### 2.7 Elementwise work dwarfs real math

**Type:** kernel efficiency. **Found by:** nsys bucketing.

4740 ms elementwise vs 1070 ms for all GEMMs + Mamba + attention + norms — **4.4×**.
Cause not established; needs `ncu`, which produced no report.

### Baseline numbers (run A)

| | median | p90 | max | spread |
|---|---|---|---|---|
| step time s | 1.79 | 2.33 | 7.14 | 4.41× |
| MFU % | 0.41 | 0.47 | 0.52 | 5.78× |
| work TFLOP | 58.89 | 67.53 | 72.46 | 1.59× |

---

## 3. Optimizations

### 3.1 Sequence packing

**Targets:** 2.1 (under-filled batches), and via 2.1 also 2.3 and 2.6.

**What it is.** Concatenate several short sequences into one microbatch until the
2048-token budget is full, with masking so sequences do not attend across boundaries.
Already present in NeMo-RL, off by default.

```
Unpacked  (# real, p padding)        Packed
0 0 0 p p p                          0 0 0 1 1 1 1 1 1 2 2 3 3 3 p
1 1 1 1 1 1
2 2 p p p p
```

**Why it should help, from first principles.** Overhead is per microbatch. Each GPU gets
2 sequences; packing merges them into 1 microbatch, halving microbatch count and therefore
halving FSDP weight all-gathers per step — for identical work.

**Semantics.** None change. The same 16 sequences produce the same gradients and the same
optimizer updates. Only the memory layout differs.

**Command.**
```bash
EXP_NAME=sft-nanov3-B-pack-gbs16 bash ~/assn1-f/run_sft_nanov3_baseline.sh \
    sft.max_num_steps=50 sft.val_at_start=false sft.val_period=1000 \
    policy.sequence_packing.enabled=true
```

**Result — 50 steps, warmup discarded. See the correctness finding below: these numbers
are real wall-clock measurements of an incorrect computation, and must not be reported as
a speedup.**

| | A baseline | B packing | change |
|---|---|---|---|
| step time median | 1.79 s | **1.45 s** | **19% faster** |
| step time p90 | 2.33 s | 1.61 s | |
| step time max/min | 4.41× | **1.15×** | |
| MFU median | 0.41% | **0.50%** | **1.22×** |
| MFU max/min | 5.78× | 1.80× | |
| steps slower than 2.2 s | **6 / 49** | **0 / 49** | |

**The unexpected result is the variance, not the mean.** Packing removed *every* stall
(6 → 0) and collapsed step-time spread from 4.41× to 1.15×. That is bottleneck 2.6 gone,
and it supports the hypothesis that the stalls came from variable-shape microbatches
churning the allocator — the same mechanism `empty_cache()` at line 571 was added to
paper over.

Warmup cost rose (25.15 s → 61.70 s), which is discarded and does not matter here.

#### RESOLVED: packing is silently incorrect on this model. Do not use it.

The 4k realistic-workload config in `~/assn1-f/configs/` flagged that packing may be
unsafe here. It is. Confirmed by direct measurement, not inference.

First, the loss comparison — same seed, same data, same batch — which was **suggestive
but not conclusive**:

```
A (no packing)  4.2480 3.0352 3.0194 2.0179 2.3047 ...  mean 2.404
B (packing)     3.9225 2.8245 2.8110 2.1899 2.3076 ...  mean 2.325
```

B is 3.3% *lower*. Cross-document attention leaks extra context and lowers loss, so this
is the direction a silent bug produces — but it is equally consistent with gradient noise
from regrouped microbatches. Twelve steps cannot separate the two.

So the model was instrumented instead. `NemotronV3Attention.forward` picks its layout from
`hidden_states.dim()`: 2 means packed (`thd`), 3 means batched (`bshd`). A one-shot probe
was inserted there and in the Mamba mixer, and three steps run each way:

| | shape the model receives | contains | `qkv_format` | `cu_seqlens` |
|---|---|---|---|---|
| packing **off** | `(1, 508, 2688)` | **one** document | `bshd` | absent — correct, nothing to separate |
| packing **on** | `(1, 2048, 2688)` | **~9** documents | `bshd` | **absent — wrong** |

```
[PACKPROBE-ATTN]  dim=3 shape=(1, 2048, 2688) qkv_format=bshd
                  kwargs=['flash_attn_kwargs'] cu_seqlens=ABSENT
[PACKPROBE-MAMBA] seq_idx_is_none=True cu_seqlens=ABSENT cu_seqlens_padded=ABSENT
```

Packing concatenates roughly nine documents into one tensor and passes **no boundary
information at all**. The layout stays `bshd`, so attention treats the whole 2048 tokens
as a single document and every document attends to all the ones before it. Mamba's
`seq_idx` is `None`, so its recurrent state also flows across boundaries.

The model *can* do this correctly — `layers.py:106` sets `qkv_format="thd"` when handed a
2-D tensor, and `layers.py:342-394` segments the SSD scan from `cu_seqlens`. The defect is
in the plumbing: NeMo-RL's DTensor path passes only HF `flash_attn_kwargs`, never
`cu_seqlens`, so none of that machinery engages.

**Consequence.** The 19% and the stall elimination are real wall-clock measurements of an
*incorrect* computation. They cannot be reported as a speedup. The mechanism — halving
microbatches halves FSDP all-gathers — is still sound and would hold if packing were
plumbed correctly; that is a possible intervention in itself, and an upstream bug report.

The probe was reverted; `layers.py` is unmodified.

---

### 3.2 Skip the per-step `empty_cache()`

**Targets:** 2.3 (30% of host CPU) and through it 2.2 (GPU idle on `host_wait`).

Already implemented in the working tree as a config key, `empty_cache_after_train`
(`nemo_rl/models/policy/__init__.py:228`), defaulting to True so stock behaviour is
unchanged. Measuring it needs no code change.

**Command** — note Hydra requires `+` to add a key the YAML does not already carry:

```bash
EXP_NAME=sft-nanov3-E-noemptycache bash ~/assn1-f/run_sft_nanov3_baseline.sh \
    sft.max_num_steps=50 sft.val_at_start=false sft.val_period=1000 \
    +policy.dtensor_cfg.empty_cache_after_train=false
```

Unlike packing, this changes no numerics: it only declines to return cached blocks to the
allocator. Loss should match run A step for step, and that is the check.

**Result (run E): 20% faster, then OOM at step 14.** The comment on line 571 was right —
the call is load-bearing.

```
torch.OutOfMemoryError: Tried to allocate 168.00 MiB. GPU 0: 125.88 MiB free.
72.31 GiB allocated by PyTorch, 2.08 GiB reserved but unallocated.
```

Step times climb just before it dies — 2.03, 1.78, 2.09 against a 1.42 median — the
allocator thrashing before it gives up.

One detail points at the real fix: only 2.08 GiB is reserved-but-unallocated against
72.31 GiB genuinely allocated. That is not classic fragmentation. It is the allocator
holding many cached blocks of *mismatched sizes*, one per distinct microbatch shape.
`empty_cache()` returns them; without it they accumulate. PyTorch's own error message
names the fix.

### 3.3 `expandable_segments` — what makes 3.2 usable

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets the allocator grow and shrink a
segment in place instead of caching fixed-size blocks, so mismatched blocks never pile up
and the per-step flush becomes unnecessary. The two changes are only useful together.

```bash
EXP_NAME=sft-nanov3-F-noec-expseg bash ~/assn1-f/run_sft_nanov3_baseline.sh \
    sft.max_num_steps=50 sft.val_at_start=false sft.val_period=1000 \
    +policy.dtensor_cfg.empty_cache_after_train=false \
    +policy.dtensor_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Result — 50 steps, no OOM:**

| | A baseline | F |
|---|---|---|
| step time median | 1.79 s | **1.20 s** (33% faster, 1.49×) |
| step time p90 | 2.33 s | 1.63 s |
| step time max/min | 4.41× | 2.44× |
| MFU median | 0.41% | **0.60%** (1.46×) |
| stalls > 2.2 s | 6 / 49 | **1 / 49** |
| loss, first 2 steps | 4.2480 3.0352 | 4.2461 3.0313 |

Neither change touches numerics — they only alter when memory is returned to the
allocator — and the early losses agree, unlike packing.

**Caveat on the loss check.** Steps 1-2 agree to ~1e-3, but step 3 is 3.0194 vs 3.0583,
1.3% apart. That is consistent with ordinary nondeterministic accumulation (reduction
order, cuBLAS algorithm choice under a different memory layout) and is *not* the
systematic one-directional shift packing showed. To settle it, run A twice and measure how
far A diverges from itself; if A-vs-A drifts comparably, F is confirmed. That control has
not been run.

### 3.4 Collectives — and what they turned out to be

This section closes the largest item under "Still open": whether bottleneck 2.4's 88%
communication is real wire time or spin-waiting on peers. It is spin-waiting, and the thing
being waited on is not the network.

Tooling for this section is `~/assn1-f/analysis/collectives_report.py`, which takes a
directory of Kineto rank traces and prints the per-rank table below. Every measurement here
re-runs that one command, so the numbers stay comparable across experiments.

#### The symptom

Collectives are 55.9% of GPU kernel time on rank 0 (NCCL 37.0% + DeepEP 18.9%) against
**5.9% in real matrix math**. The obvious conclusion is that the interconnect is the
bottleneck.

#### Evidence 1 — it is not bandwidth

This box is 8×H100 SXM with an **NV18 full mesh**: every GPU pair has 18 NVLinks, no PCIe
hop anywhere, ~450 GB/s per GPU. From the NCCL metadata in the traces:

| collective | dtype | calls | bytes | per call |
|---|---|---|---|---|
| `_allgather_base` | BFloat16 | 312 | 4.4 GB | 14 MB |
| `_reduce_scatter_base` | Float32 | 312 | 70.5 GB | 113 MB |

All-gather achieves **10.4 GB/s — 2.3% of the wire**. A 14 MB all-gather should take ~31 µs;
it takes 869 µs, 28× longer. Whatever fills those 869 µs, it is not data movement.

#### Evidence 2 — it is skew, and compute is the control

Every rank issues the same collectives on the same byte counts, so any spread in collective
*time* across ranks is waiting. GEMM and elementwise time are the control: identical work on
every rank, so their spread is the measurement noise floor.

| bucket | min | max | spread |
|---|---|---|---|
| GEMM | 101.8 ms | 112.3 ms | **1.10×** |
| elementwise | 648.0 ms | 651.6 ms | **1.01×** |
| NCCL | 639.6 ms | 1353.2 ms | **2.12×** |
| DeepEP | 326.0 ms | 871.4 ms | **2.67×** |

Compute is balanced to 1%. Collective wait varies 2.1–2.7×. So this is not load imbalance in
the model; it is arrival-time skew that the collectives absorb as a barrier.

#### Evidence 3 — an independent confirmation from the dataset swap

HellaSwag averages 118 tokens per example against SQuAD's 214 — **1.8× less arithmetic per
step**. Measured over 50 steps each: SQuAD **1.79 s**, HellaSwag **1.83 s**. Halving the work
changed the step time by 2%, because per-step cost is set by things that do not scale with
tokens.

#### Evidence 4 — re-measuring on run F, which pointed the opposite way

The traces above predate run F (captured 06:15; run F landed 11:00), and run F removed the
per-step `empty_cache()` that section 3.3 measured at 30.3% of rank-0 host CPU. Re-profiling
on the run F config was expected to shrink the skew. It widened it:

| | baseline | on run F |
|---|---|---|
| GEMM spread (control) | 1.10× | 1.11× |
| elementwise spread (control) | 1.01× | 1.01× |
| NCCL spread | 2.12× | **3.18×** |
| DeepEP spread | 2.67× | **4.36×** |

Ranks 1–7 read as 79–87% "busy", but that busy time is spinning inside NCCL. Rank 0, which
waits least, shows its true state: **GPU idle 64% of the span**.

#### Evidence 5 — the idle is 33,451 separate gaps

Merging all kernel intervals on rank 0 and measuring the holes:

```
rank 0: span 3659 ms, GPU idle 2335 ms (64%), 33451 gaps, mean gap 69.8 us
        largest single gap 85.6 ms
```

Not one blocking call — **a gap after nearly every one of the 35,030 kernels**. The GPU is
starving because the host cannot launch work fast enough. Every rank is paced by its own
Python, arrival times jitter, and the collectives bill the jitter as communication.

#### The cause

Attributing kernels and syncs to their enclosing CPU op on rank 0, over two optimizer steps:

- `Optimizer.step#AdamW.step` launched **5,968 kernels** (2,984 per step, mean 34.5 µs) — 17%
  of all launches in the trace
- **756 of the 1,402 `aten::item` calls sat inside `Optimizer.step`** — one device-to-host
  sync per parameter tensor. Each sync drains the CUDA launch queue and restarts the pipeline
  from empty.

Both come from the single-tensor AdamW path, selected in `examples/configs/sft.yaml:111-114`:

```yaml
# when using Dtensor, we need to set foreach
# and fused to False
foreach: False
fused: False
```

#### The intervention, and what the measurement said

That comment is half wrong. Measured, 50 steps each, on the run F config:

| arm | median | p90 | MFU | TFLOPS |
|---|---|---|---|---|
| single-tensor (upstream) | 1.220 s | 1.780 s | 0.580% | 45.95 |
| **`fused: true`** | **1.140 s** | **1.560 s** | **0.630%** | **50.21** |
| `foreach: true` | **OOM** | — | — | — |

`fused: true` runs correctly under DTensor and is **6.6% faster on median, 12.4% on p90**.
`foreach: true` fails, but with `Tried to allocate 168.00 MiB` at 73.45 GiB already
allocated — a memory failure, not a DTensor-compatibility one. So upstream's conclusion
holds for `foreach` for a different reason than stated, and is simply wrong for `fused`.

**Loss is unchanged within noise, and the noise was measured rather than assumed.** Running
the identical baseline config twice (A/A control) gives mean|Δ| **0.0318** over 50 steps.
Fused against that second baseline gives **0.0308** — smaller than the run-to-run variation.
Mean loss: 1.6196 and 1.6371 for the two baseline runs, 1.6349 for fused. Note the
consequence for section 3.3's open caveat: **this stack is not deterministic run-to-run even
with a fixed seed**, so small loss differences cannot validate a numerics change at this
sample size.

#### Mechanism confirmed, and the ceiling it exposed

Re-profiling with `fused` shows the change did exactly what was predicted:

| rank 0 | single-tensor | fused |
|---|---|---|
| kernels in `Optimizer.step` | 5,968 | **22** |
| optimizer host time (2 steps) | 298 ms | **21 ms** |
| `aten::item` inside optimizer | 756 | **0** |
| total kernels | 35,030 | 29,370 |
| GPU-idle gaps | 33,451 (2335 ms) | 27,879 (2416 ms) |

The optimizer's launches are gone. **The GPU idle did not shrink** — 27,879 gaps remain at
87 µs each, and rank 0's busy fraction fell to 34.8%. The step got faster because the
optimizer's own GPU work and host time went away, not because the starvation was fixed.

That is the real finding: **the training step is host-launch-bound end to end, not just in
the optimizer.** 29,370 kernels for a handful of steps, with dead air after nearly every one.
The 56% "communication" figure is a downstream symptom — collectives acting as barriers that
absorb host launch jitter — and it will not move until kernel count per step comes down.

#### Negative result worth recording

Measured compute/communication overlap is 1.3%, and the instinct is to overlap the
collectives behind compute. **You cannot hide 56% of communication behind 5.9% of compute.**
There is not enough compute to hide it in. Overlap work only becomes worth doing after
tokens-per-step rises; at this batch size it is not the lever.

#### The knob sweep, and why most of it is a negative result

With fused AdamW in place, every remaining config-level collectives knob was swept, each
arm 50 steps on the same config. **Each setting was run against a repeat of the unchanged
baseline**, because a difference is only a result if it exceeds the noise.

| arm | median | p90 | p99 | MFU |
|---|---|---|---|---|
| single-tensor AdamW | 1.220 | 1.780 | 3.320 | 0.580% |
| single-tensor AdamW *(repeat)* | **1.220** | **1.510** | 1.880 | 0.590% |
| fused AdamW | 1.140 | 1.560 | 1.780 | 0.630% |
| fused AdamW *(repeat)* | **1.140** | **1.440** | 1.640 | 0.640% |
| + `cuda_graph.modules=[moe_router]` | 1.140 | 1.450 | 1.680 | 0.620% |
| + `dispatcher_async_dispatch=true` | 1.140 | 1.440 | 1.500 | 0.630% |
| + both | 1.140 | 1.440 | 1.550 | 0.640% |
| + both + `dispatcher_num_sms=8` | 1.150 | 1.440 | 1.590 | 0.630% |
| `dispatcher=torch` | 1.230 | 1.520 | 1.980 | 0.580% |

**The repeat columns are the whole point.** Median repeats *exactly* (1.220/1.220 and
1.140/1.140), so a median difference of 0.01 s is real. p90 swings **1.780 to 1.510** and
**1.560 to 1.440** between two runs of an identical config, so the p90 noise band is
roughly +/-0.12 s and nothing smaller than that can be claimed.

Applying that discipline:

- **Real: fused AdamW.** 1.220 -> 1.140 median (6.6%), MFU 0.580/0.590% -> 0.630/0.640%
  (~8.5%). Both medians repeat exactly, so this is not noise.
- **Real: `dispatcher=torch` is worse.** 1.230 vs 1.140 median (+7.9%). Replacing DeepEP's
  all-to-all with DTensor all-gather/reduce-scatter loses, so DeepEP earns its ~1 ms
  handshake even at a few hundred tokens per rank. A hypothesis worth testing, and it
  failed.
- **Not real: CUDA graphs on `moe_router`.** Median unchanged, p90 inside the noise band.
  Only the router is capturable here, and it is a small share of the launch stream.
- **Not real: `dispatcher_async_dispatch`.** Median unchanged, p90 inside the noise band.
- **Not real: `dispatcher_num_sms=8`.** 1.150 vs 1.140, at the edge of resolution. Reserving
  20 of 132 SMs for communication is not what is costing time.
- **Blocked: `dispatcher=hybridep`.** Fails to build -- it JIT-compiles its kernels and nvcc
  errors against the CUDA 13 toolchain here. This also blocks CUDA-graph capture of
  `moe_preprocess`, which requires `dispatcher='hybridep'`. Likewise `cuda_graph=[attn]`
  requires `linear='torch'`, giving up TransformerEngine's fused attention.

An interim claim was made and withdrawn: `cuda_graph=[moe_router]` and
`dispatcher_async_dispatch` each appeared to cut p90 by 7% against a single baseline run.
The repeat baseline landed on the same p90, so the effect was measurement noise. It is
recorded here rather than deleted, because running the control is what caught it.

#### Where that leaves the lane

One real win out of nine arms, and it was not in NCCL at all. The config surface for
collectives on this stack is close to exhausted; what remains is code:

1. `reduce_dtype` is hardcoded to fp32 with no knob -- reduce-scatter carries 70.5 GB
   against all-gather's 4.4 GB for the same tensors.
2. Gradient-sync deferral exists in Automodel and is never called by NeMo-RL, so gradients
   reduce-scatter twice per step with 2 microbatches.
3. NCCL runs untuned at `RING_LL` on an NVSwitch box where NVLS is the intended algorithm.

But the measured ceiling says none of those is the main event either. The step is
host-launch-bound: 27,879 GPU-idle gaps remain after the optimizer fix, and the median is
pinned at 1.140 s across every knob tried. Until kernel count per step falls
substantially, collective time is a symptom being measured, not a cause being fixed.

#### Defects found while reading the code, not yet fixed

- `nemo_rl/models/automodel/setup.py:522` hardcodes `reduce_dtype=torch.float32` with no
  config knob, one line from `output_dtype` which does have one. That is why reduce-scatter
  carries 70.5 GB against all-gather's 4.4 GB — the same tensors at double the bytes.
- The recipe's `enable_deepep: true` is **silently ignored**. Run log line 51:
  `enable_deepep is no longer supported and is ignored. Set 'dispatcher' ... and 'experts'
  explicitly instead.` The live knobs are `dispatcher`, `dispatcher_async_dispatch` (False),
  `dispatcher_num_sms` (20 of 132 SMs reserved for communication), `enable_fsdp_optimizations`
  (False).
- NeMo-RL never calls `prepare_for_grad_accumulation` / `prepare_for_final_backward`.
  Automodel provides them in `components/moe/fsdp_mixin.py` and its own recipes call them;
  `grep` over `nemo_rl/` finds zero call sites. With 2 microbatches per GPU, gradients
  reduce-scatter twice per step when once would do.
- NCCL selects `RING_LL` for every collective and there is no NCCL tuning anywhere in the
  stack. LL protocol spends half its payload on sync flags and is meant for messages under
  ~256 KiB; these are 14–113 MB. On an NVSwitch box the intended algorithm is NVLS.
- 280 ms of DeepEP's 326 ms is in `notify_dispatch` / `cached_notify_dispatch` /
  `cached_notify_combine` — metadata handshake, not token movement — at ~1 ms per call for a
  few hundred tokens per rank.

---

### 3.5 DeepEP on Track B — the defaults are already right

Section 3.4 tuned collectives on the SQuAD-scale workload. This section does the same for
Track B (Nemotron-Cascade-2 chat, 8k context, GBS 128, DTensor + FSDP2 + EP8), where the
MoE all-to-all is a far larger share of the run. Starting point: `optim-O1`, **6.530 s
median, 10.355% MFU**.

Tooling: `analysis/overlap.py` (comm/compute overlap), `analysis/dep_report.py` (arm
table), `analysis/matched.py` (matched-step comparison), `analysis/rsbytes.py` (collective
volume). Configs are `configs/dep-A*.yaml`.

#### The symptom

`analysis/overlap.py` intersects communication and compute kernel intervals directly.
Rank 0 of the Track B best run:

```
wall span                    15647.5 ms
GPU busy (any kernel)        10486.5 ms   67.0% of span
communication                 4744.2 ms
  overlapped with compute       251.3 ms    5.3% of comm
  EXPOSED                      4492.9 ms   28.7% of span
    NCCL      1826.7 ms   overlapped 232.0 ms (12.7%)
    DeepEP    3538.4 ms   overlapped  19.3 ms ( 0.5%)
streams: 7=compute 5955.9 | 16=DeepEP 3538.4 | 18=AG 898.6 | 22=RS 878.0
```

**DeepEP is 22.6% of wall clock at 0.5% overlap**, on a stream of its own — the largest
single block of time in the run. It has never been touched.

This measurement replaces HTA's `comp_comm_overlap_pctg`, which reported 46.9% for DTensor
against 4.18% for Megatron — the opposite ordering. HTA's definition is undocumented;
`overlap.py` states its own (`comm AND compute` / `comm MINUS compute`) and is what the
numbers above and in 3.6 rely on.

#### A config defect found before any experiment

The Track B config requests DeepEP through a key the code deleted:

```yaml
backend: { experts: torch_mm, enable_deepep: True }
```

```
run.log:52  WARNING: enable_deepep is no longer supported and is ignored. Set 'dispatcher'
            (deepep/hybridep/torch) and 'experts' explicitly instead. Previously
            enable_deepep=True was equivalent to experts=gmm + dispatcher=deepep.
```

No `dispatcher` key exists anywhere in the Track B config chain, so every run has used
`BackendConfig`'s dataclass defaults:

| knob | default in force | varied before this section |
|---|---|---|
| `dispatcher` | `deepep` (resolved from `HAVE_DEEP_EP`) | no |
| `dispatcher_num_sms` | **20** of 132 SMs | no |
| `dispatcher_async_dispatch` | **False** | no |
| `dispatcher_share_token_dispatcher` | `True` | no |
| `experts` | `torch_mm` | config asked for `gmm`, never got it |

Two things follow. DeepEP **was** always running — the default happens to resolve to it, so
the dead key cost us only the `experts=gmm` half. And no DeepEP setting had ever been
changed across ~30 Track B runs: it was a constant in every experiment, never a variable,
so nothing in the existing results says what it costs or what it would do differently.

#### Protocol

Each arm is the Track B best config with **exactly one change**, 30 steps, step 1 discarded.
`dep-A0-repeat` is that config unchanged — the noise floor, because a 7% p90 "win" reported
in 3.4 did not survive its repeat control. Arms are compared over a matched step range
(`analysis/matched.py`) because `dep-A5-gmm` stopped at step 22.

| arm | change | median | p90 | vs A0 |
|---|---|---|---|---|
| `optim-O1` | earlier run of the same config | 6.530 | 7.690 | −1.4% ← **noise floor** |
| `dep-A0-repeat` | **(control)** | 6.620 | 8.040 | — |
| `dep-A3-sms32` | `dispatcher_num_sms: 32` | 6.870 | 8.080 | +3.8% |
| `dep-A1-async` | `dispatcher_async_dispatch: true` | 7.030 | 7.760 | +6.2% |
| `dep-A5-gmm` | `experts: gmm` | 8.180 | 10.100 | +23.6%, **OOM at step 22** |
| `dep-A4-torchdisp` | `dispatcher: torch` | 8.510 | 9.860 | +28.5% |

Total floating-point work matched across arms to within 1% (Σ TFLOPS×time = 150,103–151,683),
so median step time is the fair comparison and the MFU column is not needed to rank them.

Not measured: `dispatcher_num_sms: 8` and `fake_balanced_gate`. Three other users took all
8 GPUs repeatedly during this sweep; both arms OOM'd against a foreign process holding
17–20 GiB. They were abandoned rather than reported, because a timing taken under
contention is not a measurement.

#### What each arm means

**A1, async dispatch (+6.2%).** The hypothesis was that DeepEP's 0.5% overlap is caused by
the dispatch blocking the compute stream — `async_finish=False` makes the caller wait, and
`allocate_on_comm_stream=False` forces a sync back. Setting both true removed the block and
made it **worse**. The reason: within one MoE layer the chain `route → dispatch → expert
GEMM → combine` is genuinely serial, because the GEMM needs the dispatched tokens. There is
no independent work to slide into the gap, so async dispatch buys nothing and costs
bookkeeping plus longer-lived comm-stream buffers. Megatron pairs the equivalent with
`moe_shared_expert_overlap` (worth +1.6% there, 18.985% → 19.280%); `grep` finds **no
shared-expert-overlap knob on the DTensor path**.

This is the section's main conclusion: **the MoE communication cannot be hidden.** Only
making it smaller could help.

**A3, 32 SMs (+3.8%).** DeepEP reserves SMs permanently. Raising the reservation took SMs
away from compute without speeding the transfer, so the all-to-all is not SM-starved at 20.

**A4, `dispatcher: torch` (+28.5%).** The first measurement of what DeepEP is *worth*.
Replacing the all-to-all with a plain all-gather costs 28.5%, so DeepEP is earning its
22.6% of wall clock rather than wasting it. A negative result that validates a default.

**A5, `experts: gmm` (+23.6%, then OOM).** This is what the config has been asking for all
along via the dead `enable_deepep` key. Its own process reached **76.39 GiB of 79.10 GiB**
while another user held only 932 MiB, so the OOM is real and not contention. The broken key
was, by luck, protecting the run: had it still worked, Track B would not fit in memory.

#### Result

**No DeepEP setting improves Track B.** The defaults are correct for this workload, and the
22.6% of wall clock spent in MoE all-to-all is the genuine cost of splitting 128 experts
across 8 GPUs at EP=8 — not a misconfiguration. `optim-O1` at 10.355% MFU stands.

The remaining opportunity in this lane is therefore not in MoE dispatch but in the FSDP
gradient path, which 3.6 measures.

---

### 3.6 Two defects in the FSDP gradient path — found and sized, not fixed

Both were found while investigating 3.5 and are larger than anything 3.5 produced. Neither
is implemented; both are stated with their measured size and their risk.

#### 3.6.1 `defer_fsdp_grad_sync` is plumbed and inert

`defer_fsdp_grad_sync` controls whether FSDP2 reduce-scatters gradients on every microbatch
or only the last one. It defaults to `True`. Following it through:

| step | location | what happens |
|---|---|---|
| NeMo-RL reads it | `nemo_rl/models/automodel/setup.py:529` | passed into `FSDP2Config` |
| exposed as config | `nemo_rl/models/policy/__init__.py:215` | `defer_fsdp_grad_sync: NotRequired[bool]` |
| Automodel stores it | `components/distributed/fsdp2.py:107` | `self.defer_fsdp_grad_sync = config.defer_fsdp_grad_sync` |
| …and never reads it back | | **that line is the only occurrence in the file** |
| the code that honours it | `components/distributed/utils.py:236` | `get_sync_ctx()` → `model.set_requires_gradient_sync(is_optim_step)` |
| call sites in `nemo_rl/` | — | **zero**, for both `get_sync_ctx` and `set_requires_gradient_sync` |

So FSDP2 never has its sync behaviour set and falls back to syncing on every backward. The
trace confirms it (`analysis/gradsync.py`, `analysis/rsbytes.py`, rank 0, Track B best):

```
_reduce_scatter_base   1014 calls   878.0 ms   Float
_allgather_base         715 calls   722.7 ms   BFloat16
```

Reduce-scatter fires as often as all-gather — once per microbatch, not once per step. The
same trace contains **2 optimizer steps** (`Optimizer.step#AdamW.step`) against **16 backward
passes**, i.e. **8 microbatches per optimizer step**.

**Size: 7 of every 8 reduce-scatters are redundant — ~768 ms of the 878 ms, against a
15,647 ms span = 4.9% of wall clock.** Larger than any arm in 3.5.

**Risk, stated up front:** deferring holds gradients unreduced across all 8 microbatches.
The docstring says so — *"saves comm, can increase peak memory."* This config already peaks
near 76 GiB of 79 GiB (3.5, arm A5), so this may simply OOM. That would be a legitimate
reason for it to be off — but nobody chose it: the flag reads `True` and is dead.

This supersedes the 3.4 bullet about `prepare_for_grad_accumulation` /
`prepare_for_final_backward`. That conclusion was right — gradients do reduce-scatter once
per microbatch — but the mechanism named was the wrong one. The live mechanism is
`defer_fsdp_grad_sync` / `get_sync_ctx`, and the defect is sharper than "never called": it
is plumbed end-to-end and inert.

#### 3.6.2 `reduce_dtype` is hardcoded to fp32

```python
# nemo_rl/models/automodel/setup.py:520
mp_policy=MixedPrecisionPolicy(
    param_dtype=dtype,                 # bf16
    reduce_dtype=torch.float32,        # <- hardcoded, no config knob
    output_dtype=STRING_TO_DTYPE[
        config["dtensor_cfg"].get("fsdp_output_dtype", "float32")
    ],                                 # <- has a knob, one line below
),
```

Measured like-for-like (`analysis/rsbytes.py`, rank 0, Track B best):

| collective | dtype | calls | ms | my shard | full tensor |
|---|---|---|---|---|---|
| `_allgather_base` | BFloat16 | 715 | 722.7 | 7.16 GB (in) | 57.28 GB (out) |
| `_reduce_scatter_base` | Float | 1014 | 878.0 | 14.32 GB (out) | **114.56 GB (in)** |

Comparing the full-tensor side of each: **114.56 / 57.28 = 2.00× exactly.** The entire
difference is the dtype. bf16 reduction would bring it to 57.28 GB.

> **Measurement caveat, which also applies to the table in 3.4.** All-gather and
> reduce-scatter report message sizes with opposite meanings: all-gather's *input* is one
> shard and its *output* is the full tensor, while reduce-scatter's *input* is the full
> tensor and its *output* is one shard. Reading "In msg nelems" for both compares a shard
> against a full tensor and manufactures a spurious factor of N (=8). The 3.4 figures of
> 4.4 GB against 70.5 GB carry that artifact; like-for-like the ratio there is also 2×, the
> dtype alone. The conclusion of 3.4 is unaffected — it rests on bandwidth and skew, not on
> this ratio — but the ratio should not be quoted as evidence of asymmetry.

**Size.** On Track B these collectives are far closer to bandwidth-bound than on Track A:

| | GB | ms | effective | % of ~450 GB/s NVLink |
|---|---|---|---|---|
| reduce-scatter | 114.56 | 878.0 | 130 GB/s | 29% |
| all-gather | 57.28 | 722.7 | 79 GB/s | 18% |

At 29% of wire, halving the bytes plausibly buys real time — unlike Track A's 2.3%, where
the original plan predicted ~20 ms and was right to. Reduce-scatter is 878 ms of a
15,647 ms span (5.6% of wall); halving bytes saves between very little and ~439 ms, i.e.
**1.3–2.8% of wall**. Not narrowed further without measuring.

**Numerics.** This is a real precision change, not a free win, and needs a loss gate rather
than a timing comparison. Direct evidence it is safe at this scale: Megatron's config on
this exact model and data already runs `grad_reduce_in_fp32: False` and reaches 20.530% MFU.

**Ordering.** 3.6.1 mostly subsumes 3.6.2. Deferring cuts reduce-scatter from 1014 calls /
114.56 GB to ~128 calls / 14.32 GB; bf16 then halves *that*. So 3.6.1 is worth ~768 ms and
3.6.2 ~439 ms **only if 3.6.1 does not land first**. Do 3.6.1 first; if it OOMs on memory,
3.6.2 becomes the fallback and keeps its full value.

## Status

- Run A (baseline) complete and recorded: 1.79 s median, 0.41% MFU.
- Run B (packing) complete but **invalid** — packing is incorrect on this model (3.1).
- Runs C/D (packing + larger global batch) started and **stopped**; they inherited the
  same defect. A batch-size sweep is still worth running *without* packing.
- Run E (no per-step `empty_cache`) gave 20% but OOM'd at step 14 (3.2).
- Run F (E + `expandable_segments`) is the first verified win: **33% faster, MFU 1.46×,
  50 steps, no OOM** (3.3). Two config flags, no new code.
- Fused AdamW (3.4) is the second: **6.6% faster median, MFU 0.580% -> 0.630%** on top of
  run F, loss unchanged within a measured noise floor. One config flag. The p90 figure
  first reported alongside it did not survive a repeat control and was withdrawn.
- Eight further collectives arms (3.4) produced no win: CUDA graphs, async dispatch and
  SM count are all inside the noise band, `dispatcher=torch` is worse, `hybridep` will
  not build. The remaining lever is kernel count per step, not the interconnect.
- A separate realistic-workload track exists in `~/assn1-f/configs/` (Nemotron-Cascade-2
  chat, 4k/8k/16k context, chunked cross-entropy, bf16 FSDP outputs, expandable_segments),
  with matching uncommitted changes across five files. Not yet folded into this document.
- HellaSwag **is** wired up (`nemo_rl/data/datasets/response_datasets/hellaswag.py`,
  registered as `hellaswag` and `hellaswag-completion`). 39,905 train examples, 50 steps
  clean, median 1.83 s. Sections 1-3.3 are SQuAD; 3.4 uses both.
- DeepEP on Track B (3.5) produced **no win**: five arms, every one a regression against a
  1.4% noise floor. The defaults are correct for this workload, and `dispatcher: torch`
  costing +28.5% is the first measurement of what DeepEP is worth. `optim-O1` stands at
  10.355% MFU. Two arms (`num_sms: 8`, `fake_balanced_gate`) were abandoned unmeasured
  because three other users took the GPUs repeatedly during the sweep.
- Two defects in the FSDP gradient path are found and sized but **not fixed** (3.6):
  `defer_fsdp_grad_sync` is plumbed end-to-end and inert (~4.9% of wall clock, with a real
  peak-memory risk), and `reduce_dtype` is hardcoded fp32 with no config knob (1.3-2.8%).
  These are the largest remaining items in this lane and both need code, not config.
- Nsight Systems data for Track B exists (`profiles/chat8k/nsys/`, 8 per-rank `.nsys-rep`
  plus GPU metrics, captured from `sft-nanov3-cascade2chat-8k-best-1n8g.yaml` at commit
  `0fa66b0fa5`). **No Nsight Compute report exists for any track** - DeepEP hits Xid 31 MMU
  faults under `ncu` kernel replay, and the `torchdispatch` workaround config produced none
  either.

### Still open

- ~~Whether bottleneck 2.4's 88% communication is real wire time or spin-waiting on
  peers.~~ **Answered in 3.4: spin-waiting.** All-gather runs at 2.3% of NVLink
  bandwidth while compute is balanced across ranks to 1.10x and collective wait varies
  2.1-4.4x. The collectives are barriers absorbing host launch jitter. The step is
  host-launch-bound: 27,879 GPU-idle gaps remain per trace even after the optimizer fix.
- Whether allocator fragmentation is real at all. The 8 memory snapshots (29 MB each in
  `profiles/memory/`) would settle it and have not been opened.
- Why elementwise work is 4.4× the real math (2.7). Needs `ncu`, which produced no report.
- Whether `defer_fsdp_grad_sync` can actually be honoured on this config, or whether holding
  8 microbatches of unreduced gradients OOMs (3.6.1). This is the largest unclaimed item in
  the collectives lane.
- Whether bf16 gradient reduction is worth its numerics cost here (3.6.2). Megatron already
  runs `grad_reduce_in_fp32: False` on this model and data, which is evidence but not proof.
- `dispatcher_num_sms: 8` and `fake_balanced_gate` (3.5) remain unmeasured. The second is a
  diagnostic that would bound how much of DeepEP's 3,538 ms is expert load imbalance, and so
  whether any traffic-reduction work is worth starting.
- ~~Whether the 56% communication figure indicates an asymmetry between all-gather and
  reduce-scatter volume.~~ **No: like-for-like the ratio is exactly 2.00x, the dtype alone
  (3.6.2).** The larger ratio quoted in 3.4 is an artifact of comparing a shard against a
  full tensor.
