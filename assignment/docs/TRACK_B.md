# Cascade-2 [Long Context Optimizations]

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data

Local subset: `/mnt/scratch/chloe_wong/data/cascade2/chat_20k_seed0.jsonl` — 20,000 rows
sampled seed 0 from the `Chat` domain (the full dataset is ~15.9M rows / 593 GB across 8
domains). Each row is `{domain, source, messages, generator}`; `messages` is a
system/user/assistant conversation whose assistant turn contains `<think>` reasoning traces.

**Training algorithm.** Supervised fine-tuning (SFT). Loss is token-level negative
log-likelihood — standard next-token cross-entropy — masked by `token_mask × sample_mask` so
only response tokens contribute, normalised by total valid tokens in the global batch
(`nemo_rl/algorithms/loss/loss_functions.py:1230`, `NLLLossFn`). Optimizer AdamW. No RL, no
DPO, no pretraining.

**Model.** `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`. 30B total parameters, ~3B active. A
hybrid of 23 Mamba2 layers, 23 Mixture-of-Experts layers (128 routed experts, top-6, plus 1
shared expert) and 6 attention layers. Vocabulary 131,072.

**Hardware.** 1 node, 8x H100 80GB SXM, NVLink NV18 full mesh (every GPU pair has 18 links,
no PCIe hop between any two GPUs).

---

## Why this dataset, when HellaSwag is shorter

Because HellaSwag is *too* short to measure a 30B model on, and we can prove it.

Token lengths, measured with the run's own tokenizer and chat template:

| dataset | mean | median | p90 | max |
|---|---|---|---|---|
| HellaSwag (choice) | 122 | 116 | 160 | 287 |
| SQuAD | 235 | 217 | 356 | 462 |
| **Cascade-2 chat** | **1,966** | 1,394 | 4,565 | 17,298 |

A training step has **fixed costs** that do not depend on how many tokens you feed it — the
FSDP all-gather of every weight (4.4 GB), the gradient reduce-scatter (70.5 GB), the optimizer
touching all 746 parameter tensors, and ~29,000 kernel launches. It also has **variable costs**
proportional to tokens: the actual matrix math.

MFU (Model FLOPs Utilization) = useful math / (peak FLOPs x time). When tokens are few,
`time ≈ fixed cost`, the math is negligible, and MFU approaches zero no matter how good the
kernels are.

Two measurements make this concrete:

1. **HellaSwag has 1.8x less arithmetic than SQuAD and takes the same step time** (1.83 s vs
   1.79 s). Halving the work changed nothing, because the work was never what we were paying
   for.
2. **The memory optimizations that are worth +67% on Cascade-2 are worth exactly zero on
   SQuAD** — 0.640% MFU and 50.9 TFLOPS with and without, identical to three significant
   figures.

So on HellaSwag we are benchmarking the harness, not the model. Measured ceiling on HellaSwag
after every optimization that works there: **0.805% MFU**. Cascade-2 reaches **20.5%**.

Cascade-2 is also the honest workload for this model: it is NVIDIA's own SFT data for the
Nemotron-Cascade models, it is long-form and reasoning-heavy, and it exercises the MoE routing
and long-context paths that dominate real fine-tuning.

**One caveat on the sequence cap.** The cap (`max_total_sequence_length`) truncates anything
longer, so it decides how much of the data is ever seen:

| cap | examples fitting whole | tokens discarded |
|---|---|---|
| 4,096 | 68.5% | **35.5%** |
| 8,192 | 88.8% | **14.2%** |
| 16,384 | 97.8% | 3.4% |

Raising the cap is therefore a data-quality change as well as a throughput change.

---

## Baseline

The first configuration that ran at all, after 16k and 8k both failed:

```
sequence cap            4,096
global batch size       128
micro batch size        1
dynamic batching        on, 4,096 tokens per microbatch
activation checkpointing on
parallelism             FSDP2 + expert-parallel 8, no tensor/pipeline/context parallel
backend                 DTensor (PyTorch-native sharding via Automodel)
```

**Result: 6.510 s/step, 5.195% MFU, 411 TFLOPS, 30 steps.**

Getting here took three attempts, and the failures are part of the baseline:

| attempt | cap | outcome |
|---|---|---|
| `16k-try1` | 16,384 | OOM at step 1 |
| `8k-try2` | 8,192 | OOM at step 1 |
| `4k-try3` | 4,096 | **ran, 5.195% MFU** |

At this baseline the run is discarding 35.5% of its training tokens. Everything that follows
is about affording a larger cap.

**Memory budget per GPU** (arithmetic; consistent with observed OOM figures of ~71 GB
allocated):

| | per GPU |
|---|---|
| parameters, bf16, sharded /8 | 7.5 GB |
| AdamW moments, fp32, sharded /8 | 30.0 GB |
| gradients, fp32, sharded /8 | 15.0 GB |
| **fixed subtotal** | **52.5 GB** |
| **left for activations + logits** | **27.5 GB** |

That 27.5 GB is the entire budget every optimization below is fighting over. Note that
**global batch size costs no memory** — with micro-batch 1 and gradient accumulation,
activations are freed after each microbatch, so peak memory is set by tokens per *microbatch*,
not by the global batch.

---

## Profiling Results

Profiled with the torch profiler (Kineto), nsys, and Holistic Trace Analysis on the tuned 8k
DTensor configuration. Artefacts:

- `profiles/chat8k/nsys/*.nsys-rep` — open in Nsight Systems for the timeline screenshots
- `profiles/chat8k/torch/kineto/rank{0..7}.pt.trace.json.gz` — open in Perfetto
- `profiles/chat8k/analysis/` — HTA summaries, nsys summary, memory summary
- `profiles/megatron8k/` — the same for the Megatron backend

*(Nsight screenshots to be inserted here: the per-stream timeline showing DeepEP dispatch/combine
on its own stream against the compute stream, and the exposed-communication gaps.)*

### Where the time goes — rank 0, one step window (nsys)

| | ms | share of busy |
|---|---|---|
| window | 14,586.6 | |
| busy | 10,088.0 | **69.2%** |
| idle | 4,498.7 | 30.8% |
| compute | 6,039.5 | |
| communication | 4,277.5 | |
| **communication overlapped with compute** | **229.1** | **5.4% of comm** |
| **exposed communication** | **4,048.5** | **94.6% of comm** |

By kernel category:

| category | ms |
|---|---|
| **DeepEP (MoE all-to-all)** | **3,229.9** |
| Elementwise / other | 1,676.9 |
| Dense GEMM | 1,656.1 |
| NCCL (FSDP all-gather + reduce-scatter) | 1,608.3 |
| MoE expert GEMM | 1,328.7 |
| Mamba | 568.4 |
| Copy / cat | 344.2 |
| MoE routing | 242.3 |
| Attention | 148.7 |
| Norm | 97.8 |

Real math (dense GEMM + expert GEMM + Mamba + attention) = **3,702 ms**.
Communication (DeepEP + NCCL) = **4,838 ms**.

### Per-rank balance (all 8 ranks, Kineto)

Every rank issues the same collectives on the same byte counts, so spread in collective *time*
across ranks is waiting. GEMM and elementwise time are the control — identical work on every
rank, so their spread is the measurement noise floor.

| bucket | min ms | max ms | spread |
|---|---|---|---|
| GEMM | 3,115.1 | 3,302.8 | **1.06x** (control) |
| elementwise | 2,471.7 | 2,516.8 | **1.02x** (control) |
| NCCL | 1,217.2 | 2,037.0 | 1.67x |
| DeepEP | 2,728.4 | 3,538.4 | 1.30x |

### Idle composition (HTA)

Idle is 33.3% of the step. On the compute stream it is **78% `host_wait`**; on the
communication stream **99% `host_wait`**. The GPU is waiting on the CPU, not on other GPUs.

### Reading of the profile

1. **The MoE all-to-all is the single largest kernel category** — larger than dense GEMM. This
   is expert parallelism: with 128 experts split 16-per-GPU, every token must travel to
   whichever GPU owns its expert and come back.
2. **Communication is almost entirely exposed** — only 5.4% of it is hidden behind compute.
   This is the largest single opportunity on this track.
3. **Load is balanced.** Compute spread is 1.06x across ranks, so this is not a straggler
   problem the way the short-sequence workload was (where spread reached 4.36x).
4. **Still partly host-bound** — idle is overwhelmingly `host_wait`.

For contrast, the same profile on the short HellaSwag/SQuAD workload shows GEMM at 5.9% of
kernel time and collective spread at 3.2–4.4x. Cascade-2 is a fundamentally healthier run:
compute is 27% of kernel time and load is balanced. The bottleneck genuinely moved.

---

## Diagnosis, Interventions and Optimizations

Every number below is 30 steps, validation and checkpointing off, step 1 discarded.

### Optimization 1 — Chunked cross-entropy + bf16 FSDP output

**Diagnosis.** The final layer emits *logits*: one score per vocabulary entry per token. With
a 131,072-entry vocabulary and 8,192 tokens per microbatch, that is a single **4.29 GB**
tensor in fp32 — 16% of the entire 27.5 GB activation budget, with several copies live at
once (logits, softmax, gradient). This is why 8k OOM'd at step 1.

**Intervention.** Two coupled settings:
- `fsdp_output_dtype: bfloat16` stores logits (and checkpointed block outputs) in 2 bytes
  instead of 4: 4.29 GB -> 2.15 GB.
- `logprob_chunk_size: 1024` computes the loss 1,024 tokens at a time and discards each
  chunk's logits, so the peak is 0.54 GB rather than 4.29 GB. It also keeps each chunk's loss
  in fp32, which is what makes the bf16 storage numerically safe. They are a pair, not two
  independent options.

**Measured.**

| | MFU |
|---|---|
| 8k + chunked CE | 5.685% |
| **+ bf16 FSDP output** | **9.500% (+67%)** |
| same pair at 4k (control) | 4.985% vs 5.195% without — slightly worse |

The 4k control is the important one: the win is not free, it scales with sequence length. At
4k there is not enough memory pressure to pay for the extra work. This is the largest
kernel-level win on the track.

### Optimization 2 — Allocator: expandable segments + no per-step empty_cache

**Diagnosis.** PyTorch does not call CUDA for every allocation; it grabs large *segments* and
sub-allocates. When tensor shapes vary — and dynamic batching varies them every step —
segments **fragment**: enough free memory in total, no single contiguous block. Upstream's fix
was to call `torch.cuda.empty_cache()` every step, which hands everything back to CUDA and
re-acquires it next step. Profiling measured that at **30% of host CPU time**.
`expandable_segments:True` instead lets a segment grow and shrink in place, so the
fragmentation largely stops happening.

**Intervention and measurement.** A full 2x2, which is what makes the conclusion trustworthy:

| | expandable_segments | empty_cache | MFU | outcome |
|---|---|---|---|---|
| A0 | off | on | 9.295% | OOM at step 22 |
| A1 | off | **off** | 9.290% | OOM at step 3 |
| **A2** | **on** | **off** | **9.945%** | **30 steps** |
| A3 | on | on | 7.645% | 30 steps, 23% slower |

**Conclusion: neither flag works alone.** Without expandable segments the run OOMs; with them
but `empty_cache` still on you pay 23% for defragmentation you no longer need.
`expandable_segments` removes the *reason* `empty_cache` existed, and only then can it be
deleted.

### Optimization 3 — Fused optimizer

**Diagnosis.** AdamW by default updates each parameter tensor with ~8 small elementwise
operations. With 746 parameter tensors that is ~5,968 kernel launches per optimizer step, plus
one `.item()` call per tensor — **756 GPU-to-CPU synchronisations**, each of which drains the
CUDA launch queue and restarts the pipeline from empty. Upstream `sft.yaml` states "when using
Dtensor, we need to set foreach and fused to False".

**Intervention.** `optimizer.fused: true`. Verified on the profile: kernels inside
`Optimizer.step` fall from **5,968 to 22**, host time from 298 ms to 21 ms, and `.item()`
calls inside the optimizer from **756 to 0**.

**Measured.**

| | MFU | step |
|---|---|---|
| baseline | 9.720% | 6.730 s |
| **fused** | **10.355%** | **6.530 s** |
| foreach | 9.655% | 7.100 s |

The upstream comment is half wrong: `fused` runs correctly under DTensor and is faster;
`foreach` is worse here and OOMs on the short-sequence workload. Independently reproduced on
the SQuAD workload at the same +6.6%.

**Best DTensor result: 10.355% MFU, 820 TFLOPS.**

### Optimization 4 — Change the backend: DTensor to Megatron

**Definitions.** The model is 30B parameters and cannot fit on one 80 GB GPU, so it must be
split across all 8. There are two implementations of that splitting:

- **DTensor** — built into PyTorch. A tensor that knows it is sharded; PyTorch inserts the
  GPU-to-GPU communication automatically. Generic, works on any model, easy to modify.
- **Megatron** — NVIDIA's separate library where the parallelism is hand-written into the
  model implementation, with communication hand-placed and deliberately overlapped.
  Specialised and hand-tuned, but a model must be ported to it first.

One config flag switches them; model, data, batch size and GPUs stay identical.

**Diagnosis of why Megatron is faster.** It is not mysterious tuning. Its config does four
things the DTensor path does not:

| | DTensor path | Megatron |
|---|---|---|
| gradient reduction precision | **fp32** (hardcoded, `setup.py:522`, no config knob) | **bf16** (`grad_reduce_in_fp32: False`) |
| optimizer moment precision | **fp32** | **bf16** (`exp_avg_dtype`, `exp_avg_sq_dtype`) |
| gradient reduce overlapped with backward | **no** — the deferral hooks exist in Automodel and NeMo-RL never calls them | `overlap_grad_reduce: True` |
| weight gather overlapped with forward | 5.4% of comm overlapped | `overlap_param_gather: True` |
| sequence packing | **silently incorrect** (see below) | `even_bin_count`, works |

The first two change the memory budget outright:

| fixed overhead per GPU | DTensor | Megatron |
|---|---|---|
| parameters bf16 | 7.5 GB | 7.5 GB |
| optimizer moments | 30.0 GB (fp32) | **15.0 GB** (bf16) |
| gradients | 15.0 GB (fp32) | **7.5 GB** (bf16) |
| fixed total | 52.5 GB | **30.0 GB** |
| **left for activations** | **27.5 GB** | **50.0 GB** |

**Megatron has ~1.8x more activation headroom before doing anything clever.** That is why it
packs 16,384 tokens per microbatch where DTensor fights at 8,192.

**Measured.**

| run | change | MFU | TFLOPS | step |
|---|---|---|---|---|
| `megatron-try8` | first stable Megatron | 15.615% | 1236.0 | 4.570 s |
| `meg_deepep` | deepep dispatcher | 13.985% | 1107.0 | 5.140 s |
| **`meg_evenpack`** | `even_bin_count` packing | **18.985%** | 1503.0 | 3.620 s |
| **`meg_evenpack_seo`** | + `moe_shared_expert_overlap` | **19.280%** | 1526.5 | **3.590 s** |
| `meg_evenpack_gbs512` | global batch 512 | **20.530%** | 1624.9 | 13.360 s |
| `meg_evenpack_mb8k` | packing budget 8k not 16k | 15.190% | 1202.3 | 4.610 s |

Porting cost six failed attempts (ActorDied, TypeError, OOM x4) before `try8` ran.

**Sequence packing, and an important correctness note.** Packing concatenates several short
documents into one microbatch so the GPU works on real tokens instead of padding. On the
**DTensor** path we proved it is *silently incorrect*: a probe showed the packed batch arrives
as `(1, 2048, 2688)` with `qkv_format=bshd` and **`cu_seqlens` absent**, and Mamba's `seq_idx`
is `None` — roughly 9 documents concatenated with no boundary information, so every token can
attend across document boundaries. Loss came out 3.3% *lower*, which is the direction a leak
produces. It is a correctness bug, not a speedup. On **Megatron** packing works and is worth
**+36%** (13.985% -> 18.985%).

### Optimization 5 — DeepEP: the MoE all-to-all, and why none of its knobs help

This addresses open item 4 above: exposed communication is 94.6% of communication, and DeepEP
is the largest kernel category on the profile. It is the one thing on this track that had never
been varied.

**Definition.** *DeepEP* is NVIDIA's library for the one data movement Mixture-of-Experts needs.
The 128 experts in each MoE layer are split 16-per-GPU (`expert_parallel_size: 8`). Each token
is routed to 6 experts, most of which live on other GPUs, so every MoE layer must **dispatch**
tokens to the GPUs owning their experts and **combine** the results back. Both are *all-to-all*
collectives: every GPU sends different data to every other GPU, in irregular sizes, because
routing is data-dependent.

**A config defect found before any experiment.** The config requests DeepEP through a key the
code deleted:

```yaml
backend: { experts: torch_mm, enable_deepep: True }
```
```
run.log:52  WARNING: enable_deepep is no longer supported and is ignored. Set 'dispatcher'
            and 'experts' explicitly instead. Previously enable_deepep=True was equivalent
            to experts=gmm + dispatcher=deepep.
```

No `dispatcher` key exists anywhere in this track's config chain, so every run used
`BackendConfig`'s dataclass defaults:

| knob | default in force | varied before this section |
|---|---|---|
| `dispatcher` | `deepep` (resolved from `HAVE_DEEP_EP`) | no |
| `dispatcher_num_sms` | **20** of 132 SMs | no |
| `dispatcher_async_dispatch` | **False** | no |
| `experts` | `torch_mm` | config asked for `gmm`, never got it |

Two consequences. DeepEP **was** always running — the default happens to resolve to it, so the
dead key cost only the `experts=gmm` half. And no DeepEP setting had ever changed across ~30
runs on this track: it was a constant in every experiment, never a variable, so nothing in the
results above says what it costs or what it would do differently.

**Intervention.** Each arm is the best config (`optim-O1`) with **exactly one change**, 30 steps,
step 1 discarded, compared over a matched step range. `dep-A0-repeat` is the unchanged config
run again — the noise floor, included because a 7% p90 "win" claimed earlier on the short-sequence
track did not survive its repeat control.

| arm | change | median | p90 | vs control |
|---|---|---|---|---|
| `optim-O1` | earlier run of the same config | 6.530 s | 7.690 | −1.4% ← **noise floor** |
| `dep-A0-repeat` | **(control)** | 6.620 s | 8.040 | — |
| `dep-A3-sms32` | `dispatcher_num_sms: 32` | 6.870 s | 8.080 | +3.8% |
| `dep-A1-async` | `dispatcher_async_dispatch: true` | 7.030 s | 7.760 | +6.2% |
| `dep-A5-gmm` | `experts: gmm` | 8.180 s | 10.100 | +23.6%, **OOM at step 22** |
| `dep-A4-torchdisp` | `dispatcher: torch` | 8.510 s | 9.860 | +28.5% |

Total floating-point work matched across arms to within 1% (Σ TFLOPS×time = 150,103–151,683), so
median step time is the fair comparison.

**Measured: no DeepEP setting improves this track.** Three results are worth keeping regardless.

*Async dispatch made it worse, which settles the overlap question.* The hypothesis was that
DeepEP's 0.5% overlap is caused by the dispatch blocking the compute stream — `async_finish=False`
makes the caller wait and `allocate_on_comm_stream=False` forces a sync back. Setting both true
removed the block and cost 6.2%. The reason is that within one MoE layer the chain
`route → dispatch → expert GEMM → combine` is genuinely serial: the GEMM needs the dispatched
tokens. There is no independent work to slide into the gap, so unblocking the stream buys nothing
and costs bookkeeping plus longer-lived comm-stream buffers. Megatron pairs the equivalent with
`moe_shared_expert_overlap` (worth +1.6% there, 18.985% → 19.280%); `grep` finds **no
shared-expert-overlap knob on the DTensor path**. **This communication cannot be hidden. Only
making it smaller could help.**

*Turning DeepEP off costs 28.5%*, which is the first measurement of what it is worth. It earns
its share of the runtime rather than wasting it.

*`experts: gmm` OOM'd at 76.39 GiB of 79.10 GiB* with only 932 MiB held by another user, so the
failure is real and not contention. That is the setting the config has been silently asking for;
the dead key was accidentally protecting the run.

Not measured: `dispatcher_num_sms: 8` and `fake_balanced_gate` (a diagnostic that would bound how
much of DeepEP's time is expert load imbalance). Three other users took all 8 GPUs repeatedly
during this sweep and both arms OOM'd against foreign processes holding 17–20 GiB. They are
reported as not run rather than measured under contention.

### The FSDP gradient path — two defects, found and sized, not fixed

Found while investigating Optimization 5. Both are larger than anything it produced. Neither is
implemented; each is stated with its measured size and its risk.

**Defect 1: `defer_fsdp_grad_sync` is plumbed end-to-end and inert.** The flag controls whether
FSDP2 reduce-scatters gradients on every micro-batch or only the last one, and defaults to `True`:

| step | location | what happens |
|---|---|---|
| NeMo-RL reads it | `setup.py:529` | passed into `FSDP2Config` |
| exposed as config | `policy/__init__.py:215` | `defer_fsdp_grad_sync: NotRequired[bool]` |
| Automodel stores it | `fsdp2.py:107` | `self.defer_fsdp_grad_sync = config.defer_fsdp_grad_sync` |
| …and never reads it back | | **that line is its only occurrence in the file** |
| the code that honours it | `utils.py:236` | `get_sync_ctx()` → `set_requires_gradient_sync(is_optim_step)` |
| call sites in `nemo_rl/` | — | **zero**, for both functions |

FSDP2 therefore never has its sync behaviour set and falls back to syncing on every backward. The
trace confirms it: `_reduce_scatter_base` fires 1014 times against `_allgather_base`'s 715, i.e.
once per micro-batch rather than once per step, while the same trace holds **2 optimizer steps
against 16 backward passes — 8 micro-batches per optimizer step**.

**Size: 7 of every 8 reduce-scatters are redundant — ~768 ms of 878 ms, against a 15,647 ms span
= 4.9% of wall clock.** **Risk:** deferring holds gradients unreduced across all 8 micro-batches;
the docstring says *"saves comm, can increase peak memory"*, and this config already peaks near
76 GiB of 79 GiB, so it may simply OOM. That would be a legitimate reason for it to be off — but
nobody chose it: the flag reads `True` and is dead.

**Defect 2: `reduce_dtype` is hardcoded to fp32** (`setup.py:522`), with no config knob, while
`output_dtype` one line below has one:

| collective | dtype | calls | ms | full-tensor side |
|---|---|---|---|---|
| `_allgather_base` | BFloat16 | 715 | 722.7 | 57.28 GB |
| `_reduce_scatter_base` | Float | 1014 | 878.0 | **114.56 GB** |

Like-for-like the ratio is **2.00× exactly** — the dtype alone. On this track these collectives run
at 29% of NVLink wire speed (130 GB/s of ~450), so unlike the short-sequence track they are
genuinely bandwidth-sensitive and halving the bytes should buy real time: **1.3–2.8% of wall**.
This is a real precision change and needs a loss gate, not a timing comparison; the evidence that
it is safe is that Megatron's config on this exact model and data already runs
`grad_reduce_in_fp32: False` and reaches 20.530% MFU.

> **A measurement caveat that applies to any collective-volume figure.** All-gather and
> reduce-scatter report message sizes with opposite meanings: all-gather's *input* is one shard
> and its *output* the full tensor, while reduce-scatter's *input* is the full tensor and its
> *output* one shard. Reading "In msg nelems" for both compares a shard against a full tensor and
> manufactures a spurious factor of 8. Volume comparisons here use the full-tensor side of each.

**Ordering.** Defect 1 mostly subsumes Defect 2: deferring cuts reduce-scatter from 1014 calls /
114.56 GB to ~128 calls / 14.32 GB, and bf16 then halves *that*. Do Defect 1 first; if it OOMs,
Defect 2 becomes the fallback and keeps its full value.

### Quantization

**Definition.** Quantization stores and multiplies numbers in fewer bits. Training here is
bf16 (16-bit). **FP8** is 8-bit floating point — the `e4m3` format is 1 sign bit, 4 exponent
bits, 3 mantissa bits. Halving the bits halves memory traffic for those tensors and lets the
H100's tensor cores run roughly twice the throughput on the matrix multiplies. The cost is
precision, managed by *scaling*: values are multiplied into fp8's representable range before
the multiply and scaled back after. `blockwise` recipe means a separate scale per block of the
tensor rather than one per tensor, which handles outliers better.

**Status: attempted four times, never successfully run.** All four attempts OOM at step 0.

| run | configuration | outcome |
|---|---|---|
| `meg_evenpack_fp8` | `fp8: e4m3, fp8_recipe: blockwise, fp8_param: false`, `moe_router_dtype: fp32`, `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` | **OOM at step 0** |
| `meg_fp8_mb8k` | same, packing budget 8k | **OOM at step 0** |
| `meg_fp8param` | same + `fp8_param: true` | **OOM at step 0** |
| `meg_fp8_nvls0` | same + `NCCL_NVLS_ENABLE=0` | **OOM at step 0** |

**Diagnosis (hypothesis, untested).** Failing at step 0 rather than partway means the failure
is in setup, not in steady-state activation growth. Blockwise fp8 allocates scale factors and
fp8 copies of the weights *in addition to* the bf16 master copies; with `fp8_param: false` the
bf16 parameters are retained, so the fp8 path is additive rather than a replacement. The
natural next step is to reduce the packing budget and re-measure the memory high-water mark
before concluding fp8 does not fit.

**Why it matters.** On top of 19.280% MFU, fp8 is the only remaining intervention on this
track with a plausible path past 25%. It is the largest open item.

---

## Summary

| stage | MFU | TFLOPS | tokens discarded |
|---|---|---|---|
| 16k / 8k first attempts | crashed | — | — |
| **4k baseline** | 5.195% | 411 | 35.5% |
| 8k + chunked CE | 5.685% | 450 | 14.2% |
| + bf16 FSDP output | 9.500% | 752 | 14.2% |
| + allocator pair | 9.945% | 788 | 14.2% |
| + fused optimizer | **10.355%** | 820 | 14.2% |
| **+ Megatron backend, even packing, shared-expert overlap** | **19.280%** | 1526 | 14.2% |
| Megatron + global batch 512 | **20.530%** | 1625 | 14.2% |

**Baseline to best: 5.195% -> 20.530% MFU, a 3.95x improvement; 411 -> 1625 TFLOPS.**

DeepEP tuning (Optimization 5) adds no row: five arms, five regressions against a 1.4% noise
floor. It is recorded because it establishes what DeepEP is worth (+28.5% to remove), rules out
overlapping the MoE communication, and is what surfaced the two FSDP gradient-path defects.

## Open items, ranked

1. **FP8** — four attempts, all OOM at step 0, never run. The only plausible path past 25%.
2. **`hybridep` dispatcher** — three attempts, all ActorDiedError. Also gates CUDA-graph
   capture of `moe_preprocess`.
3. **16k context** — two attempts, both OOM. Would cut tokens discarded from 14.2% to 3.4%.
   The fp32 gradient hardcode on the DTensor path is 7.5 GB/GPU of avoidable waste and is a
   plausible blocker.
4. ~~**Exposed communication** — 94.6% not overlapped, DeepEP the largest kernel category.~~
   **Partly answered by Optimization 5: it cannot be overlapped.** The MoE chain
   `route -> dispatch -> expert GEMM -> combine` is serial, async dispatch costs 6.2%, and the
   DTensor path has no shared-expert-overlap knob to fill the gap. What remains is the **FSDP
   gradient path**: `defer_fsdp_grad_sync` is inert (~4.9% of wall clock) and `reduce_dtype` is
   hardcoded fp32 (1.3-2.8%). Both need code, not config, and the first carries an OOM risk on a
   config already peaking near 76 GiB of 79 GiB. These are now the largest levers on this track
   short of fp8.
5. **CPU optimizer offload** — dies at step 9 and step 6 with RuntimeError, not OOM, so
   probably fixable.
6. **Verify the tokenizer** — the track trains `A3B-Base-BF16` but tokenizes with `A3B-BF16`
   (the instruct variant) to obtain a usable chat template. Defensible, but a vocabulary or
   special-token mismatch would corrupt training silently rather than raise an error.

---

## Reproducibility Notes

**Seeds and determinism.** `sft.seed: 42` and `data.shuffle: true` fix the batch order, so every
arm sees the identical sequence of batches; this is what makes matched-step comparisons valid and
was checked directly (total floating-point work agreed across arms to within 1%). Runs are *not*
bit-deterministic: collective reduction order varies, and dynamic batching packs by token budget.
Two runs of the identical config differ by **1.4% on median step time and 1.3% on MFU** — that is
the measured noise floor (`dep-A0-repeat` vs `optim-O1`), and no result below it is reported as
an effect.

**Data.** `chat_20k_seed0.jsonl`, a 20,000-row seed-0 sample of the `Chat` domain of
<https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data>, at
`/mnt/scratch/chloe_wong/data/cascade2/`. Matched to the upstream dataset by schema (`domain`,
`source`, `messages`, `generator`) and name; **no download script or HF cache entry exists on the
box, so the provenance is inferred rather than recorded.** Measured token lengths under this
tokenizer and template: mean 1,966, median 1,394, p90 4,565.

**Environment.** 1 node, 8×H100 80GB SXM, NVLink NV18 full mesh. NeMo-RL at
`~/assn1-f/RL` (branch `assignment1`), Automodel as an editable install at
`RL/3rdparty/Automodel-workspace/Automodel` (0.6.1+72daceff), worker venvs under
`/mnt/scratch/chloe_wong/venvs/`. Profiling artefacts record `git_commit.txt` and
`git_diff.patch` per run; the profiled configuration is commit `0fa66b0fa5`.

**End-to-end run.**

```bash
# baseline (4k)
CONFIG=~/assn1-f/configs/sft-nanov3-cascade2chat-4k-1n8g.yaml \
EXP_NAME=baseline bash ~/assn1-f/run_sft_nanov3_realistic.sh

# best DTensor result (10.355% MFU)
CONFIG=~/assn1-f/configs/sft-nanov3-cascade2chat-8k-best-1n8g.yaml \
EXP_NAME=optim-O1 bash ~/assn1-f/run_sft_nanov3_realistic.sh

# best overall (20.530% MFU, Megatron backend)
CONFIG=~/assn1-f/configs/sft-nanov3-cascade2chat-8k-meg-evenpack-gbs512-1n8g.yaml \
EXP_NAME=meg_evenpack_gbs512 bash ~/assn1-f/run_sft_nanov3_realistic.sh

# profiling
PROFILE_TAG=chat8k CONFIG=~/assn1-f/configs/sft-nanov3-cascade2chat-8k-best-1n8g.yaml \
  bash ~/assn1-f/profile_sft_nanov3.sh nsys    # per-rank timelines + GPU metrics
  bash ~/assn1-f/profile_sft_nanov3.sh torch   # Kineto traces (all analysis below uses these)

# analysis (RL/analysis/, committed)
python analysis/overlap.py  /mnt/scratch/chloe_wong/profiles/chat8k/torch/kineto
python analysis/rsbytes.py  /mnt/scratch/chloe_wong/profiles/chat8k/torch/kineto
python analysis/gradsync.py /mnt/scratch/chloe_wong/profiles/chat8k/torch/kineto
python analysis/dep_report.py
```

**Caveats a reader should know.**

1. **The box is shared.** Three other users ran jobs during the DeepEP sweep, and several arms
   OOM'd against foreign processes holding 17–20 GiB. Those are reported as *not run*, never as
   results. Any run of this work should check `nvidia-smi` for foreign processes first —
   `analysis/gpu_cleanup.sh` does this and refuses to measure on a dirty box.
2. **Ray does not reliably free GPU memory between runs.** The first sweep launched each arm on
   top of the previous arm's workers. Tear down and wait for the GPUs to drain between arms.
3. **This config peaks near 76 GiB of 79 GiB.** Memory-affecting changes are one step from OOM,
   and an OOM should be checked against how much another process was holding before it is
   attributed to the setting under test.
4. **Tokenizer/model mismatch, unverified.** The track trains `A3B-Base-BF16` but tokenizes with
   `A3B-BF16` (the instruct variant) to obtain a usable chat template. The run log also warns
   `pad_token_id=None; defaulting to 0`, which it notes "can cause incorrect MoE auxiliary loss
   calculations if valid tokens share token ID 0". Defensible, but a vocabulary or special-token
   mismatch would corrupt training silently rather than raise an error. Not yet checked.
5. **14.2% of tokens are discarded** at the 8k cap. Comparisons across sequence lengths are
   therefore not comparisons on identical data; comparisons *within* 8k are.
6. **Not under version control.** `~/assn1-f` is not a git repository. `PROCESS.md`,
   `RESULTS.md`, this document, and the entire `configs/` directory — including the
   `sft-nanov3-cascade2chat-*` chain that produces every number here — exist only on that box's
   disk. Only `~/assn1-f/RL` is versioned. **This is the largest reproducibility gap in the
   project.**
7. **No Nsight Compute data exists, on any track.** DeepEP hits Xid 31 MMU faults under `ncu`
   kernel replay on every rank, and the `torchdispatch` workaround config produced no report
   either. Per-kernel SM/memory analysis is therefore unavailable, not merely missing.
