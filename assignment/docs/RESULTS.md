# All optimizations and results — EE290/194 Part B

Consolidated from every run on this box, 2026-09-28. Three tracks ran in parallel on
different workloads, which is the main reason the numbers look confusing: **the same
optimization is worth wildly different amounts depending on which workload it is measured
on.**

Sources: `/mnt/scratch/chloe_wong/runs/*/run.log` (67 runs),
`/home/palshraya/Automodel/traces/*.txt` (teammate).

---

## The headline

| workload | seq | global batch | best MFU | best TFLOPS |
|---|---|---|---|---|
| SQuAD (diagnostic) | 2048 | 16 | **0.64%** | 50.9 |
| HellaSwag (assigned) | 2048 | 16 | **0.80%** | 63.5 |
| Cascade-2 chat, DTensor | 8192 | 128 | **10.36%** | 819.6 |
| Cascade-2 chat, **Megatron** | 8192 | 128 | **19.28%** | **1526.5** |

**A 16× MFU difference, and almost none of it came from kernels.** It came from feeding the
GPUs a realistic workload: longer sequences, a bigger batch, dynamic batching, activation
checkpointing. Every kernel-level fix in this document is worth a few percent. The workload
shape was worth 16×.

That is the single most important result, and it is why the SQuAD number should be presented
as a *diagnostic baseline*, not as the system's performance.

---

## Track A — SQuAD 2048/gbs16 (the diagnostic workload)

Deliberately small and starved, which is what made the bottlenecks legible. All 50-step
runs, validation and checkpointing off, step 1 discarded.

| # | change | median step | MFU | TFLOPS | verdict |
|---|---|---|---|---|---|
| A | baseline | 1.790 s | 0.410% | 32.3 | reference |
| B | sequence packing | 1.450 s | 0.500% | 39.5 | **INVALID** — silently incorrect, see below |
| E | no per-step `empty_cache()` | 1.420 s | 0.530% | 42.2 | OOM at step 14 |
| F | E + `expandable_segments:True` | **1.200 s** | **0.600%** | 47.6 | **real, 1.49×** |
| G0 | F re-measured as optimizer baseline | 1.220 s | 0.580% | 46.0 | control |
| G0b | G0 repeated (noise control) | 1.220 s | 0.590% | 46.9 | median repeats exactly |
| G1 | `optimizer.foreach=true` | — | — | — | **OOM** |
| G2 | `optimizer.fused=true` | **1.140 s** | **0.635%** | 50.4 | **real, 6.6%** |
| J0 | G2 repeated (noise control) | 1.140 s | 0.640% | 50.9 | median repeats exactly |
| I1 | + CUDA graph `[moe_router]` | 1.140 s | 0.625% | 49.4 | no effect |
| I5 | + `dispatcher_async_dispatch` | 1.150 s | 0.630% | 50.1 | no effect |
| J1 | + both | 1.140 s | 0.640% | 50.7 | no effect |
| J2 | + both + `dispatcher_num_sms=8` | 1.150 s | 0.630% | 49.7 | no effect |
| I4 | `dispatcher=torch` | 1.230 s | 0.580% | 45.7 | **worse** |
| I2/I3 | `dispatcher=hybridep` | — | — | — | nvcc build failure |
| H1/H2 | CUDA graph `[moe_preprocess]`/`[attn]` | — | — | — | unmet prerequisites |
| — | HellaSwag instead of SQuAD | 1.830 s | 0.485% | 38.4 | workload swap, not an optimization |

**Track A total: 1.790 → 1.140 s (1.57× faster), MFU 0.410% → 0.640% (1.56×).**

Two config flags did all of it: `empty_cache_after_train=false` +
`expandable_segments:True` (F), and `optimizer.fused=true` (G2).

### Why B is thrown out

Sequence packing looked like a 19% win. A probe showed the packed batch arrives as
`(1, 2048, 2688)` with `qkv_format=bshd` and **`cu_seqlens` absent**, and Mamba's
`seq_idx` is `None`. About 9 documents are concatenated with no boundary information, so
every token attends across document boundaries. Loss came out 3.3% *lower* — the direction
a leak produces. Not a speedup; a correctness bug. Runs C and D inherited it and were
stopped.

### Why the "no effect" rows are stated that confidently

The repeat controls are the point. Median repeats *exactly* between identical runs
(1.220/1.220 and 1.140/1.140), but p90 swings **1.780 → 1.510** and **1.560 → 1.440**
between identical runs. An earlier claim that CUDA graphs and async dispatch each cut p90
by 7% did not survive the repeat baseline and was withdrawn. Only median is trustworthy at
this sample size.

---

## Track B — Cascade-2 chat 4k/8k/16k, gbs 128 (the realistic workload)

This is where the real performance is. 30-step runs, dynamic batching and activation
checkpointing on throughout.

| run | seq | chunked CE | bf16 FSDP out | expandable | no empty_cache | fused | median | MFU | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| 4k-try3 | 4096 | | | | | | 6.510 s | 5.20% | 411 |
| 4k-chunkedce-bf16out | 4096 | ✓ | ✓ | | | | 6.310 s | 4.99% | 395 |
| 8k-chunkedce | 8192 | ✓ | | | | | 8.020 s | 5.69% | 450 |
| 8k-chunkedce-expseg | 8192 | ✓ | | ✓ | | | 9.670 s | 7.14% | 565 |
| 8k-chunkedce-bf16out | 8192 | ✓ | ✓ | | | | 7.160 s | 9.50% | 752 |
| 8k-memfix / A3 | 8192 | ✓ | ✓ | ✓ | | | 8.920 s | 7.65% | 605 |
| alloc-A0 | 8192 | ✓ | ✓ | | | | 7.280 s | 9.30% | 736 |
| alloc-A1 | 8192 | ✓ | ✓ | | ✓ | | 7.930 s | 9.29% | 735 |
| **alloc-A2** | 8192 | ✓ | ✓ | ✓ | ✓ | | **6.760 s** | **9.95%** | 788 |
| optim-O0 (= A2) | 8192 | ✓ | ✓ | ✓ | ✓ | | 6.730 s | 9.72% | 770 |
| **optim-O1** | 8192 | ✓ | ✓ | ✓ | ✓ | ✓ | **6.530 s** | **10.36%** | **820** |
| optim-O2 (`foreach`) | 8192 | ✓ | ✓ | ✓ | ✓ | foreach | 7.100 s | 9.66% | 765 |
| 4k-gbs512 | 4096 | | | | | | 20.310 s | 6.64% | 526 |
| 16k-memfix | 16384 | ✓ | ✓ | ✓ | | | crashed at step 1 | 2.35% | 186 |

**Track B best: `optim-O1` — 10.36% MFU, 820 TFLOPS, 6.53 s/step.**

What each ingredient was worth, reading the table:

- **4k → 8k sequence length**: 5.20% → 5.69% MFU
- **bf16 FSDP output** (`fsdp_output_dtype: bfloat16`): 5.69% → 9.50%. **The single largest
  kernel-level win in the whole project (+67%).**
- **`expandable_segments` + no `empty_cache`** (A2 vs A3): 7.65% → 9.95% (**+30%**) — the
  same finding as Track A's run F, but worth far more here
- **fused optimizer** (O1 vs O0): 9.72% → 10.36% (**+6.6%**) — the same finding as Track A's
  G2, and almost exactly the same percentage
- **16k context**: does not fit; crashes at step 1

---

## Track C — teammate (palshraya), Automodel directly

A different stack: Automodel standalone rather than NeMo-RL, `ep_size=4` (not 8),
`global_batch_size=8`, LoRA/PEFT rather than full fine-tuning. Traces in
`/home/palshraya/Automodel/traces/`.

| trace | date | what it is | per ProfilerStep | self CUDA total |
|---|---|---|---|---|
| `nemo-hellaswag-ft` | 09-23 | full fine-tune baseline | **5.341 s** | 7.119 s |
| `nemo_triton_peft_hellaswag-ft` | 09-25 | Triton LoRA + PEFT | **1.135 s** | 3.322 s |
| `detailed-nemo_triton_peft...` | 09-26 | same, NVTX-annotated | 166.4 ms | 744 ms |
| `ma_prefetch_nemo_triton_peft...` | 09-28 | + "ma_prefetch" | 170.6 ms | 1.059 s |

**Full fine-tune → Triton LoRA: 5.341 s → 1.135 s per step, 4.7×.** That is their headline
result, and it is a workload change (training adapters instead of all weights), not a
kernel fix.

Two cautions on reading this table:

1. **The first two rows and the last two are not on the same measurement basis.** 1.135 s
   versus 166 ms for the same named config is a 7× gap that almost certainly comes from
   different profiling scope or annotation, not from a real speedup. Do not multiply these
   through.
2. **`ma_prefetch` looks like a regression and should be checked.** Against `detailed-`, its
   directly comparable predecessor: reduce-scatter **99.6 ms → 421.4 ms (4.2× worse)**,
   `record_param_comms` 500 ms → 806 ms, self CUDA total 744 ms → 1059 ms, and per-step
   170.6 ms vs 166.4 ms. Only 3 steps each, so it may be noise — but it is the opposite of
   the intended direction and is worth a repeat run before it goes in the report.

---

## The crossover test: do Track B's wins transfer to Track A?

Track B's largest kernel-level win is `fsdp_output_dtype: bfloat16` (+67% MFU there). It had
never been tried on Track A. Run on Track A on top of fused AdamW, 50 steps each:

| Track A arm | median | p90 | MFU | TFLOPS |
|---|---|---|---|---|
| fused (J0) | 1.140 s | 1.440 s | 0.640% | 50.9 |
| + bf16 FSDP output (K1) | 1.120 s | 1.430 s | **0.640%** | **50.9** |
| + bf16 output + chunked CE (K2) | 1.140 s | 1.420 s | **0.640%** | **50.9** |

**MFU and TFLOPS are identical to three significant figures.** Both of Track B's memory
optimizations are worth exactly nothing on Track A.

The reason is mechanical. `fsdp_output_dtype` and `logprob_chunk_size` both shrink the
lm_head logits tensor, which is `[tokens, 131072]`. Track A has ~3,400 tokens per step, so
that tensor is already small; Track B has ~1,000,000, so it dominates memory.

This is the cleanest statement of why the two tracks diverged:

| | Track A | Track B |
|---|---|---|
| tokens per step | ~3,400 | ~1,000,000 |
| bottleneck | **host launch latency** (27,879 GPU-idle gaps, 87 us each) | **activation memory** |
| what helps | fewer kernel launches (fused AdamW) | smaller activations (bf16 out, chunked CE) |
| what does nothing | bf16 out, chunked CE, CUDA graphs, dispatcher knobs | -- |

Same model, same parallelism (EP 8 / TP 1), same allocator flags, same optimizer, same
learning rate. Opposite bottlenecks, purely because of tokens per step.

The one optimization that transferred is **fused AdamW**, worth ~6.6% on both, because
kernel launch count per optimizer step does not depend on batch size.

### Config differences between the two best runs

Measured from the resolved config each run printed (`J0-fused-repeat` vs `optim-O1`).

Identical: model `Nemotron-3-Nano-30B-A3B-Base-BF16`, EP 8, TP 1, micro-batch 1, lr 1.5e-05,
max_grad_norm 1.0, sequence packing off, `expandable_segments:True`,
`empty_cache_after_train=false`, fused AdamW, `experts=torch_mm`, dispatcher deepep.

| setting | Track A | Track B |
|---|---|---|
| dataset | `squad` | `openai_format` (Cascade-2 chat, 20k jsonl) |
| tokenizer | `A3B-Base-BF16` | `A3B-BF16` (instruct variant) |
| chat template | `Context:/Question:/Answer:` | `default` |
| global batch size | 16 | **128** |
| max sequence length | 2048 | **8192** |
| dynamic batching | off | **on**, `train_mb_tokens: 8192` |
| `make_sequence_length_divisible_by` | 1 | 64 |
| activation checkpointing | off | **on** |
| `fsdp_output_dtype` | fp32 (default) | **bfloat16** |
| `logprob_chunk_size` | none | **1024** |

**Open question worth resolving before the report:** Track B trains the *Base* model but
tokenizes with the *instruct* tokenizer (`A3B-BF16`, not `A3B-Base-BF16`) to obtain a usable
`default` chat template. That is a reasonable choice for chat-format SFT, but the two
tokenizers should be confirmed to share vocabulary and special tokens -- a mismatch would
corrupt training silently rather than raise an error.

---

## Correction: there are four tracks, not three, and Track B is not the best

An earlier version of this document surveyed 41 runs and called `optim-O1` (10.355% MFU) the
project's best result. There are now **67 runs**. A Megatron-backend track was running in
parallel and was missed. On the *same* Cascade-2 data it roughly doubles the DTensor result.

| run | backend | step | MFU | TFLOPS | steps |
|---|---|---|---|---|---|
| `meg_evenpack_gbs512` | Megatron | 13.360 s | **20.530%** | 1624.9 | 30 |
| `meg_evenpack_seo` | Megatron | 3.590 s | **19.280%** | 1526.5 | 30 |
| `meg_evenpack` | Megatron | 3.620 s | 18.985% | 1503.0 | 30 |
| `megatron-try8` | Megatron | 4.570 s | 15.615% | 1236.0 | 30 |
| `meg_evenpack_mb8k` | Megatron | 4.610 s | 15.190% | 1202.3 | 30 |
| `meg_deepep` | Megatron | 5.140 s | 13.985% | 1107.0 | 30 |
| `optim-O1` | DTensor | 6.530 s | 10.355% | 819.6 | 30 |
| `J0-fused` (SQuAD) | DTensor | 1.140 s | 0.640% | 50.9 | 50 |

Ten of the Megatron arms crashed (fp8, fp8param, hybridep, hybridep2, nvls0, cpuoptim_selrc,
and others), so that track has its own unfinished business.

### Why Megatron is ~2x faster: it already fixes the three defects this project found

The collectives investigation (PROCESS.md 3.4) identified three defects in the NeMo-RL
DTensor path and did not fix them. Megatron's resolved config shows it does not have them:

```
distributed_data_parallel_config:
  grad_reduce_in_fp32: False      overlap_grad_reduce: True      overlap_param_gather: True
```

| defect found on the DTensor path | Megatron equivalent |
|---|---|
| `reduce_dtype` hardcoded fp32 (`setup.py:522`), no config knob. Reduce-scatter carries 70.5 GB against all-gather's 4.4 GB; gradients occupy 15.0 GB/GPU where bf16 would be 7.5 GB. | `grad_reduce_in_fp32: False` |
| `prepare_for_grad_accumulation` / `prepare_for_final_backward` exist in Automodel and are never called by NeMo-RL, so gradients reduce-scatter twice per step at 2 microbatches. | `overlap_grad_reduce: True` |
| Measured compute/communication overlap: 1.3%. | `overlap_param_gather: True` |

This is independent validation of the collectives findings: the three things identified as
missing are precisely the three things the faster backend has.

### Sequence packing: broken on DTensor, working and worth +36% on Megatron

Section 3.1 proved packing is silently incorrect on the DTensor path -- `cu_seqlens` absent,
Mamba `seq_idx` None, so tokens attend across document boundaries. Megatron packs with
`algorithm: even_bin_count` at `train_mb_tokens: 16384` and it works:

- `meg_deepep` (no even packing): 13.985%
- `meg_evenpack`: **18.985%**, i.e. **+36%**

So packing is not a bad idea; it is a broken implementation on one backend.

---

## The assigned workload finally has a before/after

HellaSwag is the team's assigned workload, and all earlier numbers in this document were
SQuAD or Cascade-2. Both verified optimizations applied to HellaSwag, 49 steps each:

| HellaSwag | median | p90 | MFU | TFLOPS |
|---|---|---|---|---|
| baseline | 1.800 s | 1.980 s | 0.480% | 38.1 |
| `empty_cache` off + `expandable_segments` + fused AdamW | **1.110 s** | **1.170 s** | **0.800%** | **63.5** |
| | **1.62x** | **1.69x** | **+67%** | **+67%** |

Loss agrees to within 0.013 over the first six steps, against a measured A/A noise floor of
0.032 mean. The p90/median ratio tightens from 1.10 to 1.05, which is the jitter reduction
the host-launch-bound diagnosis predicts.

This is a larger relative gain than the same two flags produced on SQuAD (1.57x), consistent
with HellaSwag being the shorter workload (122 tokens mean vs SQuAD's 235) and therefore even
more dominated by fixed per-step cost.

### Dataset provenance

| workload | source |
|---|---|
| HellaSwag (assigned) | https://huggingface.co/datasets/Rowan/hellaswag |
| SQuAD | https://huggingface.co/datasets/rajpurkar/squad |
| Cascade-2 chat | https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data |

`chat_20k_seed0.jsonl` is a 20,000-row seed-0 sample of the `Chat` domain. Matched by schema
(`domain`, `source`, `messages`, `generator`) and name; no download script or HF cache entry
exists on the box, so the provenance is inferred rather than recorded.

Measured token lengths, one tokenizer and template for all three:

| dataset | mean | median | p90 | x SQuAD |
|---|---|---|---|---|
| HellaSwag choice (assigned) | 122 | 116 | 160 | 0.52x |
| SQuAD | 235 | 217 | 356 | 1.00x |
| Cascade-2 chat | 1,966 | 1,394 | 4,565 | 8.35x |

---

## What was fixed in code (not config)

| fix | where | effect |
|---|---|---|
| MFU was never computed at all | `flops_tracker.py`, `flops_formulas.py` | Nemotron-H hit an unhandled branch, so `flops_tracker` was silently `None` and **zero MFU lines printed for the whole run**. Added the dispatch branch and fixed 3 bugs in the dead hybrid formula — worst was that `E` (MoE) layers were uncounted, i.e. **57% of the model's FLOPs**. Every MFU number in this document depends on this fix. |
| profiler gate coupled to nsys | `nsys.py`, `worker_group_utils.py` | torch profiler / memory snapshot could not run without nsys |
| HellaSwag dataset | `response_datasets/hellaswag.py` | 39,905 examples, registered as `hellaswag` and `hellaswag-completion` |

---

---

## Track B, round 2: DeepEP — five arms, five regressions

Every Track B run to date had DeepEP **on** (the `dispatcher` default resolves to it) and
**untuned** (the config sets the removed `enable_deepep` key, which the code ignores). So
DeepEP was a constant in every earlier experiment, never a variable. These arms make it one.

Each arm is `sft-nanov3-cascade2chat-8k-best-1n8g.yaml` with exactly one change, 30 steps,
step 1 discarded, compared over a matched step range. Total floating-point work matched
across arms to within 1%.

| arm | change | median | p90 | vs control |
|---|---|---|---|---|
| `optim-O1` | earlier run of the same config | 6.530 | 7.690 | −1.4% ← **noise floor** |
| `dep-A0-repeat` | **(control)** | 6.620 | 8.040 | — |
| `dep-A3-sms32` | `dispatcher_num_sms: 32` | 6.870 | 8.080 | +3.8% |
| `dep-A1-async` | `dispatcher_async_dispatch: true` | 7.030 | 7.760 | +6.2% |
| `dep-A5-gmm` | `experts: gmm` | 8.180 | 10.100 | +23.6%, **OOM at step 22** |
| `dep-A4-torchdisp` | `dispatcher: torch` | 8.510 | 9.860 | +28.5% |

**No DeepEP setting improves Track B.** Three results are worth keeping anyway:

- `dispatcher: torch` costing **+28.5%** is the first measurement of what DeepEP is worth.
  It earns its 22.6% of wall clock.
- `dispatcher_async_dispatch` making things **worse** shows the MoE communication cannot be
  hidden: `route → dispatch → expert GEMM → combine` is serial, so unblocking the compute
  stream has nothing to fill the gap. Megatron pairs it with `moe_shared_expert_overlap`;
  the DTensor path has no equivalent knob.
- `experts: gmm` OOM'd at **76.39 GiB of 79.10 GiB** with only 932 MiB held by another
  user. That is the setting the config has been silently asking for; the dead key was
  accidentally protecting the run.

Unmeasured: `dispatcher_num_sms: 8` and `fake_balanced_gate`, both repeatedly destroyed by
other users' jobs on the shared box. Reported as not run rather than measured under
contention.


## Found, diagnosed, not yet fixed

From the collectives investigation (PROCESS.md §3.4):

- **The step is host-launch-bound.** Rank 0's GPU is idle 64–68% of the span across
  **27,879 gaps averaging 87 µs** even after the optimizer fix. Median is pinned at 1.140 s
  across every knob tried. Collective time (56% of GPU time) is a *symptom* of this, not a
  cause: all-gather runs at **2.3% of NVLink bandwidth** while compute is balanced across
  ranks to 1.10× and collective wait varies 2.1–4.4×.
- `reduce_dtype` hardcoded to fp32 (`setup.py:522`), no config knob — reduce-scatter carries
  **70.5 GB against all-gather's 4.4 GB** for the same tensors.
- Gradient-sync deferral exists in Automodel, never called by NeMo-RL — gradients
  reduce-scatter **twice per step** with 2 microbatches.
- NCCL untuned at `RING_LL` on an NVSwitch box where NVLS is the intended algorithm.
- `enable_deepep: true` in the recipe is **silently ignored** by current Automodel.
- 280 ms of DeepEP's 326 ms is notify/handshake metadata, not token movement.

---

- **`defer_fsdp_grad_sync` is plumbed and inert.** NeMo-RL passes it (`setup.py:529`),
  Automodel stores it (`fsdp2.py:107`) and nothing ever calls `get_sync_ctx` /
  `set_requires_gradient_sync`, so FSDP2 syncs gradients on every microbatch. The Track B
  trace shows 1014 reduce-scatters where ~128 would do, at 8 microbatches per optimizer
  step: **~768 ms of a 15,647 ms span, 4.9% of wall clock.** Risk: holding 8 microbatches
  of unreduced gradients may OOM on a config already peaking near 76 GiB of 79 GiB.
- **`reduce_dtype` hardcoded to fp32** (`setup.py:522`), no config knob, while
  `output_dtype` one line below has one. Like-for-like, reduce-scatter moves exactly
  **2.00×** the bytes of all-gather — purely the dtype. On Track B these run at 29% of
  NVLink wire speed, so halving bytes is worth **1.3-2.8% of wall**. Megatron already runs
  `grad_reduce_in_fp32: False` on this model and data.

## Honest summary of who got what

- **Track B (`optim-O1`, 10.36% MFU) is the best DTensor result** — see the correction
  above: Megatron reaches 20.53% on the same data, so this is not the project's best. It is mostly the
  realistic-workload configuration: bf16 FSDP output, chunked cross-entropy, dynamic
  batching, activation checkpointing, allocator settings.
- **Track A is the diagnostic track.** Its 1.57× is real but small in absolute terms; its
  value is that it produced the evidence — the MFU fix, the packing correctness bug, the
  allocator finding, and the launch-bound diagnosis.
- **The fused optimizer was found independently on both tracks** (`optim-O1` on Track B,
  `G2` on Track A) and is worth ~6.6% on each. Track B found it first.
- **Track C's 4.7× is LoRA vs full fine-tuning**, on a different stack and parallelism.
