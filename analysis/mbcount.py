"""How many microbatches (and optimizer steps) does the Track B trace actually contain?

Needed to size the dead defer_fsdp_grad_sync flag: reduce-scatter fires once per FSDP
unit per BACKWARD, so deferring to the final microbatch would cut it by (mb-1)/mb.
"""
import collections, glob, gzip, json, os, sys

p = glob.glob(os.path.join(sys.argv[1], "rank0.pt.trace.json.gz"))[0]
ev = json.load(gzip.open(p))["traceEvents"]

steps = sorted({e["name"] for e in ev if e.get("name", "").startswith("ProfilerStep")})
print("profiler steps in trace:", len(steps), steps[:8])

# Optimizer steps: the fused AdamW kernel fires once per optimizer step.
opt = [e for e in ev if e.get("cat") == "kernel" and "adam" in e.get("name", "").lower()]
print("adam kernel launches:", len(opt))

# Backward passes: count top-level autograd engine root ops.
bwd = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation")
       and e.get("name", "") in ("autograd::engine::evaluate_function: torch::autograd::AccumulateGrad",)]
# A more robust marker: the optimizer-zero or the backward root annotation
roots = collections.Counter(e.get("name", "") for e in ev
                            if e.get("cat") == "cpu_op" and "backward" in e.get("name", "").lower())
print("\ntop backward-ish cpu_op names:")
for n, c in roots.most_common(6):
    print("  %6d  %s" % (c, n[:90]))

ker = [e for e in ev if e.get("cat") == "kernel" and e.get("dur")]
rs = [e for e in ker if "nccl" in e["name"].lower() and "reduce" in e["name"].lower()]
ag = [e for e in ker if "nccl" in e["name"].lower() and "gather" in e["name"].lower()]
print("\nreduce-scatter kernels: %d   all-gather kernels: %d" % (len(rs), len(ag)))

# Cluster reduce-scatters in time: a gap much larger than the median gap separates
# one backward pass from the next.
ts = sorted(e["ts"] for e in rs)
gaps = [b - a for a, b in zip(ts, ts[1:])]
if gaps:
    med = sorted(gaps)[len(gaps) // 2]
    big = [g for g in gaps if g > 20 * med]
    print("median gap between RS kernels: %.0f us; gaps >20x median: %d" % (med, len(big)))
    print("=> approx %d separate backward passes in the trace" % (len(big) + 1))
    if opt:
        print("=> approx %.1f microbatches per optimizer step" % ((len(big) + 1) / len(opt)))
