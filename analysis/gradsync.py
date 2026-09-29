"""Size the dead defer_fsdp_grad_sync flag from a Kineto trace.

NeMo-RL passes defer_fsdp_grad_sync=True into FSDP2Config (setup.py:529) and Automodel's
FSDP2Manager stores it (fsdp2.py:107) -- but nothing ever calls get_sync_ctx /
set_requires_gradient_sync on the DTensor path, so FSDP2 keeps its default of syncing
gradients on EVERY microbatch backward instead of only the last one.

If that is true the trace shows reduce-scatter called once per microbatch per layer.
This counts the collectives, their bytes and their time, and reports what deferring to
the final microbatch would save.
"""
import collections, glob, gzip, json, os, sys


def load(path):
    return json.load(gzip.open(path))["traceEvents"]


def main(d, rank="0"):
    p = glob.glob(os.path.join(d, "rank%s.pt.trace.json.gz" % rank))
    if not p:
        raise SystemExit("no rank%s trace under %s" % (rank, d))
    ev = load(p[0])
    ker = [e for e in ev if e.get("cat") == "kernel" and e.get("dur")]

    # NCCL kernels carry collective metadata in args on recent PyTorch.
    agg = collections.defaultdict(lambda: {"n": 0, "ms": 0.0, "bytes": 0, "dtypes": set()})
    for e in ker:
        n = e["name"].lower()
        if "nccl" not in n:
            continue
        a = e.get("args", {}) or {}
        coll = a.get("Collective name") or a.get("collective_name") or "?"
        if coll == "?":
            for tag in ("allgather", "reduce_scatter", "reducescatter", "allreduce", "broadcast"):
                if tag in n:
                    coll = tag
                    break
        r = agg[coll]
        r["n"] += 1
        r["ms"] += e["dur"] / 1e3
        # in/out msg sizes when present
        cnt = a.get("In msg nelems") or a.get("Out msg nelems") or 0
        dt = a.get("dtype", "")
        r["dtypes"].add(dt)
        width = {"Float": 4, "float": 4, "Float32": 4, "BFloat16": 2, "bf16": 2, "Half": 2}.get(dt, 0)
        r["bytes"] += int(cnt) * width

    print("=== NCCL collectives, rank %s ===" % rank)
    print("%-22s %7s %10s %12s  %s" % ("collective", "calls", "ms", "GB", "dtype"))
    tot_ms = 0.0
    for k, v in sorted(agg.items(), key=lambda x: -x[1]["ms"]):
        tot_ms += v["ms"]
        print("%-22s %7d %10.1f %12.2f  %s" % (
            k, v["n"], v["ms"], v["bytes"] / 1e9, ",".join(sorted(d for d in v["dtypes"] if d)) or "-"))
    print("%-22s %7s %10.1f" % ("TOTAL", "", tot_ms))

    # How many optimizer steps are in this trace? Use the optimizer kernels as the marker.
    steps = sum(1 for e in ker if "adam" in e["name"].lower())
    # Reduce-scatter count per FSDP unit tells us microbatches: 52 layers + root.
    rs = next((v for k, v in agg.items() if "reduce" in k.lower() and "scatter" in k.lower()), None)
    ag = next((v for k, v in agg.items() if "gather" in k.lower()), None)
    print()
    if rs and ag and ag["n"]:
        ratio = rs["n"] / ag["n"]
        print("reduce-scatter calls / all-gather calls = %.2f" % ratio)
        print("  all-gather runs once per FSDP unit per forward.")
        print("  If gradient sync were deferred to the final microbatch, reduce-scatter")
        print("  would run once per unit per STEP, i.e. the same count as all-gather")
        print("  divided by microbatches per step.")
        if ratio > 0.9:
            print("  -> ratio ~1: reduce-scatter fires on EVERY backward, i.e. every microbatch.")
            print("     Deferring would cut it to 1/mb_per_step of its current count.")
    if rs:
        print()
        print("reduce-scatter total: %d calls, %.1f ms, %.2f GB" % (rs["n"], rs["ms"], rs["bytes"] / 1e9))
        for mb in (2, 3, 4):
            print("   if %d microbatches/step: deferring saves ~%.1f ms (%.0f%% of RS)"
                  % (mb, rs["ms"] * (mb - 1) / mb, 100 * (mb - 1) / mb))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0")
