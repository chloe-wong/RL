"""Per-rank collective breakdown from a directory of Kineto traces.

Every rank runs the same collectives on the same byte counts, so any spread in
collective *time* across ranks is time spent waiting for a straggler, not time on the
wire. Comparing that spread against the spread in GEMM time is what separates "the
network is slow" from "one rank arrives late".

Usage:
    python collectives_report.py <kineto_dir> [label]

<kineto_dir> holds rank{N}.pt.trace.json.gz, as written by NRL_TORCH_PROFILE_DIR.
"""

import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys

# Bytes per element, keyed by the dtype strings Kineto writes into NCCL op args.
ELEM_SIZE = {"Float": 4, "BFloat16": 2, "Half": 2, "Byte": 1, "Int": 4, "Long": 8, "Double": 8}


def bucket(name):
    """Classify a CUDA kernel by what it is actually doing."""
    low = name.lower()
    if "nccl" in low:
        return "nccl"
    if "deep_ep" in low or "deepep" in low:
        return "deepep"
    if any(k in low for k in ("gemm", "cutlass", "sm90", "wgrad", "dgrad", "tensorop")):
        return "gemm"
    if any(k in low for k in ("chunk_scan", "chunk_state", "ssd", "causal_conv")):
        return "mamba"
    if any(k in low for k in ("flash", "fmha", "attn")):
        return "attn"
    return "other"


def read_rank(path):
    events = json.load(gzip.open(path))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise SystemExit("no kernel events in %s" % path)

    by = collections.Counter()
    for e in kernels:
        by[bucket(e["name"])] += e["dur"]
    total = sum(by.values())
    span = max(e["ts"] + e.get("dur", 0) for e in kernels) - min(e["ts"] for e in kernels)

    # NCCL metadata lives on the cpu_op side, one record per collective call.
    coll = collections.defaultdict(lambda: [0, 0])  # (op, dtype) -> [calls, elements]
    for e in events:
        args = e.get("args") or {}
        if "Collective name" not in args:
            continue
        key = (args["Collective name"], args.get("dtype", "?"))
        coll[key][0] += 1
        coll[key][1] += args.get("In msg nelems", 0) or 0

    kernel_time = collections.Counter()
    kernel_count = collections.Counter()
    for e in kernels:
        kernel_time[e["name"]] += e["dur"]
        kernel_count[e["name"]] += 1

    return {"by": by, "total": total, "span": span, "coll": coll,
            "kernel_time": kernel_time, "kernel_count": kernel_count}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(d.rstrip("/"))

    paths = sorted(glob.glob(os.path.join(d, "rank*.pt.trace.json.gz")),
                   key=lambda p: int(re.search(r"rank(\d+)", p).group(1)))
    if not paths:
        raise SystemExit("no rank*.pt.trace.json.gz under %s" % d)

    print("=" * 96)
    print("collectives report: %s   (%d ranks)" % (label, len(paths)))
    print("=" * 96)

    ranks = {}
    print("\n%-5s %10s %10s %10s %10s %10s %10s %8s" %
          ("rank", "kernel ms", "nccl ms", "deepep ms", "gemm ms", "mamba+at", "other ms", "busy%"))
    for p in paths:
        r = int(re.search(r"rank(\d+)", p).group(1))
        s = read_rank(p)
        ranks[r] = s
        b = s["by"]
        print("%-5d %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %7.1f%%" %
              (r, s["total"] / 1e3, b["nccl"] / 1e3, b["deepep"] / 1e3, b["gemm"] / 1e3,
               (b["mamba"] + b["attn"]) / 1e3, b["other"] / 1e3, 100 * s["total"] / s["span"]))

    # The headline. Compute spread is the control: it is the same work on every rank,
    # so whatever spread it shows is measurement noise. Collective spread above that
    # floor is straggler wait.
    print("\n%-12s %10s %10s %8s   %s" % ("bucket", "min ms", "max ms", "spread", "reading"))
    for name, note in (("gemm", "control: identical work on every rank"),
                       ("other", "control: identical work on every rank"),
                       ("nccl", "spread above the control floor is straggler wait"),
                       ("deepep", "spread above the control floor is straggler wait")):
        vals = [s["by"][name] for s in ranks.values()]
        lo, hi = min(vals), max(vals)
        print("%-12s %10.1f %10.1f %7.2fx   %s" % (name, lo / 1e3, hi / 1e3, hi / lo if lo else 0, note))

    slow = min(ranks, key=lambda r: ranks[r]["by"]["nccl"])
    print("\nlikely straggler: rank %d (least time inside NCCL, so the rank others wait for)" % slow)

    # Bytes actually moved, to separate a bandwidth problem from a latency problem.
    print("\n%-26s %-10s %8s %14s %12s %10s" %
          ("collective", "dtype", "calls", "elements", "GB moved", "MB/call"))
    agg = collections.defaultdict(lambda: [0, 0])
    for s in ranks.values():
        for k, (c, n) in s["coll"].items():
            agg[k][0] += c
            agg[k][1] += n
    for (op, dt), (c, n) in sorted(agg.items(), key=lambda x: -x[1][1]):
        if not n:
            continue
        gb = n * ELEM_SIZE.get(dt, 4) / 1e9
        print("%-26s %-10s %8d %14d %12.2f %10.2f" % (op, dt, c, n, gb, 1e3 * gb / c))

    # Achieved bandwidth on the two FSDP collectives, against NVLink's ~450 GB/s.
    print("\n%-16s %8s %12s %12s %14s" % ("nccl kernel", "calls", "total ms", "us/call", "GB/s achieved"))
    for want in ("AllGather", "ReduceScatter", "AllReduce"):
        calls = sum(s["kernel_count"][k] for s in ranks.values() for k in s["kernel_count"] if want in k and "nccl" in k)
        ms = sum(s["kernel_time"][k] for s in ranks.values() for k in s["kernel_time"] if want in k and "nccl" in k) / 1e3
        if not calls:
            continue
        key = {"AllGather": "_allgather_base", "ReduceScatter": "_reduce_scatter_base", "AllReduce": "allreduce"}[want]
        gb = sum(n * ELEM_SIZE.get(dt, 4) / 1e9 for (op, dt), (c, n) in agg.items() if op == key)
        print("%-16s %8d %12.1f %12.1f %14.1f" % (want, calls, ms, 1e3 * ms / calls, gb / (ms / 1e3) if ms else 0))
    print("\n(NVLink on this box is NV18 full mesh, ~450 GB/s per GPU.)")

    print("\n=== top 12 kernels on the straggler rank (%d) ===" % slow)
    s = ranks[slow]
    for n, v in s["kernel_time"].most_common(12):
        print("%9.1f ms %6.1f%% %7d  %s" % (v / 1e3, 100 * v / s["total"], s["kernel_count"][n], n[:86]))


if __name__ == "__main__":
    main()
