"""Measure communication/compute overlap directly from a Kineto trace.

HTA reports a comp_comm_overlap percentage whose definition is not obvious, and it gave
a counterintuitive answer (Megatron 4.2% vs DTensor 46.9%). This computes it explicitly:

  comm      = union of NCCL and DeepEP kernel intervals
  compute   = union of all other kernel intervals
  overlapped= |comm AND compute|         time where both are running
  exposed   = |comm MINUS compute|       time where comms run with the GPU otherwise idle

"exposed" is the number that matters: it is communication you are actually paying for in
wall-clock, because nothing else is happening.

Usage: python overlap.py <kineto_dir> [label] [rank]
"""

import collections
import glob
import gzip
import json
import os
import re
import sys


def merge(intervals):
    """Union of [start, end) intervals -> disjoint sorted list."""
    if not intervals:
        return []
    iv = sorted(intervals)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    """Intersection of two disjoint sorted interval lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def is_comm(name):
    low = name.lower()
    return "nccl" in low or "deep_ep" in low or "deepep" in low


def analyse(path, label):
    events = json.load(gzip.open(path))["traceEvents"]
    ker = [e for e in events if e.get("cat") == "kernel" and e.get("dur")]
    if not ker:
        print("%s: no kernels" % label)
        return

    comm_iv = merge([(e["ts"], e["ts"] + e["dur"]) for e in ker if is_comm(e["name"])])
    comp_iv = merge([(e["ts"], e["ts"] + e["dur"]) for e in ker if not is_comm(e["name"])])
    busy_iv = merge([(e["ts"], e["ts"] + e["dur"]) for e in ker])

    lo = min(e["ts"] for e in ker)
    hi = max(e["ts"] + e["dur"] for e in ker)
    span = hi - lo

    comm, comp, busy = total(comm_iv), total(comp_iv), total(busy_iv)
    ov = total(merge(intersect(comm_iv, comp_iv)))
    exposed = comm - ov

    f = lambda us: us / 1e3
    print("=== %s ===" % label)
    print("  wall span                 %10.1f ms" % f(span))
    print("  GPU busy (any kernel)     %10.1f ms  %5.1f%% of span" % (f(busy), 100 * busy / span))
    print("  idle                      %10.1f ms  %5.1f%% of span" % (f(span - busy), 100 * (span - busy) / span))
    print("  compute (non-comm)        %10.1f ms" % f(comp))
    print("  communication             %10.1f ms" % f(comm))
    print("    overlapped with compute %10.1f ms  %5.1f%% of comm" % (f(ov), 100 * ov / comm if comm else 0))
    print("    EXPOSED                 %10.1f ms  %5.1f%% of comm  = %.1f%% of span" %
          (f(exposed), 100 * exposed / comm if comm else 0, 100 * exposed / span))

    # split comm into NCCL vs DeepEP to see which is exposed
    for tag, pred in (("NCCL", lambda n: "nccl" in n.lower()),
                      ("DeepEP", lambda n: "deep_ep" in n.lower() or "deepep" in n.lower())):
        iv = merge([(e["ts"], e["ts"] + e["dur"]) for e in ker if pred(e["name"])])
        if not iv:
            continue
        t = total(iv)
        o = total(merge(intersect(iv, comp_iv)))
        print("      %-7s %8.1f ms   overlapped %6.1f ms (%4.1f%%)   exposed %8.1f ms" %
              (tag, f(t), f(o), 100 * o / t if t else 0, f(t - o)))

    # stream layout: are comms even on a separate stream from compute?
    st = collections.defaultdict(lambda: [0, 0])
    for e in ker:
        st[e.get("tid")][1 if is_comm(e["name"]) else 0] += e["dur"]
    print("  streams (compute ms / comm ms):")
    for tid, (c, m) in sorted(st.items(), key=lambda x: -(x[1][0] + x[1][1]))[:6]:
        print("    stream %-6s  compute %8.1f   comm %8.1f" % (tid, f(c), f(m)))
    print()


if __name__ == "__main__":
    d = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(d.rstrip("/"))
    rank = sys.argv[3] if len(sys.argv) > 3 else "0"
    p = glob.glob(os.path.join(d, "rank%s.pt.trace.json.gz" % rank))
    if not p:
        raise SystemExit("no rank%s trace under %s" % (rank, d))
    analyse(p[0], "%s (rank %s)" % (label, rank))
