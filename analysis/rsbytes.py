"""Compare all-gather and reduce-scatter volume honestly.

These two report message sizes with OPPOSITE meanings:
  all-gather     In = my shard (1/N of the tensor),  Out = the full tensor
  reduce-scatter In = the full tensor,               Out = my shard (1/N)
Reading "In msg nelems" for both therefore compares a shard against a full tensor and
manufactures a factor of N. This prints both fields so the comparison is like-for-like.
"""
import collections, glob, gzip, json, os, sys

W = {"Float": 4, "float": 4, "Float32": 4, "BFloat16": 2, "bf16": 2, "Half": 2,
     "Double": 8, "Long": 8, "Int": 4}

p = glob.glob(os.path.join(sys.argv[1], "rank0.pt.trace.json.gz"))[0]
ev = json.load(gzip.open(p))["traceEvents"]
ker = [e for e in ev if e.get("cat") == "kernel" and e.get("dur") and "nccl" in e.get("name", "").lower()]

agg = collections.defaultdict(lambda: collections.defaultdict(float))
for e in ker:
    a = e.get("args", {}) or {}
    coll = a.get("Collective name") or "?"
    dt = a.get("dtype", "")
    w = W.get(dt, 0)
    r = agg[(coll, dt)]
    r["calls"] += 1
    r["ms"] += e["dur"] / 1e3
    r["in_el"] += float(a.get("In msg nelems", 0) or 0)
    r["out_el"] += float(a.get("Out msg nelems", 0) or 0)
    r["w"] = w

print("%-22s %-10s %7s %9s %12s %12s" % ("collective", "dtype", "calls", "ms", "IN GB", "OUT GB"))
print("-" * 80)
for (coll, dt), r in sorted(agg.items(), key=lambda x: -x[1]["ms"]):
    print("%-22s %-10s %7d %9.1f %12.2f %12.2f" % (
        coll, dt or "-", r["calls"], r["ms"], r["in_el"] * r["w"] / 1e9, r["out_el"] * r["w"] / 1e9))

print()
ag = {k: v for k, v in agg.items() if "gather" in k[0].lower()}
rs = {k: v for k, v in agg.items() if "reduce_scatter" in k[0].lower()}
def tot(d, f):
    return sum(v[f] * v["w"] for v in d.values()) / 1e9
print("all-gather     : IN %.2f GB (my shard)   OUT %.2f GB (full tensor)" % (tot(ag, "in_el"), tot(ag, "out_el")))
print("reduce-scatter : IN %.2f GB (full tensor) OUT %.2f GB (my shard)" % (tot(rs, "in_el"), tot(rs, "out_el")))
a_full, r_full = tot(ag, "out_el"), tot(rs, "in_el")
if a_full:
    print()
    print("like-for-like (full-tensor side of each): AG %.2f GB vs RS %.2f GB  = %.2fx"
          % (a_full, r_full, r_full / a_full))
    print("if reduce_dtype were bf16, RS full-tensor side would be %.2f GB" % (r_full / 2))
