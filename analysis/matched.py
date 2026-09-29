"""Compare arms over a MATCHED step range.

dep-A5-gmm OOMed at step 22, so its median is over fewer steps than the others. With a
fixed seed the batch sequence is identical across arms, so comparing arms over the same
step indices is valid; comparing a 21-step median against a 29-step median is not.
"""
import os, re, statistics as st

ROOT = "/mnt/scratch/chloe_wong/runs"
ARMS = ["dep-A0-repeat", "dep-A1-async", "dep-A3-sms32", "dep-A4-torchdisp", "dep-A5-gmm"]

def steps(a):
    f = os.path.join(ROOT, a, "run.log")
    if not os.path.exists(f):
        return []
    return [float(x) for x in re.findall(r"Total step time:\s+([0-9.]+)s",
                                         open(f, errors="ignore").read())]

data = {a: steps(a) for a in ARMS}
n = min(len(v) for v in data.values() if v)
print("matched over steps 2..%d (step 1 discarded, %d steps)\n" % (n, n - 1))
base = st.median(data["dep-A0-repeat"][1:n])
print("%-18s %8s %8s %9s %s" % ("arm", "median", "p90", "vs base", "note"))
print("-" * 70)
for a in ARMS:
    v = data[a][1:n]
    if not v:
        print("%-18s  no steps" % a); continue
    s = sorted(v)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    note = "OOM at step %d" % len(data[a]) if len(data[a]) < 30 else ""
    print("%-18s %8.3f %8.3f %8.1f%%  %s" % (a, st.median(v), p90, 100 * (st.median(v) - base) / base, note))
