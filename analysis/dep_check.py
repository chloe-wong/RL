"""Cross-check step time against MFU/TFLOPS.

With a fixed seed and fixed data order, every arm should see the identical sequence of
batches, so FLOPs per step should match step-for-step across arms. If they do, MFU must
move inversely with step time; if MFU and time disagree, the batches are NOT matched and
MFU is not comparable between arms.
"""
import glob, os, re, statistics as st

ROOT = "/mnt/scratch/chloe_wong/runs"
ARMS = ["optim-O1", "dep-A0-repeat", "dep-A1-async", "dep-A3-sms32", "dep-A4-torchdisp"]

def series(a, pat):
    f = os.path.join(ROOT, a, "run.log")
    if not os.path.exists(f):
        return []
    return [float(x) for x in re.findall(pat, open(f, errors="ignore").read())]

print("%-18s %8s %8s %9s %10s %12s" % ("arm", "med t", "med MFU", "med TFLOP", "sum FLOPs", "t*TFLOPS"))
print("-" * 76)
ref = None
for a in ARMS:
    t = series(a, r"Total step time:\s+([0-9.]+)s")[1:]
    m = series(a, r"Floating Point Utilization:\s+([0-9.]+)%")
    f = series(a, r"Training FLOPS:\s+([0-9.]+) TFLOPS")
    if not t:
        print("%-18s  (no steps)" % a); continue
    n = min(len(t), len(m), len(f))
    # total work = sum over steps of TFLOPS*time  (TFLOPS is a rate)
    work = sum(f[i] * t[i] for i in range(n))
    print("%-18s %8.3f %8.3f%% %9.1f %10s %12.0f" % (
        a, st.median(t), st.median(m), st.median(f), "", work))
    if ref is None:
        ref = (a, st.median(t), work)

print()
print("If 'sum t*TFLOPS' (total floating-point work) matches across arms, the batches are")
print("identical and step time is the fair comparison. If it differs, the arms saw")
print("different work and MFU/time can legitimately disagree.")
print()
# Per-step FLOPs comparison against the base, first 10 steps
base = series("dep-A0-repeat", r"Training FLOPS:\s+([0-9.]+) TFLOPS")
for a in ARMS:
    f = series(a, r"Training FLOPS:\s+([0-9.]+) TFLOPS")
    t = series(a, r"Total step time:\s+([0-9.]+)s")[1:]
    n = min(len(f), len(base), len(t), 10)
    if n == 0:
        continue
    # FLOPs per step = TFLOPS * seconds
    fl = [f[i] * t[i] for i in range(n)]
    bl = [base[i] * series("dep-A0-repeat", r"Total step time:\s+([0-9.]+)s")[1:][i] for i in range(n)]
    diff = max(abs(x - y) / y for x, y in zip(fl, bl)) if bl else 0
    print("%-18s max per-step FLOPs deviation from base over %d steps: %.2f%%" % (a, n, 100 * diff))
