"""Summarise the DeepEP sweep against the Track B baseline.

Reports median and p90 of step time (step 1 discarded), MFU, and the resolved backend
settings actually used -- the base config's `enable_deepep` key is ignored, so the only
trustworthy record of what ran is the resolved MasterConfig printed into the log.
"""
import glob, os, re, statistics as st

ROOT = "/mnt/scratch/chloe_wong/runs"
ARMS = ["optim-O1", "dep-A0-repeat", "dep-A1-async", "dep-A2-sms8",
        "dep-A3-sms32", "dep-A4-torchdisp", "dep-A5-gmm"]

def pct(a, b):
    return "" if a is None or b is None else "%+.1f%%" % (100 * (b - a) / a)

def p90(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]

rows = []
for a in ARMS:
    log = os.path.join(ROOT, a, "run.log")
    if not os.path.exists(log):
        rows.append((a, None, None, None, None, "not run")); continue
    txt = open(log, errors="ignore").read()
    t = [float(x) for x in re.findall(r"Total step time:\s+([0-9.]+)s", txt)]
    m = [float(x) for x in re.findall(r"Floating Point Utilization:\s+([0-9.]+)%", txt)]
    body = t[1:] if len(t) > 1 else t
    cfg = re.search(r"MasterConfig\(policy=.*", txt)
    cfg = cfg.group(0)[:20000] if cfg else ""
    def g(k, d="-"):
        mm = re.search(r"'%s':\s*'?([A-Za-z0-9_.]+)'?" % k, cfg)
        return mm.group(1) if mm else d
    back = "%s/%s/sms%s/async=%s" % (g("experts"), g("dispatcher", "DEFAULT(deepep)"),
                                    g("dispatcher_num_sms", "20"),
                                    g("dispatcher_async_dispatch", "False"))
    why = "ok (%d steps)" % len(t)
    if not t:
        why = "OOM" if "out of memory" in txt.lower() else (
              "crash" if "Traceback" in txt else "no steps")
    rows.append((a, st.median(body) if body else None, p90(body) if body else None,
                 st.median(m) if m else None, back, why))

base = next((r for r in rows if r[0] == "dep-A0-repeat" and r[1]), None) \
    or next((r for r in rows if r[0] == "optim-O1" and r[1]), None)

print("%-18s %8s %8s %8s %9s  %-34s %s" % ("arm", "median", "p90", "MFU", "vs base", "backend (experts/disp/sms/async)", "status"))
print("-" * 130)
for a, med, p, mfu, back, why in rows:
    print("%-18s %8s %8s %8s %9s  %-34s %s" % (
        a,
        "%.3f" % med if med else "-",
        "%.3f" % p if p else "-",
        "%.3f%%" % mfu if mfu is not None else "-",
        pct(base[1], med) if base and med else "",
        back, why))
if base:
    print("\nbase = %s (median %.3f s, p90 %.3f s)" % (base[0], base[1], base[2]))
    print("Noise floor: compare dep-A0-repeat against optim-O1 before believing any arm.")
