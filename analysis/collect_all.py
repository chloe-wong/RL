"""Collect every SFT run on this box into one table.

For each run directory: the Hydra overrides that distinguish it, how many steps it
actually completed, and the step-time / MFU / loss statistics. Runs that crashed are
reported as crashed with the reason, because a failed arm is a result too.
"""

import glob
import os
import re
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/scratch/chloe_wong/runs"

# Overrides shared by every run; listing them per row would be noise.
BORING = re.compile(
    r"^(cluster\.|logger\.|checkpointing\.|sft\.max_num_steps|sft\.val_|"
    r"policy\.model_name)"
)


def parse(d):
    log = os.path.join(d, "run.log")
    if not os.path.exists(log):
        return None
    txt = open(log, errors="ignore").read()

    m = re.search(r"Overrides: \[(.*?)\]", txt)
    ov = []
    if m:
        for o in re.findall(r"'([^']*)'", m.group(1)):
            if not BORING.match(o):
                ov.append(o.replace("policy.dtensor_cfg.", "").replace("automodel_kwargs.backend.", "")
                            .replace("policy.optimizer.kwargs.", "opt.").replace("policy.", ""))
    cfg = re.search(r"--config\s+(\S+)", txt)
    times = [float(x) for x in re.findall(r"Total step time:\s+([0-9.]+)s", txt)]
    mfu = [float(x) for x in re.findall(r"Floating Point Utilization:\s+([0-9.]+)%", txt)]
    loss = [float(x) for x in re.findall(r"Loss:\s+([0-9.]+)", txt)]
    crashed = "Traceback (most recent call last)" in txt
    reason = ""
    if crashed:
        r = re.findall(r"^(?:torch\.)?(\w*(?:Error|Exception))[: ](.{0,90})", txt, re.M)
        seen = [f"{a}: {b.strip()}" for a, b in r if a not in ("RayTaskError",)]
        reason = seen[-1] if seen else "unknown"

    body = times[1:] if len(times) > 1 else times  # drop step 1 (warmup)
    q = (lambda v, p: sorted(v)[int(p * (len(v) - 1))]) if body else None
    return {
        "name": os.path.basename(d.rstrip("/")),
        "cfg": os.path.basename(cfg.group(1)) if cfg else "?",
        "ov": ",".join(ov) if ov else "-",
        "steps": len(times),
        "median": q(body, .5) if body else None,
        "p90": q(body, .9) if body else None,
        "mfu": st.median(mfu) if mfu else None,
        "loss_last": loss[-1] if loss else None,
        "loss_mean": st.mean(loss) if loss else None,
        "crashed": crashed,
        "reason": reason,
        "mtime": os.path.getmtime(log),
    }


rows = [r for r in (parse(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))) if r]
rows.sort(key=lambda r: r["mtime"])

print("%-42s %-34s %5s %8s %8s %7s %8s" %
      ("run", "config", "steps", "median", "p90", "MFU", "loss(mean)"))
print("-" * 125)
for r in rows:
    if r["crashed"] and r["median"] is None:
        print("%-42s %-34s %5d  CRASHED  %s" % (r["name"], r["cfg"], r["steps"], r["reason"][:48]))
        continue
    flag = "  CRASHED@%d" % r["steps"] if r["crashed"] else ""
    print("%-42s %-34s %5d %8s %8s %6s%% %8s%s" % (
        r["name"], r["cfg"], r["steps"],
        "%.3f" % r["median"] if r["median"] else "-",
        "%.3f" % r["p90"] if r["p90"] else "-",
        "%.3f" % r["mfu"] if r["mfu"] is not None else "  n/a",
        "%.4f" % r["loss_mean"] if r["loss_mean"] else "-", flag))

print("\n\n=== distinguishing overrides per run ===")
for r in rows:
    if r["ov"] != "-":
        print("  %-42s %s" % (r["name"], r["ov"][:150]))
