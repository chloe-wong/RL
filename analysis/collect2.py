"""Second pass: pull the config values that actually distinguish each run.

The alloc-*/optim-* runs select behaviour by YAML file rather than Hydra overrides, so
the overrides line is uninformative. The MasterConfig dump in the log is authoritative.
"""

import glob
import os
import re
import statistics as st

ROOT = "/mnt/scratch/chloe_wong/runs"

FIELDS = [
    ("seq", r"'max_total_sequence_length':\s*(\d+)"),
    ("gbs", r"'train_global_batch_size':\s*(\d+)"),
    ("chunk", r"'logprob_chunk_size':\s*([^,}]+)"),
    ("fsdp_out", r"'fsdp_output_dtype':\s*'([^']+)'"),
    ("alloc", r"PYTORCH_CUDA_ALLOC_CONF':\s*'([^']+)'"),
    ("nocache", r"'empty_cache_after_train':\s*(\w+)"),
    ("fused", r"'fused':\s*(\w+)"),
    ("foreach", r"'foreach':\s*(\w+)"),
    ("ac", r"'activation_checkpointing':\s*(\w+)"),
    ("dynbatch", r"'dynamic_batching':\s*\{'enabled':\s*(\w+)"),
    ("data", r"'train':\s*\{'dataset_name':\s*'([^']+)'"),
]


def row(d):
    log = os.path.join(d, "run.log")
    if not os.path.exists(log):
        return None
    txt = open(log, errors="ignore").read()
    cfg = re.search(r"MasterConfig\(policy=.*", txt)
    cfg = cfg.group(0)[:20000] if cfg else ""
    vals = {}
    for name, pat in FIELDS:
        m = re.search(pat, cfg)
        vals[name] = m.group(1).strip().strip("'") if m else "-"
    times = [float(x) for x in re.findall(r"Total step time:\s+([0-9.]+)s", txt)]
    mfu = [float(x) for x in re.findall(r"Floating Point Utilization:\s+([0-9.]+)%", txt)]
    tfl = [float(x) for x in re.findall(r"Training FLOPS:\s+([0-9.]+) TFLOPS", txt)]
    body = times[1:] if len(times) > 1 else times
    med = sorted(body)[len(body) // 2] if body else None
    return {
        "name": os.path.basename(d.rstrip("/")),
        "steps": len(times),
        "median": med,
        "mfu": st.median(mfu) if mfu else None,
        "tflops": st.median(tfl) if tfl else None,
        "crash": "Traceback (most recent call last)" in txt,
        "mtime": os.path.getmtime(log),
        **vals,
    }


rows = [r for r in (row(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))) if r]
rows.sort(key=lambda r: r["mtime"])

hdr = ("run", "data", "seq", "gbs", "dynb", "ac", "chunk", "fsdpout", "alloc", "nocache",
       "fused", "steps", "median", "MFU", "TFLOPS")
print("%-42s %-12s %6s %5s %5s %5s %6s %8s %8s %7s %6s %5s %7s %7s %8s" % hdr)
print("-" * 160)
for r in rows:
    print("%-42s %-12s %6s %5s %5s %5s %6s %8s %8s %7s %6s %5d %7s %6s%% %8s%s" % (
        r["name"], r["data"][:12], r["seq"], r["gbs"], r["dynbatch"][:5], r["ac"][:5],
        r["chunk"][:6], r["fsdp_out"][:8], "expseg" if "expand" in r["alloc"] else "-",
        r["nocache"][:7], r["fused"][:6], r["steps"],
        "%.3f" % r["median"] if r["median"] else "-",
        "%.3f" % r["mfu"] if r["mfu"] is not None else "  n/a",
        "%.1f" % r["tflops"] if r["tflops"] else "-",
        "  CRASH" if r["crash"] else ""))
