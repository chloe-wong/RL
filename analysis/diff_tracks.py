"""Diff the resolved MasterConfig of the best Track A run against the best Track B run.

Reads the config dump each run prints, so this is what actually ran, not what the YAML
files imply.
"""

import ast
import re

RUNS = {
    "A (J0-fused-repeat)": "/mnt/scratch/chloe_wong/runs/sft-nanov3-J0-fused-repeat/run.log",
    "B (optim-O1)": "/mnt/scratch/chloe_wong/runs/optim-O1/run.log",
}

KEYS = [
    ("policy", "model_name"),
    ("tokenizer", "name"),
    ("tokenizer", "chat_template"),
    ("policy", "train_global_batch_size"),
    ("policy", "train_micro_batch_size"),
    ("policy", "max_total_sequence_length"),
    ("policy", "make_sequence_length_divisible_by"),
    ("policy", "logprob_chunk_size"),
    ("policy", "max_grad_norm"),
    ("dynamic_batching", "enabled"),
    ("dynamic_batching", "train_mb_tokens"),
    ("sequence_packing", "enabled"),
    ("dtensor_cfg", "expert_parallel_size"),
    ("dtensor_cfg", "tensor_parallel_size"),
    ("dtensor_cfg", "activation_checkpointing"),
    ("dtensor_cfg", "fsdp_output_dtype"),
    ("dtensor_cfg", "empty_cache_after_train"),
    ("dtensor_cfg", "env_vars"),
    ("optimizer", "fused"),
    ("optimizer", "foreach"),
    ("optimizer", "lr"),
    ("backend", "experts"),
    ("backend", "dispatcher"),
    ("data", "dataset_name"),
]


def grab(path):
    txt = open(path, errors="ignore").read()
    m = re.search(r"MasterConfig\(policy=(\{.*)", txt)
    blob = m.group(1) if m else ""
    out = {}
    for scope, key in KEYS:
        # search for the key, preferring an occurrence near its scope word
        best = None
        for mm in re.finditer(r"'%s':\s*(\{[^{}]*\}|'[^']*'|[^,}\]]+)" % re.escape(key), blob):
            ctx = blob[max(0, mm.start() - 900):mm.start()]
            if scope == "policy" or scope.rstrip("_cfg") in ctx or scope in ctx:
                best = mm.group(1).strip()
                break
            if best is None:
                best = mm.group(1).strip()
        out[(scope, key)] = (best or "-")[:54]
    return out


vals = {name: grab(p) for name, p in RUNS.items()}
names = list(RUNS)

print("%-46s %-30s %-30s %s" % ("setting", names[0], names[1], ""))
print("-" * 118)
for k in KEYS:
    a, b = vals[names[0]][k], vals[names[1]][k]
    mark = "   <-- DIFFERENT" if a != b else ""
    print("%-46s %-30s %-30s%s" % (".".join(k), a[:30], b[:30], mark))
