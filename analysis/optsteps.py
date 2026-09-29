import glob, gzip, json, os, sys, collections
p = glob.glob(os.path.join(sys.argv[1], "rank0.pt.trace.json.gz"))[0]
ev = json.load(gzip.open(p))["traceEvents"]
c = collections.Counter(e.get("name", "") for e in ev if e.get("cat") in ("cpu_op", "user_annotation"))
print("=== optimizer / step markers ===")
for n, k in c.items():
    low = n.lower()
    if "optimizer" in low or "zero_grad" in low or "clip_grad" in low or low.startswith("nn.module"):
        print("%6d  %s" % (k, n[:95]))
print("\n=== annotations that look like step boundaries ===")
for n, k in c.most_common():
    if k <= 40 and ("step" in n.lower() or "train" in n.lower() or "forward" in n.lower()):
        print("%6d  %s" % (k, n[:95]))
