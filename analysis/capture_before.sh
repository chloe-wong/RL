#!/bin/bash
# Capture the MISSING half of the report's before/after: an Nsight Systems trace of the
# Track B configuration BEFORE the three DTensor optimizations.
#
#   before : sft-nanov3-cascade2chat-8k-chunkedce-1n8g.yaml   5.685% MFU   8.020 s/step
#   after  : sft-nanov3-cascade2chat-8k-best-1n8g.yaml       10.355% MFU   6.530 s/step
#            (already captured at profiles/chat8k/nsys/)
#
# Same 8k context, same global batch, same data order, so the pair is a like-for-like
# comparison of bf16 FSDP output + the allocator pair + fused AdamW.
#
# PROFILE_TAG=chat8k-before writes to profiles/chat8k-before/, which the nsys mode wipes
# with rm -rf at startup -- so the tag MUST differ from chat8k or the existing "after"
# capture is destroyed.
set -uo pipefail
ME=$(whoami)
LIMIT_MIB=5120   # tolerate an idle CUDA context, block on a real training job (17-20+ GiB)

foreign_mib () {
  local t=0 pid mem u
  while read -r pid mem; do
    [ -z "$pid" ] && continue
    u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && t=$((t + mem))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ',')
  echo "$t"
}

for w in $(seq 1 240); do
  f=$(foreign_mib)
  [ "$f" -le "$LIMIT_MIB" ] && break
  [ $((w % 10)) -eq 1 ] && echo "waiting: ${f} MiB foreign ($(date +%H:%M:%S))"
  sleep 60
done
f=$(foreign_mib)
if [ "$f" -gt "$LIMIT_MIB" ]; then echo "GAVE UP after 4h: ${f} MiB foreign"; exit 1; fi

bash ~/assn1-f/analysis/gpu_cleanup.sh
echo "== capture start $(date +%H:%M:%S)  (foreign ${f} MiB)"
PROFILE_TAG=chat8k-before \
CONFIG=$HOME/assn1-f/configs/sft-nanov3-cascade2chat-8k-chunkedce-1n8g.yaml \
  bash ~/assn1-f/profile_sft_nanov3.sh nsys
echo "   exit=$?  $(date +%H:%M:%S)"
O=/mnt/scratch/chloe_wong/profiles/chat8k-before/nsys
echo "== produced: $(ls $O/*.nsys-rep 2>/dev/null | wc -l) nsys-rep files"
ls -lh $O/*.nsys-rep 2>/dev/null | awk '{print "   " $5, $9}'
# The sudo sampler leaves root-owned files; a root-owned inductor cache broke a run before.
sudo chown -R "$ME": "$O" 2>/dev/null
echo "== capture done $(date +%H:%M:%S)"
