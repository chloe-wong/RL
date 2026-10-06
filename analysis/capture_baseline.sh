#!/bin/bash
# Capture the report's BASELINE for Nsight Systems, the missing half of the before/after.
#
#   before : sft-nanov3-cascade2chat-4k-1n8g.yaml        5.195% MFU   6.510 s/step   411 TFLOPS
#   after  : sft-nanov3-cascade2chat-8k-best-1n8g.yaml  10.355% MFU   6.530 s/step   820 TFLOPS
#            (already captured at profiles/chat8k/nsys/)
#
# This is the configuration TRACK_B.md's "Baseline" section documents and the row its Summary
# measures "baseline to best" from, so it is the pair the report actually needs.
#
# An earlier attempt used 8k-chunkedce for equal-sequence-length comparison; it failed on a
# genuine memory limit (76.29 GiB own, 1.46 GiB foreign). A first 4k attempt also failed, but
# on CONTENTION (58.39 GiB own, 20.6 GiB foreign) -- not the same cause.
# It OOMed at step 2 with 76.29 GiB in its own process and only 932 MiB held by anyone else:
# nsys tracing adds buffers, and that config sits at the memory edge exactly because it lacks
# the bf16 FSDP output setting. Its fragility is the reason that optimization was worth +67%.
# The 4k baseline has roughly half the activation memory and room to be traced.
#
# PROFILE_TAG=chat4k-baseline writes to profiles/chat4k-baseline/, which the nsys mode wipes
# with rm -rf at startup -- the tag MUST differ from chat8k or the existing capture is lost.
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
PROFILE_TAG=chat4k-baseline \
CONFIG=$HOME/assn1-f/configs/sft-nanov3-cascade2chat-4k-1n8g.yaml \
  bash ~/assn1-f/profile_sft_nanov3.sh nsys
echo "   exit=$?  $(date +%H:%M:%S)"
O=/mnt/scratch/chloe_wong/profiles/chat4k-baseline/nsys
echo "== produced: $(ls $O/*.nsys-rep 2>/dev/null | wc -l) nsys-rep files"
ls -lh $O/*.nsys-rep 2>/dev/null | awk '{print "   " $5, $9}'
# The sudo sampler leaves root-owned files; a root-owned inductor cache broke a run before.
sudo chown -R "$ME": "$O" 2>/dev/null
echo "== capture done $(date +%H:%M:%S)"
