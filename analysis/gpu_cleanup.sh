#!/bin/bash
# Tear down every trainer process and wait for the GPUs to actually release memory.
#
# Bracket patterns ([r]un_sft) so a pkill cannot match the ssh command line that invoked
# it -- that self-match killed the connection on the first attempt. For the same reason
# this script must NOT kill sweep_deepep.sh: the sweep calls this between arms, so that
# pattern made the sweep terminate itself. Kill trainers only.
pkill -f '[r]un_sft.py' 2>/dev/null
sleep 5
ray stop --force >/dev/null 2>&1
sleep 5
pkill -9 -f '[r]un_sft.py' 2>/dev/null
pkill -9 -f '[r]ay::'     2>/dev/null
pkill -9 -f '[r]aylet'    2>/dev/null
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "$used" -lt 2000 ] && { echo "   GPUs clear (max ${used} MiB) after $((i*4))s"; exit 0; }
  sleep 4
done
echo "   STILL BUSY: max ${used} MiB"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
exit 1
