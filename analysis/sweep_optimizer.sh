#!/bin/bash
# Does the upstream claim "when using Dtensor, we need to set foreach and fused to
# False" (examples/configs/sft.yaml:111-114) still hold, and what does it cost?
#
# Unfused AdamW launches one set of elementwise kernels per parameter tensor and
# calls .item() once per parameter -- measured on rank 0: 2,984 kernels and 756
# device syncs per optimizer step. Each sync drains the CUDA launch queue, which
# paces every rank's arrival at the next collective.
set -uo pipefail
run() {
    local name=$1; shift
    echo "############## $name : $* ##############"
    date -u +"start %H:%M:%S UTC"
    EXP_NAME=sft-nanov3-$name bash ~/assn1-f/analysis/run_collectives.sh "$@" \
        > /mnt/scratch/chloe_wong/logs/$name.out 2>&1
    local rc=$?
    date -u +"end   %H:%M:%S UTC"
    if [ $rc -eq 0 ]; then echo "$name OK"; else
        echo "$name FAILED rc=$rc -- last lines:"
        grep -A15 "Traceback (most recent call last)" /mnt/scratch/chloe_wong/runs/sft-nanov3-$name/run.log 2>/dev/null | tail -20 | cut -c1-170
    fi
    echo
}
run G0-optbase
run G1-foreach policy.optimizer.kwargs.foreach=true
run G2-fused   policy.optimizer.kwargs.fused=true
echo SWEEP_DONE
