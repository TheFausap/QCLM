#!/usr/bin/env bash
# Matched quantum-vs-decohered sweep across Hilbert-space dimensions.
# The decohered run is the SAME architecture with off-diagonal coherences zeroed
# each step => an n-state classical HMM. Identical params/optimizer/data.
set -e
cd "$(dirname "$0")/.."
STEPS=${STEPS:-1800}
BLOCK=${BLOCK:-64}
BATCH=${BATCH:-32}
LR=${LR:-3e-3}

run () {
  local dim=$1 mode=$2
  local flag=""; local tag="q_d${dim}"
  if [ "$mode" = "deco" ]; then flag="--decohere"; tag="c_d${dim}"; fi
  echo "=== RUN dim=$dim mode=$mode tag=$tag ==="
  python3 qlm/train.py --dim "$dim" --kraus 4 --block "$BLOCK" --batch "$BATCH" \
    --steps "$STEPS" --lr "$LR" --eval_every 300 --eval_batches 20 --threads 4 \
    --tag "$tag" --seed 0
}

for dim in 24 48; do
  run "$dim" quantum
  run "$dim" deco
done
echo "SWEEP_DONE"
