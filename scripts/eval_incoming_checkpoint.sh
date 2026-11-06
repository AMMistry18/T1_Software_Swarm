#!/usr/bin/env bash
# Evaluate a checkpoint dropped in experiments/checkpoints/incoming/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INCOMING="$ROOT/experiments/checkpoints/incoming"
OUT="$ROOT/experiments/results/incoming_eval"
mkdir -p "$OUT"

CKPT=""
for cand in "$INCOMING/best_model.zip" "$INCOMING"/*.zip; do
  if [[ -f "$cand" ]]; then
    CKPT="$cand"
    break
  fi
done

if [[ -z "$CKPT" ]]; then
  echo "No .zip found in $INCOMING" >&2
  echo "Drop best_model.zip there, then re-run." >&2
  exit 1
fi

echo "Using checkpoint: $CKPT"

# Prefer project venv / uv 3.12 env if present
if [[ -x /tmp/t1_ci_test/bin/python ]]; then
  PY=/tmp/t1_ci_test/bin/python
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY=python3
fi

export PYTHONPATH=src
export MPLBACKEND=Agg

"$PY" -m tar_swarm.mission_planning.cli evaluate \
  --scenario one_mapper_small \
  --checkpoint "$CKPT" \
  --output "$OUT/eval_one_mapper_small.json"

"$PY" -m tar_swarm.mission_planning.cli plan \
  --scenario one_mapper_small \
  --checkpoint "$CKPT" \
  --output "$OUT/plan_rl.json" || true

"$PY" -m tar_swarm.mission_planning.cli plan \
  --scenario one_mapper_small \
  --fallback static \
  --output "$OUT/plan_static.json" || true

"$PY" -m tar_swarm.mission_planning.cli plan \
  --scenario one_mapper_small \
  --fallback search \
  --output "$OUT/plan_search.json" || true

echo "Wrote results under $OUT"
ls -la "$OUT"
