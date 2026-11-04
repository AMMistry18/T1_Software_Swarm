#!/usr/bin/env bash
# Mirror .github/workflows/python.yml locally (needs Python 3.12+).
# Usage:
#   ./scripts/run_ci_locally.sh           # all jobs
#   ./scripts/run_ci_locally.sh test      # one of: network | planning | test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3.12 >/dev/null 2>&1 && ! command -v uv >/dev/null 2>&1; then
  echo "Need python3.12 or uv (https://docs.astral.sh/uv/) to match CI." >&2
  exit 1
fi

make_venv() {
  local dir="$1"
  rm -rf "$dir"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$dir" --python 3.12 --seed
  else
    python3.12 -m venv "$dir"
  fi
}

# Drop ROS / system site packages that can break pytest plugin discovery.
clear_env=(env -u PYTHONPATH -u ROS_DISTRO -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH)
export MPLBACKEND=Agg

run_network() {
  echo "======== job: network ========"
  local v=/tmp/t1_ci_network
  make_venv "$v"
  "${clear_env[@]}" "$v/bin/python" -m pip install -q -r requirements-network.txt pytest
  "${clear_env[@]}" PYTHONPATH=src "$v/bin/python" -m pytest -q tests/test_network.py
}

run_planning() {
  echo "======== job: planning ========"
  local v=/tmp/t1_ci_planning
  make_venv "$v"
  "${clear_env[@]}" "$v/bin/python" -m pip install -q -r requirements-planning.txt pytest ruff
  "${clear_env[@]}" "$v/bin/python" -m pip install -q -e .
  "${clear_env[@]}" "$v/bin/python" -m ruff check src/tar_swarm/mission_planning tests/test_mission_planning.py
  "${clear_env[@]}" PYTHONPATH=src "$v/bin/python" -m pytest -q tests/test_mission_planning.py
}

run_test() {
  echo "======== job: test ========"
  local v=/tmp/t1_ci_test
  make_venv "$v"
  "${clear_env[@]}" "$v/bin/python" -m pip install -q --upgrade pip
  "${clear_env[@]}" "$v/bin/python" -m pip install -q -r requirements.txt -r requirements-dev.txt -r requirements-planning.txt
  "${clear_env[@]}" "$v/bin/python" -m pip install -q -e .
  "${clear_env[@]}" "$v/bin/python" -m ruff check .
  "${clear_env[@]}" "$v/bin/python" -m pytest -q
}

job="${1:-all}"
case "$job" in
  network) run_network ;;
  planning) run_planning ;;
  test) run_test ;;
  all)
    run_network
    run_planning
    run_test
    echo "======== all CI jobs passed ========"
    ;;
  *)
    echo "Usage: $0 [network|planning|test|all]" >&2
    exit 2
    ;;
esac
