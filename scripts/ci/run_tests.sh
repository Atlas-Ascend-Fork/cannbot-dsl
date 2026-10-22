#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR=""
MODE=""
OPS_LIST=""

# torch_npu 依赖 libhccl 等动态库，需先加载 CANN 环境
source_cann_env() {
    local candidates=(
        "${ASCEND_TOOLKIT_HOME:-}/set_env.sh"
        "$HOME/Ascend/ascend-toolkit/set_env.sh"
        "/usr/local/Ascend/cann-9.2.0/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    )
    for env_sh in "${candidates[@]}"; do
        if [[ -f "$env_sh" ]]; then
            set +u
            # shellcheck disable=SC1090
            source "$env_sh"
            set -u
            return 0
        fi
    done
    echo "[WARN] ascend-toolkit set_env.sh not found; torch_npu may fail to import"
}

source_cann_env

# torch 2.12 auto-loads torch_npu through an entry point.  Loading it before
# CANN is initialized can fail on libhccl, and repeated collection can register
# the same torch library twice.  NPU tests import torch_npu explicitly later.
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

usage() {
    cat <<EOF
Usage: bash scripts/ci/run_tests.sh [options]
Options:
    --ops=op1,op2   Operators to test (comma separated). Default: all.
    --mode=ci|cpu|npu|perf|all
                    cpu: run tests not marked 'npu' (no device needed)
                    npu: run only 'npu' marked tests (device needed)
                    ci: one pytest -n 16 precision run, then serial eligible performance
                    perf: profile only passed AICore/Mix cases in CI_PERF_ENTRY_REPORT
                    all: run everything
                    Default: auto-detect via NPU device count.
    --log-dir=DIR   Directory for pytest log and junit.xml. Default: \${REPO_ROOT}/log
    Perf settings: CI_PERF_WARMUP=1, CI_PERF_REPEAT=10 (minimum 2).
                   perf skips precision checks and excludes warmups.
    -h|--help       Print this help.
EOF
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --ops=*)
                OPS_LIST="${arg#*=}"
                ;;
            --mode=*)
                MODE="${arg#*=}"
                ;;
            --log-dir=*)
                LOG_DIR="${arg#*=}"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "[ERROR] unknown argument: $arg" >&2
                usage
                exit 1
                ;;
        esac
    done
}

detect_npu_count() {
    "$PYTHON" - <<'EOF' 2>/dev/null || echo 0
try:
    import torch_npu
    print(torch_npu.npu.device_count())
except Exception:
    print(0)
EOF
}

build_test_targets() {
    local ops_list="$1"
    TEST_TARGETS=()
    if [[ -z "$ops_list" || "$ops_list" == "all" ]]; then
        TEST_TARGETS=("test")
        return
    fi
    IFS=',' read -ra ops <<< "$ops_list"
    for op in "${ops[@]}"; do
        TEST_TARGETS+=("test/${op}")
    done
}

print_summary() {
    local junit="$1"
    "$PYTHON" - "$junit" "$MODE" <<'EOF'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
try:
    root = ET.parse(path).getroot()
except Exception:
    print("[SUMMARY] junit.xml unavailable")
    sys.exit(1 if sys.argv[2] in ("perf",) else 0)

# pytest writes <testsuites><testsuite .../></testsuites>; pick the suite carrying the counts
suite = root
if suite.tag == "testsuites" and len(suite) > 0:
    suite = suite[0]

tests = int(suite.get("tests", 0))
failures = int(suite.get("failures", 0))
errors = int(suite.get("errors", 0))
skipped = int(suite.get("skipped", 0))
passed = tests - failures - errors - skipped
rate = passed / tests * 100 if tests else 0.0
print(f"[SUMMARY] TOTAL: {tests} | PASSED: {passed} | FAILED: {failures} | "
      f"ERRORS: {errors} | SKIPPED: {skipped} | RATE: {rate:.2f}%")
if failures or errors:
    sys.exit(1)
if sys.argv[2] in ("perf",) and passed == 0 and not (tests > 0 and skipped == tests):
    print("[ERROR] no NPU tests executed; performance collection is incomplete")
    sys.exit(1)
EOF
}

parse_args "$@"
[[ -n "$LOG_DIR" ]] || LOG_DIR="${REPO_ROOT}/log"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)" || exit 1

if [[ -z "$MODE" ]]; then
    if [[ "$(detect_npu_count)" -gt 0 ]]; then
        MODE="npu"
    else
        MODE="cpu"
    fi
    echo "[INFO] auto-detected mode: ${MODE}"
fi

case "$MODE" in
    ci)
        # Keep precision and performance on the same checkout and SDK cache.
        LOG_DIR="$(mktemp -d "$LOG_DIR/ci.XXXXXXXX")" || exit 1
        export CANNBOTDSL_CACHE_DIR="$LOG_DIR/sdk-cache"
        export CANNBOTDSL_CACHE_MEM_DISABLE=0 CANNBOTDSL_CACHE_DISK_DISABLE=0
        export CI_CASE_RESULTS="$LOG_DIR/precision.json"
        export CI_OBSERVE_DSL=1
        # Preserve host/device isolation across CI runs. No per-group locks remain.
        lock_root="${CI_LOCK_DIR:-/var/lock/cannbot-ci}"
        mkdir -p "$lock_root" || exit 1
        exec 9>"$lock_root/batch-compile.lock" || exit 1
        echo "[INFO] waiting for host CI slot"
        flock 9 || exit 1
        exec 8>"$lock_root/npu-test.lock" || exit 1
        flock 8 || exit 1
        echo "[INFO] host CI slot acquired"
        bash "$0" --ops="$OPS_LIST" --mode=all --log-dir="$LOG_DIR/precision"
        precision_rc=$?
        unset CI_CASE_RESULTS CI_OBSERVE_DSL
        export CI_PERF_ENTRY_REPORT="$LOG_DIR/precision.json"
        bash "$0" --ops="$OPS_LIST" --mode=perf --log-dir="$LOG_DIR"
        perf_rc=$?
        echo "[INFO] precision exit code=$precision_rc; performance exit code=$perf_rc"
        echo "[INFO] performance is informational; see uploaded tables and logs"
        exit "$precision_rc"
        ;;
    cpu)
        MARKERS=(-m "not npu")
        ;;
    npu|perf)
        MARKERS=(-m "npu")
        ;;
    all)
        MARKERS=()
        ;;
    *)
        echo "[ERROR] unsupported mode: ${MODE}. Supported: ci, cpu, npu, perf, all" >&2
        exit 1
        ;;
esac

if [[ "$MODE" == "perf" ]]; then
    MSPROF="${MSPROF:-msprof}"
    PERF_REPEAT="${CI_PERF_REPEAT:-10}"
    PERF_WARMUP="${CI_PERF_WARMUP:-1}"
    if [[ ! "$PERF_REPEAT" =~ ^[0-9]+$ || ! "$PERF_WARMUP" =~ ^[0-9]+$ ]] ||
        (( 10#$PERF_REPEAT < 2 )); then
        echo "[ERROR] CI_PERF_REPEAT must be >= 2; CI_PERF_WARMUP must be >= 0" >&2
        exit 1
    fi
    echo "[INFO] performance: warmup=$PERF_WARMUP, measured repetitions=$PERF_REPEAT; metric=min device time"
    echo "[INFO] precision already checked; profile passed AICore/Mix cases only"
    echo "[INFO] sample association: device MSTX ranges; host time is diagnostic only"
    # Each collection has its own directory, so earlier reports cannot hide a failed run.
    mkdir -p "$LOG_DIR/perf"
    LOG_DIR="$(mktemp -d "$LOG_DIR/perf/run.XXXXXXXX")" || exit 1
    echo "[INFO] performance reports: $LOG_DIR"
fi

build_test_targets "$OPS_LIST"
echo "[INFO] mode: ${MODE}, targets: ${TEST_TARGETS[*]}"
echo "[INFO] running pytest ..."

JUNIT="${LOG_DIR}/junit.xml"
cd "$REPO_ROOT"
# Bound CPU math-library threads per worker, as in the measured baseline.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
TEST_COMMAND=("$PYTHON" -m pytest "${TEST_TARGETS[@]}" "${MARKERS[@]}" --junitxml="$JUNIT")
if [[ "$MODE" != "perf" ]]; then
    workers="${CI_TEST_WORKERS:-16}"
    if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]] || (( workers > 64 )); then
        echo "[ERROR] CI_TEST_WORKERS must be an integer from 1 to 64" >&2
        exit 1
    fi
    echo "[INFO] pytest workers=$workers; dist=load; worker restart disabled"
    TEST_COMMAND+=(-n "$workers" --dist=load --max-worker-restart=0)
fi
if [[ -n "${CI_CASE_RESULTS:-}" ]]; then
    TEST_COMMAND+=(-p scripts.ci.case_results --ci-case-results="$CI_CASE_RESULTS")
    if [[ "${CI_OBSERVE_DSL:-0}" == "1" ]]; then
        TEST_COMMAND+=(--ci-observe-dsl)
    fi
fi
if [[ "$MODE" == "perf" ]]; then
    TEST_COMMAND+=(-p scripts.ci.perf_cases --ci-perf-cases="$LOG_DIR/cases.jsonl")
    TEST_COMMAND+=(--ci-perf-repeat="$PERF_REPEAT" --ci-perf-warmup="$PERF_WARMUP")
    if [[ -z "${CI_PERF_ENTRY_REPORT:-}" ]]; then
        echo "[ERROR] CI_PERF_ENTRY_REPORT is required; run --mode=ci for precision and performance"
        exit 1
    fi
    selection="$LOG_DIR/performance-selection.json"
    "$PYTHON" -m scripts.ci.case_results --select-perf="$CI_PERF_ENTRY_REPORT" \
        --output="$selection" || exit 1
    if "$PYTHON" -c 'import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))["selected"]))' "$selection"; then
        echo "[INFO] no eligible AICore/Mix cases; skip Profiler"
        exit 0
    fi
    TEST_COMMAND+=(-p scripts.ci.case_results --ci-perf-selection="$selection"
        --ci-perf-entry-report="$CI_PERF_ENTRY_REPORT")
    if ! command -v "$MSPROF" >/dev/null 2>&1; then
        echo "[ERROR] msprof not found; install CANN profiling tools or set MSPROF" >&2
        exit 1
    fi
    TEST_COMMAND=("$MSPROF" --ai-core=on --aic-metrics=PipeUtilization --l2=off --task-time=on --msproftx=on
        --output="$LOG_DIR/profiling" "${TEST_COMMAND[@]}")
fi
set +e
"$PYTHON" -m scripts.ci.timing --output="$LOG_DIR/ci_timing.csv" \
    --phase=pytest --scope="$MODE" -- "${TEST_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/pytest.log"
rc=${PIPESTATUS[0]}
set -e

summary_rc=0
print_summary "$JUNIT" || summary_rc=$?
if [[ "$MODE" == "perf" ]]; then
    perf_rc=0
    "$PYTHON" "$REPO_ROOT/scripts/ci/perf_summary.py" \
        --profile-dir="$LOG_DIR/profiling" --cases="$LOG_DIR/cases.jsonl" \
        --output="$LOG_DIR/perf_summary.csv" || perf_rc=$?
    "$PYTHON" - "$LOG_DIR/performance-status.json" "$rc" "$perf_rc" <<'EOF'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"profiler_exit_code": int(sys.argv[2]),
    "summary_exit_code": int(sys.argv[3])}), encoding="utf-8")
EOF
    if [[ -s "$LOG_DIR/perf_summary.csv" && -n "${ATOMGIT_OUTPUT:-}" ]]; then
        mkdir -p "$LOG_DIR/results"
        cp "$LOG_DIR"/perf_*.csv "$LOG_DIR/results/"
        "$PYTHON" -m zipfile -c "$LOG_DIR/perf_results.zip" "$LOG_DIR"/results/*.csv
        echo "perf_summary=$LOG_DIR/perf_summary.csv" >> "$ATOMGIT_OUTPUT"
        echo "perf_results=$LOG_DIR/results" >> "$ATOMGIT_OUTPUT"
        echo "perf_archive=$LOG_DIR/perf_results.zip" >> "$ATOMGIT_OUTPUT"
        echo "perf_run_id=$(basename "$LOG_DIR")" >> "$ATOMGIT_OUTPUT"
    fi
    # An empty selection is a profiling failure, rather than a successful skip.
    [[ "$rc" -ne 5 ]] || exit 1
    [[ "$rc" -eq 0 ]] || exit "$rc"
    [[ "$summary_rc" -eq 0 && "$perf_rc" -eq 0 ]] || exit 1
fi
[[ "$summary_rc" -eq 0 ]] || exit "$summary_rc"
if [[ "$rc" -eq 5 ]]; then
    echo "[INFO] no tests collected in this mode, treated as pass"
    exit 0
fi
exit "$rc"
