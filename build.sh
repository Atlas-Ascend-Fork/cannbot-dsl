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

# CI 流水线统一入口：平台侧只需调用本脚本。
# Precision tests compile on demand with pytest -n 16; profile eligible cases afterward.
# 流程：环境校验 → 变更分析 → 跑相关算子的测试。
#
# 用法（平台侧 job 内）:
#   bash build.sh --mode=npu     # 设备环境，跑 npu 用例
#   bash build.sh --mode=cpu     # 无设备环境，跑非 npu 用例
#   bash build.sh --mode=perf    # 用 msprof 采集相关 npu 用例的性能
#   bash build.sh --mode=all
# 可选: --base=origin/master  变更对比基线（默认 origin/master）
#       --changed-list=FILE   直接使用外部变更清单（跳过 git diff）

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
MODE=""
BASE="origin/master"
CHANGED_LIST=""

# 自动激活 conda 环境（self-hosted runner 为独立进程，默认不激活 conda）。
# CANNBOT_CONDA_ENV 指定环境名；CANNBOT_CONDA_BASE 显式指定 conda 安装路径
# （root 运行下 PATH 无 conda 时必需）；置空或设 CANNBOT_NO_CONDA=1 可禁用。
CANNBOT_CONDA_ENV="${CANNBOT_CONDA_ENV:-cannbot}"
if [[ "${CANNBOT_NO_CONDA:-0}" != "1" && -n "$CANNBOT_CONDA_ENV" ]]; then
    CONDA_BASE="${CANNBOT_CONDA_BASE:-}"
    if [[ -z "$CONDA_BASE" ]] && command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    fi
    # Some self-hosted runners have conda installed but do not expose its
    # executable on PATH. Probe the standard image locations as a fallback.
    if [[ -z "$CONDA_BASE" ]]; then
        for candidate in /opt/conda /opt/miniconda3 /opt/anaconda3 /usr/local/miniconda3; do
            if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
                CONDA_BASE="$candidate"
                break
            fi
        done
    fi
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        if conda activate "$CANNBOT_CONDA_ENV" 2>/dev/null; then
            echo "[INFO] conda env activated: $CANNBOT_CONDA_ENV"
        else
            echo "[WARN] conda env '$CANNBOT_CONDA_ENV' not found, using default python"
        fi
    fi
fi

usage() {
    cat <<EOF
Usage: bash build.sh --mode=ci|cpu|npu|perf|all [--base=<branch>] [--changed-list=FILE]
EOF
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --mode=*) MODE="${arg#*=}" ;;
            --base=*) BASE="${arg#*=}" ;;
            --changed-list=*) CHANGED_LIST="${arg#*=}" ;;
            -h|--help) usage; exit 0 ;;
            *) echo "[ERROR] unknown argument: $arg" >&2; usage; exit 1 ;;
        esac
    done
    case "$MODE" in
        ci|cpu|npu|perf|all) ;;
        *) echo "[ERROR] --mode is required and must be ci|cpu|npu|perf|all" >&2; usage; exit 1 ;;
    esac
}

parse_args "$@"

# 1. 环境校验（只校验不安装，环境由平台预置）
echo "=== [1/3] env check ==="
bash "$REPO_ROOT/install_deps.sh" || exit 1

# 2. 变更分析：生成受影响算子
echo "=== [2/3] parse changed ops ==="
if [[ -z "$CHANGED_LIST" ]]; then
    CHANGED_LIST="$REPO_ROOT/log/changed_list.txt"
    mkdir -p "$REPO_ROOT/log"
    git diff --name-status "${BASE}...HEAD" > "$CHANGED_LIST" 2>/dev/null \
        || git diff --name-status HEAD~1 HEAD > "$CHANGED_LIST"
fi
[[ -s "$CHANGED_LIST" ]] || { echo "[INFO] no changed files, skip tests"; exit 0; }

ops=$("$PYTHON" "$REPO_ROOT/scripts/ci/parse_changed_ops.py" "$CHANGED_LIST") || exit 1
echo "affected ops: [$ops]"
if [[ -z "$ops" ]]; then
    echo "[INFO] no operator affected, skip tests"
    exit 0
fi

# 3. 跑测试（ops 输出以 ';' 分隔，转为 ',' 供 run_tests.sh --ops 使用）
echo "=== [3/3] run tests (mode=$MODE, ops=$ops) ==="
ops_csv="${ops//;/,}"
bash "$REPO_ROOT/scripts/ci/run_tests.sh" --ops="${ops_csv}" --mode="$MODE"
