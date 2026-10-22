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

# 环境校验脚本：只从本地 wheel 修复依赖，不访问 pip 镜像。
# cannbotdsl / torch / torch_npu 不在 pip 镜像上，由平台/CI 预置环境按本地 wheel 安装
# （参考目录：cannbotdsl -> /opt/cannbot-dsl/whl，torch(_npu) -> /opt/cannbot-dsl/cann）。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
CANNBOTDSL_WHEEL_ROOT="${CANNBOTDSL_WHEEL_ROOT:-/opt/cannbot-dsl}"
CANNBOTDSL_WHEEL_ROOTS=(
    "$CANNBOTDSL_WHEEL_ROOT"
    "${HOME:-}/.cache/cannbot-bootstrap"
)

# 1. 可选地加载 CANN 环境（torch_npu 依赖 libhccl 等动态库）
source_cann_env() {
    local candidates=(
        "${ASCEND_TOOLKIT_HOME:-}/set_env.sh"
        "$HOME/Ascend/ascend-toolkit/set_env.sh"
        "/usr/local/Ascend/cann-9.2.0/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    )
    for env_sh in "${candidates[@]}"; do
        if [[ -f "$env_sh" ]]; then
            echo "[INFO] sourcing CANN env: ${env_sh}"
            # set_env.sh 对未定义变量做追加，需临时放开 set -u
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

# 2. cannbotdsl 不可导入时，尝试从本地 wheel 修复。
check_cannbotdsl() {
    "$PYTHON" - <<'EOF'
import cannbotdsl
EOF
}

if ! check_cannbotdsl >/dev/null 2>&1
then
    local_wheel=""
    for wheel_root in "${CANNBOTDSL_WHEEL_ROOTS[@]}"; do
        if [[ -d "$wheel_root" ]]; then
            local_wheel="$(find "$wheel_root" -type f \
                \( -iname 'cannbot_dsl-*.whl' -o -iname 'cannbot-dsl-*.whl' \
                -o -iname 'cannbotdsl-*.whl' -o -iname 'cannbotdsl_*.whl' \) \
                -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
        fi
        [[ -n "$local_wheel" ]] && break
    done
    if [[ -n "$local_wheel" ]]; then
        echo "[INFO] installing local cannbotdsl wheel: ${local_wheel}"
        if ! "$PYTHON" -m pip install --quiet --no-deps --force-reinstall "$local_wheel"; then
            echo "[ERROR] failed to install cannbotdsl wheel: ${local_wheel}" >&2
            exit 1
        fi
        if ! check_cannbotdsl >/dev/null 2>&1; then
            echo "[ERROR] installed cannbotdsl wheel cannot be imported" >&2
            exit 1
        fi
    else
        echo "[ERROR] cannbotdsl is unavailable and no local wheel was found." >&2
        echo "[ERROR] searched: ${CANNBOTDSL_WHEEL_ROOTS[*]}" >&2
        exit 1
    fi
fi

# 3. 校验关键依赖（find_spec 只查存在性，不触发 torch_npu 加载依赖）
if ! "$PYTHON" - <<'EOF'
import importlib.util

required = {
    "cannbotdsl": "local wheel under /opt/cannbot-dsl",
    "torch": "local wheel under /opt/cannbot-dsl",
    "torch_npu": "local wheel under /opt/cannbot-dsl",
    "numpy": "pip",
    "ml_dtypes": "pip",
    "pytest": "pip",
    "xdist": "pip install pytest-xdist==3.8.0",
    "yaml": "pip",
}
missing = [f"{m} ({src})" for m, src in required.items() if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] missing modules:")
    for m in missing:
        print(f"  - {m}")
    print("[ERROR] please install them in the target conda env before running CI.")
    raise SystemExit(1)
print("[INFO] all required modules importable")
EOF
then
    exit 1
fi

# 4. 探测 NPU 设备（决定是否运行 npu 标记的用例）
npu_count=$("$PYTHON" - <<'EOF' 2>/dev/null || echo 0
try:
    import torch_npu
    print(torch_npu.npu.device_count())
except Exception:
    print(0)
EOF
)
if [[ "$npu_count" -gt 0 ]]; then
    echo "[INFO] NPU device detected: ${npu_count} device(s)"
else
    echo "[INFO] No NPU device detected; npu-marked tests will be skipped."
fi
