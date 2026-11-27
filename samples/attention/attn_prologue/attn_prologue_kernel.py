# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按静态 tiling 选择 AttnPrologue 的设备流水，并提供 host 启动入口。"""

__all__ = ["run"]

# fmt: off

import cannbotdsl as cbd

from .attn_prologue_kernel_l1_reuse import L1ReuseAttnPrologueKernel
from .attn_prologue_kernel_non_l1_reuse import (
    BranchBResidentAttnPrologueKernel,
    BranchBStreamingAttnPrologueKernel,
    GenericBResidentAttnPrologueKernel,
    GenericBStreamingAttnPrologueKernel,
)


def select_attn_prologue_kernel(tiling):
    """按 L1 复用、A 阶段编排和 B 阶段缓冲策略选择唯一的 Kernel 类。"""
    if tiling.l1_reuse:
        return L1ReuseAttnPrologueKernel
    if tiling.use_branch_pipeline:
        if tiling.b_resident:
            return BranchBResidentAttnPrologueKernel
        return BranchBStreamingAttnPrologueKernel
    if tiling.b_resident:
        return GenericBResidentAttnPrologueKernel
    return GenericBStreamingAttnPrologueKernel


def run(tiling):
    """用静态 tiling 创建无实例参数的 host 启动函数。"""

    @cbd.host
    def launch(gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr, gm_gamma_kv, gm_sin,
               gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d, gm_ws_qa, gm_ws_kv,
               norm_eps: cbd.dtypes.float32, gm_qr_bf16):
        """按静态 block_dim 启动完整 device 流水。"""
        kernel_cls = select_attn_prologue_kernel(tiling)
        op = kernel_cls(tiling)
        op[tiling.block_dim](gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr,
                             gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d,
                             gm_ws_qa, gm_ws_kv, norm_eps, gm_qr_bf16)

    return launch
