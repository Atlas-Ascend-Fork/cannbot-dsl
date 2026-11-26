# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""attn_epilogue 的 Torch 接口、注册和编译缓存。"""

# fmt: off

from functools import cache

import cannbotdsl as cbd
from cannbotdsl.aot import export
import torch
import torch_npu

if __package__:
    from .attn_epilogue_kernel import (
        GROUP_INPUT_DIM, HEAD_DIM, HEADS_PER_GROUP, MM1_OUTPUT_DIM, MX_SCALE_GROUP_SIZE, MX_SCALE_PAIR_SIZE,
        OUTPUT_DIM, ROPE_DIM,
    )
    from .attn_epilogue_tiling import FP4_NZ_BYTE_SHAPE, FP4_NZ_FORMAT, FP4_NZ_SHAPE, Tiling
else:
    from attn_epilogue_kernel import (
        GROUP_INPUT_DIM, HEAD_DIM, HEADS_PER_GROUP, MM1_OUTPUT_DIM, MX_SCALE_GROUP_SIZE, MX_SCALE_PAIR_SIZE,
        OUTPUT_DIM, ROPE_DIM,
    )
    from attn_epilogue_tiling import FP4_NZ_BYTE_SHAPE, FP4_NZ_FORMAT, FP4_NZ_SHAPE, Tiling


def _get_compile_params(*inputs):
    """读取实际输入规格，元组用于编译缓存，不保留 Tensor 引用。"""
    formats = []
    for index, tensor in enumerate(inputs):
        fmt = torch_npu.get_npu_format(tensor)
        if index == 1 and fmt in (FP4_NZ_FORMAT, "FRACTAL_NZ_C0_16"):
            fmt = FP4_NZ_FORMAT
        elif fmt == torch_npu.Format.NCHW or (index >= 3 and fmt != torch_npu.Format.FRACTAL_NZ):
            # NCHW 等普通布局复用 ND 编译。保留权重格式用于校验。
            fmt = torch_npu.Format.ND
        formats.append(fmt)
    return (
        tuple(tuple(x.shape) for x in inputs), tuple(x.dtype for x in inputs), tuple(formats),
    )


@cache
def _compile(shapes, dtypes, formats, *, device, block_num):
    """编译一次，并返回执行时使用的 GM 规格。"""
    kernel_type, tiling = Tiling.select_kernel(shapes, dtypes, formats, block_num)
    ng, m_align16 = tiling["ng"], tiling["m_align16"]
    mm2_k = ng * MM1_OUTPUT_DIM
    gm_specs = (
        ((ng * m_align16, GROUP_INPUT_DIM), torch.uint8, torch_npu.Format.ND),
        ((ng * m_align16, GROUP_INPUT_DIM // MX_SCALE_GROUP_SIZE), torch.uint8, torch_npu.Format.ND),
        ((m_align16, mm2_k), torch.uint8, torch_npu.Format.ND),
        ((m_align16, mm2_k // MX_SCALE_GROUP_SIZE), torch.uint8, torch_npu.Format.ND),
        ((tiling["t"], OUTPUT_DIM), torch.bfloat16, torch_npu.Format.ND),
    )
    dtype_map = {
        torch.bfloat16: cbd.dtypes.bfloat16, torch.float32: cbd.dtypes.float32,
        torch.float8_e4m3fn: cbd.dtypes.float8_e4m3fn,
        torch.float8_e8m0fnu: cbd.dtypes.float8_e8m0, torch.uint8: cbd.dtypes.uint8,
    }
    specs = []
    for index, (shape, dtype, fmt) in enumerate((*zip(shapes, dtypes, formats), *gm_specs)):
        if index == 1 and fmt == FP4_NZ_FORMAT:
            shape = (ng, *FP4_NZ_BYTE_SHAPE)
        storage_format = "nz" if fmt == torch_npu.Format.FRACTAL_NZ else "nd"
        # FP8 NZ 保留物理布局并按字节搬运，L1 计算时使用 FP8 视图。
        element_type = cbd.dtypes.uint8 if storage_format == "nz" and index in (1, 2) else dtype_map[dtype]
        specs.append(cbd.TensorSpec(shape, element_type, storage_format=storage_format))
    return cbd.compile(kernel_type(tiling).__call__, *specs), gm_specs


@export("attn_epilogue")
def export_attn_epilogue():
    """通过运行时相同的 Torch 规格导出 Native 二进制，无需创建 NPU Tensor。"""
    import json
    import os

    ng8_tokens, small_ng_tokens = (72, 80, 96, 120), (1, 16)
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            entries = json.load(stream)["attn_epilogue"]
        cases = []
        for entry in entries:
            block_num, ng = entry.get("block_num", 32), entry.get("ng", 8)
            if "first_nz" in entry or "second_nz" in entry:
                raise ValueError("use weight_format to specify the shared WOA/WOB layout")
            if entry.get("weight_fp4") or entry.get("second_fp4"):
                raise ValueError("WOB must use torch.float8_e4m3fn; use first_fp4 to select FP4 WOA")
            first_fp4 = entry.get("first_fp4", False)
            weight_format = entry.get("weight_format", "nz" if first_fp4 else "nd")
            if weight_format not in ("nd", "nz") or (first_fp4 and weight_format != "nz"):
                raise ValueError("weight_format must be nd or nz; FP4 WOA requires nz")
            tokens = entry.get("tokens", ng8_tokens if ng == 8 else small_ng_tokens)
            cases.extend((block_num, ng, t, weight_format == "nz", first_fp4) for t in tokens)
    else:
        cases = [
            (block_num, ng, t, weight_nz, first_fp4)
            for block_num in (28, 32)
            for ng in (8, 2, 1)
            for weight_nz, first_fp4 in ((False, False), (True, False), (True, True))
            for t in (ng8_tokens if ng == 8 else small_ng_tokens)
        ]

    nd, nz = torch_npu.Format.ND, torch_npu.Format.FRACTAL_NZ
    fp8, scale = torch.float8_e4m3fn, torch.float8_e8m0fnu
    device = torch.device("npu:0")
    # Native 构建需要实际提交编译请求，不能复用进程里已有的缓存句柄。
    _compile.cache_clear()
    try:
        for block_num, ng, t, weight_nz, first_fp4 in dict.fromkeys(cases):
            mm2_k = ng * MM1_OUTPUT_DIM
            wa_scale_groups, wb_scale_groups = GROUP_INPUT_DIM // MX_SCALE_PAIR_SIZE, mm2_k // MX_SCALE_PAIR_SIZE
            weight_format = nz if weight_nz else nd
            wa_scale_shape = (ng, MM1_OUTPUT_DIM, wa_scale_groups, 2)
            if first_fp4:
                wa_dtype, wa_format, wa_shape = torch.uint8, FP4_NZ_FORMAT, (ng, *FP4_NZ_SHAPE)
            elif weight_nz:
                wa_dtype, wa_format, wa_shape = fp8, weight_format, (ng, GROUP_INPUT_DIM, MM1_OUTPUT_DIM)
                wa_scale_shape = (ng, wa_scale_groups, MM1_OUTPUT_DIM, 2)
            else:
                wa_dtype, wa_format, wa_shape = fp8, weight_format, (ng, MM1_OUTPUT_DIM, GROUP_INPUT_DIM)
            wb_shape = (mm2_k, OUTPUT_DIM) if weight_nz else (OUTPUT_DIM, mm2_k)
            wb_scale_shape = (
                (wb_scale_groups, OUTPUT_DIM, 2) if weight_nz else (OUTPUT_DIM, wb_scale_groups, 2)
            )
            launch, _ = _compile(
                ((t, ng * HEADS_PER_GROUP, HEAD_DIM), wa_shape, wb_shape, wa_scale_shape, wb_scale_shape,
                 (t, ROPE_DIM), (t, ROPE_DIM)),
                (torch.bfloat16, wa_dtype, fp8, scale, scale, torch.float32, torch.float32),
                (nd, wa_format, weight_format, nd, nd, nd, nd),
                device=device, block_num=block_num,
            )
            launch.close()
    finally:
        _compile.cache_clear()


@torch.library.custom_op("cannbotdsl::attn_epilogue", mutates_args=(), device_types="npu")
def _attn_epilogue(
    x: torch.Tensor, woa: torch.Tensor, wob: torch.Tensor, descale_woa: torch.Tensor, descale_wob: torch.Tensor,
    rope_sin: torch.Tensor, rope_cos: torch.Tensor,
) -> torch.Tensor:
    shapes, dtypes, formats = _get_compile_params(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
    launch, gm_specs = _compile(
        shapes, dtypes, formats,
        device=x.device, block_num=cbd.get_platform_info(stream=torch.npu.current_stream(x.device)).cube_core_num,
    )
    gm_aq, gm_asc, gm_yq, gm_ysc, gm_out = (
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype, _ in gm_specs
    )
    if formats[1] == FP4_NZ_FORMAT:
        woa = torch.from_dlpack(woa).view(woa.shape[0], *FP4_NZ_BYTE_SHAPE)
    elif formats[1] == torch_npu.Format.FRACTAL_NZ:
        woa = woa.view(torch.uint8)
    if formats[2] == torch_npu.Format.FRACTAL_NZ:
        wob = wob.view(torch.uint8)
    launch(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, gm_aq, gm_asc, gm_yq, gm_ysc, gm_out)
    return gm_out


@_attn_epilogue.register_fake
def _attn_epilogue_fake(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
    """仅推导输出元信息，保留 T 的符号维度和输入设备。"""
    return x.new_empty((x.shape[0], OUTPUT_DIM), dtype=torch.bfloat16)


def attn_epilogue(
    x: torch.Tensor, woa: torch.Tensor, wob: torch.Tensor, descale_woa: torch.Tensor, descale_wob: torch.Tensor,
    rope_sin: torch.Tensor, rope_cos: torch.Tensor,
) -> torch.Tensor:
    """将 BF16 [T, NG*8, 512] 投影为 BF16 [T, 5120]。WOA 支持 FP8/FP4，WOB 使用 FP8。

    两级权重同为 ND 或同为 NZ。FP4 WOA 只接受官方格式转换生成的 uint8 C0_16 NZ。
    """
    return _attn_epilogue(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
