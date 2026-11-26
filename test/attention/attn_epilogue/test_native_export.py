# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-side regression for the attn_epilogue Native export profiles."""

import ast
import json
from pathlib import Path
import runpy

import pytest


def test_attn_epilogue_export_profiles(monkeypatch, tmp_path):
    """Check export coverage and runtime TensorSpecs without an NPU."""
    from bisect import bisect_right
    from functools import cache
    from math import gcd
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[3]
    builder = runpy.run_path(str(root / "packaging/native/build.py"))
    operators, _ = builder["load_catalog"]()
    source_names = operators["attn_epilogue"]
    assert source_names == [
        "attention/attn_epilogue/attn_epilogue.py",
        "attention/attn_epilogue/attn_epilogue_tiling.py",
        "attention/attn_epilogue/attn_epilogue_kernel.py",
    ]
    source, tiling_source, _ = (builder["OPS_ROOT"] / name for name in source_names)
    tree = ast.parse(source.read_text())
    functions = {fn.name: fn for fn in tree.body if isinstance(fn, ast.FunctionDef)}
    collector = functions["export_attn_epilogue"]
    (registration,) = collector.decorator_list
    assert (
        registration.func.id == "export"
        and ast.literal_eval(registration.args[0]) == "attn_epilogue"
    )
    collector.decorator_list.clear()

    torch = SimpleNamespace(
        device=str,
        bfloat16="bf16",
        float32="fp32",
        float8_e4m3fn="fp8",
        float8_e8m0fnu="e8m0",
        float4_e2m1fn_x2="fp4",
        int8="int8",
        uint8="uint8",
    )
    dsl_dtypes = SimpleNamespace(
        bfloat16="cbd.bf16",
        float32="cbd.fp32",
        float8_e4m3fn="cbd.fp8",
        float8_e8m0="cbd.e8m0",
        uint8="cbd.uint8",
    )
    nd, nz, fp4_nz = 2, 29, 50
    capacities = dict(l1=524288, l0a=65536, l0b=65536, l0c=262144, ub=253952)
    requests, compiled, closed = [], [], []

    def compile_kernel(_, *specs):
        compiled.append(specs)
        return SimpleNamespace(close=lambda: closed.append(specs))

    namespace = {
        "cache": cache,
        "torch": torch,
        "torch_npu": SimpleNamespace(
            Format=SimpleNamespace(ND=nd, FRACTAL_NZ=nz, NCHW=0),
            get_npu_format=lambda tensor: tensor.format,
        ),
        "cbd": SimpleNamespace(
            dtypes=dsl_dtypes,
            TensorSpec=lambda shape, dtype, storage_format: (
                shape,
                dtype,
                storage_format,
            ),
            compile=compile_kernel,
            get_mem_size=capacities.__getitem__,
        ),
        "AttnEpilogueKernel": lambda _: SimpleNamespace(__call__=None),
        "bisect_right": bisect_right,
        "gcd": gcd,
        "_align_up": lambda size, align: (size + align - 1) // align * align,
        "_ceil_div": lambda dividend, divisor: (dividend + divisor - 1) // divisor,
        "HEADS_PER_GROUP": 8,
        "HEAD_DIM": 512,
        "GROUP_INPUT_DIM": 4096,
        "MM1_OUTPUT_DIM": 1024,
        "OUTPUT_DIM": 5120,
        "ROPE_DIM": 64,
        "MX_SCALE_GROUP_SIZE": 32,
        "MX_SCALE_PAIR_SIZE": 64,
        "FP4_TILE_N": 16,
        "FP4_TILE_BYTES": 256,
        "FP4_VALUES_PER_BYTE": 2,
        "BF16_BYTES": 2,
        "FP32_BYTES": 4,
        "MIX_VECTOR_PARTS": 2,
        "UB_ALIGNMENT": 32,
        "QUANT_CHUNK_ELEMENTS": 256,
        "QUANT_RECIPROCAL_ELEMS_PER_BYTE": 8,
    }
    # Execute the actual tiling validation and selection on the host.
    tiling_tree = ast.parse(tiling_source.read_text())
    tiling_tree.body = [
        node
        for node in tiling_tree.body
        if isinstance(node, ast.ClassDef)
        or (isinstance(node, ast.Assign) and node.targets[0].id != "BF16_BYTES")
    ]
    exec(compile(tiling_tree, str(tiling_source), "exec"), namespace)
    select_kernel = namespace["Tiling"].select_kernel

    def record_select_kernel(shapes, dtypes, formats, block_num):
        requests.append(((shapes, dtypes, formats), block_num))
        return select_kernel(shapes, dtypes, formats, block_num)

    monkeypatch.setattr(
        namespace["Tiling"], "select_kernel", staticmethod(record_select_kernel)
    )
    tree = ast.Module(
        body=[functions["_get_compile_params"], functions["_compile"], collector],
        type_ignores=[],
    )
    exec(compile(tree, str(source), "exec"), namespace)
    export_fn = namespace["export_attn_epilogue"]

    def input_metadata(
        t, ng, weight_nz, first_fp4, *, wa_scale_format=nd, packed_nz_format=fp4_nz
    ):
        wa_dtype, wa_shape = torch.float8_e4m3fn, (ng, 1024, 4096)
        wa_format = nz if weight_nz else nd
        wa_scale = (ng, 1024, 64, 2)
        if first_fp4:
            wa_dtype, wa_shape, wa_format = (
                torch.uint8,
                (ng, 1024, 2048),
                packed_nz_format,
            )
        elif weight_nz:
            wa_shape, wa_scale = (ng, 4096, 1024), (ng, 64, 1024, 2)
        wb_shape, wb_scale = (5120, ng * 1024), (5120, ng * 16, 2)
        if weight_nz:
            wb_shape, wb_scale = (ng * 1024, 5120), (ng * 16, 5120, 2)
        tensors = (
            ((t, ng * 8, 512), torch.bfloat16, nd),
            (wa_shape, wa_dtype, wa_format),
            (wb_shape, torch.float8_e4m3fn, nz if weight_nz else nd),
            (wa_scale, torch.float8_e8m0fnu, wa_scale_format),
            (wb_scale, torch.float8_e8m0fnu, nd),
            ((t, 64), torch.float32, nd),
            ((t, 64), torch.float32, nd),
        )
        return namespace["_get_compile_params"](
            *(
                SimpleNamespace(shape=shape, dtype=dtype, format=fmt)
                for shape, dtype, fmt in tensors
            )
        )

    monkeypatch.delenv("CANNBOTDSL_DS41_PROFILES", raising=False)
    export_fn()
    assert len(requests) == 48
    assert set(requests) == {
        (input_metadata(t, ng, use_nz, first_fp4), block_num)
        for ng, tokens in ((1, (1, 16)), (2, (1, 16)), (8, (72, 80, 96, 120)))
        for t in tokens
        for use_nz, first_fp4 in ((False, False), (True, False), (True, True))
        for block_num in (28, 32)
    }
    assert len(compiled) == 48 and compiled == closed
    assert not namespace["_compile"].cache_info().currsize

    profiles = tmp_path / "profiles.json"
    monkeypatch.setenv("CANNBOTDSL_DS41_PROFILES", str(profiles))
    profiles.write_text(
        json.dumps(
            {
                "attn_epilogue": [
                    {
                        "block_num": 32,
                        "ng": 2,
                        "first_fp4": True,
                        "tokens": [17, 17, 257],
                    },
                    {"block_num": 28, "ng": 1, "weight_format": "nz", "tokens": [7]},
                    {
                        "block_num": 32,
                        "ng": 8,
                        "first_fp4": True,
                        "weight_format": "nz",
                        "tokens": [72],
                    },
                ]
            }
        )
    )
    requests.clear()
    compiled.clear()
    closed.clear()
    export_fn()
    expected = [
        (input_metadata(17, 2, True, True), 32),
        (input_metadata(257, 2, True, True), 32),
        (input_metadata(7, 1, True, False), 28),
        (input_metadata(72, 8, True, True, packed_nz_format="FRACTAL_NZ_C0_16"), 32),
    ]
    assert requests == expected and compiled == closed
    aot_specs = compiled.copy()
    compiled.clear()
    for params, block_num in expected:
        namespace["_compile"](*params, device="npu:0", block_num=block_num)
    assert compiled == aot_specs
    cache_info = namespace["_compile"].cache_info()
    nchw_scale = input_metadata(7, 1, True, False, wa_scale_format=0)
    namespace["_compile"](*nchw_scale, device="npu:0", block_num=28)
    assert namespace["_compile"].cache_info().hits == cache_info.hits + 1
    assert compiled == aot_specs
    assert aot_specs[0][1:3] == (
        ((2, 128, 16384), dsl_dtypes.uint8, "nd"),
        ((2048, 5120), dsl_dtypes.uint8, "nz"),
    )
    assert aot_specs[0][3] == ((2, 1024, 64, 2), dsl_dtypes.float8_e8m0, "nd")
    assert aot_specs[3][1] == ((8, 128, 16384), dsl_dtypes.uint8, "nd")
    assert aot_specs[3][3] == ((8, 1024, 64, 2), dsl_dtypes.float8_e8m0, "nd")
    assert aot_specs[0][7:] == (
        ((64, 4096), dsl_dtypes.uint8, "nd"),
        ((64, 128), dsl_dtypes.uint8, "nd"),
        ((32, 2048), dsl_dtypes.uint8, "nd"),
        ((32, 64), dsl_dtypes.uint8, "nd"),
        ((17, 5120), dsl_dtypes.bfloat16, "nd"),
    )

    for profile, message in (
        ({"second_fp4": True}, "WOB must use torch.float8_e4m3fn"),
        ({"first_fp4": True, "weight_format": "nd"}, "FP4 WOA requires nz"),
        ({"first_nz": True, "second_nz": False}, "use weight_format"),
    ):
        profiles.write_text(json.dumps({"attn_epilogue": [profile]}))
        with pytest.raises(ValueError, match=message):
            export_fn()

    shapes, dtypes, formats = expected[-1][0]
    for dtype, layout, message in (
        (torch.float4_e2m1fn_x2, fp4_nz, "WOA must use MXFP8"),
        (torch.int8, nz, "WOA must use MXFP8"),
        (torch.uint8, nz, "FP4 WOA requires uint8 FRACTAL_NZ_C0_16"),
        (torch.uint8, nd, "FP4 WOA requires uint8 FRACTAL_NZ_C0_16"),
    ):
        with pytest.raises(ValueError, match=message):
            namespace["_compile"](
                shapes,
                (dtypes[0], dtype, *dtypes[2:]),
                (formats[0], layout, *formats[2:]),
                device="npu:0",
                block_num=32,
            )
    for params in (input_metadata(17, 2, True, False), expected[-1][0]):
        shapes, dtypes, formats = params
        with pytest.raises(
            ValueError, match="WOA and WOB must both use ND or both use NZ"
        ):
            namespace["_compile"](
                shapes,
                dtypes,
                (*formats[:2], nd, *formats[3:]),
                device="npu:0",
                block_num=32,
            )
    shapes, dtypes, formats = input_metadata(17, 2, False, False)
    with pytest.raises(ValueError, match="WOA and WOB must both use ND or both use NZ"):
        namespace["_compile"](
            shapes,
            dtypes,
            (*formats[:2], nz, *formats[3:]),
            device="npu:0",
            block_num=32,
        )
    with pytest.raises(ValueError, match="activation must use torch.bfloat16 ND"):
        namespace["_compile"](
            shapes, dtypes, (nz, *formats[1:]), device="npu:0", block_num=32
        )
    assert compiled == aot_specs
