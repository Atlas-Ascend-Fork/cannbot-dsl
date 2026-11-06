# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile-only Native profiles for the public mega_mla launch contracts."""

import json
from numbers import Integral
import os

import cannbotdsl as dsl
from cannbotdsl.aot import export

from .mega_mla import MegaMlaOp
from .mega_mla_kernel import MegaMlaKernel

if __package__:
    from .attention_geometry import (
        choose_compressed_cache_decode_strategy,
        validate_attention_geometry,
        validate_query_heads,
    )
else:
    from attention_geometry import (
        choose_compressed_cache_decode_strategy,
        validate_attention_geometry,
        validate_query_heads,
    )
if __package__:
    from .attention_sequence_filter import AttentionSequenceFilter
else:
    from attention_sequence_filter import AttentionSequenceFilter
if __package__:
    from .attention_sequence_inputs import AttentionSequenceConfig
else:
    from attention_sequence_inputs import AttentionSequenceConfig
if __package__:
    from .attention_workspace import AttentionWorkspaceLayout
else:
    from attention_workspace import AttentionWorkspaceLayout


def _native_profiles():
    """Enumerate finite profiles; unlisted runtime schemas retain normal JIT."""
    defaults = dict(
        tokens=8,
        heads=16,
        k=640,
        grid=32,
        page_size=128,
        compressed_page_size=128,
        win_len=128,
        window_pages=4,
        compressed_pages=42,
        window_page_stride=None,
        compressed_page_stride=None,
        cmp_decode_strategy="auto",
        batches=0,
        window_table_pages=None,
        compressed_table_pages=None,
    )
    entries = [
        dict(heads=heads, grid=grid, k=k, batches=batches)
        for heads in (8, 16, 64)
        for grid in (28, 32)
        for k in (640, 2176)
        for batches in (0, 8)
    ]
    path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if path:
        with open(path, encoding="utf-8") as stream:
            entries = json.load(stream).get("mega_mla", entries)
    if not isinstance(entries, list) or not entries:
        raise ValueError("mega_mla profiles must be a nonempty list")
    profiles, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - set(defaults):
            raise ValueError("Unknown or invalid mega_mla export profile fields")
        profile = dict(defaults, **entry)
        for name in (
            "tokens",
            "heads",
            "k",
            "grid",
            "page_size",
            "compressed_page_size",
            "win_len",
            "window_pages",
            "compressed_pages",
        ):
            value = profile[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"mega_mla {name} must be a positive integer")
        validate_query_heads(profile["heads"])
        validate_attention_geometry(
            k=profile["k"],
            page_size=profile["page_size"],
            win_len=profile["win_len"],
            grid=profile["grid"],
            tokens=profile["tokens"],
        )
        batches = profile["batches"]
        if (
            isinstance(batches, bool)
            or not isinstance(batches, Integral)
            or batches < 0
        ):
            raise ValueError("mega_mla batches must be a nonnegative integer")
        for name, fallback, minimum in (
            (
                "window_page_stride",
                profile["page_size"] * 544,
                profile["page_size"] * 544,
            ),
            (
                "compressed_page_stride",
                profile["compressed_page_size"] * 320,
                profile["compressed_page_size"] * 320,
            ),
            ("window_table_pages", profile["window_pages"], 1),
            ("compressed_table_pages", profile["compressed_pages"], 1),
        ):
            value = profile[name]
            value = fallback if value is None else value
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"Invalid mega_mla {name}")
            profile[name] = value
        profile["cmp_decode_strategy"] = choose_compressed_cache_decode_strategy(
            profile["tokens"],
            profile["compressed_pages"] * profile["compressed_page_size"],
            profile["k"] - profile["win_len"],
            profile["cmp_decode_strategy"],
            heads=profile["heads"],
            grid=profile["grid"],
            page_size=profile["compressed_page_size"],
            win_len=profile["win_len"],
            has_lengths=bool(batches),
        )
        key = tuple(profile.items())
        if key not in seen:
            seen.add(key)
            profiles.append(profile)
    return profiles


def _compile_native_profile(profile):
    """Build the same receiver, workspace layout and specs as the host entry."""
    t, h, k = profile["tokens"], profile["heads"], profile["k"]
    ps, cps, kw = (
        profile["page_size"],
        profile["compressed_page_size"],
        profile["win_len"],
    )
    nw, nc, batch = (
        profile["window_pages"],
        profile["compressed_pages"],
        profile["batches"],
    )
    strategy = profile["cmp_decode_strategy"]
    layout = AttentionWorkspaceLayout(
        t,
        h,
        k,
        nc,
        ps,
        strategy,
        filter_indices=bool(batch),
        win_len=kw,
        compressed_page_size=cps,
    )
    owner = MegaMlaOp(
        k,
        ps,
        kw,
        profile["grid"],
        tokens=t,
        heads=h,
        cmp_decode_strategy=strategy,
    )
    owner.kernel = MegaMlaKernel(
        k,
        ps,
        kw,
        tokens=t,
        heads=h,
        grid=profile["grid"],
        gather_batch=owner.gather_batch,
        cmp_decode_strategy=strategy,
        workspace_layout=layout,
        compressed_page_size=cps,
    )
    dt = dsl.dtypes
    shapes = [
        ((t, 5120), dt.bfloat16),
        ((1, 1280), dt.float32),
        ((1, 512), dt.float32),
        ((1280, 5120), dt.float8_e4m3fn),
        ((512, 5120), dt.float8_e4m3fn),
        ((h * 512, 1280), dt.float8_e4m3fn),
        ((1280, 80, 2), dt.float8_e8m0),
        ((512, 80, 2), dt.float8_e8m0),
        ((h * 512, 20, 2), dt.float8_e8m0),
        ((nw, ps, 1, 544), dt.uint8),
        ((nc, cps, 1, 320), dt.uint8),
        ((t, kw), dt.int32),
        ((t, k - kw), dt.int32),
        ((1, h), dt.float32),
        ((t, 32), dt.float32),
        ((t, 32), dt.float32),
        ((t,), dt.int32),
        (((batch, profile["window_table_pages"]) if batch else (t, kw)), dt.int32),
        (
            ((batch, profile["compressed_table_pages"]) if batch else (t, k - kw)),
            dt.int32,
        ),
        ((t, kw), dt.int32),
        ((t, k - kw), dt.int32),
        ((t, h, 512), dt.bfloat16),
        ((layout.nbytes // 4,), dt.float32),
        ((layout.nbytes // 2,), dt.bfloat16),
        ((layout.nbytes,), dt.float8_e4m3fn),
        ((layout.nbytes,), dt.float8_e8m0),
        ((layout.nbytes // 4,), dt.int32),
    ]
    entry = owner._launch_mega_mla
    if batch:
        config = AttentionSequenceConfig(
            t,
            batch,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        )
        owner.kernel.sequence_filter = AttentionSequenceFilter(
            config, ps, kw, k - kw, compressed_page_size=cps
        )
        shapes += [((batch + 1,), dt.int32)] + [((batch,), dt.int32)] * 3
        entry = owner._launch_mega_mla_with_seqlens
    strides = {
        9: (profile["window_page_stride"], 544, 544, 1),
        10: (profile["compressed_page_stride"], 320, 320, 1),
    }
    specs = [
        dsl.TensorSpec(
            shape,
            dtype,
            stride=strides.get(i),
            storage_format="nz" if i in (3, 4, 5) else "nd",
        )
        for i, (shape, dtype) in enumerate(shapes)
    ]
    return dsl.compile(entry, *specs)


@export("mega_mla")
def export_mega_mla():
    """Collect Native variants without allocating tensors or launching kernels."""
    profiles = _native_profiles()
    for profile in profiles:
        handle = _compile_native_profile(profile)
        handle.close()
