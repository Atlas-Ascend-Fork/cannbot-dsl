# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TP1 inverse RoPE and two-stage MXFP8 projection. ND weights/scales; arbitrary T >= 1."""

from functools import lru_cache

import cannbotdsl
from cannbotdsl import dtypes
import torch

from cannbotdsl.aot import export
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.sync import (
    global_sync_all,
    cube_sync_all,
    PIPE,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    cube_sync_block_wait,
)

if __package__:
    from . import attn_epilogue_common as common
else:
    import attn_epilogue_common as common


def core_budget(ref):
    """Return (device, AIC blocks, AIV blocks) without caching stream quotas."""
    if ref.device.type not in ("npu", "privateuseone"):
        raise ValueError("attn_epilogue requires NPU inputs")
    device = ref.device.index
    if device is None:
        device = torch.npu.current_device()
    info = cannbotdsl.get_platform_info(stream=torch.npu.current_stream(device))
    cube, vector = int(info.cube_core_num), int(info.vector_core_num)
    if cube <= 0 or vector < 2:
        raise RuntimeError(f"Invalid MIX core quota: AIC={cube}, AIV={vector}")
    # A MIX block uses one AIC and two AIV on the supported architecture.
    return device, min(cube, vector // 2), vector


def _tile_fits(base_m, base_n, block_num):
    """Check operand and accumulator capacities for a tiling candidate."""
    scale_k = common.NG * common.O_LORA // 32
    depth = 1 if block_num == 32 else 2
    step_k = 2 if block_num == 32 else common.STEPK
    if max(base_m, base_n) * common.BASE_K * 2 > 65536:
        return False
    if base_m * base_n * 4 * depth > 262144:
        return False
    return (base_m + base_n) * (common.BASE_K * step_k * 2 + scale_k) <= 524288


@lru_cache(maxsize=1024)
def tiling_for_cores(block_num, t=72, m_pad=None):
    """Select M/N tiles from the stream core quota and Cube buffer limits."""
    block_num, t = int(block_num), int(t)
    if block_num <= 0 or t <= 0:
        raise ValueError("core count and T must be positive")
    aligned_t = (t + 15) // 16 * 16
    if m_pad is not None and (int(m_pad) < t or int(m_pad) % 16):
        raise ValueError("m_pad must cover T and be a multiple of 16")
    candidates = []
    widths = range(16, 257, 16) if block_num == 32 else (192,)
    for base_m in range(16, min(aligned_t, 256) + 1, 16):
        legal_n = [n for n in widths if _tile_fits(base_m, n, block_num)]
        if not legal_n:
            continue
        m_tiles = (t + base_m - 1) // base_m
        score, selected = 0, []
        for groups, n_size, k in (
            (common.NG, common.O_LORA, common.F),
            (1, common.DIM, common.NG * common.O_LORA),
        ):
            stage = []
            for n in legal_n:
                tiles = groups * m_tiles * ((n_size + n - 1) // n)
                waves = (tiles + block_num - 1) // block_num
                cost = waves * k * (2 * base_m * n + 256 * (base_m + n))
                stage.append((cost, waves, n))
            best = min(stage)
            score += best[0]
            selected.append(best[2])
        candidates.append((score, -base_m, *selected))
    if not candidates:
        raise ValueError("no TP1 tile fits the configured Cube buffers")
    _, negative_m, n1, n2 = min(candidates)
    return -negative_m, n1, n2


def _matmul_resources(base_m, base_n1, base_n2, k=common.NG * common.O_LORA):
    """Fit double-buffered L1 operands and one or two L0C accumulators."""
    width = max(base_n1, base_n2)
    step_k = (
        common.STEPK
        if (base_m + width) * (common.BASE_K * common.STEPK * 2 + k // 32) <= 524288
        else common.STEPK // 2
    )
    depths = [min(2, 262144 // (base_m * n * 4)) for n in (base_n1, base_n2)]
    if (
        min(depths) < 1
        or (base_m + width) * (common.BASE_K * step_k * 2 + k // 32) > 524288
    ):
        raise ValueError("TP1 matmul tile exceeds Cube buffers")
    return step_k, depths[0], depths[1]


@kernel
class AttnEpilogueTP1Kernel:
    def __init__(self, shape, m_pad, block_num, tiling):
        self.block_num = int(block_num)
        self.base_n1 = int(tiling.first_n)
        self.base_n2 = int(tiling.second_n)
        self.split_tiles = self.base_n1 != self.base_n2
        self.reuse_n = max(self.base_n1, self.base_n2)
        self.t = int(shape.t)
        self.ng = int(shape.ng)
        self.f = int(shape.f)
        self.o_lora = int(shape.o_lora)
        self.dim = int(shape.dim)
        self.m_pad = int(m_pad)
        self.base_m = int(tiling.rows) if tiling.rows else self.m_pad
        self.step_k, self.depth1, self.depth2 = _matmul_resources(
            self.base_m, self.base_n1, self.base_n2, self.ng * self.o_lora
        )
        mm1_tiles = (
            self.ng
            * ((self.t + self.base_m - 1) // self.base_m)
            * ((self.o_lora + self.base_n1 - 1) // self.base_n1)
        )
        # All launched Cubes must consume the first slab notification. Keep
        # the established path for other core budgets and shared MM objects.
        self.stream_k = (
            2048
            if self.block_num == 32 and self.split_tiles and mm1_tiles >= self.block_num
            else 0
        )
        lanes = self.block_num * 2 // self.ng
        rows_per_lane = (self.t + lanes - 1) // lanes if self.stream_k else 1
        batches = (rows_per_lane + 3) // 4
        self.batch_rows = (rows_per_lane + batches - 1) // batches
        if not self.split_tiles:
            self.mm = common.MatmulMx(
                self.ng * self.o_lora,
                self.base_m,
                self.base_n1,
                common.BASE_K,
                step_k=self.step_k,
                prefetch_tiles=1,
                prefetch_scales=True,
                l0c_depth=self.depth1,
            )

    def __call__(
        self,
        gm_o,
        gm_cos,
        gm_sin,
        woaq,
        dwa,
        wobq,
        dwb,
        gm_aq,
        gm_asc,
        gm_y,
        gm_yq,
        gm_ysc,
        gm_out,
    ):
        t = self.t
        ng = self.ng
        m_pad = self.m_pad
        k2 = ng * self.o_lora

        rq = common.RopeQuant(ng, stream_k=self.stream_k, batch_rows=self.batch_rows)
        rq.run(
            common.RopeInputs(gm_o, gm_cos, gm_sin),
            common.QuantOutput(gm_aq, gm_asc),
            t,
            m_pad,
        )
        mm1 = self._first_matmul()
        self._prefetch_weight(mm1, woaq, dwa, ng, self.o_lora)
        if const_expr(not self.stream_k):
            global_sync_all()
        qy = common.QuantTile(self.base_m, self.base_n1, 65536 + rq.extra_ub)
        self._tile_mm1_quant(
            mm1,
            qy,
            common.MxOperands(gm_aq, woaq, gm_asc, dwa),
            common.QuantOutput(gm_yq, gm_ysc),
        )
        # Protect aliased Cube buffers without waiting for other blocks or
        # their Vec quantization; Wb prefetch can overlap that work.
        cube_sync_all()
        mm2 = self._second_matmul()
        self._prefetch_weight(mm2, wobq, dwb, 1, self.dim)
        if const_expr(self.stream_k):
            # All QuantY data/scales must be in GM before any MM2 read.
            # MTE3 orders the stores; the reader waits on its MTE2 pipe.
            vec_sync_block_arrive(PIPE.MTE3, 13, mode=0)
            vec_sync_block_wait(PIPE.MTE3, 13, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 7)
            cube_sync_block_wait(PIPE.MTE2, 7)
        else:
            global_sync_all()
        self._tile_mm(
            mm2,
            common.MxOperands(gm_yq, wobq, gm_ysc, dwb),
            gm_out,
            common.MatmulShape(1, self.dim, k2, m_pad),
        )

    @jit
    def _first_matmul(self):
        if const_expr(self.split_tiles):
            mm = common.MatmulMx(
                self.ng * self.o_lora,
                self.base_m,
                self.base_n1,
                common.BASE_K,
                step_k=self.step_k,
                prefetch_tiles=2,
                prefetch_scales=True,
                reuse_n=self.reuse_n,
                stream_k=self.stream_k,
                l0c_depth=self.depth1,
            )
        else:
            mm = self.mm
        return mm

    @jit
    def _second_matmul(self):
        if const_expr(self.split_tiles):
            mm = common.MatmulMx(
                self.ng * self.o_lora,
                self.base_m,
                self.base_n2,
                common.BASE_K,
                step_k=self.step_k,
                prefetch_tiles=1,
                prefetch_scales=True,
                reuse_n=self.reuse_n,
                l0c_depth=self.depth2,
            )
        else:
            mm = self.mm
        return mm

    @jit
    def _tile_mm(self, mm, operands, out, shape):
        a, b = operands.a, operands.b
        sa, sb = operands.a_scale, operands.b_scale
        gpr, n, k, m = shape.groups, shape.n, shape.k, shape.m
        gk = k // common.MX_DIVISOR
        sa3 = sa.view(sa.shape[0], gk, 2)
        bi = get_block_idx()
        nb = self.block_num
        n_tiles = (n + mm.base_n - 1) // mm.base_n
        m_tiles = (self.t + mm.base_m - 1) // mm.base_m
        for tile in dsl_range(bi, gpr * m_tiles * n_tiles, nb):
            gl = tile // (m_tiles * n_tiles)
            m0 = tile // n_tiles % m_tiles
            n0 = tile % n_tiles
            a_g = common.matrix_view(a, gl * m, (gl + 1) * m, 0, k)
            b_g = common.matrix_view(b, gl * n, (gl + 1) * n, 0, k)
            sa_g = common.scale_view(sa3, gl * m, (gl + 1) * m, 0, gk)
            sb_g = common.scale_view(sb, gl * n, (gl + 1) * n, 0, gk)
            out_g = common.matrix_view(out, 0, m, gl * n, (gl + 1) * n)
            mm.tile(
                common.MxOperands(a_g, b_g, sa_g, sb_g),
                out_g,
                common.TileIndex(m0, n0, k, tile < nb),
            )

    @jit
    def _tile_mm1_quant(self, mm, quant, operands, output):
        a, b = operands.a, operands.b
        sa, sb = operands.a_scale, operands.b_scale
        q, scales = output.data, output.scales
        sa3 = sa.view(sa.shape[0], self.f // common.MX_DIVISOR, 2)
        bi = get_block_idx()
        n_tiles = (self.o_lora + mm.base_n - 1) // mm.base_n
        m_tiles = (self.t + mm.base_m - 1) // mm.base_m
        for tile in dsl_range(bi, self.ng * m_tiles * n_tiles, self.block_num):
            gl = tile // (m_tiles * n_tiles)
            m0 = tile // n_tiles % m_tiles
            n0 = tile % n_tiles
            a_g = common.matrix_view(
                a, gl * self.m_pad, (gl + 1) * self.m_pad, 0, self.f
            )
            b_g = common.matrix_view(
                b, gl * self.o_lora, (gl + 1) * self.o_lora, 0, self.f
            )
            sa_g = common.scale_view(
                sa3,
                gl * self.m_pad,
                (gl + 1) * self.m_pad,
                0,
                self.f // common.MX_DIVISOR,
            )
            sb_g = common.scale_view(
                sb,
                gl * self.o_lora,
                (gl + 1) * self.o_lora,
                0,
                self.f // common.MX_DIVISOR,
            )
            acc = mm.compute(
                common.MxOperands(a_g, b_g, sa_g, sb_g),
                common.TileIndex(m0, n0, self.f, tile < self.block_num),
            )
            actual_m = min(mm.base_m, self.m_pad - m0 * mm.base_m)
            actual_n = min(mm.base_n, self.o_lora - n0 * mm.base_n)
            quant.run(
                acc,
                common.QuantOutput(q, scales),
                common.TileExtent(
                    m0 * mm.base_m,
                    gl * self.o_lora + n0 * mm.base_n,
                    actual_m,
                    actual_n,
                ),
            )

    @jit
    def _prefetch_weight(self, mm, b, sb, groups, n):
        bi = get_block_idx()
        nn = (n + mm.base_n - 1) // mm.base_n
        mn = (self.t + mm.base_m - 1) // mm.base_m
        if bi < groups * mn * nn:
            gl = bi // (mn * nn)
            mm.prefetch(
                common.matrix_view(b, gl * n, (gl + 1) * n, 0, b.shape[1]),
                bi % nn,
                common.scale_view(
                    sb, gl * n, (gl + 1) * n, 0, b.shape[1] // common.MX_DIVISOR
                ),
            )


class AttnEpilogueTP1:
    def __init__(self, shape, m_pad, block_num, tiling):
        self.shape = shape
        self.block_num = int(block_num)
        self.m_pad = int(m_pad)
        self.tiling = tiling

    @host
    def run(
        self,
        gm_o,
        gm_cos,
        gm_sin,
        woaq,
        dwa,
        wobq,
        dwb,
        gm_aq,
        gm_asc,
        gm_y,
        gm_yq,
        gm_ysc,
        gm_out,
    ):
        op = AttnEpilogueTP1Kernel(self.shape, self.m_pad, self.block_num, self.tiling)
        op[self.block_num](
            gm_o,
            gm_cos,
            gm_sin,
            woaq,
            dwa,
            wobq,
            dwb,
            gm_aq,
            gm_asc,
            gm_y,
            gm_yq,
            gm_ysc,
            gm_out,
        )


_COMPILED = {}


def _compiled(shape, m_pad, *, device, block_num):
    t, ng, f, o_lora, dim = shape.t, shape.ng, shape.f, shape.o_lora, shape.dim
    base_m, base_n1, base_n2 = tiling_for_cores(block_num, t=t, m_pad=m_pad)
    key = (
        device,
        block_num,
        t,
        ng,
        f,
        o_lora,
        dim,
        m_pad,
        base_m,
        base_n1,
        base_n2,
        common.BASE_K,
        common.STEPK,
    )
    if key not in _COMPILED:
        fp8_dtype = dtypes.float8_e4m3fn
        k2 = ng * o_lora
        _COMPILED[key] = cannbotdsl.compile(
            AttnEpilogueTP1(
                shape,
                m_pad,
                block_num,
                common.ProjectionTiles(base_m, base_n1, base_n2),
            ).run,
            cannbotdsl.TensorSpec((t * ng, f), dtypes.bfloat16),
            cannbotdsl.TensorSpec((t, common.ROPE_DIM), dtypes.float32),
            cannbotdsl.TensorSpec((t, common.ROPE_DIM), dtypes.float32),
            cannbotdsl.TensorSpec((ng * o_lora, f), fp8_dtype),
            cannbotdsl.TensorSpec(
                (ng * o_lora, f // common.MX_DIVISOR, 2), dtypes.uint8
            ),
            cannbotdsl.TensorSpec((dim, k2), fp8_dtype),
            cannbotdsl.TensorSpec((dim, k2 // common.MX_DIVISOR, 2), dtypes.uint8),
            cannbotdsl.TensorSpec((ng * m_pad, f), dtypes.uint8),
            cannbotdsl.TensorSpec((ng * m_pad, f // common.QUANT_GROUP), dtypes.uint8),
            cannbotdsl.TensorSpec((m_pad, k2), dtypes.bfloat16),
            cannbotdsl.TensorSpec((m_pad, k2), dtypes.uint8),
            cannbotdsl.TensorSpec((m_pad, k2 // common.QUANT_GROUP), dtypes.uint8),
            cannbotdsl.TensorSpec((m_pad, dim), dtypes.bfloat16),
        )
    return _COMPILED[key]


def _attn_epilogue_out(
    o, rope_cos, rope_sin, woaq, dwa, wobq, dwb, aq, asc, y, yq, ysc, out, m_pad=None
):
    """Launch TP=1 on the input device's current stream; return padded out."""
    if len(o.shape) != 2 or int(o.shape[1]) != common.F:
        raise ValueError(f"o must have shape (NG*T, {common.F})")
    if int(o.shape[0]) % common.NG:
        raise ValueError("input row count must be divisible by NG")
    t = int(o.shape[0]) // common.NG
    if len(out.shape) != 2:
        raise ValueError("out must be a matrix")
    m_pad = int(out.shape[0]) if m_pad is None else int(m_pad)
    if t < 1:
        raise ValueError("T must be positive")
    if m_pad < t or m_pad % 16:
        raise ValueError("m_pad must cover T and be a multiple of 16")
    common.validate_inputs(
        t,
        common.RopeInputs(o, rope_cos, rope_sin),
        common.ProjectionWeights(woaq, dwa, wobq, dwb),
    )
    k2 = common.NG * common.O_LORA
    common.validate_tensors(
        o.device,
        (
            ("aq", aq, (common.NG * m_pad, common.F), torch.uint8),
            (
                "asc",
                asc,
                (common.NG * m_pad, common.F // common.QUANT_GROUP),
                torch.uint8,
            ),
            ("y", y, (m_pad, k2), torch.bfloat16),
            ("yq", yq, (m_pad, k2), torch.uint8),
            ("ysc", ysc, (m_pad, k2 // common.QUANT_GROUP), torch.uint8),
            ("out", out, (m_pad, common.DIM), torch.bfloat16),
        ),
    )
    device, blocks, _ = core_budget(o)
    _compiled(
        common.ProjectionShape(t, common.NG, common.F, common.O_LORA, common.DIM),
        int(m_pad),
        device=device,
        block_num=blocks,
    )(o, rope_cos, rope_sin, woaq, dwa, wobq, dwb, aq, asc, y, yq, ysc, out)
    return out


def attn_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
    """Single-device projection with internal scratch; return BF16 [T, 5120]."""
    t = common.validate_network_inputs(
        common.RopeInputs(o, rope_cos, rope_sin),
        common.ProjectionWeights(woa, descale_woa, wob, descale_wob),
    )
    if t < 1:
        raise ValueError("T must be positive")
    with torch.npu.device(o.device):
        scratch = common.get_scratch(
            o.device, t, torch.npu.is_current_stream_capturing()
        )
        m = (t + 15) // 16 * 16
        out = torch.empty((m, common.DIM), dtype=torch.bfloat16, device=o.device)
        _attn_epilogue_out(
            o.view(common.NG * t, common.F),
            rope_cos,
            rope_sin,
            woa.view(common.NG * common.O_LORA, common.F),
            descale_woa.view(torch.uint8).view(
                common.NG * common.O_LORA, common.F // common.MX_DIVISOR, 2
            ),
            wob,
            descale_wob.view(torch.uint8),
            *scratch,
            out,
        )
        return out[:t]


@export("attn_epilogue")
def export_attn_epilogue():
    """Compile the default single-card profiles without NPU execution.

    ``T`` enters the compiled program (tiling candidates, operand shapes), so
    the catalogue enumerates representative token counts for both supported
    MIX launch quotas. ``CANNBOTDSL_DS41_PROFILES`` may override the list with
    an ``attn_epilogue`` array of ``{"block_num": int, "tokens": [int, ...]}``
    entries.
    """
    import json
    import os

    default_tokens = (1, 72, 128, 256, 512, 1024, 2048, 4096, 8192)
    profiles = [(quota, t) for quota in (28, 32) for t in default_tokens]
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            entries = json.load(stream).get("attn_epilogue")
        if entries is None:
            raise ValueError(
                "CANNBOTDSL_DS41_PROFILES must define an 'attn_epilogue' list"
            )
        profiles = [
            (int(entry.get("block_num", 32)), int(t))
            for entry in entries
            for t in entry.get("tokens", default_tokens)
        ]
    handles = []
    try:
        for block_num, t in dict.fromkeys(profiles):
            shape = common.ProjectionShape(
                t, common.NG, common.F, common.O_LORA, common.DIM
            )
            handles.append(
                _compiled(
                    shape,
                    (t + 15) // 16 * 16,
                    device=0,
                    block_num=block_num,
                )
            )
    finally:
        for handle in handles:
            handle.close()
        _COMPILED.clear()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_attn_epilogue", "DEF")
_GRAPH_LIBRARY.define(
    "attn_epilogue("
    "Tensor o, Tensor woa, Tensor wob, Tensor descale_woa, Tensor descale_wob, "
    "Tensor rope_sin, Tensor rope_cos"
    ") -> Tensor"
)


def _attn_epilogue_meta(inputs, weights):
    """Infer the [T, DIM] BF16 output from tensor metadata only.

    FakeTensor tracing runs this implementation, so it must not call NPU-only
    APIs such as ``torch_npu.get_npu_format`` (they have no fake kernel).
    """
    o, rope_cos, rope_sin = inputs.source, inputs.cos, inputs.sin
    woa, descale_woa = weights.first, weights.first_scale
    wob, descale_wob = weights.second, weights.second_scale
    if not isinstance(o, torch.Tensor) or o.ndim != 3:
        raise ValueError("o must have shape (T, 64, 512)")
    if tuple(o.shape[1:]) != (common.NG * common.F // common.HEAD_DIM, common.HEAD_DIM):
        raise ValueError("o must have shape (T, 64, 512)")
    if o.dtype != torch.bfloat16:
        raise TypeError("o must be bfloat16")
    t = o.shape[0]
    if tuple(woa.shape) != (common.NG, common.O_LORA, common.F):
        raise ValueError("woa must have shape (8, 1024, 4096)")
    if woa.dtype != torch.float8_e4m3fn:
        raise TypeError("woa must be float8_e4m3fn")
    if tuple(wob.shape) != (common.DIM, common.NG * common.O_LORA):
        raise ValueError("wob must have shape (5120, 8192)")
    if wob.dtype != torch.float8_e4m3fn:
        raise TypeError("wob must be float8_e4m3fn")
    if tuple(descale_woa.shape) != (
        common.NG,
        common.O_LORA,
        common.F // common.MX_DIVISOR,
        2,
    ):
        raise ValueError("descale_woa must have shape (8, 1024, 64, 2)")
    if descale_woa.dtype != torch.float8_e8m0fnu:
        raise TypeError("descale_woa must be float8_e8m0fnu")
    if tuple(descale_wob.shape) != (
        common.DIM,
        common.NG * common.O_LORA // common.MX_DIVISOR,
        2,
    ):
        raise ValueError("descale_wob must have shape (5120, 128, 2)")
    if descale_wob.dtype != torch.float8_e8m0fnu:
        raise TypeError("descale_wob must be float8_e8m0fnu")
    if tuple(rope_sin.shape) != (t, common.ROPE_DIM):
        raise ValueError("rope_sin must have shape (T, 64)")
    if rope_sin.dtype != torch.float32:
        raise TypeError("rope_sin must be float32")
    if tuple(rope_cos.shape) != (t, common.ROPE_DIM):
        raise ValueError("rope_cos must have shape (T, 64)")
    if rope_cos.dtype != torch.float32:
        raise TypeError("rope_cos must be float32")
    return torch.empty((t, common.DIM), dtype=torch.bfloat16, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "attn_epilogue", "Meta")
def _attn_epilogue_meta_dispatch(*args):
    """Adapt the dispatcher schema to the grouped metadata inputs."""
    o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos = args
    return _attn_epilogue_meta(
        common.RopeInputs(o, rope_cos, rope_sin),
        common.ProjectionWeights(woa, descale_woa, wob, descale_wob),
    )


torch.library.impl(_GRAPH_LIBRARY, "attn_epilogue", "PrivateUse1")(attn_epilogue)

attn_epilogue_op = torch.ops.cannbotdsl_attn_epilogue.attn_epilogue

# While tracing, dynamo records the registered dispatcher call instead of inlining
# the host wrapper, so FakeTensor never reaches torch_npu.get_npu_format.
torch.compiler.substitute_in_graph(attn_epilogue, can_constant_fold_through=False)(
    lambda *args, **kwargs: attn_epilogue_op(*args, **kwargs)
)
