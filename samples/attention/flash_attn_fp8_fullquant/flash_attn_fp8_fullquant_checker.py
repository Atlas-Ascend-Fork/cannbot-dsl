# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashAttention FP8 FullQuant 的输入规格校验。"""

from bisect import bisect_right
from dataclasses import dataclass
import math

import torch

M_BASE_SIZE = 128
S2_BASE_SIZE = 256
D = 128
PAGE_BLOCK_ROWS = 128
KEY_SCALE_ROWS = 4
# K-scale 使用固定双槽；Kernel 为两个槽分别生成静态路径。
KEY_SCALE_BUFFER_SLOTS = 2


@dataclass(frozen=True)
class FlashAttnTilingInfo:
    """Launcher 和 Host 输入准备使用的已校验 Tiling 结果。"""

    layout: str
    mask_mode: str
    quant_scale_p: float | torch.Tensor
    scale: float
    block_dim: int
    batch_size: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    block_size: int
    cumulative_query_lengths: tuple[int, ...] | None
    key_value_lengths: tuple[int, ...] | None
    key_value_lengths_per_batch: tuple[int, ...]
    query_lengths: tuple[int, ...]
    ragged: bool
    max_query_tiles_per_head: int
    query_padding_rows: int
    causal_offsets: tuple[int, ...]
    mask_tail: int
    depths: dict[str, int]
    widened_block_table: torch.Tensor
    core_task_ranges: torch.Tensor
    query_sequence_boundaries: torch.Tensor
    key_value_length_table: torch.Tensor


def mask_tail_for(deltas, m_base_size=M_BASE_SIZE, s2_base_size=S2_BASE_SIZE):
    """计算 on 模式下仍需执行因果掩码的最少尾块数。

    对 KV 块 ``j``，可见边界为 ``lim0 = m*m_base_size + delta - j*s2_base_size``。
    令 ``r`` 分别取 ``delta % s2_base_size`` 和 ``(delta + m_base_size) % s2_base_size``；当任一
    ``r`` 落在 ``[m_base_size+1, s2_base_size-2]`` 时，一个 Q tile 会跨越两个边界块，
    必须保留 2 个 mask 尾块，否则 1 个即可。
    """
    if s2_base_size < m_base_size:
        raise ValueError(
            "Mask_tail_for is derived for m_base_size <= s2_base_size, "
            f"got m_base_size={m_base_size} s2_base_size={s2_base_size}. "
            f"A KV block narrower than the q tile straddles more than one "
            f"block per row; re-derive before using the gating build here."
        )
    for d in deltas:
        for r in ((d % s2_base_size), ((d + m_base_size) % s2_base_size)):
            if m_base_size + 1 <= r <= s2_base_size - 2:
                return 2
    return 1


def get_effective_cube_core_count(device=None):
    """返回当前 NPU 设备实际可用的 Cube 核数。"""
    import torch_npu  # noqa: F401

    if device is None:
        device_index = int(torch.npu.current_device())
    else:
        npu_device = torch.device(device)
        if npu_device.type != "npu":
            raise ValueError(f"Device must be an NPU device, got {npu_device}")
        device_index = (
            int(torch.npu.current_device())
            if npu_device.index is None
            else npu_device.index
        )
    cube_core_count = int(torch.npu.get_device_properties(device_index).cube_core_num)
    if cube_core_count <= 0:
        raise RuntimeError(
            f"Device npu:{device_index} reports invalid Cube core count "
            f"{cube_core_count}"
        )
    return cube_core_count


def default_block_dim(device=None):
    """返回当前 NPU 设备的默认 Kernel 发射核数。"""
    return get_effective_cube_core_count(device)


def widen_block_table(block_table, key_value_lengths, s2_base_size, block_size):
    """通过 Tensor gather 补齐 KV 尾块，不在 Host 读取物理页号。"""
    blocks_per_tile = s2_base_size // block_size
    required_width = max(
        (length + s2_base_size - 1) // s2_base_size * blocks_per_tile
        for length in key_value_lengths
    )
    width = max(block_table.shape[1], required_width)
    valid_blocks = torch.tensor(
        [(length + block_size - 1) // block_size for length in key_value_lengths],
        dtype=torch.int64,
        device=block_table.device,
    ).reshape(-1, 1)
    columns = torch.arange(width, dtype=torch.int64, device=block_table.device).reshape(
        1, -1
    )
    indices = torch.minimum(columns, valid_blocks - 1)
    return torch.gather(block_table, 1, indices)


def key_value_lengths_for_batches(batch, s2, key_value_lengths=None):
    """返回每个 batch 的 KV 长度。"""
    if key_value_lengths is None:
        if s2 is None:
            return None
        return [s2] * batch
    out = [int(x) for x in key_value_lengths]
    if len(out) != batch:
        raise ValueError(
            f"Actual_seq_kv must have one PER-BATCH KV LENGTH per batch; got "
            f"{len(out)} entries for batch={batch}"
        )
    return out


def batch_tile_params(
    batch,
    max_query_tiles_per_head,
    m_base_size=None,
    cumulative_query_lengths=None,
    s2=None,
    key_value_lengths=None,
):
    """返回每个 batch 的 Q tile 数、token 起点和因果偏移。

    TND 下 ``cumulative_query_lengths`` 是 Q 的累计长度，单 batch 长度为
    ``cumulative_query_lengths[b] - cumulative_query_lengths[b-1]``；因果偏移为 ``delta_b=S2_b-S1_b``。
    """
    if cumulative_query_lengths is None:
        tiles_per_batch = [max_query_tiles_per_head] * batch
        starts = [0] * batch
    else:
        if len(cumulative_query_lengths) != batch:
            raise ValueError(
                f"Cumulative_query_lengths must have one entry per batch; got {len(cumulative_query_lengths)} "
                f"for batch={batch}"
            )
        starts, tiles_per_batch, prev = [], [], 0
        for b in range(batch):
            if cumulative_query_lengths[b] < prev:
                raise ValueError(
                    f"Cumulative_query_lengths must be non-decreasing; entry {b} is "
                    f"{cumulative_query_lengths[b]} after {prev}"
                )
            starts.append(prev)
            tiles_per_batch.append(
                -(-(cumulative_query_lengths[b] - prev) // m_base_size)
            )
            prev = cumulative_query_lengths[b]

    step = m_base_size if m_base_size is not None else 1
    key_value_lengths_per_batch = key_value_lengths_for_batches(
        batch, s2, key_value_lengths
    )
    if s2 is None:
        deltas = [-1] * batch
    elif cumulative_query_lengths is None:
        deltas = [
            key_value_lengths_per_batch[b] - max_query_tiles_per_head * step
            for b in range(batch)
        ]
    else:
        deltas = [
            key_value_lengths_per_batch[b] - (cumulative_query_lengths[b] - starts[b])
            for b in range(batch)
        ]
    return tiles_per_batch, starts, deltas


def query_sequence_boundary_table(
    batch, max_query_tiles_per_head, m_base_size=None, cumulative_query_lengths=None
):
    """生成 Q token 前缀和表。"""
    step = m_base_size if m_base_size is not None else 1

    if cumulative_query_lengths is None:
        lens = [max_query_tiles_per_head * step] * batch
    else:
        if len(cumulative_query_lengths) != batch:
            raise ValueError(
                f"Cumulative_query_lengths must have one entry per batch; got {len(cumulative_query_lengths)} "
                f"for batch={batch}"
            )
        lens = []
        previous = 0
        for current in cumulative_query_lengths:
            lens.append(current - previous)
            previous = current
    row0, token_prefix = [0], 0
    for n in lens:
        token_prefix += n
        row0.append(token_prefix)
    return torch.tensor([row0], dtype=torch.int64)


def key_value_length_table(batch, s2=None, key_value_lengths=None):
    """生成每个 batch 的 KV 长度表。"""
    if s2 is None:
        raise ValueError(
            "Key_value_length_table needs s2; the kernel derives "
            "delta from "
            "it and has no other source"
        )
    return torch.tensor(
        [key_value_lengths_for_batches(batch, s2, key_value_lengths)], dtype=torch.int64
    )


def floor_sum(n, m, a, b):
    """以 O(log n) 复杂度计算 floor 求和。"""
    ans = 0
    while True:
        if a >= m:
            ans += (n - 1) * n // 2 * (a // m)
            a %= m
        if b >= m:
            ans += n * (b // m)
            b %= m
        y = a * n + b
        if y < m:
            return ans
        n = y // m
        b = y % m
        m, a = a, m


def _ramp_prefix(tile_count, query_tile_rows, kv_tile_rows, causal_offset):
    """计算前 ``tile_count`` 个 Q tile 的累计代价。"""
    if tile_count <= 0:
        return 0
    floor_sum_offset = query_tile_rows + causal_offset + kv_tile_rows - 1
    if floor_sum_offset < 0:
        raise ValueError(
            f"Causal_offset={causal_offset} is too negative for "
            f"query_tile_rows={query_tile_rows} "
            f"kv_tile_rows={kv_tile_rows}"
        )
    return floor_sum(tile_count, kv_tile_rows, query_tile_rows, floor_sum_offset)


def _flatten_point(
    tile_count, query_tile_rows, kv_tile_rows, causal_offset, kv_block_capacity
):
    """查找累计代价达到目标值的首个 tile。"""
    lower_bound, upper_bound = 0, tile_count
    while lower_bound < upper_bound:
        midpoint = (lower_bound + upper_bound) // 2
        visible_kv_block_count = -(
            -((midpoint + 1) * query_tile_rows + causal_offset) // kv_tile_rows
        )
        if visible_kv_block_count >= kv_block_capacity:
            upper_bound = midpoint
        else:
            lower_bound = midpoint + 1
    return lower_bound


def balanced_split_analytic(
    query_tiles_per_batch,
    query_head_count,
    query_tile_rows,
    kv_tile_rows,
    causal_offsets,
    kv_block_capacities,
    cube_core_count,
    mask_mode,
):
    """按估算代价生成各核的均衡切分区间。

    全注意力的每个 Q tile 扫描本 batch 的全部 KV block；因果模式的代价为
    ``cost(m)=min(capacity, ceil(((m+1)*query_tile_rows+causal_offset)/kv_tile_rows))``。
    函数利用 floor-sum 计算前缀代价，再按累计工作量而不是 tile 数量切分到各 Cube 核。
    """
    batch_count = len(query_tiles_per_batch)
    if len(causal_offsets) != batch_count or len(kv_block_capacities) != batch_count:
        raise ValueError(
            f"Causal_offsets ({len(causal_offsets)}) and "
            f"kv_block_capacities ({len(kv_block_capacities)}) must have "
            f"one entry per batch ({batch_count})"
        )

    segment_batch_indices = []
    segment_tile_prefix = [0]
    segment_work_prefix = [0]
    ramp_end_tile_indices = (
        [
            _flatten_point(tile_count, query_tile_rows, kv_tile_rows, offset, capacity)
            for tile_count, offset, capacity in zip(
                query_tiles_per_batch, causal_offsets, kv_block_capacities
            )
        ]
        if mask_mode == "on"
        else []
    )

    def batch_prefix_workload(batch_index, tile_count):
        capacity = kv_block_capacities[batch_index]
        if mask_mode == "off":
            return tile_count * capacity
        ramp_tiles = min(tile_count, ramp_end_tile_indices[batch_index])
        return (
            _ramp_prefix(
                ramp_tiles, query_tile_rows, kv_tile_rows, causal_offsets[batch_index]
            )
            + (tile_count - ramp_tiles) * capacity
        )

    for batch_index, tile_count in enumerate(query_tiles_per_batch):
        batch_workload = batch_prefix_workload(batch_index, tile_count)
        for _ in range(query_head_count):
            segment_batch_indices.append(batch_index)
            segment_tile_prefix.append(segment_tile_prefix[-1] + tile_count)
            segment_work_prefix.append(segment_work_prefix[-1] + batch_workload)

    total_tile_count = segment_tile_prefix[-1]
    total_workload = segment_work_prefix[-1]

    def prefix_workload(tile_index):
        """计算指定 tile 前的累计代价。"""
        if tile_index >= total_tile_count:
            return total_workload

        segment_index = bisect_right(segment_tile_prefix, tile_index) - 1
        batch_index = segment_batch_indices[segment_index]
        tile_offset_in_segment = tile_index - segment_tile_prefix[segment_index]
        return segment_work_prefix[segment_index] + batch_prefix_workload(
            batch_index, tile_offset_in_segment
        )

    core_tile_starts, core_tile_ends = [], []
    next_tile_start = 0
    for core_index in range(cube_core_count):
        if core_index == cube_core_count - 1 or next_tile_start >= total_tile_count:
            next_tile_end = (
                total_tile_count
                if next_tile_start < total_tile_count
                else next_tile_start
            )
        else:
            workload_before_core = prefix_workload(next_tile_start)
            target_prefix_workload = workload_before_core + (
                total_workload - workload_before_core
            ) / (cube_core_count - core_index)

            search_lower_bound = next_tile_start + 1
            search_upper_bound = total_tile_count
            while search_lower_bound < search_upper_bound:
                midpoint = (search_lower_bound + search_upper_bound) // 2
                if prefix_workload(midpoint) < target_prefix_workload:
                    search_lower_bound = midpoint + 1
                else:
                    search_upper_bound = midpoint
            next_tile_end = search_lower_bound
            if next_tile_end > next_tile_start + 1 and abs(
                prefix_workload(next_tile_end - 1) - target_prefix_workload
            ) < abs(prefix_workload(next_tile_end) - target_prefix_workload):
                next_tile_end -= 1
            next_tile_end = min(next_tile_end, total_tile_count)
        core_tile_starts.append(next_tile_start)
        core_tile_ends.append(next_tile_end)
        next_tile_start = next_tile_end

    if core_tile_starts[0] != 0 or core_tile_ends[-1] != total_tile_count:
        raise ValueError("Split must cover every tile")
    return torch.tensor([core_tile_starts, core_tile_ends], dtype=torch.int64)


def _validate_tensor_inputs(
    q_fp8, k_cache_u8, v_cache_u8, block_table, deq_q, deq_v, n_kv_heads, layout
):
    """校验公共入口的 Tensor 类型、维度和相互关系。"""
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Q_fp8 must be float8_e4m3fn, got {q_fp8.dtype}")
    expected_q_rank = 3 if layout == "TND" else 4
    if q_fp8.ndim != expected_q_rank:
        raise ValueError(
            f"{layout} q_fp8 must be rank {expected_q_rank}, got "
            f"shape={tuple(q_fp8.shape)}"
        )
    if q_fp8.shape[-1] != D:
        raise ValueError(f"This build requires head_dim={D}, got {q_fp8.shape[-1]}")

    if (
        not isinstance(n_kv_heads, int)
        or isinstance(n_kv_heads, bool)
        or n_kv_heads <= 0
    ):
        raise ValueError(f"N_kv_heads must be a positive integer, got {n_kv_heads!r}")
    n_q_heads = q_fp8.shape[-2] if layout == "TND" else q_fp8.shape[1]
    if n_q_heads % n_kv_heads:
        raise ValueError(
            f"N_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )

    for name, cache in (("k_cache_u8", k_cache_u8), ("v_cache_u8", v_cache_u8)):
        if cache.dtype != torch.uint8 or cache.ndim != 4:
            raise TypeError(
                f"Tensor {name} must be a rank-4 uint8 tensor, got "
                f"dtype={cache.dtype}, shape={tuple(cache.shape)}"
            )
    if tuple(k_cache_u8.shape) != tuple(v_cache_u8.shape):
        raise ValueError(
            f"K/V cache shapes must match, got {tuple(k_cache_u8.shape)} "
            f"and {tuple(v_cache_u8.shape)}"
        )
    if k_cache_u8.shape[1] != n_kv_heads:
        raise ValueError(
            f"Cache has {k_cache_u8.shape[1]} KV heads but n_kv_heads={n_kv_heads}"
        )
    if k_cache_u8.shape[-1] != D:
        raise ValueError(f"Cache head_dim must be {D}, got {k_cache_u8.shape[-1]}")
    if k_cache_u8.shape[2] <= KEY_SCALE_ROWS:
        raise ValueError(
            f"Cache row count must include data plus {KEY_SCALE_ROWS} scale rows"
        )

    if block_table.dtype != torch.int64 or block_table.ndim != 2:
        raise TypeError(
            f"Block_table must be a rank-2 int64 tensor, got "
            f"dtype={block_table.dtype}, shape={tuple(block_table.shape)}"
        )
    for name, scale_tensor in (("deq_q", deq_q), ("deq_v", deq_v)):
        if scale_tensor.dtype != torch.float32:
            raise TypeError(f"Tensor {name} must be float32, got {scale_tensor.dtype}")
    if deq_v.numel() != n_kv_heads:
        raise ValueError(
            f"Deq_v must contain one value per KV head: expected "
            f"{n_kv_heads}, got {deq_v.numel()}"
        )


def _validate_block_table(block_table, kv_lengths_per_batch, block_size):
    """仅校验 PageAttention 表的形状和容量；物理块编号由调用者保证合法。"""
    if block_table.shape[0] != len(kv_lengths_per_batch):
        raise ValueError(
            f"Block_table has {block_table.shape[0]} rows but batch is "
            f"{len(kv_lengths_per_batch)}"
        )
    for batch_idx, kv_length in enumerate(kv_lengths_per_batch):
        required_blocks = -(-kv_length // block_size)
        if required_blocks > block_table.shape[1]:
            raise ValueError(
                f"Batch {batch_idx} needs {required_blocks} block-table "
                f"entries but only {block_table.shape[1]} are available"
            )


def validate_and_resolve(
    q_fp8,
    k_cache_u8,
    v_cache_u8,
    block_table,
    deq_q,
    deq_v,
    s1,
    s2,
    n_kv_heads,
    *,
    scale,
    block_dim,
    mask_mode,
    layout,
    actual_seq,
    actual_seq_kv,
    quant_scale_p,
    m_base_size,
    s2_base_size,
    page_block_rows,
    depths,
) -> FlashAttnTilingInfo:
    for name, value in (("s1", s1), ("s2", s2)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Parameter {name} must be a positive Host integer")
    """完成公共输入规格、Host 序列参数和核数校验。"""
    # 序列长度属于 Host 参数；不隐式读取设备 Tensor 中的标量。
    for name, values in (("actual_seq", actual_seq), ("actual_seq_kv", actual_seq_kv)):
        if isinstance(values, torch.Tensor):
            raise TypeError(f"Parameter {name} must be a Host sequence of integers")
        if values is not None and any(
            isinstance(value, torch.Tensor) for value in values
        ):
            raise TypeError(f"Parameter {name} must contain Host integers, not tensors")
    if isinstance(scale, torch.Tensor):
        raise TypeError("Parameter scale must be a Host scalar")
    if layout not in ("BNSD", "TND"):
        raise ValueError(f"Layout must be 'BNSD' or 'TND', got {layout!r}")
    if mask_mode not in ("on", "off"):
        raise ValueError(f"Mask_mode must be on or off, got {mask_mode!r}")
    _validate_tensor_inputs(
        q_fp8,
        k_cache_u8,
        v_cache_u8,
        block_table,
        deq_q,
        deq_v,
        n_kv_heads,
        layout,
    )

    if isinstance(quant_scale_p, torch.Tensor):
        if quant_scale_p.numel() != 1 or quant_scale_p.dtype != torch.float32:
            raise ValueError("Quant_scale_p must be a single-element FP32 tensor")
        # Tensor 仅校验规格；调用者保证其值有限且为正，Host 不读取内容。
    else:
        quant_scale_p = float(quant_scale_p)
        if not math.isfinite(quant_scale_p) or quant_scale_p <= 0:
            raise ValueError("Quant_scale_p must be finite and positive")

    query_device = q_fp8.device if q_fp8.device.type == "npu" else None
    available_cube_cores = get_effective_cube_core_count(query_device)
    if block_dim is None:
        block_dim = available_cube_cores
    else:
        valid_block_dim = isinstance(block_dim, int) and not isinstance(block_dim, bool)
        if not valid_block_dim or not 1 <= block_dim <= available_cube_cores:
            raise ValueError(
                f"Block_dim must be an integer in [1, {available_cube_cores}], "
                f"got {block_dim!r}"
            )

    if layout == "TND":
        if actual_seq is None:
            raise ValueError("TND needs actual_seq (a prefix sum, length B)")
        cumulative_query_lengths = tuple(int(x) for x in actual_seq)
        if not cumulative_query_lengths:
            raise ValueError("Actual_seq must contain at least one batch")
        previous = 0
        for current in cumulative_query_lengths:
            if current <= previous:
                raise ValueError(
                    "Actual_seq must be a strictly increasing prefix sum, got "
                    f"{list(cumulative_query_lengths)}"
                )
            previous = current
        batch_size = len(cumulative_query_lengths)
        total_query, n_q_heads, head_dim = map(int, q_fp8.shape)
        if total_query != cumulative_query_lengths[-1]:
            raise ValueError(
                f"Q has {total_query} tokens but actual_seq ends at "
                f"{cumulative_query_lengths[-1]}; actual_seq is a PREFIX SUM, "
                "not per-batch lengths"
            )
        query_lengths_list = []
        previous = 0
        for current in cumulative_query_lengths:
            query_lengths_list.append(current - previous)
            previous = current
        query_lengths = tuple(query_lengths_list)
        if s1 != max(query_lengths):
            raise ValueError(
                f"S1={s1} must be the LONGEST batch under TND (under TND it "
                "is otherwise unused, so a value that disagrees with actual_seq "
                f"is a misunderstanding rather than a setting); actual_seq gives {max(query_lengths)}"
            )
    else:
        if actual_seq is not None:
            raise ValueError("Actual_seq is meaningful only under TND")
        cumulative_query_lengths = None
        batch_size, n_q_heads, input_s1, head_dim = map(int, q_fp8.shape)
        if input_s1 != s1:
            raise ValueError(f"Q has {input_s1} rows but s1={s1}")
        query_lengths = (s1,) * batch_size

    expected_deq_q = (
        q_fp8.shape[0] * q_fp8.shape[1]
        if layout == "TND"
        else batch_size * n_q_heads * s1
    )
    if deq_q.numel() != expected_deq_q:
        raise ValueError(
            "Deq_q must contain one value per Q token/head: expected "
            f"{expected_deq_q}, got {deq_q.numel()}"
        )

    if actual_seq_kv is None:
        key_value_lengths = None
    else:
        key_value_lengths = tuple(int(x) for x in actual_seq_kv)
        if len(key_value_lengths) != batch_size:
            raise ValueError(
                "Actual_seq_kv must have one PER-BATCH KV LENGTH per batch; "
                f"got {len(key_value_lengths)} entries for batch={batch_size}. "
                "Note it is NOT a prefix sum, unlike actual_seq"
            )
        if any(value <= 0 for value in key_value_lengths):
            raise ValueError(
                "Actual_seq_kv entries are per-batch KV LENGTHS and must be "
                f"positive; got {list(key_value_lengths)}"
            )
        if (
            cumulative_query_lengths is not None
            and key_value_lengths == cumulative_query_lengths
            and batch_size > 1
        ):
            raise ValueError(
                f"Actual_seq_kv equals actual_seq ({list(cumulative_query_lengths)}); "
                "actual_seq is a PREFIX SUM but actual_seq_kv is PER-BATCH KV LENGTHS. "
                "If the KV lengths really are the S1 lengths, pass "
                f"{list(query_lengths)}"
            )
        if s2 != max(key_value_lengths):
            raise ValueError(
                f"S2={s2} must equal max(actual_seq_kv)={max(key_value_lengths)}; "
                "with per-batch KV lengths the sweep cap is per batch and s2 is "
                "otherwise unused, so a value that disagrees is a misunderstanding "
                "rather than a setting"
            )

    block_size = int(k_cache_u8.shape[2]) - KEY_SCALE_ROWS
    if s2_base_size % block_size:
        raise ValueError(
            "A KV block must be a whole number of physical cache blocks; "
            f"S2_BASE_SIZE={s2_base_size} is not a multiple of block_size={block_size}"
        )
    key_value_lengths_per_batch = (
        key_value_lengths if key_value_lengths is not None else (s2,) * batch_size
    )
    _validate_block_table(
        block_table,
        key_value_lengths_per_batch,
        block_size,
    )

    if m_base_size != block_size:
        raise ValueError(
            f"This build fixes m_base_size={m_base_size} "
            "(flash_attn_fp8_fullquant.m_base_size) and the Q tiling assumes "
            f"it matches block_size={block_size}"
        )
    if cumulative_query_lengths is None and s1 % m_base_size:
        raise ValueError(
            f"No Q tail handling yet under BNSD: s1={s1} must be a multiple "
            f"of m_base_size={m_base_size} (TND accepts ragged batches; see "
            "the narrowing store in `_finish`)"
        )

    ragged = bool(
        cumulative_query_lengths is not None
        and any(length % m_base_size for length in query_lengths)
    )
    if s2_base_size % page_block_rows:
        raise ValueError(
            f"KV block {s2_base_size} is not a multiple of cube chunk {page_block_rows}"
        )
    invalid_causal_batches = []
    if mask_mode != "off":
        for index, query_length in enumerate(query_lengths):
            kv_length = key_value_lengths_per_batch[index]
            if kv_length < query_length:
                invalid_causal_batches.append((index, query_length, kv_length))
    if invalid_causal_batches:
        raise ValueError(
            "Causal mode 3 needs key_value_length >= query_length in EVERY batch "
            "(the query is a suffix of the key sequence, so a batch with fewer keys "
            "than queries has query rows with no key at or before them); violations "
            f"as (batch, query_length, key_value_length): {invalid_causal_batches}"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Scale must be finite and positive, got {scale}")

    if depths["key_dequant_scale_ub"] != KEY_SCALE_BUFFER_SLOTS:
        raise ValueError(
            f"K-scale ping-pong requires key_dequant_scale_ub depth {KEY_SCALE_BUFFER_SLOTS}"
        )
    max_query_tiles_per_head = -(-s1 // m_base_size)
    query_padding_rows = m_base_size if ragged else 0
    widened_table = widen_block_table(
        block_table,
        key_value_lengths_per_batch,
        s2_base_size,
        block_size,
    )
    kv_block_capacities = tuple(
        -(-length // s2_base_size) for length in key_value_lengths_per_batch
    )
    tiles_per_batch, _, causal_offsets = batch_tile_params(
        batch_size,
        max_query_tiles_per_head,
        m_base_size=m_base_size,
        cumulative_query_lengths=cumulative_query_lengths,
        s2=s2,
        key_value_lengths=key_value_lengths,
    )
    core_task_ranges = balanced_split_analytic(
        query_tiles_per_batch=tiles_per_batch,
        query_head_count=n_q_heads,
        query_tile_rows=m_base_size,
        kv_tile_rows=s2_base_size,
        causal_offsets=causal_offsets,
        kv_block_capacities=kv_block_capacities,
        cube_core_count=block_dim,
        mask_mode=mask_mode,
    )
    query_boundaries = query_sequence_boundary_table(
        batch_size,
        max_query_tiles_per_head,
        m_base_size=m_base_size,
        cumulative_query_lengths=cumulative_query_lengths,
    )
    kv_length_table = key_value_length_table(
        batch_size,
        s2=s2,
        key_value_lengths=key_value_lengths,
    )
    causal_offsets = tuple(
        key_value_lengths_per_batch[index] - query_lengths[index]
        for index in range(batch_size)
    )
    mask_tail = (
        0
        if mask_mode == "off"
        else mask_tail_for(
            causal_offsets,
            m_base_size=m_base_size,
            s2_base_size=s2_base_size,
        )
    )
    return FlashAttnTilingInfo(
        layout=layout,
        mask_mode=mask_mode,
        quant_scale_p=quant_scale_p,
        scale=scale,
        block_dim=block_dim,
        batch_size=batch_size,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        cumulative_query_lengths=cumulative_query_lengths,
        key_value_lengths=key_value_lengths,
        key_value_lengths_per_batch=tuple(key_value_lengths_per_batch),
        query_lengths=tuple(query_lengths),
        ragged=ragged,
        max_query_tiles_per_head=max_query_tiles_per_head,
        query_padding_rows=query_padding_rows,
        causal_offsets=causal_offsets,
        mask_tail=mask_tail,
        depths=dict(depths),
        widened_block_table=widened_table,
        core_task_ranges=core_task_ranges,
        query_sequence_boundaries=query_boundaries,
        key_value_length_table=kv_length_table,
    )
