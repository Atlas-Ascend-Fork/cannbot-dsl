# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Flash Attention functional precision matrix.

Formula:
    O = softmax(Q @ K^T * scale) @ V

mask_mode:
  * 0 -- full (no-mask) attention.
  * 3 -- right-context causal.  Query position *i* attends to key position *j*
    iff *j <= i + (S2 - S1)*.
  * 4 -- sliding window, bounded by win_left / win_right.  -1 means the bound
    is open on that side.

The matrix is 8 cases: batch 2-3, logical length <= 257, and every case carries
at least one length that is not a multiple of 16, so the Q/KV tail tiles are
exercised rather than only the aligned body.  These eight cases cover:

  * dtype      : float16, bfloat16
  * D          : 64, 128
  * layout_q   : BNSD [B,N,S,D], BSND [B,S,N,D], TND [T,N,D]
  * layout_kv  : BNSD, BSND, TND, PA_BNBD, PA_BBND, PA_NZ
  * layout_out : BNSD, BSND, TND -- including BNSD -> BSND conversion
  * mask_mode  : 0 (full), 3 (causal), 4 (window)
  * group size : MHA (g=1), GQA, MQA
  * ragged TND, physical padding with seqused_*, empty Q rows,
    reversed page order, non-contiguous PA cache

Paged-KV cases live in this same module.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "samples" / "flash_attn"))

from cannbotdsl import dtypes
from flash_attn import flash_attn
from flash_attn_metadata import flash_attn_metadata


# (output atol, lse atol).  The rtol stays at the same order in both cases.
_TOLERANCE = {
    torch.float16: (1e-3, 2e-3),
    torch.bfloat16: (2e-2, 2e-2),
}

_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
}


# ---------------------------------------------------------------------------
# Test cases
#
# One flat row per case; the parameter order is fixed by the parametrize
# decorator of ``test_flash_attn_functional_precision``:
#
#   nq, nkv, d, dtype, layout_q, layout_kv, layout_out, mask,
#   q_lengths, kv_lengths, q_used, kv_used,
#   win_left, win_right, block_size, reverse_pages, noncontiguous_pa
#
# ``None`` / ``-1`` / ``False`` mark a slot the case does not use:
#   * q_used / kv_used : None means "every physical row is live", i.e. the
#     lengths themselves, so no seqused_* is passed for that tensor.
#   * win_left / win_right : only read when mask == 4.
#   * block_size / reverse_pages / noncontiguous_pa : only read for PA_* KV.
# ---------------------------------------------------------------------------

_CASES = [
    # MHA, dense BNSD in and out, no mask.  Both Q and KV end mid-tile.
    pytest.param(
        2, 2, 64, torch.float16, "BNSD", "BNSD", "BNSD", 0,
        (33, 65), (65, 97),
        None, None, -1, -1, None, False, False,
        id="func_dense_bnsd_full_mha_d64_fp16_tail",
    ),
    # GQA (g=2) with the causal diagonal derived from sk - sq, BSND throughout.
    pytest.param(
        4, 2, 128, torch.bfloat16, "BSND", "BSND", "BSND", 3,
        (33, 67), (65, 97),
        None, None, -1, -1, None, False, False,
        id="func_dense_bsnd_causal_gqa_d128_bf16_tail",
    ),
    # MQA, ragged TND, with both window bounds set.
    pytest.param(
        4, 1, 128, torch.float16, "TND", "TND", "TND", 4,
        (31, 65, 97), (63, 95, 129),
        None, None, 31, 7, None, False, False,
        id="func_dense_tnd_window_mqa_d128_fp16_ragged",
    ),
    # BNSD -> BSND output conversion, physical padding, and an all-padding
    # batch (q_used == 0) whose output row must stay zero and lse +inf.
    pytest.param(
        8, 1, 128, torch.bfloat16, "BNSD", "BNSD", "BSND", 0,
        (65, 33, 17), (127, 97, 35),
        (64, 0, 17), (126, 65, 33), -1, -1, None, False, False,
        id="func_dense_bnsd_to_bsnd_mqa_d128_bf16_padding",
    ),
    # Ragged causal TND where Q and KV lengths disagree per batch, so a shared
    # causal delta would mask the wrong triangle.
    pytest.param(
        4, 2, 128, torch.float16, "TND", "TND", "TND", 3,
        (63, 65, 33), (64, 32, 97),
        (63, 0, 31), (63, 31, 95), -1, -1, None, False, False,
        id="func_dense_tnd_causal_gqa_d128_fp16_empty_tail",
    ),
    # PA_BNBD: reversed physical page order, non-contiguous cache, partial tail
    # page, MQA, and seqused_kv shorter than the page run.
    pytest.param(
        4, 1, 64, torch.bfloat16, "BNSD", "PA_BNBD", "BNSD", 0,
        (3, 5), (31, 17),
        None, (29, 15), -1, -1, 16, True, True,
        id="func_pa_bnbd_bnsd_full_mqa_d64_bf16_tail",
    ),
    # PA_BBND: block_size 48 is not a power of two and not 128-aligned, so the
    # page boundary never lines up with the KV tile boundary.
    pytest.param(
        4, 1, 128, torch.float16, "BSND", "PA_BBND", "BSND", 3,
        (1, 3), (65, 49),
        None, (63, 47), -1, -1, 48, True, False,
        id="func_pa_bbnd_bsnd_causal_mqa_d128_fp16_tail",
    ),
    # PA_NZ: 1024-token physical pages holding only 255 live tokens, TND Q, and
    # an open right window (win_right == -1).
    pytest.param(
        4, 1, 128, torch.bfloat16, "TND", "PA_NZ", "TND", 4,
        (3, 5), (129, 257),
        (2, 5), (127, 255), 7, -1, 1024, True, False,
        id="func_pa_nz_tnd_window_mqa_d128_bf16_tail",
    ),
]


def _pack_pages(k_dense, v_dense, block_table, layout, block_size):
    batch, heads, seq, dim = k_dense.shape
    pages = int(block_table.max().item()) + 1
    k_cache = torch.zeros((pages, heads, block_size, dim), dtype=k_dense.dtype)
    v_cache = torch.zeros_like(k_cache)
    for b in range(batch):
        for logical_page in range(block_table.shape[1]):
            start = logical_page * block_size
            end = min(start + block_size, seq)
            if start >= end:
                continue
            page = int(block_table[b, logical_page])
            k_cache[page, :, :end - start] = k_dense[b, :, start:end]
            v_cache[page, :, :end - start] = v_dense[b, :, start:end]
    if layout == "PA_BNBD":
        return k_cache, v_cache
    return k_cache.permute(0, 2, 1, 3).contiguous(), v_cache.permute(0, 2, 1, 3).contiguous()


def _pack_pages_nz(k_dense, v_dense, block_table, block_size):
    batch, heads, seq, dim = k_dense.shape
    d0 = 32 // k_dense.element_size()
    d1 = dim // d0
    pages = int(block_table.max().item()) + 1
    k_cache = torch.zeros((pages, heads, d1, block_size, d0), dtype=k_dense.dtype)
    v_cache = torch.zeros_like(k_cache)
    for b in range(batch):
        for logical_page in range(block_table.shape[1]):
            start = logical_page * block_size
            end = min(start + block_size, seq)
            if start >= end:
                continue
            page = int(block_table[b, logical_page])
            k_page = k_dense[b, :, start:end].reshape(heads, end - start, d1, d0)
            v_page = v_dense[b, :, start:end].reshape(heads, end - start, d1, d0)
            k_cache[page, :, :, :end - start] = k_page.permute(0, 2, 1, 3)
            v_cache[page, :, :, :end - start] = v_page.permute(0, 2, 1, 3)
    return k_cache, v_cache


def _effective_lengths(lengths, used, layout, block_size=None):
    provided = used is not None
    physical = tuple(lengths)
    effective = tuple(used) if provided else physical
    page_count = None
    if layout.startswith("PA_"):
        page_count = math.ceil(max(lengths) / block_size)
        physical = (page_count * block_size,) * len(lengths)
    elif layout != "TND" and len(set(physical)) > 1:
        physical = (max(physical),) * len(lengths)
    return physical, effective, provided, page_count


def _query_by_layout(q_dense, q_phys, layout_q):
    if layout_q == "TND":
        return torch.cat([
            q_dense[b, :, :length].transpose(0, 1)
            for b, length in enumerate(q_phys)
        ])
    if layout_q == "BSND":
        return q_dense.permute(0, 2, 1, 3).contiguous()
    return q_dense


def _paged_kv(dense_kv, block_table, layout_kv, block_size, noncontiguous_pa):
    k_dense, v_dense = dense_kv
    if layout_kv == "PA_NZ":
        k, v = _pack_pages_nz(k_dense, v_dense, block_table, block_size)
    else:
        k, v = _pack_pages(k_dense, v_dense, block_table, layout_kv, block_size)
    if noncontiguous_pa:
        d = k.shape[-1]
        width = 2 * d if noncontiguous_pa == 2 else d + 8
        k_base = torch.zeros((*k.shape[:-1], width), dtype=k_dense.dtype)
        v_base = torch.zeros_like(k_base)
        k_view = k_base[..., ::2] if noncontiguous_pa == 2 else k_base[..., :d]
        v_view = v_base[..., ::2] if noncontiguous_pa == 2 else v_base[..., :d]
        k_view.copy_(k)
        v_view.copy_(v)
        assert not k_view.is_contiguous() and not v_view.is_contiguous()
        k, v = k_view, v_view
    return k, v


def _dense_kv_by_layout(k_dense, v_dense, kv_phys, layout_kv):
    if layout_kv == "TND":
        k = torch.cat([
            k_dense[b, :, :length].transpose(0, 1)
            for b, length in enumerate(kv_phys)
        ])
        v = torch.cat([
            v_dense[b, :, :length].transpose(0, 1)
            for b, length in enumerate(kv_phys)
        ])
        return k, v
    if layout_kv == "BSND":
        return (
            k_dense.permute(0, 2, 1, 3).contiguous(),
            v_dense.permute(0, 2, 1, 3).contiguous(),
        )
    return k_dense, v_dense


def _sequence_tensor(used, physical, provided, required=False):
    if required or provided or used != physical:
        return torch.tensor(used, dtype=torch.int32, device="npu")
    return None


def _random_dense(q_shape, kv_shape, dtype):
    generator = torch.Generator().manual_seed(20260916)
    return (
        torch.randn(*q_shape, dtype=dtype, generator=generator),
        torch.randn(*kv_shape, dtype=dtype, generator=generator),
        torch.randn(*kv_shape, dtype=dtype, generator=generator),
    )


def _materialize_npu_inputs(dense, physical, layouts, paging):
    q_dense, k_dense, v_dense = dense
    q_phys, kv_phys = physical
    layout_q, layout_kv = layouts
    block_size, page_count, reverse_pages, noncontiguous_pa = paging
    q = _query_by_layout(q_dense, q_phys, layout_q)
    block_table = None
    if layout_kv.startswith("PA_"):
        block_table = torch.arange(
            len(q_phys) * page_count, dtype=torch.int32
        ).reshape(len(q_phys), page_count)
        if reverse_pages:
            block_table = block_table.flip(1).contiguous()
        k, v = _paged_kv(
            (k_dense, v_dense), block_table, layout_kv, block_size,
            noncontiguous_pa,
        )
    else:
        k, v = _dense_kv_by_layout(k_dense, v_dense, kv_phys, layout_kv)
    return q, k, v, block_table


def _launch_attrs(layouts, maxima, mask_config, cumulative, used_tensors):
    layout_q, layout_kv, layout_out = layouts
    max_q, max_kv = maxima
    mask, win_left, win_right = mask_config
    cu_q, cu_k = cumulative
    q_used_npu, kv_used_npu = used_tensors
    return dict(
        max_seqlen_q=max_q,
        max_seqlen_kv=max_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        mask_mode=mask,
        win_left=win_left,
        win_right=win_right,
        cu_seqlens_q=cu_q.npu() if layout_q == "TND" else None,
        cu_seqlens_kv=cu_k.npu() if layout_kv == "TND" else None,
        seqused_q=q_used_npu,
        seqused_kv=kv_used_npu,
    )


def _attention_mask(mask):
    if mask:
        return torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
        ).npu()
    return None



def _to_npu_cache(tensor):
    # .npu() alone packs non-dense CPU views; transfer their backing storage
    # first and restore the view so these tests exercise real device strides.
    if tensor.is_contiguous():
        return tensor.npu()
    root = tensor
    while root._base is not None:
        root = root._base
    device_root = root.npu()
    result = device_root.as_strided(
        tensor.shape, tensor.stride(), tensor.storage_offset() - root.storage_offset(),
    )
    assert result.stride() == tensor.stride()
    return result


def _input_result(identity, cpu_dense, inputs, launch, lengths):
    nq, nkv, d, dtype, layout_q, layout_out, mask, win_left, win_right = identity
    q, k, v = inputs
    attrs, attn_mask, block_table = launch
    q_phys, kv_phys, q_used, kv_used, cu_q = lengths
    return dict(
        nq=nq, nkv=nkv, d=d, dtype=dtype,
        layout_q=layout_q, layout_out=layout_out, mask=mask,
        win_left=win_left, win_right=win_right,
        cpu_dense=cpu_dense,
        npu=(q.npu(), _to_npu_cache(k), _to_npu_cache(v)),
        attrs=attrs,
        attn_mask=attn_mask,
        block_table=block_table.npu() if block_table is not None else None,
        q_phys=q_phys,
        kv_phys=kv_phys,
        q_used=q_used,
        kv_used=kv_used,
        cu_q=cu_q,
    )


def make_inputs(
    nq, nkv, d, dtype, layout_q, layout_kv, layout_out, mask,
    q_lengths, kv_lengths, q_used=None, kv_used=None,
    win_left=-1, win_right=-1, block_size=None,
    reverse_pages=False, noncontiguous_pa=False,
):
    """Materialize one matrix entry: NPU inputs, metadata attrs, CPU golden views.

    Every layout but TND needs one physical row count per tensor, so ragged
    ``q_lengths`` / ``kv_lengths`` are padded up to the batch maximum and the
    live lengths move into ``seqused_*``.  PA_* KV instead materializes its
    physical pages out of a dense ``(B, Nkv, max_kv, D)`` source, and the
    physical length becomes the whole page run -- the tail page's unused tokens
    stay zero and must not be attended.

    Returns a dict consumed by ``flash_attn_golden`` and ``run_flash_attn``.
    """
    batch = len(q_lengths)

    q_phys, q_used, q_used_given, _ = _effective_lengths(
        q_lengths, q_used, layout_q,
    )
    is_pa = layout_kv.startswith("PA_")
    kv_phys, kv_used, kv_used_given, page_count = _effective_lengths(
        kv_lengths, kv_used, layout_kv, block_size,
    )

    max_q = max(q_phys)
    max_kv = max(kv_phys)
    q_dense, k_dense, v_dense = _random_dense(
        (batch, nq, max_q, d), (batch, nkv, max_kv, d), dtype,
    )
    cu_q = torch.tensor((0,) + q_phys, dtype=torch.int32).cumsum(0).int()
    cu_k = torch.tensor((0,) + kv_phys, dtype=torch.int32).cumsum(0).int()
    q, k, v, block_table = _materialize_npu_inputs(
        (q_dense, k_dense, v_dense), (q_phys, kv_phys),
        (layout_q, layout_kv),
        (block_size, page_count, reverse_pages, noncontiguous_pa),
    )

    if is_pa:
        # Invalid cache rows must never influence PV, including a reused
        # L1 slot and the hardware's final 16-wide contraction block.
        for b, length in enumerate(kv_used):
            for logical_page in range(page_count):
                p = int(block_table[b, logical_page])
                used_rows = max(0, min(block_size, length - logical_page * block_size))
                for cache in (k, v):
                    if layout_kv == "PA_BNBD":
                        cache[p, :, used_rows:, :] = float("nan")
                    elif layout_kv == "PA_BBND":
                        cache[p, used_rows:, :, :] = float("nan")
                    else:
                        cache[p, :, :, used_rows:, :] = float("nan")

    q_used_npu = _sequence_tensor(q_used, q_phys, q_used_given)
    kv_used_npu = _sequence_tensor(kv_used, kv_phys, kv_used_given, is_pa)
    attrs = _launch_attrs(
        (layout_q, layout_kv, layout_out), (max_q, max_kv),
        (mask, win_left, win_right), (cu_q, cu_k),
        (q_used_npu, kv_used_npu),
    )
    attn_mask = _attention_mask(mask)
    return _input_result(
        identity=(nq, nkv, d, dtype, layout_q, layout_out, mask, win_left, win_right),
        cpu_dense=(q_dense, k_dense, v_dense),
        inputs=(q, k, v),
        launch=(attrs, attn_mask, block_table),
        lengths=(q_phys, kv_phys, q_used, kv_used, cu_q),
    )


def _golden_visibility(columns, diagonal, mask, win_left, win_right):
    visible = torch.ones(diagonal.shape[0], columns.shape[1], dtype=torch.bool)
    if mask == 3:
        visible = columns <= diagonal
    elif mask == 4:
        if win_left >= 0:
            visible &= columns >= diagonal - win_left
        if win_right >= 0:
            visible &= columns <= diagonal + win_right
    return visible


def _golden_output_layout(output, layout_out, q_phys):
    if layout_out == "TND":
        return torch.cat([
            output[b, :, :length].transpose(0, 1)
            for b, length in enumerate(q_phys)
        ])
    if layout_out == "BSND":
        return output.permute(0, 2, 1, 3).contiguous()
    return output


def _golden_lse(layout_q, nq, q_phys):
    if layout_q == "TND":
        return torch.full(
            (nq, sum(q_phys)), torch.inf, dtype=torch.float32
        )
    return torch.full(
        (len(q_phys), nq, max(q_phys)),
        torch.inf,
        dtype=torch.float32,
    )


def flash_attn_golden(data):
    """CPU golden for one matrix entry: fp32 attention, scored in 256-row blocks.

    Masking is per row against the *effective* KV length, so the causal
    diagonal sits at ``j = i + (sk - sq)`` and the window is
    ``i + (sk - sq) - win_left <= j <= i + (sk - sq) + win_right``.  A row with
    no visible key -- ``sq == 0``, ``sk == 0``, or an empty window -- keeps
    ``lse = +inf`` and a zero output row, which is the padding contract.
    """
    nq, nkv, d, dtype = data["nq"], data["nkv"], data["d"], data["dtype"]
    layout_q, layout_out, mask = (
        data["layout_q"], data["layout_out"], data["mask"]
    )
    win_left, win_right = data["win_left"], data["win_right"]
    q, k, v = data["cpu_dense"]
    output = torch.zeros_like(q)
    lse = _golden_lse(layout_q, nq, data["q_phys"])

    group = nq // nkv
    for batch_index, (sq, sk) in enumerate(zip(data["q_used"], data["kv_used"])):
        if sq == 0 or sk == 0:
            continue
        columns = torch.arange(sk)[None, :]
        for head in range(nq):
            q_head = q[batch_index, head, :sq]
            kv_head = head // group
            k_head = k[batch_index, kv_head, :sk].float()
            v_head = v[batch_index, kv_head, :sk].float()
            for start in range(0, sq, 256):
                end = min(start + 256, sq)
                scores = q_head[start:end].float() @ k_head.T
                scores *= d ** -0.5
                diagonal = torch.arange(start, end)[:, None] + sk - sq
                visible = _golden_visibility(
                    columns, diagonal, mask, win_left, win_right,
                )
                scores.masked_fill_(~visible, -torch.inf)
                logsum = torch.logsumexp(scores, dim=-1)
                logsum = logsum.masked_fill(torch.isneginf(logsum), torch.inf)
                if layout_q == "TND":
                    offset = int(data["cu_q"][batch_index])
                    lse[head, offset + start:offset + end] = logsum
                else:
                    lse[batch_index, head, start:end] = logsum
                probabilities = torch.softmax(scores, dim=-1).nan_to_num()
                output[batch_index, head, start:end] = (
                    probabilities @ v_head
                ).to(dtype)

    expected = _golden_output_layout(output, layout_out, data["q_phys"])
    return expected, lse


def run_flash_attn(data, lse_enabled, metadata=None):
    if metadata is None:
        metadata = flash_attn_metadata(
            data["nq"], data["nkv"], data["d"],
            batch_size=len(data["q_phys"]), **data["attrs"],
        )
    return flash_attn(
        *data["npu"],
        metadata=metadata,
        attn_mask=data["attn_mask"],
        block_table=data["block_table"],
        softmax_scale=data["d"] ** -0.5,
        return_softmax_lse=lse_enabled,
        dtype=_TORCH_TO_DSL[data["dtype"]],
        **data["attrs"],
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "nq,nkv,d,dtype,layout_q,layout_kv,layout_out,mask,"
    "q_lengths,kv_lengths,q_used,kv_used,"
    "win_left,win_right,block_size,reverse_pages,noncontiguous_pa",
    _CASES,
)
def test_flash_attn_functional_precision(
    nq, nkv, d, dtype, layout_q, layout_kv, layout_out, mask,
    q_lengths, kv_lengths, q_used, kv_used,
    win_left, win_right, block_size, reverse_pages, noncontiguous_pa,
    pytestconfig, monkeypatch,
):
    pytest.importorskip("torch_npu")
    if layout_kv.startswith("PA_"):
        import flash_attn as implementation

        def forbid_materialize(*args, **kwargs):
            raise AssertionError("PA must read physical pages inside the kernel")

        monkeypatch.setattr(implementation, "_materialize_pa_kv", forbid_materialize, raising=False)
    execution_mode = "eager"
    for option in pytestconfig.getoption("override_ini") or ():
        name, separator, value = option.partition("=")
        if separator and name == "mode":
            execution_mode = value
    if execution_mode not in ("eager", "aclgraph"):
        raise pytest.UsageError(
            "mode must be eager or aclgraph; use -o mode=aclgraph"
        )
    if execution_mode == "aclgraph":
        if not hasattr(torch.npu, "NPUGraph"):
            pytest.skip("torch_npu does not provide NPUGraph")

    data = make_inputs(
        nq, nkv, d, dtype, layout_q, layout_kv, layout_out, mask,
        q_lengths, kv_lengths, q_used, kv_used,
        win_left, win_right, block_size, reverse_pages, noncontiguous_pa,
    )
    expected_output, expected_lse = flash_attn_golden(data)
    output_tolerance, lse_tolerance = _TOLERANCE[dtype]
    metadata = None
    if execution_mode == "aclgraph":
        metadata = flash_attn_metadata(
            data["nq"], data["nkv"], data["d"],
            batch_size=len(data["q_phys"]), **data["attrs"],
        )

    for lse_enabled in (False, True):
        graph = None
        if execution_mode == "aclgraph":
            # Compile first; keep metadata and input storage stable for replay.
            run_flash_attn(data, lse_enabled, metadata=metadata)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output, lse = run_flash_attn(
                    data, lse_enabled, metadata=metadata
                )
        else:
            output, lse = run_flash_attn(data, lse_enabled)

        for _ in range(6 if graph is not None else 1):
            if graph is not None:
                graph.replay()
                torch.npu.synchronize()
            torch.testing.assert_close(
                output.cpu(), expected_output,
                atol=output_tolerance, rtol=output_tolerance,
            )
            if not lse_enabled:
                assert lse is None
                continue
            assert lse.dtype == torch.float32 and lse.is_contiguous()
            assert lse.shape == expected_lse.shape
            actual_lse = lse.cpu()
            assert torch.equal(
                torch.isposinf(actual_lse), torch.isposinf(expected_lse)
            )
            finite = torch.isfinite(expected_lse)
            torch.testing.assert_close(
                actual_lse[finite], expected_lse[finite],
                atol=lse_tolerance, rtol=2e-3,
            )


@pytest.mark.npu
def test_flash_attn_keeps_metadata_on_device(monkeypatch):
    pytest.importorskip("torch_npu")
    data = make_inputs(
        2, 2, 64, torch.float16, 'BNSD', 'BNSD', 'BNSD', 0,
        (33, 65), (65, 97), None, None, -1, -1, None, False, False,
    )
    expected_output, expected_lse = flash_attn_golden(data)
    metadata = flash_attn_metadata(
        data['nq'], data['nkv'], data['d'],
        batch_size=len(data['q_phys']), **data['attrs'],
    )

    def forbid_host_read(*args, **kwargs):
        raise AssertionError('FlashAttention must not read metadata on Host CPU')

    import flash_attn as implementation

    original_launch = implementation.FlashAttnLauncher.launch

    def check_launch(self, *args, **kwargs):
        # B=2, Nkv=2, one M block and one KV block: four AICs suffice.
        assert self.block_dim == 4
        return original_launch.__get__(self, type(self))(*args, **kwargs)

    with monkeypatch.context() as guard:
        for name in ('cpu', 'item', 'tolist'):
            guard.setattr(torch.Tensor, name, forbid_host_read)
        guard.setattr(implementation.FlashAttnLauncher, 'launch', check_launch)
        output, lse = run_flash_attn(data, True, metadata=metadata)

    host_metadata = metadata.cpu()
    assert torch.count_nonzero(host_metadata[4:16]) == 0
    sections = int(host_metadata[0])
    fd_start = 16 + sections * 36 * 16
    fa_records = host_metadata[16:fd_start].reshape(sections, 36, 16)
    fd_records = host_metadata[fd_start:fd_start + sections * 72 * 16].reshape(sections, 72, 16)
    assert torch.count_nonzero(fa_records[:, 4:]) == 0
    assert torch.count_nonzero(fd_records[:, 8:]) == 0
    torch.testing.assert_close(
        output.cpu(), expected_output, atol=1e-3, rtol=1e-3,
    )
    torch.testing.assert_close(
        lse.cpu(), expected_lse, atol=2e-3, rtol=2e-3,
    )
