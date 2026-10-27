# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SparseFlashMLA NPU precision and routing tests with a self-contained CPU golden.

Golden and input-generation definitions are based on
ops-transformer/attention/sparse_flash_mla/tests/pytest/sparse_flash_mla_golden.py.
Keep the upstream RUN_MODE and arithmetic (including dtype rounding) intact.
The CPU golden additionally accepts sinks=None as a zero-mass (-inf) sink.
Tests are opt-in: CANNBOTDSL_RUN_SMLA_TESTS=1 pytest test/sparse_flash_mla.
SMLA_CASES selects the named in-file cases.
"""

import ast
import datetime
import itertools
import json
import logging
import random
import sys
from functools import lru_cache
from pathlib import Path

import pytest

import math
import numpy as np
import os
import torch

# Keep SMLA tests available for explicit validation without running them in CI.
pytestmark = pytest.mark.skipif(
    os.environ.get("CANNBOTDSL_RUN_SMLA_TESTS") != "1",
    reason="SMLA tests are opt-in; set CANNBOTDSL_RUN_SMLA_TESTS=1 to run",
)


logger = logging.getLogger(__name__)

DATA_RANGE_LEFT = -10
DATA_RANGE_RIGHT = 10

# 三种运行模式
# 0 不切S2
# 1 切S2
# 2 AMLA

RUN_MODE = 1


def _nonfinite_fill(shape, left, right):
    """Return a deterministic tensor for a non-finite data range.

    The finite path deliberately remains at each call site so existing random
    generation (and its seeded behavior) is unchanged.
    """
    left, right = float(left), float(right)
    if math.isfinite(left) and math.isfinite(right):
        return None
    if math.isnan(left) or math.isnan(right):
        if math.isnan(left) and math.isnan(right):
            value = math.nan
        else:
            raise ValueError(
                f"data range must use [nan, nan] for a constant NaN, got [{left}, {right}]"
            )
    elif left == right:
        value = left
    else:
        raise ValueError(
            f"non-finite data range must be constant, got [{left}, {right}]"
        )
    return torch.full(shape, value, dtype=torch.float32)


def is_empty(obj):
    if isinstance(obj, list):
        return len(obj) == 0
    if isinstance(obj, torch.Tensor):
        return obj.nelement() == 0
    return obj is None


def restore_cmp_kv_length(cmp_kv_length, cmp_ratio, cmp_residual=0, cmp_mask_mode=3):
    """Restore the pre-compression length from CMP-KV inputs, as arch35 does."""
    if cmp_mask_mode == 0 or int(cmp_ratio) == 1:
        cmp_residual = 0
    return int(cmp_kv_length) * int(cmp_ratio) + int(cmp_residual)


# np.random.seed(42)
# torch.manual_seed(42)


class GeneralizedSFA:
    def __init__(
        self,
        layout_q,
        layout_kv,
        q_type,
        ori_kv_type,
        cmp_kv_type,
        B,
        S1,
        S2,
        T1,
        N1,
        N2,
        D,
        K1,
        K,
        block_num1,
        block_num2,
        block_size1,
        block_size2,
        cu_seqlens_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        ori_topk_length,
        cmp_topk_length,
    ):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.S2 = S2
        self.T1 = T1
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K1 = K1
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_ori_kv = seqused_ori_kv
        self.seqused_cmp_kv = seqused_cmp_kv
        self.cmp_residual_kv = cmp_residual_kv
        # self.seqused_kv = seqused_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right
        self.ori_topk_length = ori_topk_length
        self.cmp_topk_length = cmp_topk_length

    @staticmethod
    def _get_batch_consistency_reduce_size(ori_s2_size, cmp_s2_size):
        """Return the S2 reduction-block size used by batch-consistency metadata."""
        total_s2_size = ori_s2_size + cmp_s2_size
        # 先向上整除 32，再向上对齐到 128-token 基础块，与 metadata 和 kernel 保持一致。
        raw_reduce_size = (total_s2_size + 31) // 32
        return max(((raw_reduce_size + 127) // 128) * 128, 128)

    @staticmethod
    def _update_s2_tile(
        q_fp32,
        k_tile,
        softmax_scale,
        softmax_dtype,
        score_max,
        sumexp,
        acc_o,
    ):
        """Use one 128-token S2 tile to update the online-softmax state."""
        mm1_res = torch.matmul(q_fp32, k_tile.T)
        scale_res = mm1_res * softmax_scale

        score_max_pre = score_max.clone()
        cur_score_max = scale_res.max(dim=-1)[0]
        score_max = torch.max(score_max, cur_score_max)
        score_max_pre = torch.exp(score_max_pre - score_max)

        acc_s = torch.exp(scale_res - score_max.unsqueeze(1))
        sumexp = sumexp * score_max_pre + acc_s.sum(dim=-1)
        acc_s_cast = acc_s.to(dtype=softmax_dtype).to(dtype=torch.float32)
        mm2_res = torch.matmul(acc_s_cast, k_tile)
        acc_o = acc_o * score_max_pre.unsqueeze(1) + mm2_res
        return score_max, sumexp, acc_o

    def _calculate_local_s2_block(
        self,
        q_fp32,
        k_tiles_fp32,
        initial_score_max,
    ):
        """Calculate one NPU reduction block and retain its local softmax state."""
        score_max = initial_score_max.clone()
        sumexp = torch.ones_like(score_max, dtype=torch.float32)
        acc_o = torch.zeros(
            (q_fp32.shape[0], k_tiles_fp32[0].shape[-1]), dtype=torch.float32
        )

        for k_tile in k_tiles_fp32:
            score_max, sumexp, acc_o = self._update_s2_tile(
                q_fp32,
                k_tile,
                self.softmax_scale,
                self.q_type,
                score_max,
                sumexp,
                acc_o,
            )

        return score_max, sumexp, acc_o / sumexp.unsqueeze(1)

    def _calculate_batch_consistency(
        self,
        q_fp32,
        ori_k_fp32,
        cmp_k_fp32,
        sinks,
    ):
        """Follow the NPU S2 split and fixed FD reduction order."""
        ori_s2_size = 0 if ori_k_fp32 is None else ori_k_fp32.shape[0]
        cmp_s2_size = 0 if cmp_k_fp32 is None else cmp_k_fp32.shape[0]
        reduce_size = self._get_batch_consistency_reduce_size(ori_s2_size, cmp_s2_size)

        # ORI、CMP 分别划分规约块，ORI 尾块结束后再开始 CMP 的规约。
        blocks = []
        for k_tensor in (ori_k_fp32, cmp_k_fp32):
            if k_tensor is None or k_tensor.shape[0] == 0:
                continue
            blocks.extend(
                list(block.split(128, dim=0))
                for block in k_tensor.split(reduce_size, dim=0)
            )

        merged_lse = None
        merged_sum = None
        merged_o = None
        for block_id, k_tiles in enumerate(blocks):
            initial_max = (
                sinks.clone()
                if block_id == 0 and sinks is not None
                else torch.full((q_fp32.shape[0],), -torch.inf, dtype=torch.float32)
            )
            local_max, local_sum, local_o = self._calculate_local_s2_block(
                q_fp32, k_tiles, initial_max
            )
            if merged_lse is None:
                merged_lse = local_max
                merged_sum = local_sum
                merged_o = local_o
                continue

            global_max = torch.max(merged_lse, local_max)
            prev_weight = torch.exp(merged_lse - global_max) * merged_sum
            cur_weight = torch.exp(local_max - global_max) * local_sum
            global_sum = prev_weight + cur_weight
            merged_o = (
                merged_o * prev_weight.unsqueeze(1) + local_o * cur_weight.unsqueeze(1)
            ) / global_sum.unsqueeze(1)

            # FD persists the merged state as (LSE, 1, normalized O), then
            # consumes it in the next step of the deterministic left fold.
            merged_lse = global_max + torch.log(global_sum)
            merged_sum = torch.ones_like(global_sum)

        if merged_lse is None:
            raise ValueError("batch-consistency calculation requires non-empty KV")
        return merged_o, merged_lse + torch.log(merged_sum)

    def calculate_by_bnsd(
        self,
        q_bnsd,
        ori_k_bnsd,
        ori_sparse_indices_bnsd,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        template_idx,
        cmp_k_bnsd=None,
        cmp_sparse_indices_bnsd=None,
        seqused_q=None,
        ori_topk_length_bnsd=None,
        cmp_topk_length_bnsd=None,
        return_softmax_lse=False,
        batch_consistency=False,
    ):
        if sinks is None:
            # A missing sink contributes no probability mass in every RUN_MODE.
            sinks = torch.full(
                (self.N1,), -torch.inf, dtype=torch.float32, device=q_bnsd.device
            )
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        softmax_lse = None
        if return_softmax_lse:
            softmax_lse = torch.zeros(
                (
                    q_bnsd.shape[0],
                    ori_k_bnsd.shape[1],
                    q_bnsd.shape[2],
                    q_bnsd.shape[1] // ori_k_bnsd.shape[1],
                ),
                dtype=torch.float32,
            )
        B = q_bnsd.shape[0]
        if seqused_q is not None:
            act_q = seqused_q
        elif self.layout_q in ["TND"]:
            act_q = prefix_sum_to_original(cu_seqlens_q)
        else:
            act_q = [q_bnsd.shape[2]] * B
        G = int(self.N1 / self.N2)
        s2_base_size = 128

        for i_B in range(B):
            print(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            if seqused_ori_kv is not None:
                cur_ori_act_kv = seqused_ori_kv[i_B]
            elif cu_seqlens_ori_kv is not None:
                cur_ori_act_kv = cu_seqlens_ori_kv[i_B + 1] - cu_seqlens_ori_kv[i_B]
            else:
                cur_ori_act_kv = ori_k_bnsd.size()[2]
            if seqused_cmp_kv is not None:
                cur_cmp_act_kv = seqused_cmp_kv[i_B]
            elif cu_seqlens_cmp_kv is not None:
                cur_cmp_act_kv = cu_seqlens_cmp_kv[i_B + 1] - cu_seqlens_cmp_kv[i_B]
            else:
                cur_cmp_act_kv = 0
            cur_cmp_residual = (
                cmp_residual_kv[i_B] if cmp_residual_kv is not None else 0
            )
            restored_cmp_act_kv = (
                restore_cmp_kv_length(
                    cur_cmp_act_kv, self.cmp_ratio, cur_cmp_residual, self.cmp_mask_mode
                )
                if template_idx in (1, 2, 4)
                else 0
            )
            for i_N2 in range(self.N2):
                print(f"    i_N2 = {i_N2}/{self.N2}")
                cur_sinks = sinks[i_N2 * G : (i_N2 + 1) * G]
                cur_sinks_expand = cur_sinks.unsqueeze(1)
                for i_S1 in range(cur_act_q):
                    milestones = [
                        int(cur_act_q * pct / 100) for pct in range(10, 101, 10)
                    ]
                    milestones = list(dict.fromkeys(milestones))
                    if i_S1 in milestones:
                        current_pct = (i_S1 / cur_act_q) * 100
                        print(
                            f"      进度：{current_pct:.1f}% | 步数：{i_S1:>{len(str(cur_act_q))}}/{cur_act_q}"
                        )
                    q_curr = q_bnsd[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :]
                    q_curr_fp32 = q_curr.to(dtype=torch.float32)

                    if template_idx == 1 or template_idx == 2 or template_idx == 4:
                        if template_idx == 1:
                            threshold = 0
                            if self.cmp_mask_mode == 3:
                                threshold = math.floor(
                                    (restored_cmp_act_kv - cur_act_q + i_S1 + 1)
                                    / (self.cmp_ratio)
                                )
                            elif self.cmp_mask_mode == 0:
                                threshold = cur_cmp_act_kv
                            threshold = min(threshold, cur_cmp_act_kv)
                            if threshold <= 0:
                                empty_flag = True
                            else:
                                empty_flag = False
                            cur_cmp_k = (
                                cmp_k_bnsd[i_B, i_N2, :threshold, :]
                                if not empty_flag
                                else []
                            )

                        else:
                            cmp_topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                            empty_flag, cur_cmp_k = self.gather_cmp_kv(
                                cmp_k_bnsd,
                                cmp_topk_id,
                                i_B,
                                i_N2,
                                i_S1,
                                restored_cmp_act_kv,
                                cur_act_q,
                                self.cmp_mask_mode,
                                self.K,
                                cmp_topk_length_bnsd,
                                self.cmp_ratio,
                                cur_cmp_act_kv,
                            )
                        if is_empty(cur_cmp_k):
                            cmp_s2_loop_time = 0
                            cur_cmp_k_fp32 = []
                        else:
                            cmp_s2_loop_time = math.ceil(
                                cur_cmp_k.size(0) / s2_base_size
                            )
                            cur_cmp_k_fp32 = cur_cmp_k.to(dtype=torch.float32)
                    else:
                        empty_flag = True
                        cur_cmp_k = []
                        cur_cmp_k_fp32 = []
                        cmp_s2_loop_time = 0

                    if template_idx == 3 or template_idx == 4:
                        # ORI_SPARSE / ORI_CMP_SPARSE: gather ori_kv via ori_sparse_indices
                        ori_topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        empty_flag_ori, cur_ori_k_bnsd = self.gather_ori_kv(
                            ori_k_bnsd,
                            ori_topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_ori_act_kv,
                            cur_act_q,
                            self.ori_mask_mode,
                            self.K1,
                            ori_topk_length_bnsd,
                        )
                        if is_empty(cur_ori_k_bnsd):
                            ori_s2_loop_time = 0
                        else:
                            ori_s2_loop_time = math.ceil(
                                cur_ori_k_bnsd.size(0) / s2_base_size
                            )
                    else:
                        if self.ori_mask_mode == 4:
                            ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                            if self.ori_win_right == -1:
                                ori_win_end = cur_ori_act_kv
                            else:
                                ori_win_end = min(
                                    max(ori_threshold + self.ori_win_right, 0),
                                    cur_ori_act_kv,
                                )
                            if self.ori_win_left == -1:
                                ori_win_start = 0
                            else:
                                ori_win_start = max(
                                    ori_threshold - self.ori_win_left - 1, 0
                                )
                        elif self.ori_mask_mode == 3:
                            ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                            ori_win_end = max(ori_threshold, 0)
                            ori_win_start = 0
                        elif self.ori_mask_mode == 0:
                            ori_win_start = 0
                            ori_win_end = cur_ori_act_kv

                        if ori_win_start >= ori_win_end:
                            cur_ori_k_bnsd = []
                            empty_flag_ori = True
                        else:
                            cur_ori_k_bnsd = ori_k_bnsd[
                                i_B, i_N2, ori_win_start:ori_win_end, :
                            ]
                            empty_flag_ori = False

                    if is_empty(cur_ori_k_bnsd) and is_empty(cur_cmp_k):
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = torch.zeros(
                            [G, self.D], dtype=torch.float
                        )
                        continue

                    cur_attn_out = attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :]
                    if batch_consistency:
                        ori_k_for_reduce = (
                            None
                            if is_empty(cur_ori_k_bnsd)
                            else cur_ori_k_bnsd.to(dtype=torch.float32)
                        )
                        cmp_k_for_reduce = (
                            None
                            if is_empty(cur_cmp_k)
                            else cur_cmp_k.to(dtype=torch.float32)
                        )
                        acc_o, final_lse = self._calculate_batch_consistency(
                            q_curr_fp32,
                            ori_k_for_reduce,
                            cmp_k_for_reduce,
                            cur_sinks,
                        )
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = acc_o.to(
                            dtype=q_bnsd.dtype
                        )
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = final_lse
                        continue

                    if RUN_MODE == 0:
                        if empty_flag:
                            k_concat = (
                                cur_ori_k_bnsd
                                if not is_empty(cur_ori_k_bnsd)
                                else cur_cmp_k
                            )
                        else:
                            if is_empty(cur_ori_k_bnsd):
                                k_concat = cur_cmp_k
                            else:
                                k_concat = torch.concat(
                                    [cur_ori_k_bnsd, cur_cmp_k], dim=0
                                )

                        k_concat_fp32 = k_concat.to(dtype=torch.float32)
                        v_concat_fp32 = k_concat_fp32.clone()

                        mm1_res = torch.matmul(q_curr_fp32, k_concat_fp32.T)
                        scale_res = mm1_res * self.softmax_scale
                        softmax_res, x_max, x_sum = self.sinks_softmax(
                            scale_res, cur_sinks_expand
                        )
                        mm2_res = torch.matmul(
                            softmax_res.to(dtype=q_bnsd.dtype).to(dtype=torch.float),
                            v_concat_fp32,
                        )
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = mm2_res.to(
                            dtype=q_bnsd.dtype
                        )
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = x_max[:, 0] + torch.log(
                                x_sum[:, 0] + 1e-10
                            )
                    elif RUN_MODE == 1:
                        if empty_flag_ori:
                            ori_s2_loop_time = 0
                            total_s2_loop_time = cmp_s2_loop_time
                            row_sum = torch.empty((G), dtype=torch.float32).uniform_(
                                1.0, 1.0
                            )
                            row_max = torch.empty((G, 1), dtype=torch.float32)
                            row_max = cur_sinks
                        else:
                            ori_s2_loop_time = math.ceil(
                                cur_ori_k_bnsd.size(0) / s2_base_size
                            )
                            total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time
                            cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                            row_sum = torch.empty((G), dtype=torch.float32).uniform_(
                                1.0, 1.0
                            )
                            row_max = torch.empty((G, 1), dtype=torch.float32)
                            row_max = cur_sinks

                        for i_S2 in range(total_s2_loop_time):
                            if i_S2 < ori_s2_loop_time:  # ori_kv
                                if i_S2 < ori_s2_loop_time - 1:
                                    k_tile = cur_ori_k_bnsd_fp32[
                                        i_S2 * s2_base_size : (i_S2 + 1) * s2_base_size,
                                        :,
                                    ]
                                else:
                                    k_tile = cur_ori_k_bnsd_fp32[
                                        i_S2 * s2_base_size :, :
                                    ]
                            else:  # cmp_kv
                                if i_S2 < total_s2_loop_time - 1:
                                    k_tile = cur_cmp_k_fp32[
                                        (i_S2 - ori_s2_loop_time) * s2_base_size : (
                                            i_S2 - ori_s2_loop_time + 1
                                        )
                                        * s2_base_size,
                                        :,
                                    ]
                                else:
                                    k_tile = cur_cmp_k_fp32[
                                        (i_S2 - ori_s2_loop_time) * s2_base_size :, :
                                    ]
                            v_tile = k_tile.clone()
                            mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                            scale_res = (
                                mm1_res * self.softmax_scale
                            )  # 外层for S1 循环，据实拷入数据，因此不需要mask

                            row_max_old = row_max.clone()
                            row_max_tmp = torch.max(scale_res, dim=1)[0]
                            # row_max_tmp = row_max_tmp.unsqueeze(1)
                            row_max = torch.max(row_max, row_max_tmp)
                            update_mul = torch.exp(row_max_old - row_max)
                            row_max_expand = row_max.unsqueeze(1)
                            update_mul_expand = update_mul.unsqueeze(1)
                            cur_softmax_res = torch.exp(scale_res - row_max_expand)
                            row_sum = update_mul * row_sum + torch.sum(
                                cur_softmax_res, dim=1
                            )
                            cur_softmax_res = cur_softmax_res.to(dtype=q_bnsd.dtype).to(
                                dtype=torch.float
                            )
                            cur_o = torch.matmul(cur_softmax_res, v_tile)
                            cur_attn_out = cur_attn_out * update_mul_expand + cur_o
                        row_sum_expand = row_sum.unsqueeze(1)
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = (
                            cur_attn_out / row_sum_expand
                        ).to(dtype=q_bnsd.dtype)
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = row_max + torch.log(
                                row_sum + 1e-10
                            )
                    elif RUN_MODE == 2:
                        ori_s2_loop_time = math.ceil(
                            cur_ori_k_bnsd.size(0) / s2_base_size
                        )
                        total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time

                        cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                        row_sum = torch.empty((G), dtype=torch.float32).uniform_(
                            1.0, 1.0
                        )  # due to sinks
                        row_max = torch.empty((G, 1), dtype=torch.float32)
                        row_max = cur_sinks
                        O_flash = torch.empty(
                            (G, q_bnsd.shape[3]), dtype=torch.float32
                        ).uniform_(2 ** (-80), 2 ** (-80))  # 注意D轴

                        rcof_old = torch.empty((G), dtype=torch.float32).uniform_(
                            1.0, 1.0
                        )

                        for i_S2 in range(total_s2_loop_time):
                            if i_S2 < ori_s2_loop_time:  # ori_kv
                                if i_S2 < ori_s2_loop_time - 1:
                                    k_tile = cur_ori_k_bnsd_fp32[
                                        i_S2 * s2_base_size : (i_S2 + 1) * s2_base_size,
                                        :,
                                    ]
                                else:
                                    k_tile = cur_ori_k_bnsd_fp32[
                                        i_S2 * s2_base_size :, :
                                    ]
                            else:  # cmp_kv
                                if i_S2 < total_s2_loop_time - 1:
                                    k_tile = cur_cmp_k_fp32[
                                        (i_S2 - ori_s2_loop_time) * s2_base_size : (
                                            i_S2 + 1
                                        )
                                        * s2_base_size,
                                        :,
                                    ]
                                else:
                                    k_tile = cur_cmp_k_fp32[
                                        (i_S2 - ori_s2_loop_time) * s2_base_size :, :
                                    ]
                            v_tile = k_tile.clone()

                            mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                            scale_res = (
                                mm1_res * self.softmax_scale
                            )  # 外层for S1 循环，据实拷入数据，因此不需要mask

                            row_max_old = row_max.clone()
                            row_max_tmp = torch.max(scale_res, dim=1)[0]
                            row_max = torch.max(row_max, row_max_tmp)
                            update_mul = torch.exp(row_max_old - row_max)
                            row_max_expand = row_max.unsqueeze(1)
                            log_2 = torch.log(torch.tensor(2.0))

                            if i_S2 == 0:
                                ni = torch.round((-row_max) / log_2)
                                ni_old = ni.clone()
                            else:
                                ni_old = ni.clone()
                                ni = torch.round((-row_max) / log_2)

                            cur_softmax_res = torch.exp(scale_res - row_max_expand)
                            row_sum = update_mul * row_sum + torch.sum(
                                cur_softmax_res, dim=1
                            )
                            tmp_scale = torch.exp(row_max / log_2 + ni) * log_2
                            tmp_scale_16 = tmp_scale.to(dtype=q_bnsd.dtype).to(
                                dtype=torch.float
                            )
                            tmp_scale_16_expand = tmp_scale_16.unsqueeze(1)
                            cur_softmax_res = cur_softmax_res * tmp_scale_16_expand
                            cur_softmax_res = cur_softmax_res.to(dtype=q_bnsd.dtype).to(
                                dtype=torch.float
                            )
                            cur_o = torch.matmul(cur_softmax_res, v_tile)
                            rcof = tmp_scale / tmp_scale_16
                            eps = (rcof_old / rcof - 1) * 1.5

                            if i_S2 != 0:
                                N = (
                                    torch.clamp((ni - ni_old), min=-30).to(
                                        torch.float32
                                    )
                                    + eps
                                    + 1e-6
                                )
                                N_up = (
                                    N * torch.tensor(2 ** (23), dtype=torch.float32)
                                ).to(torch.int32)
                                N_up_expand = N_up.unsqueeze(1)
                                O_int = O_flash.view(torch.int32) + N_up_expand
                                O_fp = O_int.view(torch.float32)
                                O_flash = O_fp + cur_o
                            else:
                                O_flash = O_flash + cur_o
                            rcof_old = rcof
                        row_sum_expand = row_sum.unsqueeze(1)
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = (
                            O_flash / row_sum_expand / tmp_scale_16_expand
                        ).to(dtype=q_bnsd.dtype)
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = row_max + torch.log(
                                row_sum + 1e-10
                            )
                    else:
                        raise ValueError(f"unsupported RUN_MODE:{RUN_MODE}")
        return attn_out, softmax_lse

    def gather_ori_kv(
        self,
        k_tensor,
        topk_id,
        i_B,
        i_N2,
        i_S1,
        cur_ori_act_kv,
        cur_act_q,
        mask_mode,
        K,
        topk_length_bnsd,
        sparse_block_size=1,
    ):
        s2_sparse = list()
        ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
        left_bound = 0
        right_bound = cur_ori_act_kv
        if mask_mode == 3:
            left_bound = 0
            right_bound = min(max(ori_threshold, 0), cur_ori_act_kv)
        elif mask_mode == 4:
            if self.ori_win_left == -1:
                left_bound = 0
            else:
                left_bound = max(ori_threshold - self.ori_win_left - 1, 0)
            if self.ori_win_right == -1:
                right_bound = cur_ori_act_kv
            else:
                right_bound = min(
                    max(ori_threshold + self.ori_win_right, 0), cur_ori_act_kv
                )
        elif mask_mode == 0:
            pass
        left_bound = min(left_bound, cur_ori_act_kv)

        valid_count = min(K, math.ceil(right_bound / sparse_block_size))
        if topk_length_bnsd is not None:
            valid_count = min(topk_length_bnsd[i_B, 0, i_S1, 0], valid_count)

        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = (
                begin_idx + sparse_block_size
                if begin_idx + sparse_block_size <= cur_ori_act_kv
                else cur_ori_act_kv
            )
            if begin_idx >= right_bound:
                continue
            if begin_idx < left_bound:
                continue
            if end_idx <= right_bound:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, right_bound))

        empty_flag = False
        if len(s2_sparse) == 0:
            cur_ori_k = torch.tensor([])
            empty_flag = True
        else:
            cur_ori_k = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, cur_ori_k

    def gather_cmp_kv(
        self,
        k_tensor,
        topk_id,
        i_B,
        i_N2,
        i_S1,
        restored_cmp_act_kv,
        cur_act_q,
        mask_mode,
        K,
        topk_length_bnsd,
        cmp_ratio,
        cur_cmp_act_kv=None,
        sparse_block_size=1,
    ):
        s2_sparse = list()
        if cur_cmp_act_kv is None:
            cur_cmp_act_kv = math.floor(restored_cmp_act_kv / cmp_ratio)
        # Actual lengths arrive as scalar tensors; read this integer once so
        # the per-index bounds check does not launch a tensor comparison.
        cur_cmp_act_kv = int(cur_cmp_act_kv)
        threshold = 0
        if mask_mode == 3:
            threshold = math.floor(
                (restored_cmp_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
            )
        elif mask_mode == 0:
            threshold = cur_cmp_act_kv
        threshold = min(threshold, cur_cmp_act_kv)

        valid_count = min(K, math.ceil(threshold / sparse_block_size))
        if topk_length_bnsd is not None:
            valid_count = min(topk_length_bnsd[i_B, 0, i_S1, 0], valid_count)

        # Read integer indices once; preserve order and the first -1 terminator.
        topk_values = topk_id.tolist()
        for i_valid in range(valid_count):
            cur_topk_id = topk_values[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = (
                begin_idx + sparse_block_size
                if begin_idx + sparse_block_size <= cur_cmp_act_kv
                else cur_cmp_act_kv
            )
            if begin_idx >= threshold:
                continue
            if end_idx <= threshold:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, threshold))

        empty_flag = False
        if len(s2_sparse) == 0:
            cur_cmp_k = []
            empty_flag = True
        else:
            cur_cmp_k = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, cur_cmp_k

    def sinks_softmax(self, x, sinks):  # [G, S2] [G, 1]
        x = x.to(dtype=torch.float)
        x_concat = torch.cat([x, sinks], dim=1)
        x_max = x_concat.max(dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True) + torch.exp(sinks - x_max)
        ans = y / x_sum
        return ans, x_max, x_sum

    def trans_shape_to_bnsd(self, tensor, shape, layout, act_seq=None, seqused=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = shape[3]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout in ["TND"]:
            shape[0]
            N = shape[1]
            D = shape[2]
            B = len(act_seq) - 1  # TND act_q is cumulative
            max_s1 = get_max_adjacent_diff(act_seq)
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            seqused_per_batch = (
                seqused if seqused is not None else act_seq_per_batch
            )  # padding
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            # t_start = 0
            for b_index in range(B):
                t_start = int(act_seq[b_index])
                cur_act_seq = int(act_seq_per_batch[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                # t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_seqused, :] = tensor[
                        t_start : t_start + cur_seqused, n_index, :
                    ]
                # t_start += cur_act_seq
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_topk_length_shape_to_bnsd(
        self, tensor, shape, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            tensor = tensor.reshape(B, 1, S, 1)
            return tensor, [B, 1, S, 1]
        elif layout in ["TND"]:
            shape[0]
            B = len(cu_seqlens_q) - 1
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            new_tensor = torch.zeros((B, 1, max_s1, 1), dtype=tensor.dtype)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                new_tensor[b_index, 0, 0:cur_seqused, :] = tensor[
                    t_start : t_start + cur_seqused, :
                ]
            return new_tensor, [B, 1, max_s1, 1]

    def trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["TND"]:
            T = act_seq[-1]
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.zeros((T, N, D), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[
                        b_index, n_index, :cur_act_seq, :
                    ]
                t_start += cur_act_seq
            return output
        else:
            return tensor

    def lse_trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:  # B N S G
            return tensor
        elif layout in ["TND"]:  # B N2 S1 G  --> T1 N2 G --> N2 T1 G
            T = act_seq[-1]
            B = tensor.shape[0]
            N2 = tensor.shape[1]
            G = tensor.shape[3]
            output = torch.zeros((N2, T, G), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(
                act_seq
            )  # prefix_sum_to_original 还原成每个batch的真实长度
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N2):
                    output[n_index, t_start:t_end, :] = tensor[
                        b_index, n_index, :cur_act_seq, :
                    ]
                    # output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :cur_act_seq, :]
                t_start += cur_act_seq
            return output
        else:
            return tensor

    def forward(
        self,
        q,
        ori_k,
        ori_sparse_indices,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        template_idx,
        cmp_k=None,
        cmp_sparse_indices=None,
        seqused_q=None,
        ori_topk_length=None,
        cmp_topk_length=None,
        return_softmax_lse=False,
        batch_consistency=False,
    ):
        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(
            q, q.shape, self.layout_q, cu_seqlens_q, seqused_q
        )
        ori_k_bnsd, ori_k_bnsd_shape = self.trans_shape_to_bnsd(
            ori_k, ori_k.shape, self.layout_kv, cu_seqlens_ori_kv, seqused_ori_kv
        )
        if template_idx == 1 or template_idx == 2 or template_idx == 4:
            cmp_k_bnsd, cmp_k_bnsd_shape = self.trans_shape_to_bnsd(
                cmp_k, cmp_k.shape, self.layout_kv, cu_seqlens_cmp_kv, seqused_cmp_kv
            )
        else:
            cmp_k_bnsd = None

        cmp_sparse_indices_bnsd = None
        if template_idx == 2 or template_idx == 4:
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = (
                self.trans_shape_to_bnsd(
                    cmp_sparse_indices,
                    cmp_sparse_indices.shape,
                    self.layout_q,
                    cu_seqlens_q,
                    seqused_q,
                )
            )

        ori_sparse_indices_bnsd = None
        if template_idx == 3 or template_idx == 4:
            ori_sparse_indices_bnsd, ori_sparse_indices_bnsd_shape = (
                self.trans_shape_to_bnsd(
                    ori_sparse_indices,
                    ori_sparse_indices.shape,
                    self.layout_q,
                    cu_seqlens_q,
                    seqused_q,
                )
            )
        ori_topk_length_bnsd = None
        if (template_idx == 3 or template_idx == 4) and ori_topk_length is not None:
            ori_topk_length_bnsd, _ = self.trans_topk_length_shape_to_bnsd(
                ori_topk_length, ori_topk_length.shape, self.layout_q, cu_seqlens_q
            )

        cmp_topk_length_bnsd = None
        if (template_idx == 2 or template_idx == 4) and cmp_topk_length is not None:
            cmp_topk_length_bnsd, _ = self.trans_topk_length_shape_to_bnsd(
                cmp_topk_length, cmp_topk_length.shape, self.layout_q, cu_seqlens_q
            )

        attn_out, softmax_lse = self.calculate_by_bnsd(
            q_bnsd,
            ori_k_bnsd,
            ori_sparse_indices_bnsd,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            sinks,
            template_idx,
            cmp_k_bnsd,
            cmp_sparse_indices_bnsd,
            seqused_q,
            ori_topk_length_bnsd,
            cmp_topk_length_bnsd,
            return_softmax_lse,
            batch_consistency,
        )
        attn_out = self.trans_bnsd_to_target_layout(
            attn_out, self.layout_q, cu_seqlens_q
        )
        if return_softmax_lse:
            softmax_lse = self.lse_trans_bnsd_to_target_layout(
                softmax_lse, self.layout_q, cu_seqlens_q
            )
        return attn_out, softmax_lse


def prefix_sum_to_original(cu_seqlens_q):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens_q (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens_q)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens_q.ndim != 1:
        raise TypeError(
            f"输入必须是一维tensor，当前维度：{cu_seqlens_q.ndim}，形状：{cu_seqlens_q.shape}"
        )

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens_q) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens_q)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    # 切片实现：cu_seqlens_q[1:] 取第2个到最后一个元素，cu_seqlens_q[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

    return original_tensor


def get_max_adjacent_diff(cu_seqlens_q):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens_q (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens_q) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens_q[1] - cu_seqlens_q[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens_q) - 1):
        current_diff = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff


def gen_sparse_indices_bsnd(
    cmp_ratio,
    B,
    S1,
    S2,
    N2,
    K,
    cu_seqlens_q,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    mask_mode,
    sparse_indices_mode,
    kv_topk_mode,
    topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if mask_mode != 0:
        kv_topk_mode = "no"  # mask_mode != 0时，kv_topk_mode只能为no
    if sparse_indices_mode is None:  # 不传默认取full, 兼容老用例
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(
            f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}"
        )
    if kv_topk_mode not in ["fullK", "random", "no"]:
        raise ValueError(
            f"kv_topk_mode only support no/fullK/random, which is {kv_topk_mode}"
        )

    # 生成sparse_indices
    sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    if topk_length_override is not None:
        topk_length = topk_length_override
    elif kv_topk_mode != "no":
        topk_length = torch.zeros((B, S1, N2), dtype=torch.int32)
    else:
        topk_length = None

    for i_B in range(B):
        # CMP and ORI have independent causal bases.  For CMP, arch35
        # reconstructs the pre-compression length solely from CMP inputs.
        is_cmp_kv = seqused_cmp_kv is not None
        if is_cmp_kv:
            cur_cmp_act_kv = int(seqused_cmp_kv[i_B])
            cur_cmp_residual = (
                cmp_residual_kv[i_B] if cmp_residual_kv is not None else 0
            )
            cur_act_kv = restore_cmp_kv_length(
                cur_cmp_act_kv, cmp_ratio, cur_cmp_residual, mask_mode
            )
        else:
            cur_cmp_act_kv = None
            cur_act_kv = int(seqused_ori_kv[i_B]) if seqused_ori_kv is not None else S2
        if seqused_q is not None:
            cur_act_q = seqused_q[i_B]
        else:
            cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]

        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                cur_valid_left = 0
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor(
                        (cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
                    )
                elif mask_mode == 0:
                    cur_valid_s2_max = (
                        cur_cmp_act_kv
                        if is_cmp_kv
                        else math.floor(cur_act_kv / cmp_ratio)
                    )
                elif mask_mode == 4:
                    ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
                    if ori_win_left == -1:
                        left_bound = 0
                    else:
                        left_bound = max(ori_threshold - ori_win_left - 1, 0)
                    if ori_win_right == -1:
                        right_bound = cur_act_kv
                    else:
                        right_bound = min(
                            max(ori_threshold + ori_win_right, 0), cur_act_kv
                        )
                    left_bound = min(left_bound, cur_act_kv)
                    cur_valid_left = max(0, left_bound)
                    cur_valid_s2_max = max(0, right_bound - cur_valid_left)
                else:
                    raise ValueError(
                        f"mask_mode only support 0/3/4, which is {mask_mode}"
                    )
                cur_valid_s2_max = max(0, cur_valid_s2_max)
                if (
                    sparse_indices_mode == "random"
                ):  # sparse_indices random模式, 长度随机值生成
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(
                            1, cur_valid_s2_max, (1, 1)
                        )[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max

                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_indices[i_B, i_S1, i_N2, :valid_blocks_topk] = (
                    block_indices[0:valid_blocks_topk] + cur_valid_left
                )

                if topk_length is not None and topk_length_override is None:
                    if kv_topk_mode == "fullK":
                        topk_length[i_B, i_S1, i_N2] = min(cur_valid_s2_max_update, K)
                    elif kv_topk_mode == "random":
                        # torch.randint范围 [1, K + 1)左开右闭
                        topk_length[i_B, i_S1, i_N2] = min(
                            torch.randint(1, K + 1, (1, 1))[0], cur_valid_s2_max_update
                        )
    return sparse_indices, topk_length


def gen_sparse_indices_tnd(
    cmp_ratio,
    B,
    T1,
    N2,
    K,
    cu_seqlens_q,
    seqused_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    mask_mode,
    sparse_indices_mode,
    kv_topk_mode,
    topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if mask_mode != 0:
        kv_topk_mode = "no"  # mask_mode != 0时，kv_topk_mode只能为no
    if sparse_indices_mode is None:
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(
            f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}"
        )
    if kv_topk_mode not in ["full", "fullK", "random", "no"]:
        raise ValueError(
            f"kv_topk_mode only support no/full/random, which is {kv_topk_mode}"
        )

    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    if topk_length_override is not None:
        topk_length = topk_length_override
    elif kv_topk_mode != "no":
        topk_length = torch.zeros((T1, N2), dtype=torch.int32)
    else:
        topk_length = None

    for i_B in range(B):
        if seqused_q is not None:
            cur_act_q = seqused_q[i_B]
        else:
            cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        is_cmp_kv = seqused_cmp_kv is not None or cu_seqlens_cmp_kv is not None
        if is_cmp_kv:
            cur_cmp_act_kv = (
                int(seqused_cmp_kv[i_B])
                if seqused_cmp_kv is not None
                else int(cu_seqlens_cmp_kv[i_B + 1] - cu_seqlens_cmp_kv[i_B])
            )
            cur_cmp_residual = (
                cmp_residual_kv[i_B] if cmp_residual_kv is not None else 0
            )
            cur_act_kv = restore_cmp_kv_length(
                cur_cmp_act_kv, cmp_ratio, cur_cmp_residual, mask_mode
            )
        elif seqused_ori_kv is not None:
            cur_cmp_act_kv = None
            cur_act_kv = seqused_ori_kv[i_B]
        else:
            cur_cmp_act_kv = None
            cur_act_kv = cu_seqlens_ori_kv[i_B + 1] - cu_seqlens_ori_kv[i_B]

        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                cur_valid_left = 0
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor(
                        (cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
                    )
                elif mask_mode == 0:
                    cur_valid_s2_max = (
                        cur_cmp_act_kv
                        if is_cmp_kv
                        else math.floor(cur_act_kv / cmp_ratio)
                    )
                elif mask_mode == 4:
                    ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
                    if ori_win_left == -1:
                        left_bound = 0
                    else:
                        left_bound = max(ori_threshold - ori_win_left - 1, 0)
                    if ori_win_right == -1:
                        right_bound = cur_act_kv
                    else:
                        right_bound = min(
                            max(ori_threshold + ori_win_right, 0), cur_act_kv
                        )
                    left_bound = min(left_bound, cur_act_kv)
                    cur_valid_left = max(0, left_bound)
                    cur_valid_s2_max = max(0, right_bound - cur_valid_left)
                else:
                    raise ValueError(
                        f"mask_mode only support 0/3/4, which is {mask_mode}"
                    )
                cur_valid_s2_max = max(0, cur_valid_s2_max)

                if (
                    sparse_indices_mode == "random"
                ):  # sparse_indices random模式, 长度随机值生成
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(
                            1, cur_valid_s2_max, (1, 1)
                        )[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max
                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = (
                    block_indices[0:valid_blocks_topk] + cur_valid_left
                )

                if topk_length is not None and topk_length_override is None:
                    if kv_topk_mode == "fullK":
                        topk_length[s1_prefix + i_S1] = min(cur_valid_s2_max_update, K)
                    elif kv_topk_mode == "random":
                        # torch.randint范围 [1, K + 1)左开右闭
                        topk_length[s1_prefix + i_S1] = min(
                            torch.randint(1, K + 1, (1, 1))[0], cur_valid_s2_max_update
                        )
    return sparse_indices, topk_length


def convert_sparse_indices_to_pa_offsets(
    sparse_indices, block_table, block_size, layout_q, cu_seqlens_q
):
    """Convert logical sparse indices to physical PA token offsets for mode 2."""
    if sparse_indices is None or block_table is None:
        return sparse_indices

    physical_offsets = sparse_indices.clone()
    for batch_idx in range(block_table.shape[0]):
        if layout_q == "BSND":
            logical_slice = sparse_indices[batch_idx]
            physical_slice = physical_offsets[batch_idx]
        elif layout_q == "TND":
            q_start = int(cu_seqlens_q[batch_idx])
            q_end = int(cu_seqlens_q[batch_idx + 1])
            logical_slice = sparse_indices[q_start:q_end]
            physical_slice = physical_offsets[q_start:q_end]
        else:
            raise ValueError(f"unsupported layout_q for sparse indices: {layout_q}")

        valid_mask = logical_slice >= 0
        if not valid_mask.any().item():
            continue
        logical_values = logical_slice[valid_mask].to(torch.int64)
        logical_block_ids = torch.div(logical_values, block_size, rounding_mode="floor")
        in_block_offsets = logical_values % block_size
        physical_block_ids = block_table[
            batch_idx, logical_block_ids.to(torch.long)
        ].to(torch.int64)
        if (physical_block_ids < 0).any().item():
            raise ValueError("sparse index maps to an invalid PA block table entry")
        physical_slice[valid_mask] = (
            physical_block_ids * block_size + in_block_offsets
        ).to(physical_offsets.dtype)

    return physical_offsets


def gen_ori_kv(
    layout_q,
    layout_kv,
    ori_kv_type,
    B,
    S1,
    S2,
    T1,
    T2,
    N2,
    D,
    K1,
    block_num1,
    block_size1,
    cu_seqlens_q,
    seqused_q,
    cu_seqlens_ori_kv,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_mask_mode,
    ori_sparse_indices_mode,
    ori_kv_topk_mode,
    template_idx,
    data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT,
    ori_topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if layout_kv == "PA_BBND":
        ori_max_s2 = max(seqused_ori_kv)
        ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

        ori_k_bnsd = _nonfinite_fill(
            (B, N2, ori_max_s2, D), data_range_left, data_range_right
        )
        if ori_k_bnsd is None:
            ori_k_bnsd = (
                torch.rand((B, N2, ori_max_s2, D))
                * (data_range_left - data_range_right)
                + data_range_left
            )
        ori_k_bnsd = ori_k_bnsd.to(ori_kv_type)
        ori_block_num_per_batch = []
        ori_block_num_sum = 0

        for cur_ori_act_kv in seqused_ori_kv:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
            ori_block_num_per_batch.append(cur_ori_kv_block_num)
            ori_block_num_sum += cur_ori_kv_block_num

        if block_num1 < ori_block_num_sum:
            raise ValueError(
                f"ori_kv actual_block_num < needed_block_num, which is {block_num1} < {ori_block_num_sum}"
            )

        ori_block_id_list = np.arange(block_num1)
        ori_block_id_list = np.random.permutation(ori_block_id_list).astype(np.int32)
        cur_block_id = 0
        ori_block_table = np.full(
            (B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32
        )
        batch_idx = 0
        for cur_block_id_threshold in ori_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                ori_block_table[batch_idx][i_block_id] = ori_block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1

        # [B, S, N, D] expand to [B, ori_max_block_num_per_batch * block_size1, N, D]
        ori_k_expand = torch.zeros(
            (B, N2, ori_max_block_num_per_batch * block_size1, D), dtype=ori_kv_type
        )
        ori_k_expand[:, :, :ori_max_s2, :] = ori_k_bnsd
        ori_k_in_pa_shape = torch.zeros(
            (block_num1, block_size1, N2, D), dtype=ori_kv_type
        )

        for i_B in range(B):
            for i_block, cur_block_id in enumerate(ori_block_table[i_B]):
                block_start_pos = i_block * block_size1
                if cur_block_id == -1:
                    continue
                else:
                    for i_N2 in range(N2):
                        ori_k_in_pa_shape[cur_block_id, :, i_N2, :] = ori_k_expand[
                            i_B,
                            i_N2,
                            block_start_pos : block_start_pos + block_size1,
                            :,
                        ]

        ori_block_table = torch.tensor(ori_block_table).to(torch.int32)
        ori_k = ori_k_bnsd
    elif layout_kv == "TND":
        ori_block_table = None
        ori_max_s2 = get_max_adjacent_diff(cu_seqlens_ori_kv)
        ori_k_tnd = _nonfinite_fill((T2, N2, D), data_range_left, data_range_right)
        if ori_k_tnd is None:
            ori_k_tnd = (
                torch.rand((T2, N2, D)) * (data_range_left - data_range_right)
                + data_range_left
            )
        ori_k_tnd = ori_k_tnd.to(ori_kv_type)
        ori_k_in_pa_shape = ori_k_tnd
        ori_k = ori_k_tnd
    elif layout_kv == "BSND":
        ori_block_table = None
        ori_max_s2 = S2
        ori_k_bnsd = _nonfinite_fill(
            (B, ori_max_s2, N2, D), data_range_left, data_range_right
        )
        if ori_k_bnsd is None:
            ori_k_bnsd = (
                torch.rand((B, ori_max_s2, N2, D))
                * (data_range_left - data_range_right)
                + data_range_left
            )
        ori_k_bnsd = ori_k_bnsd.to(ori_kv_type)
        ori_k_in_pa_shape = ori_k_bnsd
        ori_k = ori_k_bnsd

    ori_sparse_indices = None
    ori_topk_length = None

    # ORI_SPARSE or ORI_CMP_SPARSE mode: generate ori_sparse_indices
    if template_idx == 3 or template_idx == 4:
        if K1 is None:
            raise ValueError("K1 must be provided for ORI_SPARSE/ORI_CMP_SPARSE mode")
        if layout_q == "BSND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_bsnd(
                1,
                B,
                S1,
                S2,
                N2,
                K1,
                cu_seqlens_q,
                seqused_q,
                seqused_ori_kv,
                None,
                None,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                topk_length_override=ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )
        elif layout_q == "TND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_tnd(
                1,
                B,
                T1,
                N2,
                K1,
                cu_seqlens_q,
                seqused_q,
                cu_seqlens_ori_kv,
                None,
                seqused_ori_kv,
                None,
                None,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                topk_length_override=ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )

    return (
        ori_k_in_pa_shape,
        ori_block_table,
        ori_k,
        ori_sparse_indices,
        ori_topk_length,
    )


def gen_cmp_kv(
    layout_q,
    layout_kv,
    cmp_kv_type,
    B,
    S1,
    S2,
    T1,
    T2,
    T3,
    N2,
    D,
    K,
    block_num2,
    block_size2,
    cu_seqlens_q,
    seqused_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    cmp_ratio,
    cmp_mask_mode,
    cmp_sparse_indices_mode,
    cmp_kv_topk_mode,
    template_idx,
    data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT,
    cmp_topk_length_override=None,
):
    if cmp_ratio is None:
        raise ValueError("cmp_ratio can't be None")
    if cmp_ratio < 1 or cmp_ratio > 128:
        raise ValueError(f"cmp_ratio should be in range [1, 128], but got {cmp_ratio}")

    if layout_kv == "PA_BBND":
        cmp_max_s2 = max(int(length) for length in seqused_cmp_kv)
        cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)

        cmp_k = _nonfinite_fill(
            (B, N2, cmp_max_s2, D), data_range_left, data_range_right
        )
        if cmp_k is None:
            cmp_k = (
                torch.rand((B, N2, cmp_max_s2, D))
                * (data_range_left - data_range_right)
                + data_range_left
            )
        cmp_k = cmp_k.to(cmp_kv_type)
        cmp_block_num_per_batch = []
        cmp_block_num_sum = 0
        for cur_cmp_act_kv in seqused_cmp_kv:
            cur_cmp_act_kv = int(cur_cmp_act_kv)
            cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
            cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
            cmp_block_num_sum += cur_cmp_kv_block_num
        if block_num2 < cmp_block_num_sum:
            raise ValueError(
                f"cmp_kv actual_block_num < needed_block_num, which is {block_num2} < {cmp_block_num_sum}"
            )

        cmp_block_id_list = np.arange(block_num2)
        cmp_block_id_list = np.random.permutation(cmp_block_id_list).astype(np.int32)
        cur_block_id = 0
        cmp_block_table = np.full(
            (B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32
        )
        batch_idx = 0
        for cur_block_id_threshold in cmp_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                cmp_block_table[batch_idx][i_block_id] = cmp_block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1

        cmp_k_expand = torch.zeros(
            (B, N2, cmp_max_block_num_per_batch * block_size2, D), dtype=cmp_kv_type
        )
        cmp_k_expand[:, :, :cmp_max_s2, :] = cmp_k
        cmp_k_in_pa_shape = torch.zeros(
            (block_num2, block_size2, N2, D), dtype=cmp_kv_type
        )
        for i_B in range(B):
            for i_block, cur_block_id in enumerate(cmp_block_table[i_B]):
                block_start_pos = i_block * block_size2
                if cur_block_id == -1:
                    continue
                else:
                    for i_N2 in range(N2):
                        cmp_k_in_pa_shape[cur_block_id, :, i_N2, :] = cmp_k_expand[
                            i_B,
                            i_N2,
                            block_start_pos : block_start_pos + block_size2,
                            :,
                        ]

        cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
        if cmp_block_table.shape[1] == 0:
            cmp_block_table = None
    elif layout_kv == "TND":
        T3 = int(T3)
        cmp_block_table = None
        cmp_k = _nonfinite_fill((T3, N2, D), data_range_left, data_range_right)
        if cmp_k is None:
            cmp_k = (
                torch.rand((T3, N2, D)) * (data_range_left - data_range_right)
                + data_range_left
            )
        cmp_k = cmp_k.to(cmp_kv_type)
        cmp_k_in_pa_shape = cmp_k
    elif layout_kv == "BSND":
        cmp_block_table = None
        # Keep the legacy full-capacity default, while allowing an independently
        # supplied CMP actual length to exceed ORI-derived capacity.
        cmp_max_s2 = max(int(length) for length in seqused_cmp_kv)
        cmp_k = _nonfinite_fill(
            (B, cmp_max_s2, N2, D), data_range_left, data_range_right
        )
        if cmp_k is None:
            cmp_k = (
                torch.rand((B, cmp_max_s2, N2, D))
                * (data_range_left - data_range_right)
                + data_range_left
            )
        cmp_k = cmp_k.to(cmp_kv_type)
        cmp_k_in_pa_shape = cmp_k

    # generate cmp_sparse_indices
    cmp_sparse_indices = None
    cmp_topk_length = None

    if template_idx == 2 or template_idx == 4:
        if layout_q == "BSND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_bsnd(
                cmp_ratio,
                B,
                S1,
                S2,
                N2,
                K,
                None,
                seqused_q,
                seqused_ori_kv,
                seqused_cmp_kv,
                cmp_residual_kv,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                topk_length_override=cmp_topk_length_override,
            )
        elif layout_q == "TND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_tnd(
                cmp_ratio,
                B,
                T1,
                N2,
                K,
                cu_seqlens_q,
                seqused_q,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                seqused_ori_kv,
                seqused_cmp_kv,
                cmp_residual_kv,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                topk_length_override=cmp_topk_length_override,
            )

    return (
        cmp_k_in_pa_shape,
        cmp_sparse_indices,
        cmp_block_table,
        cmp_k,
        cmp_topk_length,
    )


def resolve_compressed_actual_lengths(
    seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, cmp_ratio, cmp_mask_mode=3
):
    """Resolve compressed actual lengths exactly as SMLA input generation does."""
    if seqused_ori_kv is None or cmp_ratio is None:
        return seqused_cmp_kv, cmp_residual_kv
    if cmp_ratio < 1:
        raise ValueError(f"cmp_ratio should be in range [1, 128], but got {cmp_ratio}")
    if seqused_cmp_kv is None:
        if torch.is_tensor(seqused_ori_kv):
            seqused_cmp_kv = seqused_ori_kv // cmp_ratio
        else:
            seqused_cmp_kv = [value // cmp_ratio for value in seqused_ori_kv]
    if cmp_mask_mode == 0 or int(cmp_ratio) == 1:
        return seqused_cmp_kv, None
    if cmp_residual_kv is None:
        if torch.is_tensor(seqused_ori_kv):
            cmp_residual_kv = seqused_ori_kv % cmp_ratio
        else:
            cmp_residual_kv = [value % cmp_ratio for value in seqused_ori_kv]
    return seqused_cmp_kv, cmp_residual_kv


def resolve_query_actual_lengths(layout_q, batch_size, sequence_length, seqused_q):
    """Resolve the query actual lengths exactly as SMLA input generation does."""
    if layout_q == "BSND" and seqused_q is None:
        return [int(sequence_length)] * int(batch_size)
    return seqused_q


def gen_data(params, prepare_device_storage=True, generate_golden=True):
    # 从字典中提取参数
    layout_q = params.get("layout_q")
    layout_kv = params.get("layout_kv")
    q_type = params.get("q_type")
    ori_kv_type = params.get("ori_kv_type")
    cmp_kv_type = params.get("cmp_kv_type")
    B = params.get("B")
    S1 = params.get("S1")
    T1 = params.get("T1")
    T2 = params.get("T2")
    T3 = params.get("T3")
    N1 = params.get("N1")
    N2 = params.get("N2")
    D = params.get("D")
    K1 = params.get("K1")
    K = params.get("K")
    block_num1 = params.get("block_num1")
    block_num2 = params.get("block_num2")
    block_size1 = params.get("block_size1")
    block_size2 = params.get("block_size2")
    cu_seqlens_q = params.get("cu_seqlens_q")
    cu_seqlens_ori_kv = params.get("cu_seqlens_ori_kv")
    cu_seqlens_cmp_kv = params.get("cu_seqlens_cmp_kv")
    seqused_ori_kv = params.get("seqused_ori_kv")
    seqused_cmp_kv = params.get("seqused_cmp_kv")
    cmp_residual_kv = params.get("cmp_residual_kv")
    softmax_scale = params.get("softmax_scale")
    cmp_ratio = params.get("cmp_ratio")
    ori_mask_mode = params.get("ori_mask_mode")
    cmp_mask_mode = params.get("cmp_mask_mode")
    ori_win_left = params.get("ori_win_left")
    ori_win_right = params.get("ori_win_right")
    case_name = params.get("testcase_name")
    S2 = params.get("S2")
    q_datarange = params.get("q_datarange")
    ori_kv_datarange = params.get("ori_kv_datarange")
    cmp_kv_datarange = params.get("cmp_kv_datarange")
    seqused_q = params.get("seqused_q")
    ori_kv_topk_mode = params.get("ori_kv_topk_mode")
    cmp_kv_topk_mode = params.get("cmp_kv_topk_mode")
    ori_sparse_indices_mode = params.get("ori_sparse_indices_mode")
    cmp_sparse_indices_mode = params.get("cmp_sparse_indices_mode")
    return_softmax_lse = params.get("return_softmax_lse")
    topk_value_mode = int(params.get("topk_value_mode", 1))
    batch_consistency = params.get("batch_consistency", False)
    template_mode = params.get("template_mode")

    # 路由到各算子的逻辑。必须先于Tensor转换和压缩长度推导执行，
    # 否则SWA会从共享参数中生成无效的cmp_kv相关Tensor。
    template_idx = 0
    mapping = {"SWA": 0, "HCA": 1, "CSA": 2, "ORI_SPARSE": 3, "ORI_CMP_SPARSE": 4}
    if template_mode is not None and template_mode in mapping:
        template_run_mode = template_mode
        template_idx = mapping[template_mode]
        if template_mode == "SWA":
            cmp_ratio = None
            K1 = None
            K = None
        elif template_mode == "HCA":
            cmp_ratio = int(cmp_ratio) if cmp_ratio is not None else 1
            K1 = None
            K = None
        elif template_mode == "CSA":
            cmp_ratio = int(cmp_ratio) if cmp_ratio is not None else 1
            K = int(K) if K is not None else None
        elif template_mode == "ORI_SPARSE":
            cmp_ratio = 1
            K1 = int(K1) if K1 is not None else None
            K = None
        elif template_mode == "ORI_CMP_SPARSE":
            cmp_ratio = int(cmp_ratio) if cmp_ratio is not None else 1
            K1 = int(K1) if K1 is not None else None
            K = int(K) if K is not None else None
    else:
        if K is None or K == ["None"]:
            if cmp_ratio is None:
                template_idx = 0  # SWA
                template_run_mode = "SWA"
            else:
                template_idx = 1  # HCA
                template_run_mode = "HCA"
                cmp_ratio = int(cmp_ratio)
        else:
            template_idx = 2  # CSA
            template_run_mode = "CSA"
            cmp_ratio = 1 if cmp_ratio is None else int(cmp_ratio)
            K = int(K)

    # 支持外部指定topk_length
    ori_topk_length_override = params.get("ori_topk_length", None)
    cmp_topk_length_override = params.get("cmp_topk_length", None)

    if template_idx == 0:
        # SWA only consumes ori_kv. Do not materialize or retain any compressed-KV
        # tensor/shape metadata, even when it was supplied by a shared case source.
        cmp_mask_mode = 0
        cmp_kv_type = None
        T3 = None
        block_num2 = None
        block_size2 = None
        cu_seqlens_cmp_kv = None
        seqused_cmp_kv = None
        cmp_residual_kv = None
        cmp_topk_length_override = None
    if (
        isinstance(ori_topk_length_override, list)
        and len(ori_topk_length_override) == 1
        and ori_topk_length_override[0] is None
    ):
        ori_topk_length_override = None
    if (
        isinstance(cmp_topk_length_override, list)
        and len(cmp_topk_length_override) == 1
        and cmp_topk_length_override[0] is None
    ):
        cmp_topk_length_override = None
    if ori_topk_length_override is not None and not isinstance(
        ori_topk_length_override, torch.Tensor
    ):
        if layout_q == "BSND":
            ori_topk_length_override = torch.tensor(
                ori_topk_length_override, dtype=torch.int32
            ).reshape(B, S1, N2)
        else:
            ori_topk_length_override = torch.tensor(
                ori_topk_length_override, dtype=torch.int32
            ).reshape(T1, N2)
    if cmp_topk_length_override is not None and not isinstance(
        cmp_topk_length_override, torch.Tensor
    ):
        if layout_q == "BSND":
            cmp_topk_length_override = torch.tensor(
                cmp_topk_length_override, dtype=torch.int32
            ).reshape(B, S1, N2)
        else:
            cmp_topk_length_override = torch.tensor(
                cmp_topk_length_override, dtype=torch.int32
            ).reshape(T1, N2)

    B = int(B)
    S1 = int(S1)
    S2 = int(S2)
    N1 = int(N1)
    N2 = int(N2)
    D = int(D)
    seqused_q = resolve_query_actual_lengths(layout_q, B, S1, seqused_q)

    if layout_q == "TND" and T1 is None:
        raise ValueError("T1 must be provided when layout_kv is TND")
    if layout_kv == "TND" and T2 is None:
        raise ValueError("T2 must be provided when layout_kv is TND")
    if layout_kv == "BSND" and S2 is None:
        raise ValueError("S2 must be provided when layout_kv is BSND")

    if cu_seqlens_q is not None:
        cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    if cu_seqlens_ori_kv is not None:
        cu_seqlens_ori_kv = torch.tensor(cu_seqlens_ori_kv).to(torch.int32)
    if cu_seqlens_cmp_kv is not None:
        cu_seqlens_cmp_kv = torch.tensor(cu_seqlens_cmp_kv).to(torch.int32)
    if seqused_ori_kv is not None:
        seqused_ori_kv = torch.tensor(seqused_ori_kv).to(torch.int32)
    if seqused_cmp_kv is not None:
        seqused_cmp_kv = torch.tensor(seqused_cmp_kv).to(torch.int32)
    if cmp_residual_kv is not None:
        cmp_residual_kv = torch.tensor(cmp_residual_kv).to(torch.int32)
    if seqused_q is not None:
        seqused_q = torch.tensor(seqused_q).to(torch.int32)

    if layout_kv == "PA_BBND" and len(seqused_ori_kv) != B:
        raise ValueError(
            f"len(seqused_ori_kv) != B, which is {len(seqused_ori_kv)} != {B}"
        )
    elif layout_kv == "TND":
        if len(cu_seqlens_ori_kv) != B + 1:
            raise ValueError(
                f"len(cu_seqlens_ori_kv) != B, which is {len(cu_seqlens_ori_kv)} != {B + 1}"
            )
        if cu_seqlens_ori_kv[-1] != T2:
            raise ValueError(
                f"cu_seqlens_ori_kv[-1] != T2, which is {cu_seqlens_ori_kv[-1]} != {T2}"
            )
    else:
        pass

    # 获取最长 q
    if layout_q == "TND":
        seq_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = torch.max(seq_lens).item()
    else:
        max_seqlen_q = S1

    # generate q
    if layout_q == "BSND":
        S1, B = int(S1), int(B)
        q = _nonfinite_fill((B, S1, N1, D), q_datarange[0], q_datarange[1])
        if q is None:
            q = (
                torch.rand((B, S1, N1, D)) * (q_datarange[1] - q_datarange[0])
                + q_datarange[0]
            )
        q = q.to(q_type)
        if seqused_q is None:
            B * [S1]
            seqused_q = torch.tensor(B * [S1]).to(torch.int32)
        else:
            seqused_q = torch.tensor(seqused_q).to(torch.int32)
    elif layout_q == "TND":
        T1, B = int(T1), int(B)
        q = _nonfinite_fill((T1, N1, D), q_datarange[0], q_datarange[1])
        if q is None:
            q = (
                torch.rand((T1, N1, D)) * (q_datarange[1] - q_datarange[0])
                + q_datarange[0]
            )
        q = q.to(q_type)
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(
                f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}"
            )
        else:
            prefix_sum_to_original(cu_seqlens_q)
            get_max_adjacent_diff(cu_seqlens_q)
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    # generate ori_kv/cmp_kv (only support PA_BBND)
    if layout_kv in ["PA_BBND", "TND", "BSND"]:
        pass
    else:
        raise ValueError(f"layout_kv is not support {layout_kv}")

    if template_idx in (1, 2, 4):
        seqused_cmp_kv, cmp_residual_kv = resolve_compressed_actual_lengths(
            seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, cmp_ratio, cmp_mask_mode
        )
    print("template_run_mode: ", template_run_mode)
    if layout_kv == "PA_BBND":
        block_size1, block_num1 = int(block_size1), int(block_num1)
        if template_idx != 0 and template_idx != 3:
            block_size2, block_num2 = int(block_size2), int(block_num2)

    ori_k_in_pa_shape, ori_block_table, ori_k, ori_sparse_indices, ori_topk_length = (
        gen_ori_kv(
            layout_q,
            layout_kv,
            ori_kv_type,
            B,
            S1,
            S2,
            T1,
            T2,
            N2,
            D,
            K1,
            block_num1,
            block_size1,
            cu_seqlens_q,
            seqused_q,
            cu_seqlens_ori_kv,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            ori_mask_mode,
            ori_sparse_indices_mode,
            ori_kv_topk_mode,
            template_idx,
            ori_kv_datarange[0],
            ori_kv_datarange[1],
            ori_topk_length_override=ori_topk_length_override,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
        )
    )

    if template_idx == 1 or template_idx == 2 or template_idx == 4:
        (
            cmp_k_in_pa_shape,
            cmp_sparse_indices,
            cmp_block_table,
            cmp_k,
            cmp_topk_length,
        ) = gen_cmp_kv(
            layout_q,
            layout_kv,
            cmp_kv_type,
            B,
            S1,
            S2,
            T1,
            T2,
            T3,
            N2,
            D,
            K,
            block_num2,
            block_size2,
            cu_seqlens_q,
            seqused_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            cmp_ratio,
            cmp_mask_mode,
            cmp_sparse_indices_mode,
            cmp_kv_topk_mode,
            template_idx,
            cmp_kv_datarange[0],
            cmp_kv_datarange[1],
            cmp_topk_length_override=cmp_topk_length_override,
        )
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k = None
        cmp_topk_length = None

    # CPU Golden always consumes logical token indices. Only the sparse-index
    # tensors passed to a PA kernel use physical token offsets in mode 2.
    ori_sparse_indices_for_op = ori_sparse_indices
    cmp_sparse_indices_for_op = cmp_sparse_indices
    if topk_value_mode == 2 and layout_kv == "PA_BBND":
        ori_sparse_indices_for_op = convert_sparse_indices_to_pa_offsets(
            ori_sparse_indices,
            ori_block_table,
            block_size1,
            layout_q,
            cu_seqlens_q,
        )
        cmp_sparse_indices_for_op = convert_sparse_indices_to_pa_offsets(
            cmp_sparse_indices,
            cmp_block_table,
            block_size2,
            layout_q,
            cu_seqlens_q,
        )

    sinks = _nonfinite_fill((N1,), q_datarange[0], q_datarange[1])
    if sinks is None:
        sinks = (
            torch.rand((N1,)) * (q_datarange[1] - q_datarange[0]) / 10
            + q_datarange[0] / 10
        )
    else:
        sinks = sinks / 10
    sinks = sinks.to(torch.float)
    if not params.get("use_sinks", True):
        sinks = None

    # Pytest materializes the A5 PA stride on device. TTK supplies the same
    # storage/stride through its CSV allocation and only needs logical CPU data.
    properties = torch.npu.get_device_properties() if prepare_device_storage else None
    if (
        properties is not None
        and "Ascend950" in properties.name
        and layout_kv == "PA_BBND"
    ):
        key_stride = 10  # 0轴非连续增加stride
        blocksize_with_stride = block_size1 + key_stride  # 整个非连续的长度
        blockFusion1 = torch.zeros(
            (block_num1, blocksize_with_stride * N2 * D), dtype=ori_kv_type
        )
        ori_key_flat = ori_k_in_pa_shape.view(block_num1, block_size1 * N2 * D)
        blockFusion1[:, : block_size1 * N2 * D] = ori_key_flat
        blockFusion1 = blockFusion1.npu()
        ori_k_in_pa_shape = blockFusion1[:, : block_size1 * N2 * D].view(
            block_num1, block_size1, N2, D
        )

        if template_idx != 0 and template_idx != 3:
            blocksize_with_stride = block_size2 + key_stride
            blockFusion2 = torch.zeros(
                (block_num2, blocksize_with_stride * N2 * D), dtype=cmp_kv_type
            )
            cmp_key_flat = cmp_k_in_pa_shape.view(block_num2, block_size2 * N2 * D)
            blockFusion2[:, : block_size2 * N2 * D] = cmp_key_flat
            blockFusion2 = blockFusion2.npu()
            cmp_k_in_pa_shape = blockFusion2[:, : block_size2 * N2 * D].view(
                block_num2, block_size2, N2, D
            )

    golden_state = {
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "q_type": q_type,
        "ori_kv_type": ori_kv_type,
        "cmp_kv_type": cmp_kv_type,
        "B": B,
        "S1": S1,
        "S2": S2,
        "T1": T1,
        "N1": N1,
        "N2": N2,
        "D": D,
        "K1": K1,
        "K": K,
        "block_num1": block_num1,
        "block_num2": block_num2,
        "block_size1": block_size1,
        "block_size2": block_size2,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
        "seqused_q": seqused_q,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "softmax_scale": softmax_scale,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": ori_mask_mode,
        "cmp_mask_mode": cmp_mask_mode,
        "ori_win_left": ori_win_left,
        "ori_win_right": ori_win_right,
        "ori_topk_length": ori_topk_length,
        "cmp_topk_length": cmp_topk_length,
        "q": q,
        "ori_k": ori_k,
        "cmp_k": cmp_k,
        "ori_sparse_indices": ori_sparse_indices,
        "cmp_sparse_indices": cmp_sparse_indices,
        "topk_value_mode": topk_value_mode,
        # A -inf sink contributes zero probability in the unchanged CPU model.
        # The operator input remains None when use_sinks is disabled.
        "sinks": sinks if sinks is not None else torch.full((N1,), -torch.inf),
        "template_idx": template_idx,
        "return_softmax_lse": return_softmax_lse,
        "batch_consistency": batch_consistency,
    }
    if generate_golden:
        generate_cpu_golden({"golden_state": golden_state})
        cpu_result = golden_state["cpu_output"]
        softmax_lse = golden_state["softmax_lse"]
    else:
        cpu_result = None
        softmax_lse = None

    max_seqlen_ori_kv = 0
    if seqused_ori_kv is not None:
        max_seqlen_ori_kv = seqused_ori_kv.max().item()
    elif cu_seqlens_ori_kv is not None:
        max_seqlen_ori_kv = get_max_adjacent_diff(cu_seqlens_ori_kv)

    max_seqlen_cmp_kv = 0
    if seqused_cmp_kv is not None:
        max_seqlen_cmp_kv = seqused_cmp_kv.max().item()
    elif cu_seqlens_cmp_kv is not None:
        max_seqlen_cmp_kv = get_max_adjacent_diff(cu_seqlens_cmp_kv)

    # ORI_SPARSE/ORI_CMP_SPARSE: seqused (actualLength) not passed to op, determined by sparse_indices and topkLength
    # cu_seqlens still passed for TND kv layout (needed for data addressing in kernel)
    # no_actual_length = template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
    # seqused_ori_kv = None if no_actual_length else seqused_ori_kv
    # seqused_cmp_kv = None if no_actual_length else seqused_cmp_kv
    cu_seqlens_ori_kv = cu_seqlens_ori_kv if layout_kv == "TND" else None
    cu_seqlens_cmp_kv = cu_seqlens_cmp_kv if layout_kv == "TND" else None
    input_data = {
        # 命名用变量
        "B": B,
        "S1": S1,
        "S2": S2,
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "N1": N1,
        "N2": N2,
        "K1": K1,
        "K": K,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "template_run_mode": template_run_mode,
        "case_name": case_name,
        # 固定参数
        "params": params,
        # 输入张量
        "metadata_input": {
            "N1": N1,
            "N2": N2,
            "D": D,
            "num_heads_q": N1,
            "num_heads_kv": N2,
            "head_dim": D,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
            "seqused_q": seqused_q,
            "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv,
            "cmp_residual_kv": cmp_residual_kv,
            "ori_topk_length": ori_topk_length if ori_topk_length is not None else None,
            "cmp_topk_length": cmp_topk_length if cmp_topk_length is not None else None,
            "B": B,
            "batch_size": B,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_ori_kv": max_seqlen_ori_kv,
            "max_seqlen_cmp_kv": max_seqlen_cmp_kv,
            "K1": K1,
            "K": K,
            "ori_topk": K1 if ori_sparse_indices is not None else 0,
            "cmp_topk": K if cmp_sparse_indices is not None else 0,
            "cmp_ratio": cmp_ratio,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "has_ori_kv": (ori_k if layout_kv == "TND" else ori_k_in_pa_shape)
            is not None,
            "has_cmp_kv": cmp_k_in_pa_shape is not None,
        },
        "input": {
            "q": q,
            "ori_kv": ori_k if layout_kv == "TND" else ori_k_in_pa_shape,
            "cmp_kv": cmp_k_in_pa_shape,
            "ori_sparse_indices": ori_sparse_indices_for_op,
            "cmp_sparse_indices": cmp_sparse_indices_for_op,
            "ori_block_table": ori_block_table,
            "cmp_block_table": cmp_block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
            "seqused_q": seqused_q,
            "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv,
            "cmp_residual_kv": cmp_residual_kv,
            "sinks": sinks,
            "softmax_scale": softmax_scale,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "topk_value_mode": topk_value_mode,
        },
        "golden_state": golden_state,
        "cpu_output": cpu_result,
        "softmax_lse": softmax_lse,
    }
    return input_data


def generate_cpu_golden(input_data):
    """Calculate CPU reference from input-stage state without regenerating random tensors."""
    state = input_data["golden_state"]
    test_smla = GeneralizedSFA(
        state["layout_q"],
        state["layout_kv"],
        state["q_type"],
        state["ori_kv_type"],
        state["cmp_kv_type"],
        state["B"],
        state["S1"],
        state["S2"],
        state["T1"],
        state["N1"],
        state["N2"],
        state["D"],
        state["K1"],
        state["K"],
        state["block_num1"],
        state["block_num2"],
        state["block_size1"],
        state["block_size2"],
        state["cu_seqlens_q"],
        state["seqused_ori_kv"],
        state["seqused_cmp_kv"],
        state["cmp_residual_kv"],
        state["softmax_scale"],
        state["cmp_ratio"],
        state["ori_mask_mode"],
        state["cmp_mask_mode"],
        state["ori_win_left"],
        state["ori_win_right"],
        state["ori_topk_length"],
        state["cmp_topk_length"],
    )
    cpu_output, softmax_lse = test_smla.forward(
        state["q"],
        state["ori_k"],
        state["ori_sparse_indices"],
        state["cu_seqlens_q"],
        state["cu_seqlens_ori_kv"],
        state["cu_seqlens_cmp_kv"],
        state["seqused_ori_kv"],
        state["seqused_cmp_kv"],
        state["cmp_residual_kv"],
        state["sinks"],
        state["template_idx"],
        state["cmp_k"],
        state["cmp_sparse_indices"],
        state["seqused_q"],
        state["ori_topk_length"],
        state["cmp_topk_length"],
        state["return_softmax_lse"],
        state.get("batch_consistency", False),
    )
    state["cpu_output"] = cpu_output
    state["softmax_lse"] = softmax_lse
    input_data["cpu_output"] = cpu_output
    input_data["softmax_lse"] = softmax_lse
    return cpu_output, softmax_lse


def save_test_case(input_data, output_dir):
    """
    保存单条测试用例到文件
    """
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    case_name = f"{input_data['template_run_mode']}_layoutQ_{input_data['layout_q']}_layoutKV_{input_data['layout_kv']}_B_{input_data['B']}_S1_{input_data['S1']}_S2_{input_data['S2']}_T1_{input_data['T1']}_N1_{input_data['N1']}_N2_{input_data['N2']}_{input_data['case_name']}"

    # 生成文件名
    input_filename = f"smla_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)

    # 保存数据
    torch.save(input_data, input_filepath)
    print(f"测试用例已保存到: {input_filepath}")
    return input_filepath


# Precision comparator copied from ops-transformer result_compare_method.py.
def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (start + idx, expect_data[j], real_data[j], diff_abs, diff_rate)
                )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (start + idx, expect_data[j], real_data[j], diff_abs, diff_rate)
                )

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


# fuzz 中 precision_method == 1的精度对比方式
def check_result(expect, npu_result):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025
    max_error_idx = 10000000

    real_data = npu_result.cpu().to(torch.float32).numpy()
    data_compe = expect.cpu().to(torch.float32).numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.size, data_compe.size)
        )
        return result, 0.0, max_error
    overflows_count = (
        data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    )

    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                data_compe[np.isinf(data_compe)][0:10],
                data_compe[np.isnan(data_compe)][0:10],
            )
        )

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    if (
        "nan" in str(real_data)
        or "inf" in str(real_data)
        or "nan" in str(data_compe)
        or "inf" in str(data_compe)
    ):
        pass

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
        # assert 1==0
    return result, fulfill_percent


_PARAM_DEFAULTS = {
    "testcase_name": [None],
    "layout_q": [None],
    "layout_kv": [None],
    "q_type": [None],
    "ori_kv_type": [None],
    "cmp_kv_type": [None],
    "B": [None],
    "S1": [None],
    "S2": [None],
    "T1": [None],
    "T2": [None],
    "T3": [None],
    "N1": [None],
    "N2": [None],
    "D": [None],
    "K1": [None],
    "K": [None],
    "block_num1": [None],
    "block_num2": [None],
    "block_size1": [None],
    "block_size2": [None],
    "seqused_q": [None],
    "cu_seqlens_q": [None],
    "cu_seqlens_ori_kv": [None],
    "cu_seqlens_cmp_kv": [None],
    "seqused_ori_kv": [None],
    "seqused_cmp_kv": [None],
    "cmp_residual_kv": [None],
    "softmax_scale": [None],
    "cmp_ratio": [None],
    "ori_mask_mode": [None],
    "cmp_mask_mode": [None],
    "ori_win_left": [None],
    "ori_win_right": [None],
    "ori_kv_topk_mode": ["no"],
    "cmp_kv_topk_mode": ["no"],
    "ori_sparse_indices_mode": ["full"],
    "cmp_sparse_indices_mode": ["full"],
    "actlen_mode": ["full"],
    "template_mode": [None],
    "q_datarange": ["[-2, 2]"],
    "ori_kv_datarange": ["[-2, 2]"],
    "cmp_kv_datarange": ["[-2, 2]"],
    "random_seq": [False],
    "return_softmax_lse": [False],
    "ori_topk_length": [None],
    "cmp_topk_length": [None],
}


def _as_choices(value):
    return value if isinstance(value, list) else [value]


def _expand_param_groups(groups):
    cases = []
    for group in groups:
        values = {
            name: _as_choices(group.get(name, default))
            for name, default in {**_PARAM_DEFAULTS, **group}.items()
        }
        names = tuple(values)
        cases.extend(
            dict(zip(names, combination))
            for combination in itertools.product(*(values[name] for name in names))
        )
    return cases


def _parse_range(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [-10, 10] if value is None else value


def _fill_cu_seqlens(total, max_length, batch, random_seq):
    per_batch = min(max_length, total // batch)
    result = [0]
    for _ in range(batch):
        length = random.randint(0, per_batch) if random_seq else per_batch
        result.append(result[-1] + length)
    result[-1] = total
    return result


def _seqused_from_cu(cu_seqlens):
    return [
        cu_seqlens[index] - cu_seqlens[index - 1] for index in range(1, len(cu_seqlens))
    ]


def _cu_from_seqused(seqused):
    result = [0]
    for length in seqused:
        result.append(result[-1] + length)
    return result


def _materialize_case(params):
    case = params.copy()
    for name in ("q_datarange", "ori_kv_datarange", "cmp_kv_datarange"):
        case[name] = _parse_range(case.get(name))

    layout_q = case["layout_q"]
    layout_kv = case["layout_kv"]
    batch = case["B"]
    random_seq = case["random_seq"]
    if case["cu_seqlens_q"] is None:
        total_q = case["T1"] if layout_q == "TND" else batch * case["S1"]
        case["cu_seqlens_q"] = _fill_cu_seqlens(total_q, case["S1"], batch, random_seq)
    if case["cu_seqlens_ori_kv"] is None and layout_kv == "TND":
        case["cu_seqlens_ori_kv"] = _fill_cu_seqlens(
            case["T2"], case["S2"], batch, random_seq
        )
    if (
        case["cu_seqlens_cmp_kv"] is None
        and layout_kv == "TND"
        and case["cmp_ratio"] is not None
    ):
        case["seqused_ori_kv"] = _seqused_from_cu(case["cu_seqlens_ori_kv"])
        case["seqused_cmp_kv"] = [
            length // case["cmp_ratio"] for length in case["seqused_ori_kv"]
        ]
        case["cu_seqlens_cmp_kv"] = _cu_from_seqused(case["seqused_cmp_kv"])
        case["T3"] = case["cu_seqlens_cmp_kv"][-1]
    if case["seqused_ori_kv"] is None and layout_kv != "TND":
        case["seqused_ori_kv"] = [
            random.randint(0, case["S2"]) if random_seq else case["S2"]
            for _ in range(batch)
        ]

    ori_blocks = 0
    cmp_blocks = 0
    if layout_kv == "PA_BBND":
        for ori_length in case["seqused_ori_kv"]:
            ori_blocks += math.ceil(ori_length / case["block_size1"])
            if case["cmp_ratio"] is not None and case["block_size2"] is not None:
                cmp_length = math.floor(ori_length / case["cmp_ratio"])
                cmp_blocks += math.ceil(cmp_length / case["block_size2"])
    case["block_num1"] = ori_blocks
    case["block_num2"] = cmp_blocks
    return case


def _to_npu(tensor):
    return tensor.npu() if tensor is not None else None


def _metadata_tensor(tensor):
    if tensor is None:
        return None
    return tensor.to(dtype=torch.int32).reshape(-1).contiguous()


def _run_npu(input_data):
    SMLA, SMLA_METADATA = _load_operators()
    params = input_data["params"]
    metadata_input = input_data["metadata_input"]
    tensor_input = input_data["input"]
    layout_q = tensor_input["layout_q"]
    if not isinstance(layout_q, str):
        layout_q = layout_q[0]
    layout_kv = tensor_input["layout_kv"]
    npu = {
        name: _to_npu(value)
        for name, value in tensor_input.items()
        if isinstance(value, torch.Tensor) or value is None
    }
    ori_topk_length = _to_npu(metadata_input.get("ori_topk_length"))
    cmp_topk_length = _to_npu(metadata_input.get("cmp_topk_length"))
    ori_sparse_indices = npu.get("ori_sparse_indices")
    cmp_sparse_indices = npu.get("cmp_sparse_indices")
    cmp_ratio = metadata_input["cmp_ratio"]
    cmp_mask_mode = tensor_input["cmp_mask_mode"]
    ori_win_left = tensor_input["ori_win_left"]
    ori_win_right = tensor_input["ori_win_right"]

    metadata = SMLA_METADATA.sparse_flash_mla_metadata(
        num_heads_q=metadata_input["N1"],
        num_heads_kv=metadata_input["N2"],
        head_dim=metadata_input["D"],
        cu_seqlens_q=_metadata_tensor(npu.get("cu_seqlens_q")),
        cu_seqlens_ori_kv=_metadata_tensor(npu.get("cu_seqlens_ori_kv")),
        cu_seqlens_cmp_kv=_metadata_tensor(npu.get("cu_seqlens_cmp_kv")),
        seqused_q=_metadata_tensor(npu.get("seqused_q")),
        seqused_ori_kv=_metadata_tensor(npu.get("seqused_ori_kv")),
        seqused_cmp_kv=_metadata_tensor(npu.get("seqused_cmp_kv")),
        cmp_residual_kv=_metadata_tensor(npu.get("cmp_residual_kv")),
        ori_topk_length=_metadata_tensor(ori_topk_length),
        cmp_topk_length=_metadata_tensor(cmp_topk_length),
        batch_size=metadata_input["B"],
        max_seqlen_q=metadata_input["max_seqlen_q"],
        max_seqlen_ori_kv=metadata_input["max_seqlen_ori_kv"],
        max_seqlen_cmp_kv=metadata_input["max_seqlen_cmp_kv"],
        ori_topk=metadata_input.get("K1") if ori_sparse_indices is not None else 0,
        cmp_topk=metadata_input["K"] if cmp_sparse_indices is not None else 0,
        cmp_ratio=cmp_ratio if cmp_ratio is not None else 1,
        ori_mask_mode=tensor_input["ori_mask_mode"],
        cmp_mask_mode=cmp_mask_mode if cmp_mask_mode is not None else 3,
        **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
        **({"ori_win_right": ori_win_right} if ori_win_right is not None else {}),
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=npu.get("ori_kv") is not None,
        has_cmp_kv=npu.get("cmp_kv") is not None,
    )
    if npu.get("cmp_kv") is None:
        # Upstream generation can retain unused CMP descriptors in ORI-only
        # state. They do not describe an input cache and must not reach the API.
        for name in (
            "cmp_sparse_indices",
            "cmp_block_table",
            "cu_seqlens_cmp_kv",
            "seqused_cmp_kv",
            "cmp_residual_kv",
        ):
            npu[name] = None
        cmp_sparse_indices = None
        cmp_topk_length = None
    public_kwargs = dict(
        ori_kv=npu.get("ori_kv"),
        cmp_kv=npu.get("cmp_kv"),
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=npu.get("ori_block_table"),
        cmp_block_table=npu.get("cmp_block_table"),
        cu_seqlens_q=npu.get("cu_seqlens_q") if layout_q == "TND" else None,
        cu_seqlens_ori_kv=npu.get("cu_seqlens_ori_kv"),
        cu_seqlens_cmp_kv=npu.get("cu_seqlens_cmp_kv"),
        seqused_q=npu.get("seqused_q"),
        seqused_ori_kv=npu.get("seqused_ori_kv"),
        seqused_cmp_kv=npu.get("seqused_cmp_kv"),
        cmp_residual_kv=npu.get("cmp_residual_kv"),
        ori_topk_length=_public_topk(ori_topk_length, npu["q"]),
        cmp_topk_length=_public_topk(cmp_topk_length, npu["q"]),
        sinks=npu["sinks"],
        metadata=metadata,
        softmax_scale=tensor_input["softmax_scale"],
        cmp_ratio=cmp_ratio if cmp_ratio is not None else 1,
        ori_mask_mode=tensor_input["ori_mask_mode"],
        cmp_mask_mode=cmp_mask_mode if cmp_mask_mode is not None else 0,
        **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
        **({"ori_win_right": ori_win_right} if ori_win_right is not None else {}),
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=1,
        return_softmax_lse=params.get("return_softmax_lse") or False,
    )
    admission = SMLA.route_admission(npu["q"], **public_kwargs)
    print(f"public_route={admission['route']}; reasons={admission.get('reasons')}")
    assert admission["route"] == params["expected_route"], admission
    actual_calls = []
    sparse_kernel = SMLA.sparse_n64_kernel
    generic_kernel = SMLA.generic_sparse_flash_mla

    def run_sparse(*args, **kwargs):
        actual_calls.append("asc_sparse_n64_kernel")
        return sparse_kernel(*args, **kwargs)

    def run_generic(*args, **kwargs):
        actual_calls.append("generic")
        return generic_kernel(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SMLA, "sparse_n64_kernel", run_sparse)
        patch.setattr(SMLA, "generic_sparse_flash_mla", run_generic)
        output, lse = SMLA.sparse_flash_mla(npu["q"], **public_kwargs)
    torch.npu.synchronize()
    print(f"actual_public_route={actual_calls}; expected={params['expected_route']}")
    assert actual_calls == [params["expected_route"]], actual_calls
    if layout_kv == "PA_BBND":
        for operator in (
            SMLA.sparse_flash_mla,
            SMLA.generic_sparse_flash_mla,
            SMLA.sparse_n64_kernel,
        ):
            with pytest.raises(ValueError):
                operator(
                    npu["q"],
                    **{**public_kwargs, "layout_kv": "PA", "return_softmax_lse": False},
                )
        with pytest.raises(ValueError, match="PA_BBND"):
            SMLA_METADATA.sparse_flash_mla_metadata(
                num_heads_q=metadata_input["N1"], layout_kv="PA"
            )
    return output, lse


@lru_cache(maxsize=1)
def _load_operators():
    """Resolve this checkout's samples, never an external arena/operator tree."""
    import importlib

    samples = Path(__file__).resolve().parents[2] / "samples"
    for name in ("sparse_flash_mla", "sparse_flash_mla_metadata"):
        sys.path.insert(0, str(samples / name))
    return (
        importlib.import_module("sparse_flash_mla"),
        importlib.import_module("sparse_flash_mla_metadata"),
    )


def _public_topk(value, query):
    if value is None:
        return None
    return value.to(torch.int32).reshape(query.shape[:-2]).contiguous()


def _branch_case(
    name,
    mode,
    route,
    *,
    layout="TND",
    paged=False,
    q_lengths=(3, 2),
    kv_lengths=(256, 128),
    heads=64,
    ratio=4,
    ori_k=65,
    cmp_k=33,
    masks=(4, 3),
    window=(31, 0),
    pages=(64, 32),
    random_topk=True,
    lse=False,
    dtype=torch.bfloat16,
    q_seqused=True,
):
    """Compose small, multi-feature cases; never alter upstream golden math."""
    case = {key: value[0] for key, value in _PARAM_DEFAULTS.items()}
    batch = len(q_lengths)
    s1, s2 = max(q_lengths), max(kv_lengths)
    has_cmp = mode in ("HCA", "CSA", "ORI_CMP_SPARSE")
    has_ori_indices = mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
    has_cmp_indices = mode in ("CSA", "ORI_CMP_SPARSE")
    ratio = ratio if has_cmp else 1
    cmp_lengths = [length // ratio for length in kv_lengths] if has_cmp else None

    def prefix(lengths):
        return [0, *itertools.accumulate(lengths)]

    case.update(
        testcase_name=name,
        template_mode=mode,
        expected_route=route,
        layout_q=layout,
        layout_kv="PA_BBND" if paged else layout,
        B=batch,
        S1=s1,
        S2=s2,
        T1=sum(q_lengths),
        T2=sum(kv_lengths),
        T3=sum(cmp_lengths) if has_cmp else None,
        N1=heads,
        N2=1,
        D=512,
        q_type=dtype,
        ori_kv_type=dtype,
        cmp_kv_type=dtype if has_cmp else None,
        K1=ori_k if has_ori_indices else None,
        K=cmp_k if has_cmp_indices else None,
        cu_seqlens_q=prefix(q_lengths),
        cu_seqlens_ori_kv=prefix(kv_lengths) if layout == "TND" and not paged else None,
        cu_seqlens_cmp_kv=prefix(cmp_lengths)
        if has_cmp and layout == "TND" and not paged
        else None,
        seqused_q=list(q_lengths) if q_seqused else None,
        seqused_ori_kv=list(kv_lengths),
        seqused_cmp_kv=cmp_lengths,
        cmp_ratio=ratio if has_cmp else None,
        cmp_residual_kv=[length % ratio for length in kv_lengths] if has_cmp else None,
        block_size1=pages[0] if paged else None,
        block_size2=pages[1] if paged and has_cmp else None,
        ori_mask_mode=masks[0],
        cmp_mask_mode=masks[1] if has_cmp else 0,
        ori_win_left=window[0],
        ori_win_right=window[1],
        ori_kv_topk_mode=("random" if random_topk else "fullK")
        if has_ori_indices
        else "no",
        cmp_kv_topk_mode=("random" if random_topk else "fullK")
        if has_cmp_indices
        else "no",
        ori_sparse_indices_mode="full",
        cmp_sparse_indices_mode="full",
        q_datarange=[-2, 2],
        ori_kv_datarange=[-2, 2],
        cmp_kv_datarange=[-2, 2],
        softmax_scale=0.035 if name.startswith(("02_", "19_")) else 512**-0.5,
        return_softmax_lse=lse,
    )
    return case


# Twenty-two logical cases for NPU precision checks with CPU golden comparison.
# Each description is the coverage intent, not a claim of measured line coverage.
TEST_CASES = [
    # SWA: ragged sequence metadata, Q/head tails, two-sided/causal windows, pages.
    _branch_case(
        "01_swa_tnd_ragged_head33_window",
        "SWA",
        "generic",
        kv_lengths=(257, 129),
        heads=33,
        window=(31, 2),
    ),
    _branch_case(
        "02_swa_bsnd_padding_head96_scale",
        "SWA",
        "generic",
        layout="BSND",
        q_lengths=(2, 3),
        kv_lengths=(193, 257),
        heads=96,
        window=(63, 0),
    ),
    _branch_case(
        "03_swa_pa_page_tail_fp16",
        "SWA",
        "generic",
        paged=True,
        q_lengths=(2, 1),
        kv_lengths=(129, 65),
        heads=128,
        window=(15, 1),
        dtype=torch.float16,
    ),
    # HCA: dense CMP, ratio and length restoration, both layouts, paged storage.
    _branch_case("04_hca_tnd_ratio4_head65", "HCA", "generic", heads=65),
    _branch_case(
        "05_hca_bsnd_ratio2_noncausal",
        "HCA",
        "generic",
        layout="BSND",
        q_lengths=(2, 3),
        heads=32,
        ratio=2,
        masks=(0, 0),
        window=(-1, -1),
    ),
    _branch_case(
        "06_hca_pa_ratio8_head128",
        "HCA",
        "generic",
        paged=True,
        q_lengths=(1, 2),
        heads=128,
        ratio=8,
        pages=(32, 16),
        window=(63, 1),
    ),
    # CSA: sparse CMP tails/unequal lengths, mixed masks and paged Q layouts.
    _branch_case(
        "07_csa_tnd_ratio4_topk33_head96",
        "CSA",
        "generic",
        kv_lengths=(512, 256),
        heads=96,
    ),
    _branch_case(
        "08_csa_bsnd_ratio2_noncausal",
        "CSA",
        "asc_sparse_n64_kernel",
        layout="BSND",
        q_lengths=(2, 3),
        ratio=2,
        cmp_k=65,
        masks=(0, 0),
        window=(-1, -1),
    ),
    _branch_case(
        "09_csa_tnd_pa_ratio8_head33",
        "CSA",
        "asc_sparse_n64_kernel",
        paged=True,
        q_lengths=(2, 1),
        kv_lengths=(512, 256),
        heads=33,
        ratio=8,
        cmp_k=17,
    ),
    _branch_case(
        "10_csa_bsnd_pa_causal_head128",
        "CSA",
        "generic",
        layout="BSND",
        paged=True,
        q_lengths=(1, 2),
        heads=128,
        cmp_k=17,
        masks=(3, 3),
        window=(-1, -1),
    ),
    # ORI sparse: generic cases deliberately retain seqused/page/mask constraints.
    _branch_case(
        "11_ori_generic_tnd_head65_topk129",
        "ORI_SPARSE",
        "generic",
        kv_lengths=(193, 129),
        heads=65,
        ori_k=129,
        masks=(0, 0),
        window=(-1, -1),
    ),
    _branch_case(
        "12_ori_generic_bsnd_causal_head96",
        "ORI_SPARSE",
        "generic",
        layout="BSND",
        q_lengths=(2, 3),
        kv_lengths=(257, 193),
        heads=96,
        ori_k=129,
        masks=(3, 0),
        window=(-1, -1),
    ),
    _branch_case(
        "13_ori_generic_pa_window_head33",
        "ORI_SPARSE",
        "asc_sparse_n64_kernel",
        paged=True,
        q_lengths=(2, 1),
        kv_lengths=(193, 129),
        heads=33,
        window=(31, 2),
    ),
    _branch_case(
        "14_ori_hp_tnd_head33_topk129_lse",
        "ORI_SPARSE",
        "asc_sparse_n64_kernel",
        q_lengths=(5,),
        kv_lengths=(257,),
        heads=33,
        ori_k=129,
        masks=(0, 0),
        window=(-1, -1),
        lse=True,
    ),
    _branch_case(
        "15_ori_generic_bsnd_head64_window",
        "ORI_SPARSE",
        "asc_sparse_n64_kernel",
        layout="BSND",
        q_lengths=(3,),
        kv_lengths=(193,),
        heads=64,
        ori_k=128,
        random_topk=False,
        masks=(4, 0),
        window=(127, 0),
    ),
    # ORI+CMP sparse: dual topk, separate tails, ratios, generic and HP execution.
    _branch_case(
        "16_oricmp_generic_tnd_ratio4_head96",
        "ORI_CMP_SPARSE",
        "generic",
        kv_lengths=(512, 256),
        heads=96,
        ori_k=129,
        masks=(0, 0),
        window=(-1, -1),
    ),
    _branch_case(
        "17_oricmp_generic_bsnd_pa_ratio2",
        "ORI_CMP_SPARSE",
        "generic",
        layout="BSND",
        paged=True,
        q_lengths=(2, 3),
        heads=128,
        ratio=2,
        pages=(32, 16),
        window=(31, 1),
    ),
    _branch_case(
        "18_oricmp_hp_tnd_ratio4_head32_lse",
        "ORI_CMP_SPARSE",
        "asc_sparse_n64_kernel",
        q_lengths=(7,),
        kv_lengths=(512,),
        heads=32,
        ori_k=129,
        cmp_k=65,
        masks=(0, 0),
        window=(-1, -1),
        lse=True,
    ),
    _branch_case(
        "19_oricmp_hp_bsnd_ratio2_head63",
        "ORI_CMP_SPARSE",
        "asc_sparse_n64_kernel",
        layout="BSND",
        q_lengths=(5,),
        kv_lengths=(384,),
        heads=63,
        ratio=2,
        ori_k=193,
        cmp_k=129,
        random_topk=False,
        masks=(0, 0),
        window=(-1, -1),
    ),
    _branch_case(
        "20_oricmp_hp_tnd_ratio8_head64_lse",
        "ORI_CMP_SPARSE",
        "asc_sparse_n64_kernel",
        q_lengths=(9,),
        kv_lengths=(1024,),
        heads=64,
        ratio=8,
        ori_k=257,
        cmp_k=65,
        masks=(0, 0),
        window=(-1, -1),
        lse=True,
    ),
    _branch_case(
        "21_oricmp_hp_bsnd_b2_fp16_seqused_lse",
        "ORI_CMP_SPARSE",
        "asc_sparse_n64_kernel",
        layout="BSND",
        q_lengths=(2, 2),
        kv_lengths=(192, 128),
        heads=32,
        ratio=2,
        ori_k=65,
        cmp_k=33,
        masks=(0, 0),
        window=(-1, -1),
        lse=True,
        dtype=torch.float16,
    ),
    _branch_case(
        "22_oricmp_hp_tnd_b2_partial_seqused_lse",
        "ORI_CMP_SPARSE",
        "asc_sparse_n64_kernel",
        q_lengths=(3, 2),
        kv_lengths=(256, 128),
        heads=33,
        ratio=4,
        ori_k=65,
        cmp_k=33,
        masks=(0, 0),
        window=(-1, -1),
        lse=True,
        q_seqused=False,
    ),
]


# Explicit boundaries complement random lengths: empty, single-token, tile-1,
# tile, capacity, and ORI-only/CMP-only rows within one dual-cache launch.
TEST_CASES[10]["ori_topk_length"] = [0, 1, 127, 128, 129]
TEST_CASES[18]["ori_topk_length"] = [0, 1, 127, 128, 193]
TEST_CASES[18]["cmp_topk_length"] = [1, 0, 128, 129, 129]


def _combined_precision_cases():
    """Compose 58 additional cases; each row exercises several independent axes."""
    layouts = (("TND", False), ("BSND", False), ("TND", True), ("BSND", True))
    heads = (1, 18, 64, 65)
    mask_pairs = ((0, 0), (3, 0), (0, 3), (3, 3), (4, 0), (4, 3))
    ratios = (1, 2, 3, 4, 8, 16, 64, 128)
    windows = ((0, 0), (1, 0), (15, 1), (-1, 2), (31, -1), (-1, -1))
    pages = ((1, 1), (3, 5), (7, 16), (512, 128))
    widths = (1, 16, 17, 127, 128, 129, 257)
    cases = []

    def compose(mode, layout, paged, head, profile, serial):
        if profile == 0:
            q_lengths, kv_lengths = (1,), (129,)
        elif profile == 1 and layout == "TND":
            q_lengths, kv_lengths = (2, 0, 3), (257, 129, 193)
        elif profile == 1:
            q_lengths, kv_lengths = (1, 3), (257, 129)
        elif profile == 2:
            q_lengths, kv_lengths = (3, 2), (259, 131)
        else:
            q_lengths, kv_lengths = (6,), (1033,)
        masks = (0, 0) if profile == 3 else mask_pairs[serial % len(mask_pairs)]
        if mode in ("SWA", "ORI_SPARSE"):
            masks = (masks[0], 0)
        window = windows[serial % len(windows)] if masks[0] == 4 else (-1, -1)
        ratio = ratios[(serial // 4 + profile * 3) % len(ratios)]
        route = (
            "asc_sparse_n64_kernel"
            if head <= 64 and mode in ("ORI_SPARSE", "ORI_CMP_SPARSE", "CSA")
            else "generic"
        )
        name = (
            f"{23 + serial:02d}_combined_{mode.lower()}_"
            f"{layout.lower()}{'_pa' if paged else ''}_n{head}_p{profile}"
        )
        case = _branch_case(
            name,
            mode,
            route,
            layout=layout,
            paged=paged,
            q_lengths=q_lengths,
            kv_lengths=kv_lengths,
            heads=head,
            ratio=ratio,
            ori_k=257 if profile == 3 else widths[serial % len(widths)],
            cmp_k=257 if profile == 3 else widths[(serial + 2) % len(widths)],
            masks=masks,
            window=window,
            pages=pages[(serial // 4 + profile) % 4],
            random_topk=serial % 3 != 0,
            lse=route == "asc_sparse_n64_kernel"
            and (serial // 4 + profile // 2) % 2 == 0,
            dtype=torch.float16 if (serial // 4 + profile) % 2 else torch.bfloat16,
            q_seqused=not (layout == "TND" and profile == 2),
        )
        case["softmax_scale"] = (0.0, 0.035, 512**-0.5, 0.5)[
            (serial // 4 + profile) % 4
        ]
        if route == "asc_sparse_n64_kernel" and profile != 3 and serial % 3 == 0:
            case["use_sinks"] = False
        if profile == 3:
            # All-zero, one-token, tile-1, tile and tile+1 lengths share a launch.
            for side, key in (("ori", "K1"), ("cmp", "K")):
                if case[key] is None:
                    continue
                lengths = []
                capacities = (
                    case["seqused_ori_kv"] if side == "ori" else case["seqused_cmp_kv"]
                )
                for batch, capacity in enumerate(capacities):
                    count = q_lengths[batch] if layout == "TND" else case["S1"]
                    for row in range(count):
                        valid = row < q_lengths[batch]
                        bound = (0, 1, 127, 128, 129, case[key])[row % 6]
                        lengths.append(min(bound, capacity, case[key]) if valid else 0)
                case[f"{side}_topk_length"] = lengths
        case["coverage_profile"] = (
            "single_query",
            "ragged_or_padding",
            "ratio_residual_and_lengths",
            "empty_topk_and_tile_boundaries",
        )[profile]
        return case

    # 3 sparse modes x 4 layout pairs x 4 head profiles = 48 cases.
    for mode_index, mode in enumerate(("ORI_SPARSE", "ORI_CMP_SPARSE", "CSA")):
        for layout_index, (layout, paged) in enumerate(layouts):
            for head_index, head in enumerate(heads):
                profile = (head_index + layout_index + mode_index) % 4
                cases.append(compose(mode, layout, paged, head, profile, len(cases)))
    # Dense modes already use the generic route; five mixed profiles each suffice.
    for mode in ("SWA", "HCA"):
        for profile, head in enumerate((1, 18, 64, 65, 128)):
            layout, paged = layouts[profile % len(layouts)]
            cases.append(compose(mode, layout, paged, head, profile % 4, len(cases)))
    return cases


TEST_CASES.extend(_combined_precision_cases())


def _selected_cases():
    selection = os.environ.get("SMLA_CASES", "all").strip()
    if not selection or selection == "all":
        return TEST_CASES
    catalogue = {case["testcase_name"]: case for case in TEST_CASES}
    return [catalogue[name.strip()] for name in selection.split(",")]


PARAM_CASES = _selected_cases()


def _make_test_data(case_params):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    return gen_data(_materialize_case(case_params), prepare_device_storage=False)


def _expected_lse_shape(params, tensor_input):
    """Return the public high-performance LSE shape for either layout."""
    query = tensor_input["q"]
    if params["layout_q"] == "TND":
        return (1, query.shape[0], query.shape[1])
    return (query.shape[0], 1, query.shape[1], query.shape[2])


@pytest.mark.npu
@pytest.mark.parametrize(
    "case_params", PARAM_CASES, ids=lambda case: case["testcase_name"]
)
def test_sparse_flash_mla(case_params):
    """Public routing + metadata, full output/LSE comparison with upstream rules.

    BF16: rtol=.0078125/atol=.0001; FP32: rtol=.005/atol=.000025.
    Both use the upstream 99.5% pass-rate and max-error criterion unchanged.
    """
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    data = _make_test_data(case_params)
    output, lse = _run_npu(data)
    expected = data["cpu_output"]
    assert output.shape == expected.shape
    assert output.dtype == data["input"]["q"].dtype
    result, percent = check_result(expected, output.cpu())
    assert result == "Pass", f"output precision: {percent}%"
    if case_params["return_softmax_lse"]:
        expected_lse = data["softmax_lse"]
        expected_lse_shape = _expected_lse_shape(case_params, data["input"])
        assert lse.shape == expected_lse_shape
        assert lse.is_contiguous()
        assert lse.numel() == expected_lse.numel() and lse.dtype == torch.float32
        result, percent = check_result(
            expected_lse, lse.cpu().reshape(expected_lse.shape)
        )
        assert result == "Pass", f"LSE precision: {percent}%"
    else:
        assert lse is None or lse.numel() == 0


def _sinks_test_inputs(dtype, layout="TND", random_inputs=False):
    rows, heads, dim, kv_rows = 4, 64, 512, 129
    q = torch.zeros(rows, heads, dim, dtype=dtype)
    kv = torch.ones(kv_rows, 1, dim, dtype=dtype)
    indices = torch.arange(kv_rows, dtype=torch.int32).view(1, 1, -1).repeat(rows, 1, 1)
    lengths = torch.tensor([kv_rows, 0, 1, kv_rows], dtype=torch.int32).view(rows, 1)
    cu_q = torch.tensor([0, rows], dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], dtype=torch.int32)
    used_q = torch.tensor([3], dtype=torch.int32)
    init = dict(
        layout_q="TND",
        layout_kv="TND",
        q_type=dtype,
        ori_kv_type=dtype,
        cmp_kv_type=dtype,
        B=1,
        S1=rows,
        S2=kv_rows,
        T1=rows,
        N1=heads,
        N2=1,
        D=dim,
        K1=kv_rows,
        K=kv_rows,
        block_num1=None,
        block_num2=None,
        block_size1=None,
        block_size2=None,
        cu_seqlens_q=cu_q,
        seqused_ori_kv=torch.tensor([kv_rows], dtype=torch.int32),
        seqused_cmp_kv=torch.tensor([kv_rows], dtype=torch.int32),
        cmp_residual_kv=None,
        softmax_scale=0.125,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=-1,
        ori_win_right=-1,
        ori_topk_length=lengths,
        cmp_topk_length=lengths,
    )
    model = GeneralizedSFA(**init)
    forward = dict(
        q=q,
        ori_k=kv,
        ori_sparse_indices=indices,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_kv,
        cu_seqlens_cmp_kv=cu_kv,
        seqused_ori_kv=init["seqused_ori_kv"],
        seqused_cmp_kv=init["seqused_cmp_kv"],
        cmp_residual_kv=None,
        template_idx=4,
        cmp_k=kv,
        cmp_sparse_indices=indices,
        seqused_q=used_q,
        ori_topk_length=lengths,
        cmp_topk_length=lengths,
        return_softmax_lse=True,
    )
    if layout == "BSND":
        model.layout_q = model.layout_kv = layout
        for name in (
            "q",
            "ori_k",
            "cmp_k",
            "ori_sparse_indices",
            "cmp_sparse_indices",
            "ori_topk_length",
            "cmp_topk_length",
        ):
            forward[name] = forward[name].unsqueeze(0)
        for name in ("cu_seqlens_q", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv"):
            forward[name] = None
    else:
        assert layout == "TND"
    if random_inputs:
        generator = torch.Generator().manual_seed(674)
        for name in ("q", "ori_k"):
            value = forward[name]
            value.copy_((torch.randn(value.shape, generator=generator) * 0.2).to(dtype))
    return model, forward


def _sinks_npu_inputs(args):
    layout = "BSND" if args["q"].ndim == 4 else "TND"
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    print("NPU_STAGE set_device_start", flush=True)
    torch.npu.set_device(0)
    print("NPU_STAGE set_device_returned", flush=True)
    torch.set_num_threads(4)
    public, metadata_module = _load_operators()
    device = {
        name: _to_npu(value)
        for name, value in args.items()
        if torch.is_tensor(value) or value is None
    }
    for name in ("ori_topk_length", "cmp_topk_length"):
        device[name] = device[name].reshape(-1)
    md = metadata_module.sparse_flash_mla_metadata(
        cu_seqlens_q=device["cu_seqlens_q"],
        cu_seqlens_ori_kv=device["cu_seqlens_ori_kv"],
        cu_seqlens_cmp_kv=device["cu_seqlens_cmp_kv"],
        seqused_q=device["seqused_q"],
        seqused_ori_kv=device["seqused_ori_kv"],
        seqused_cmp_kv=device["seqused_cmp_kv"],
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        batch_size=1,
        max_seqlen_q=4,
        max_seqlen_ori_kv=129,
        max_seqlen_cmp_kv=129,
        ori_topk=129,
        cmp_topk=129,
        cmp_ratio=1,
        ori_topk_length=device["ori_topk_length"],
        cmp_topk_length=device["cmp_topk_length"],
        ori_mask_mode=0,
        cmp_mask_mode=0,
        layout_q=layout,
        layout_kv=layout,
        has_ori_kv=True,
        has_cmp_kv=True,
    )
    kw = dict(
        ori_kv=device["ori_k"],
        cmp_kv=device["cmp_k"],
        ori_sparse_indices=device["ori_sparse_indices"],
        cmp_sparse_indices=device["cmp_sparse_indices"],
        cu_seqlens_q=device["cu_seqlens_q"],
        cu_seqlens_ori_kv=device["cu_seqlens_ori_kv"],
        cu_seqlens_cmp_kv=device["cu_seqlens_cmp_kv"],
        seqused_q=device["seqused_q"],
        ori_topk_length=device["ori_topk_length"],
        cmp_topk_length=device["cmp_topk_length"],
        seqused_ori_kv=device["seqused_ori_kv"],
        seqused_cmp_kv=device["seqused_cmp_kv"],
        metadata=md,
        softmax_scale=0.125,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        layout_q=layout,
        layout_kv=layout,
        return_softmax_lse=True,
    )
    return public, device, kw


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_n64_sinks_match_golden(dtype, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "RUN_MODE", 1)
    model, args = _sinks_test_inputs(dtype)
    public, device, kw = _sinks_npu_inputs(args)
    f = torch.finfo(torch.float32)
    values = [
        0.0,
        -0.0,
        -math.inf,
        math.inf,
        math.nan,
        -1e4,
        1e4,
        f.min,
        f.max,
        1.0,
        -1.0,
        f.tiny,
    ]
    mixed = torch.tensor(
        [values[i % len(values)] for i in range(64)], dtype=torch.float32
    )
    results = {}
    for name, sink in [
        ("mixed", mixed),
        ("none", None),
        ("neg_inf", torch.full((64,), -math.inf)),
    ]:
        expected, expected_lse = model.forward(**args, sinks=sink)
        print("NPU_STAGE run_start", name, flush=True)
        out, lse = public.sparse_flash_mla(
            device["q"], sinks=None if sink is None else sink.npu(), **kw
        )
        print("NPU_STAGE run_returned", name, flush=True)
        torch.npu.synchronize()
        print("NPU_STAGE sync_returned", name, flush=True)
        assert (
            public.route_admission(device["q"], **kw)["route"]
            == "asc_sparse_n64_kernel"
        )
        out, lse = out.cpu(), lse.cpu()
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.float(),
            expected.float(),
            atol=8e-4 if dtype == torch.bfloat16 else 2e-4,
            rtol=3.01 / 128 if dtype == torch.bfloat16 else 0.005,
            equal_nan=True,
        )
        torch.testing.assert_close(
            lse, expected_lse, atol=1e-5, rtol=3.1e-5, equal_nan=True
        )
        finite_lse = torch.isfinite(expected_lse)
        finite_out = torch.isfinite(expected)
        print(
            "SINKS_CHECK",
            json.dumps(
                dict(
                    dtype=str(dtype),
                    profile=name,
                    lse_max_abs=float(
                        (lse[finite_lse] - expected_lse[finite_lse]).abs().max()
                    ),
                    out_max_abs=float(
                        (out[finite_out].float() - expected[finite_out].float())
                        .abs()
                        .max()
                    ),
                )
            ),
            flush=True,
        )
        results[name] = out, lse
    for actual, expected in zip(results["none"], results["neg_inf"]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    positive_inf_heads = torch.isposinf(mixed)
    assert torch.isnan(results["mixed"][0][[0, 2]][:, positive_inf_heads]).all()
    assert torch.isnan(results["mixed"][1][0, [0, 2]][:, positive_inf_heads]).all()
    assert torch.count_nonzero(results["mixed"][0][[1, 3]]) == 0
    assert torch.count_nonzero(results["mixed"][1][0, [1, 3]]) == 0
    out, disabled_lse = public.sparse_flash_mla(
        device["q"], sinks=mixed.npu(), **{**kw, "return_softmax_lse": False}
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), results["mixed"][0], atol=0, rtol=0, equal_nan=True
    )
    assert disabled_lse.shape == (0,)
    # Finite, nonzero scores retain the existing output/LSE arithmetic.
    generator = torch.Generator().manual_seed(670)
    args["q"].copy_((torch.randn(args["q"].shape, generator=generator) * 0.2).to(dtype))
    args["ori_k"].copy_(
        (torch.randn(args["ori_k"].shape, generator=generator) * 0.2).to(dtype)
    )
    device["q"].copy_(args["q"])
    device["ori_k"].copy_(args["ori_k"])
    device["cmp_k"].copy_(args["cmp_k"])
    sink = torch.linspace(-3.0, 3.0, 64)
    expected, expected_lse = model.forward(**args, sinks=sink)
    out, lse = public.sparse_flash_mla(device["q"], sinks=sink.npu(), **kw)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu().float(),
        expected.float(),
        atol=8e-4 if dtype == torch.bfloat16 else 2e-4,
        rtol=3.01 / 128 if dtype == torch.bfloat16 else 0.005,
        equal_nan=True,
    )
    torch.testing.assert_close(
        lse.cpu(), expected_lse, atol=1e-5, rtol=3.1e-5, equal_nan=True
    )
    print(
        "SINKS_CHECK",
        json.dumps(
            dict(
                dtype=str(dtype),
                profile="finite_nonzero",
                lse_max_abs=float((lse.cpu() - expected_lse).abs().max()),
                out_max_abs=float((out.cpu().float() - expected.float()).abs().max()),
            )
        ),
        flush=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout,profile",
    [
        ("TND", "none"),
        ("BSND", "none"),
        ("TND", "neg_inf"),
        ("TND", "zero"),
        ("BSND", "pos_inf"),
        ("BSND", "nan"),
        ("BSND", "mixed"),
    ],
)
def test_n64_sinks_precision(dtype, layout, profile, monkeypatch):
    """Run the NPU kernel and compare output/LSE with the CPU reference."""
    monkeypatch.setattr(sys.modules[__name__], "RUN_MODE", 1)
    model, args = _sinks_test_inputs(dtype, layout=layout, random_inputs=True)
    public, device, kw = _sinks_npu_inputs(args)
    f = torch.finfo(torch.float32)
    if profile == "none":
        sink = None
    elif profile == "mixed":
        values = [
            -math.inf,
            0.0,
            -0.0,
            math.inf,
            math.nan,
            f.min,
            f.max,
            -1e4,
            1e4,
            1.0,
            -1.0,
            f.tiny,
        ]
        sink = torch.tensor(
            [values[h % len(values)] for h in range(64)], dtype=torch.float32
        )
    else:
        value = {
            "neg_inf": -math.inf,
            "zero": 0.0,
            "pos_inf": math.inf,
            "nan": math.nan,
        }[profile]
        sink = torch.full((64,), value, dtype=torch.float32)
    expected, expected_lse = model.forward(**args, sinks=sink)
    print("NPU_STAGE run_start", layout, profile, flush=True)
    out, lse = public.sparse_flash_mla(
        device["q"], sinks=None if sink is None else sink.npu(), **kw
    )
    print("NPU_STAGE run_returned", layout, profile, flush=True)
    torch.npu.synchronize()
    print("NPU_STAGE sync_returned", layout, profile, flush=True)
    assert public.route_admission(device["q"], **kw)["route"] == "asc_sparse_n64_kernel"
    assert out.dtype == dtype and lse.dtype == torch.float32 and lse.is_contiguous()
    actual, actual_lse = out.cpu(), lse.cpu()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        equal_nan=True,
        atol=8e-4 if dtype == torch.bfloat16 else 2e-4,
        rtol=3.01 / 128 if dtype == torch.bfloat16 else 0.005,
    )
    torch.testing.assert_close(
        actual_lse, expected_lse, atol=1e-5, rtol=3.1e-5, equal_nan=True
    )
    flat_out, flat_lse = actual.reshape(4, 64, 512), actual_lse.reshape(4, 64)
    assert torch.count_nonzero(flat_out[[1, 3]]) == 0
    assert torch.count_nonzero(flat_lse[[1, 3]]) == 0
    if sink is None:
        explicit_out, explicit_lse = public.sparse_flash_mla(
            device["q"],
            sinks=torch.full((64,), -math.inf, device=device["q"].device),
            **kw,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(explicit_out.cpu(), actual, atol=0, rtol=0)
        torch.testing.assert_close(explicit_lse.cpu(), actual_lse, atol=0, rtol=0)
    else:
        nan_heads = torch.isposinf(sink) | torch.isnan(sink)
        if nan_heads.any():
            assert torch.isnan(flat_out[[0, 2]][:, nan_heads]).all()
            assert torch.isnan(flat_lse[[0, 2]][:, nan_heads]).all()
    finite_out, finite_lse = torch.isfinite(expected), torch.isfinite(expected_lse)
    print(
        "SINKS_CHECK",
        json.dumps(
            dict(
                dtype=str(dtype),
                layout=layout,
                profile=profile,
                lse_max_abs=float(
                    (actual_lse[finite_lse] - expected_lse[finite_lse]).abs().max()
                ),
                out_max_abs=float(
                    (actual[finite_out].float() - expected[finite_out].float())
                    .abs()
                    .max()
                ),
            )
        ),
        flush=True,
    )
