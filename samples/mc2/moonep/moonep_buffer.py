# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonepBuffer：moonep 五算子统一封装（构造即拥有全部通信资源 + ElasticBuffer 收编）。

把 test/moonep/test_moonep_mem.py run() 与 test/moonep/test_moonep_ep.py 中由调用方
手做的全部通信基建收进单一对象：堆布局计算（hccl_buffer_size 注入）、HCCL 子组
创建、ctx 定型（window_slots）、权重/梯度槽视图、home 行切片与 view(epn, W) 样板，
以及 ElasticBuffer 的 nmtr 推导与 dispatch/combine 参数接线（vtopk.view(S,K)、
num_experts=2E、scores.float()）。接口形态对标 GPU MoonEP 官方 moonep.Buffer
（plan 句柄贯穿五算子、destroy 显式释放）。

构造级通信域分离保证：EB 建在 WORLD 组、
moonep ctx 建在独建子组——两个 HCCL 通信器由构造即分离，不依赖调用方自觉；
check_no_shared_elastic_buffer 只作兜底。

Usage（五算子链端到端，FFN 侧全 NPU 算子、零 host 同步——形态 A：bf16
三矩阵，缺省 slot 数据流）:
    import torch_npu
    buf = MoonepBuffer(8192, 3584, 3072, 16, 224,  # (S, H, Hp, K, E)——H 亦是专家 FFN 输入维
                       moonep_group=mp_group, ep_group=ep_group)  # 两个独立子组
    # 构造即保证通信域分离：EB@WORLD + moonep ctx@独建子组；
    # nmtr = S（EB per-rank 容量，发送侧 per-peer 上界）由构造器定型
    epn = buf.epn
    # —— 权重矩阵（常规 NPU 参数张量）：w1/w3 [epn, Hp, H]（gate/up）、
    #    w2 [epn, H, Hp]（down）；交织形态可融合 W13 = cat(w1, w3, dim=?) 按
    #    每专家行 [gate; up] 拼成 [epn, 2W] 传 2 组（一次 GMM 免消费端 concat）
    w1 = torch.empty(epn, 3072, 3584, dtype=torch.bfloat16, device="npu")  # 如 nn.Parameter
    w3, w2 = ..., ...

    for step in ...:
        plan = buf.plan(tpe, topk)                  # MoonepPlan（可解包五件套）
        # 首步显式传 local（视图按各组行宽创建并缓存；此后参数更新直接写
        # weights[g][:epn]，再传 home 行切片调 prefetch 即零拷贝直推）；
        # N-D 直入免调用方展平
        weights, scales = buf.prefetch_weight(plan, local_weights=(w1, w3, w2))
        wv1 = weights[0].view(2 * epn, 3072, 3584)   # home+副本全量行
        wv2 = weights[2].view(2 * epn, 3584, 3072)
        recv, recv_w, handle = buf.dispatch(plan, x, topk_weights=scores, do_cpu_sync=False)
        #   ↑ do_cpu_sync=False 免 host 同步：recv 按 worst-case 容量分配，
        #     有效行数恒 S*K（plan 均衡不变量），device 侧切片即可
        # 零拷贝（次步起）：传 home 行切片（data_ptr 同址 ⇒ sync 跳过）
        buf.prefetch_weight(plan, local_weights=[w[:epn] for w in weights])
        n, cap = S * K, recv.shape[0]               # 两值 host 侧可得（零同步）
        # —— FFN 前向：分组 GMM + SwiGLu 全 NPU 算子（cu_seqlens 保持 device）——
        #   分组边界经 offset 直消费 device int32，全程无 .cpu() 同步
        gate = torch_npu.npu_grouped_matmul(recv[:n], wv1, offset=plan.cu_seqlens)
        ...
        out, out_w = buf.combine(combine_in, handle, topk_weights=recv_w)
        # —— 反向：dispatch 散梯度（plan 复用免重规划）+ combine 裸和 ——
        g_recv, _, handle_b = buf.dispatch(plan, g_out, topk_weights=None)
        ...
        # —— FFN 反向：分组 GEMM 反传产出 full-dW（行数 2*epn，张量由 GMM
        #     反向分配：前 epn 行 home 梯度种子、后 epn 行副本梯度）直入 ——
        #     副本行入槽由 reduce_grad 内部完成；返回归并结果 home 视图
        dw1 = torch._grouped_mm(g_recv[:n].mT, gate)
        dws = buf.reduce_grad(plan, local_grads=[dw1, dw3, dw2])
        # dws[g] 即 [epn, Hp, H] 全量 dW → optimizer.step()
    buf.destroy()                                   # 幂等；缺省不销毁子组/EB（池化保活）

形态 B（交织 W13 + down，2 组——PR #378）::

        weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))
        # weights = ([2*epn, 2W], [2*epn, W])——GMM 一次算完 gate+up，
        # 免消费端 concat；反向仍一次调用：
        dws = buf.reduce_grad(plan, local_grads=[dw13, dw2])
        # dws = ([epn, 2W], [epn, W])；优化器持融合参数直用，需分开时
        # dw13_home[:, :W] / dw13_home[:, W:] 零拷贝切

形态 C（mxfp8 + ue8m0 scale——R3 完整形态）::

        buf = MoonepBuffer(S, H, Hp, K, E,
                           moonep_group=mp_group, ep_group=ep_group,
                           weight_dtype="e4m3fn", grad_dtype="fp32")
        weights, scales = buf.prefetch_weight(plan, local_weights=(w13_fp8, w2_fp8),
                                       local_scales=(s13, s2))
        # scales = ([2*epn, Sg13], [2*epn, Sg2])——量化 GMM 按
        # (weights, scales) 成对消费；scale 无 reduce（ue8m0
        # 无梯度）；Sg 须为 32B 倍数（E3，不足 pad 到 ceil(Sg/32)*32）

其他:
    - 多槽流水（slots=2）：视图随 slot 轮转（权重+scale 同槽切换；梯度
      槽区单窗口、不随 slots 倍增——反向按层串行）。
    - async_finish=True：prefetch/reduce 跳过末尾屏障的 host 同步，返回发起流
      torch.npu.Event（侧流并发形态）；调用方 wait_event 汇合，跨 rank 槽位/
      staging 复用安全由调用方在合适位置（如下一轮迭代的 barrier）负责。
    - 容量对拍：heap_required_bytes <= heap_bytes（HCCL 2× 分配机制下实际
      可用 ≈ 2 × heap_config_mb MiB）。
    - 依赖 cann_ops_transformer whl（ElasticBuffer）：import 失败秒级报错并给
      安装指引。
"""

from __future__ import annotations

import os
import warnings
from typing import NamedTuple

import torch
import torch.distributed as dist

import moonep_plan
import prefetch_weight
import reduce_grad
from moonep_context import (
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    SCALE_ALIGN_BYTES,
    check_heap_capacity,
    create_moonep_context,
    hccl_buffer_size,
    pick_weight_chunk,
    reduce_region_end,
    sym_barrier,
)

try:
    from cann_ops_transformer.ops import ElasticBuffer
except ImportError as _exc:
    raise ImportError(
        "MoonepBuffer 需要 cann_ops_transformer"
        "（ElasticBuffer dispatch/combine）。安装：pip install <cann_ops_transformer whl>"
        "（A5 环境见 venvs/mooep）。"
    ) from _exc

# reduce VL=64 尾块约束（moonep_context VL 同源）：W 非 64 倍数时尾块越界 UB
_W_ALIGN = 64
_UE8M0_BLOCK = 32  # mxfp8 block scale 粒度（每 32 权重元素 1 个 ue8m0 字节）
# UB 分块 fail-fast 试算界（fp32 更严口径——bf16 更宽，取严者构造期报错）
_CHUNK_TRIAL_UB = 16384
_HCCL_2X_MIB = 2 * 1024**2  # HCCL 2× 分配机制的对账单位（2 MiB）


def _normalize_grad_dtype(grad_dtype):
    """grad_dtype 归一化："bf16"/"fp32"（或 torch.dtype）→ torch dtype。

    构造期声明定型（缺省字面 bf16——显存优先口径；
    fp32 显式声明，master-weight 精度优先）：定型值绑定两处——①自建组
    路径的堆容量预留口径（梯度槽区 fp32 = bf16 的 2×，建组时刻堆已定死）；
    ② local_grads 的 dtype 校验基（prefetch/reduce 均无 dtype 参数——
    构造已传，不在调用点重复）。
    """
    if isinstance(grad_dtype, str):
        if grad_dtype == "bf16":
            return torch.bfloat16
        if grad_dtype == "fp32":
            return torch.float32
        raise TypeError(
            f"grad_dtype 仅支持 'bf16'/'fp32'（或 torch.bfloat16/torch.float32），"
            f"got {grad_dtype!r}"
        )
    if grad_dtype in (torch.bfloat16, torch.float32):
        return grad_dtype
    raise TypeError(
        f"grad_dtype 仅支持 'bf16'/'fp32'（或 torch.bfloat16/torch.float32），"
        f"got {grad_dtype!r}"
    )


def _normalize_weight_dtype(weight_dtype):
    """weight_dtype 归一化："bf16"/"e4m3fn"/"e5m2"（或 torch.dtype）→ torch dtype。

    构造期容量口径 + 首次 prefetch 的 dtype 校验基。fp8
    （e4m3fn/e5m2）= mxfp8 数据面 ⟹ 一并推断 scale 区预留与"首调必传
    local_scales"硬契约（双向：scales ⟺ fp8——quant 参数已并入 dtype
    语义，无半状态、无浪费预留；无 block scale 的 int8 形态不在范围）。
    dtype 终态：uint8 退役（mxfp8 请显式 e4m3fn/e5m2）。
    """
    if isinstance(weight_dtype, str):
        if weight_dtype == "bf16":
            return torch.bfloat16
        if weight_dtype == "e4m3fn":
            return torch.float8_e4m3fn
        if weight_dtype == "e5m2":
            return torch.float8_e5m2
        raise TypeError(
            "weight_dtype 仅支持 'bf16'/'e4m3fn'/'e5m2'（或 torch 对应"
            " dtype；uint8 已退役——mxfp8 用 e4m3fn/e5m2），"
            f"got {weight_dtype!r}"
        )
    if weight_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        return weight_dtype
    raise TypeError(
        "weight_dtype 仅支持 'bf16'/'e4m3fn'/'e5m2'（或 torch 对应"
        f" dtype；uint8 已退役——mxfp8 用 e4m3fn/e5m2），got {weight_dtype!r}"
    )


def _scale_upper_bytes(w_elems):
    """单组 scale 行宽上界（字节）：ceil(W/32) 个 ue8m0 block scale、32B 对齐。

    构造期容量预留口径（堆按 3 组上界预留；实际 Sg 由首调 local_scales
    定型且 ≤ 本上界——组宽 ≤ W ⇒ Sg ≤ W/32 ≤ 本上界）。
    """
    raw = (int(w_elems) + _UE8M0_BLOCK - 1) // _UE8M0_BLOCK
    return (raw + SCALE_ALIGN_BYTES - 1) // SCALE_ALIGN_BYTES * SCALE_ALIGN_BYTES


class MoonepPlan(NamedTuple):
    """plan 五件套句柄（moonep_plan 输出的纯数据包装，零同步契约保持）。

    NamedTuple 选型：元组解包兼容（``dst_n, cu_seqlens, experts_to_copy,
    src_n, vtopk = plan`` 仍成立，迁移面最小）+ 属性访问 + 零运行时开销；
    不添加任何同步类方法（plan 返回路径零同步，溢出检查延迟到
    prefetch/reduce 尾部——既有模块行为透传）。语义对标 GPU MoonEPCommPlan。

    Fields:
        dst_n: [N] int32 零占位（vtopk-only 快路径不计算）。
        cu_seqlens: [2*epn] int32 分组偏移（GMM 分段边界）。
        experts_to_copy: [R, epn] int32 副本清单（-1 填充；每行至多一个
            远程 home rank），prefetch/reduce 共用。
        src_n: [N] int32 零占位（vtopk-only 快路径不计算）。
        vtopk: [N] int32 虚拟专家 id（扩展空间 2E，供 moe_ep dispatch 路由）。
    """

    dst_n: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    src_n: torch.Tensor
    vtopk: torch.Tensor


class MoonepBuffer:
    """moonep 五算子的统一封装宿主（构造即拥有全部通信资源）。

    构造内化：对称堆大小决策（业务形状 → HEAP_BASE +
    reduce_region_end + LAYOUT_TAIL_BYTES → hccl_buffer_size 注入，HCCL 2×
    分配机制）→ HCCL 子组（use_local_synchronization，全 world rank）→
    MoonepContext（plan 区容量校验）→ window_slots/weight_dtype/grad_dtype
    定型（模块侧 E1/E5 校验基）。运行期区域寻址权保留在既有模块公式内
    （prefetch_weight_off/reduce_stg_off/grad_slot_off，变宽组经 widths
    前缀和、scale 组随 slot 窗口前缀和）；本类只透传算子并缓存其返回
    视图（(weights, scales) 二元组），不自绘布局。

    构造内化（EB 增量）：nmtr 推导（S——EB 发送侧 per-peer 上界）+
    ElasticBuffer@WORLD。通信域分离由构造保证：EB 与 ctx 分属两个 HCCL
    通信器（mem 实证组合），check_no_shared_elastic_buffer 的 group_name
    比对天然通过——通信域分离由构造不变量保证。

    与 GPU moonep.Buffer 的有意差异：
    - 类名不用裸名 Buffer（避开 cannbotdsl.buffer.Buffer——DSL 的 UB/GM Buffer）；
    - 权重视图经 prefetch_weight 以 (weights, scales) 裸二元组返回
      （2/3 组自适应——含交织 2 组与 mxfp8+scale 形态），views(slot) 为
      缓存访问器、weight_view(g, slot) 为解包糖；梯度槽视图内部化
      （reduce 数据流内步），不对外暴露（GPU 侧 hidden_nvsh_buffer_view
      为单视图 @property）；
    - dispatch 返回 3 元组（recv_x, recv_w, handle），EB 的 meta 元组位
      丢弃（现有全部调用均弃置，YAGNI；如后续需要再扩展返回形态）；
    - destroy 缺省不释放子组/EB（池化保活到进程退出），与 GPU 的确定性
      VMM 释放形态不同——根因是 EB Destroy 平台行为（~120s/个），非设计偏好。

    生命周期：视图/plan 张量的使用须先于 destroy()；destroy 幂等。
    """

    @staticmethod
    def get_moon_ep_ccl_buffer_size(
        S,
        H,
        Hp,
        K,
        E,
        *,
        R=None,
        slots=1,
        weight_dtype="bf16",
        grad_dtype="bf16",
    ) -> int:
        """计算 moonep 对称堆所需的 hccl_buffer_size 配置值（MB）。

        在创建 moonep_group 前调用——返回值直接注入
        ``ProcessGroupHCCL.Options().hccl_config["hccl_buffer_size"]``。

        Args:
            S/H/Hp/K/E: 与 MoonepBuffer 构造同名参数。
            R: 组内 rank 数（缺省 dist.get_world_size()）。
            slots/weight_dtype/grad_dtype: 与 MoonepBuffer 构造同名参数。

        Returns:
            int: hccl_buffer_size 配置值（MB）。
        """
        import torch.distributed as dist

        if R is None:
            R = dist.get_world_size()
        w = int(H) * int(Hp)
        epn = int(E) // int(R)
        gdtype = _normalize_grad_dtype(grad_dtype)
        wdtype = _normalize_weight_dtype(weight_dtype)
        grad_elem = 2 if gdtype == torch.bfloat16 else 4
        weight_elem = 2 if wdtype == torch.bfloat16 else 1
        scale_upper = _scale_upper_bytes(w)
        scale_widths = (
            (scale_upper,) * 3 if prefetch_weight._is_fp8_weight(wdtype) else None
        )
        heap_need = (
            HEAP_BASE
            + reduce_region_end(
                int(R),
                epn,
                w,
                3,
                int(slots),
                grad_bytes_per_elem=grad_elem,
                scale_widths=scale_widths,
                weight_bytes=weight_elem,
            )
            + LAYOUT_TAIL_BYTES
        )
        return int(heap_need // _HCCL_2X_MIB) + 2

    @staticmethod
    def get_moe_ep_ccl_buffer_size(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        topk: int,
    ) -> int:
        """EB（dispatch/combine）通信域所需 buffer 大小（MB）。

        透传 ``ElasticBuffer.get_moe_ep_ccl_buffer_size``——EB 内部
        按算子参数自动计算，当前返回默认值。

        Args:
            world_size: 组内 rank 数。
            num_max_tokens_per_rank: 每 rank 最大 token 数（= S）。
            hidden: token 隐层数（= H）。
            num_experts: 专家总数（= E）。
            topk: 每 token top-k（= K）。

        Returns:
            int: hccl_buffer_size 配置值（MB）。
        """
        from cann_ops_transformer.ops import ElasticBuffer

        return ElasticBuffer.get_moe_ep_ccl_buffer_size(
            world_size, num_max_tokens_per_rank, hidden, num_experts, topk
        )

    def __init__(
        self,
        S: int,
        H: int,
        Hp: int,
        K: int,
        E: int,
        *,
        num_ep_ranks: int | None = None,
        moonep_group: torch.distributed.ProcessGroup,
        ep_group: torch.distributed.ProcessGroup,
        weight_dtype: str | torch.dtype = "bf16",
        grad_dtype: str | torch.dtype = "bf16",
        slots: int = 1,
        pull_dual: int | None = None,
        pull_seg_k: int | None = None,
    ) -> None:
        """构造并持有 moonep 全部通信资源（集体调用——内含 dist.new_group）。

        前置条件（调用方保证）：``dist.init_process_group(backend="hccl")``
        已完成、``torch.npu.set_device(rank)`` 已绑定；全部 EP rank 以相同
        参数同序执行（构造内含集合通信）。

        前五个 positional ``(S, H, Hp, K, E)``：``(S, H, K, E)`` 与 MoonEP
        ``Buffer(S, H, K, E, ...)`` 同名同序，``Hp``（专家 FFN 中间维）插在
        H 后；H 既是 EB token 维（dispatch/combine 载荷宽）也是专家 FFN
        输入维（专家权重即 (H, Hp) 形态）；``num_ep_ranks`` 显式参数
        （None 回退组 world size，显式给出则校验一致）。

        Args:
            S: 每 rank 输入 token 数（plan/dispatch 入参形状校验口径）。
            H: hidden size（= MoonEP 的 H，同名同序）——EB token 维
                （dispatch/combine 载荷宽），同时是专家 FFN 输入维。
            Hp: 专家 FFN 中间维（intermediate/pipeline）；专家权重形态
                ``(Hp, H) / (H, Hp)``，权重行宽上界 ``W = H × Hp``。堆按 3W
                组宽上界预留（覆盖 3 等宽 / 交织 (2W, W) / 2 等宽三种组型）。
            K: 每 token top-k（同上；EB num_topk 亦取此值）。
            E: EP 组专家总数；E % R == 0（R = world size），E <= MAX_E、
                R <= 64 由既有模块守卫（moonep_plan / ctx）承接。
            moonep_group: moonep ctx（plan/prefetch/reduce）通信组——
                ``ProcessGroup``（**仅支持组对象，不接受 str/None**）。
                调用方须先调用 ``MoonepBuffer.get_moon_ep_ccl_buffer_size()``
                计算堆容量并注入 ``hccl_buffer_size`` 后建组。
            ep_group: EB（dispatch/combine）通信组——``ProcessGroup``。
                调用方须先调用 ``MoonepBuffer.get_moe_ep_ccl_buffer_size()``
                计算通信 buffer 大小后建组。须与 moonep_group 分属两个
                HCCL 通信器（通信域分离不变量，构造期自检）。
            weight_dtype: 权重容量口径 "bf16"/"e4m3fn"/"e5m2"（或
                torch.dtype），缺省 bf16。fp8 = mxfp8 数据面 ⟹ 首调必传
                local_scales（双向硬契约）并预留 scale 区（dtype 终态
                终态：uint8 已退役）。
            grad_dtype: 梯度槽 dtype 定型值 "bf16"/"fp32"，**缺省字面
                bf16**（显存优先 + 当前唯一可信口径；fp32 显式
                声明为 master-weight 精度口径）。定型值 = 槽区容量预留
                口径 + local_grads 校验基。
            slots: 多槽窗口池层数（层间流水双槽传 2），>= 1；构造期定型
                ctx.window_slots——缺此定型 slots>1 时槽区基址按 slots=1
                计算，落在发布区 slot-1 段内被 prefetch 踩（实证偏差 15-48%）。

        组数契约（运行期自适应）：权重/梯度组数不构造期固定——调用方按
        3 组等宽 ``(W, W, W)`` 或交织 2 组 ``(2W, W)``（gate/up 融合为 W13）
        传权重/梯度均可；堆按 3W 总宽上限预留（交织两组 2W+W=3W 同字节、
        2 等宽组更小），首次算子调用定型 layout_groups 并由容量校验兜底。
        """
        # —— 守卫（全部先于 dist.new_group：违例构造为纯本地失败，不产生集合调用）——
        # R 取注入通信域的组内 rank 数（子组 != WORLD 时 epn 须按组内分片
        # 计算——如 WORLD=16、EP 子组=8 时 epn = E/8 而非 E/16）
        R = dist.get_world_size(moonep_group)
        # ---- C4: num_ep_ranks 显式参数（与 MoonEP 同形；与组 world size 校验）----
        if num_ep_ranks is None:
            num_ep_ranks = R
        elif int(num_ep_ranks) != R:
            raise ValueError(
                f"num_ep_ranks={num_ep_ranks} 与组 world size={R} 不符"
                "（参照 MoonEP 显式声明，防跨组误用）"
            )
        # ---- 通信域校验（仅接受 ProcessGroup——由调用方经静态方法预算后建组）----
        if not isinstance(moonep_group, torch.distributed.ProcessGroup):
            raise TypeError(
                f"moonep_group 须为 torch.distributed.ProcessGroup，"
                f"got {type(moonep_group).__name__}——请先调用 "
                "MoonepBuffer.get_moon_ep_ccl_buffer_size() 计算 buffer 大小"
                "并以 ProcessGroupHCCL.Options 注入后建组"
            )
        if not isinstance(ep_group, torch.distributed.ProcessGroup):
            raise TypeError(
                f"ep_group 须为 torch.distributed.ProcessGroup，"
                f"got {type(ep_group).__name__}——请先调用 "
                "MoonepBuffer.get_moe_ep_ccl_buffer_size() 计算 buffer 大小"
                "并以 ProcessGroupHCCL.Options 注入后建组"
            )
        owns_moonep = False
        owns_ep = False
        if S < 1:
            raise ValueError(f"S 须 >= 1，got {S}")
        if K < 1:
            raise ValueError(f"K 须 >= 1，got {K}")
        if E < 1:
            # E=0 时 E%R==0 成立会静默通过（0 模任意正数余 0），epn=0
            # 在下游 numel//epn 除零；负数经 E%R==0 同样漏网
            raise ValueError(f"E 须 >= 1，got {E}")
        if E % R != 0:
            raise ValueError(f"E={E} 必须能被 R={R} 整除（epn = E // R）")
        if slots < 1:
            raise ValueError(f"slots 须 >= 1，got {slots}")
        if H < 1:
            raise ValueError(f"H 须 >= 1，got {H}")
        if Hp < 1:
            raise ValueError(f"Hp 须 >= 1，got {Hp}")
        w = int(H) * int(Hp)
        gdtype = _normalize_grad_dtype(grad_dtype)
        wdtype = _normalize_weight_dtype(weight_dtype)
        if w % _W_ALIGN != 0:
            raise ValueError(
                f"W={w} 须为 {_W_ALIGN} 的倍数（reduce VL=64 尾块越界防护，"
                "本类必配 reduce 路径故构造期检查）"
            )
        # ---- 堆容量计算（参数校验后——w/gdtype/wdtype 已就绪）----
        pick_weight_chunk(w, _CHUNK_TRIAL_UB)  # 不可整除即 ValueError（fail-fast）
        epn = E // R
        grad_elem = 2 if gdtype == torch.bfloat16 else 4
        weight_elem = 2 if wdtype == torch.bfloat16 else 1
        scale_upper = _scale_upper_bytes(w)
        scale_widths = (
            (scale_upper,) * 3 if prefetch_weight._is_fp8_weight(wdtype) else None
        )
        heap_need = (
            HEAP_BASE
            + reduce_region_end(
                R,
                epn,
                w,
                3,
                slots,
                grad_bytes_per_elem=grad_elem,
                scale_widths=scale_widths,
                weight_bytes=weight_elem,
            )
            + LAYOUT_TAIL_BYTES
        )
        # HCCL 2× 机制：配置 N MB → 实际可用 ≈ 2×N MiB
        buffer_mb = int(heap_need // _HCCL_2X_MIB) + 2
        # —— 集体段（全 EP rank 同序执行）——
        self._group = moonep_group
        self._owns_group = owns_moonep  # destroy 时是否释放自建组
        self._ep_group = ep_group
        self._owns_ep_group = owns_ep
        self._ctx = create_moonep_context(group=self._group)
        # ★ 现状注入点收编：window_slots / weight_dtype / grad_dtype / pull 流水
        # 参数构造期定型（调用方只传 slots=N 业务参数与 dtype/pull；模块侧 E1/E5 校验基）
        self._ctx.window_slots = slots
        self._ctx.weight_dtype = wdtype
        self._ctx.grad_dtype = gdtype
        # 对称堆容量前置校验（fail-fast——注入组堆不足时构造期即报错）
        check_heap_capacity(self._ctx, heap_need, "MoonepBuffer 构造")
        # —— 定型状态（构造期只读）——
        self._S, self._K, self._E = int(S), int(K), int(E)
        self._H = int(H)
        self._epn = epn
        self._Hp = int(Hp)
        self._w_elems = w  # 仅内部布局口径（properties 不再对外暴露 w_elems）
        self._slots = int(slots)
        # pull 流水旋钮为构造参数（env 仅作缺省——单一真值源，构造后只读）
        self._pull_dual = int(
            pull_dual
            if pull_dual is not None
            else os.environ.get("MOONEP_PULL_DUAL", "1")
        )
        self._pull_seg_k = int(
            pull_seg_k
            if pull_seg_k is not None
            else os.environ.get("MOONEP_PULL_K", "5")
        )
        self._ctx.pull_dual = self._pull_dual
        self._ctx.pull_seg_k = self._pull_seg_k
        self._views_cache = {}  # slot -> prefetch 返回的 (weights, scales) 二元组
        self._weight_dtype = wdtype
        self._grad_dtype = gdtype
        self._grad_elem_bytes = grad_elem
        self._heap_need = heap_need
        self._buffer_mb = buffer_mb
        self._rank = self._ctx.rank
        self._rank_size = self._ctx.rank_size
        self._destroyed = False
        # —— EB 步骤 10：nmtr 定型 + ElasticBuffer@WORLD ——
        # nmtr = S（与 torchtitan 生产口径一致，override/common/moonep.py 的
        # "S is the per-peer send bound"）：EB 窗口 per-peer 槽位承载的是 token
        # 行（扩展为 (token,expert) 对发生在接收侧 recv 区），per-peer 到达
        # ≤ S 恒成立——与路由分布、与 plan 均衡与否无关。历史公式
        # max(S, int(S*K/R*1.25)+64) 的"接收侧 per-peer 泊松余量"是对不受
        # nmtr 约束的量（扩展对流量）预留的：其出处（3ee880219，10-17）诞生于
        # plan 尚在修均衡 bug 的窗口期（10-11 "balanced 实测定位"），且 10-24
        # 的推导有缺陷，已废弃。
        # plan 均衡不变量（每 rank 扩展行恰 S*K）由 recv 分配承载
        # （do_cpu_sync=True 精确 / False worst-case R·nmtr·min(K,epn) ≥ S·K）。
        self._nmtr = self._S
        # EB 建在 WORLD 组：与上面的独建子组分属两个 HCCL 通信器——通信域分离
        # 由构造保证（check_no_shared_elastic_buffer 只作兜底）
        self._eb = ElasticBuffer(
            ep_group,
            num_max_tokens_per_rank=self._nmtr,
            hidden=self._H,
            num_topk=self._K,
        )

    # ------------------------------------------------------------------
    # 内部守卫
    # ------------------------------------------------------------------

    def _require_alive(self):
        """destroy 后调用拒绝（比 GPU 的裸 assert 更可测——有意偏差）。"""
        if self._destroyed or self._ctx is None:
            raise RuntimeError(
                "MoonepBuffer 已 destroy：视图/plan 张量的使用须先于 destroy"
            )
        return self._ctx

    # ------------------------------------------------------------------
    # plan / 视图
    # ------------------------------------------------------------------

    def plan(
        self, tokens_per_expert: torch.Tensor, topk_experts_sk: torch.Tensor
    ) -> MoonepPlan:
        """规划入口（透传 moonep_plan，包 MoonepPlan 句柄）。

        入参/返回语义与 moonep_plan 完全一致（vtopk-only 快路径；返回路径
        零同步，溢出检查延迟到 prefetch/reduce 尾部）。构造期一致性守卫：
        tpe 须为 [E]、topk 须为 [S, K]（防以异形 E 规划导致视图/清单错位）。

        返回后 weight_view/dispatch 方可调用（视图随 prefetch 首调创建；
        既有 last_plan_shape 守卫自然生效，错误消息照旧传播）。
        """
        ctx = self._require_alive()
        if tokens_per_expert.shape[0] != self._E:
            raise ValueError(
                f"tokens_per_expert.shape[0]={tokens_per_expert.shape[0]} "
                f"!= 构造期 E={self._E}"
            )
        if tuple(topk_experts_sk.shape) != (self._S, self._K):
            raise ValueError(
                f"topk_experts_sk.shape={tuple(topk_experts_sk.shape)} "
                f"!= 构造期 (S, K)=({self._S}, {self._K})"
            )
        return MoonepPlan(
            *moonep_plan.moonep_plan(ctx, tokens_per_expert, topk_experts_sk)
        )

    def views(self, slot: int = 0) -> tuple:
        """prefetch 返回的权重侧视图二元组（缓存访问器）。

        返回 ``(weights, scales)`` 裸二元组：``weights[g]`` = 组 g 权重视图 ``[2*epn, Wg]``
        （home+副本全量行）；``scales[g]`` = 组 g scale 视图
        ``[2*epn, Sg]`` float8_e8m0fnu（mxfp8 形态；bf16 形态为 ``()``）。
        **须先成功调用
        prefetch_weight**（组结构与行宽运行期自适应——2 组交织或 3 组
        等宽，本方法只是缓存访问器；同 data_ptr 幂等）。
        """
        self._require_alive()
        if not 0 <= slot < self._slots:
            raise ValueError(f"slot={slot} 超出槽位数 {self._slots}")
        views = self._views_cache.get(int(slot))
        if views is None:
            raise RuntimeError(
                f"slot={slot} 尚无视图——视图随 prefetch_weight 首调创建"
                "（先传 local_weights 调用一次，取返回的 (weights, scales) 二元组）"
            )
        return views

    def weight_view(self, group: int = 0, slot: int = 0) -> torch.Tensor:
        """views 二元组的解包糖：组 g 权重视图 [2*epn, Wg]（GEMM 直读）。

        组数语义：3 组等宽传 ``(w1, w3, w2)`` 时 ``weight_view(0/1/2)``
        = gate/up/down（每组 [2*epn, Hp, H] 或 [2*epn, H, Hp]）；2 组交织传
        ``(w13, w2)`` 时 ``weight_view(0)`` = W13（每专家一行 2W，gate 前
        up 后）、``weight_view(1)`` = down。前 epn 行 home 参数、后 epn 行
        预取副本（prefetch 后含远端行）。用法：
        ``torch_npu.npu_grouped_matmul(x, weight_view(g).view(2*epn, Hp, H),
        offset=plan.cu_seqlens)``——scale 配对见 ``views()[1]``。"""
        weights = self.views(slot)[0]
        if not 0 <= group < len(weights):
            raise ValueError(f"group={group} 超出组数 {len(weights)}")
        return weights[group]

    # ------------------------------------------------------------------
    # moonep 三算子
    # ------------------------------------------------------------------

    def prefetch_weight(
        self,
        plan: MoonepPlan,
        async_finish: bool = False,
        *,
        local_weights: list | tuple,
        local_scales: list | tuple | None = None,
        slot: int = 0,
    ) -> tuple | tuple[tuple, torch.Event]:
        """home 直推预取副本专家权重（+ 可选 mxfp8 scale 组）。

        参数序与 MoonEP 同位（``prefetch_weight(plan, async_finish, *,
        local_*_weight)``——其 3 个命名投影收敛为 1 个同序元组）。

        Args:
            plan: buf.plan 返回的 MoonepPlan。
            async_finish: False（缺省）返回即全组本轮直推完成；True 跳过
                末尾屏障的 host 同步，返回发起流 torch.npu.Event，调用方
                wait_event 汇合（跨 rank 槽位复用安全由调用方负责）。
            local_weights: 2 或 3 组权重（list/tuple，每组 ``[epn, Wg]``
                或 N-D ``[epn, Hp, H]`` 直入——**组数与各组行宽运行期
                自适应**：三矩阵等宽传 ``(w1, w3, w2)``；交织两矩阵传
                ``(w13, w2)``（w13 每专家一行 2W，gate 前 up 后——一次
                GMM 算完免消费端 concat）。dtype 须与构造 weight_dtype
                一致。**必选**（外部参数传本体；参数驻留视图
                home 行时 sync_transport 同址跳过自动零拷贝——C8：
                不支持 None 取缓存形态，复用视图数据会造成潜在逻辑
                错误且难排查）。
            local_scales: 与 local_weights 同组同序的 scale 组（每组
                ``[epn, ...]`` 连续 float8_e8m0fnu（OCP MX 标准 ue8m0
                编码），行宽 Sg = 每专家 scale 字节数；
                交织形态传融合 scale 或调用方自行拼接）。**E2 双向硬
                契约：scales ⟺ fp8**——fp8 权重（e4m3fn/e5m2）必传、
                bf16 不传也不可传；Sg 须为 32B 倍数。
            slot: 多槽窗口池槽位（层间流水轮转；scale 视图同槽随权重
                切换）。

        Returns:
            **变长返回**：``async_finish=False`` 只返回
            ``(weights, scales)`` 裸二元组（weights = 各组 ``[2*epn, Wg]`` 视图元组、scales =
            各组 ``[2*epn, Sg]`` e8m0fnu 视图元组，未量化 ``()``）；
            ``async_finish=True`` 返回 ``((weights, scales), event)``——
            event 为发起流完成事件。
        """
        ctx = self._require_alive()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(
                f"prefetch_weight 需要 MoonepPlan（buf.plan 返回），"
                f"got {type(plan).__name__}"
            )
        if local_weights is None:
            raise TypeError(
                "local_weights 为必选参数（C7+C8 裁策）——外部参数传本体"
                "（sync_transport 同址跳过自动零拷贝），不再支持 None 取缓存形态"
                "（复用视图数据会造成潜在逻辑错误且难排查）"
            )
        local_weights = tuple(local_weights)
        result = prefetch_weight.prefetch_weight(
            ctx,
            plan.experts_to_copy,
            local_weights,
            self._epn,
            local_scales=local_scales,
            slot=int(slot),
            slots=self._slots,
            async_finish=async_finish,
        )
        # 变长返回：模块层已按 async_finish 定形——缓存视图后原样透传
        if async_finish:
            views, event = result
            self._views_cache[int(slot)] = views  # (weights, scales) 二元组
            return views, event
        self._views_cache[int(slot)] = result  # (weights, scales) 二元组
        return result

    def reduce_grad(
        self,
        plan: MoonepPlan,
        async_finish: bool = False,
        *,
        local_grads: list | tuple,
    ) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], torch.Event]:
        """副本梯度归并回 home（pull 协议，透传 reduce_grad.reduce_grad）。

        参数序与 MoonEP 同位（``reduce_grad(plan, async_finish, *,
        local_*_grad)``——3 命名投影收敛为 1 同序列表）。**不支持
        scales**（梯度只有 bf16/fp32；ue8m0 无梯度）。

        梯度搬运四步全在本方法内（①副本行入槽 → ②组屏障 → ③owner
        远拉原子直累 home 行 → ④收尾）；prefetch_weight 不搬任何梯度。

        Args:
            plan: buf.plan 返回的 MoonepPlan（副本清单 experts_to_copy
                与 prefetch 共用）。
            async_finish: False（缺省）阻塞到全组归并完成（返回 None）；
                True 返回发起流 torch.npu.Event（wait_event 汇合后读
                归并结果）。
            local_grads: 2 或 3 组 full-dW 张量（**必传 keyword**；
                list/tuple；每组 ``[2*epn, Wg]`` 或 N-D ``[2*epn, Hp, H]``
                直入——GMM 对全量权重视图反传的原生形态，行序与 prefetch
                视图一致：前 epn 行 home 种子、后 epn 行副本 dW）。组数/
                各组行宽从张量解析（交织两矩阵传 ``[dw13, dw2]`` 仍一次
                调用）；dtype 须与构造 grad_dtype 定型值一致。副本
                行入槽由内部完成（梯度槽视图不对外暴露）。

        Returns:
            **变长返回**：``async_finish=False``（缺省）只返回
            home_views（各组归并结果的 [epn, Wg] 视图——同存储，归并后
            即全量 dW 的直接读柄）；仅 ``async_finish=True`` 时返回
            ``(home_views, event)`` 二元组。

        前置时序契约：本方法晚于本层权重最后一次读、先于下一层
        prefetch（单槽；双槽流水安全）。梯度槽区单窗口（不随 slots
        倍增——反向按层串行，与 MoonEP 单 reduce 池同构）。
        """
        ctx = self._require_alive()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(
                f"reduce_grad 需要 MoonepPlan（buf.plan 返回），"
                f"got {type(plan).__name__}"
            )
        if local_grads is None:
            raise TypeError("local_grads 为必传 keyword（2/3 组 full-dW 张量）")
        if isinstance(local_grads, torch.Tensor):
            # C15：单张量无成功路径（组数恒 1 ∉ {2,3}）——原 isinstance
            # 包装是只通向报错的死分支；显式报错防 list(tensor) 行迭代歧义。
            raise TypeError(
                "local_grads 须为 2/3 组 list/tuple（每组 [2*epn, ...]）——不支持单张量"
            )
        grads = list(local_grads)
        # 模块侧解析组数/行宽（[2*epn, Wg] 校验 + dtype 校验 + 容量
        # 兜底）；async_finish 与模块侧同名同义。
        # 变长返回：模块层已按 async_finish 定形——原样透传
        # （sync 返回 home_views 单值 / async 返回 (home_views, event)）
        return reduce_grad.reduce_grad(
            ctx,
            plan.experts_to_copy,
            grads,
            self._epn,
            slot=0,
            slots=self._slots,
            async_finish=async_finish,
        )

    # ------------------------------------------------------------------
    # dispatch / combine（EB 收编，前反向同构复用）
    # ------------------------------------------------------------------

    def dispatch(
        self,
        plan: MoonepPlan,
        x: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        *,
        do_cpu_sync: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, object]:
        """EB dispatch：token 按虚拟专家路由分发（前向载荷 / 反向散梯度同构）。

        内部自动接线（现状两测试的手工旋钮收编）：``topk_idx =
        plan.vtopk.view(S, K)``、``num_experts = 2E``（虚拟专家扩展空间）、
        ``num_max_tokens_per_rank = nmtr``（= S）、
        ``topk_weights = topk_weights.float()``。

        Args:
            plan: buf.plan 返回的 MoonepPlan（反向复用前向同一 plan，免重规划）。
            x: [S, hidden] 张量——前向为 token 载荷（bf16），反向为输出梯度
                （combine⁻¹ 散梯度形态）。
            topk_weights: [S, K] 路由权重（fp32 或可 ``.float()``）。None = 反向
                形态：rw 不随车，返回的 recv_w 为 None——加权由调用方复用
                前向 recv_w 或由 GMM 反向侧施加（mem 实证路径）。
            do_cpu_sync: True（缺省）为 EB 官方精确分配路径——dispatch 首调
                CPU 同步获取实际到达行数后精确分配 recv（无 worst-case 预留，
                显存最优，代价一次 CPU 同步）；False 免同步（ep 测试实证路径，
                recv 按 worst-case 容量分配）。透传可调，缺省取显存最优；
                固定形状生产链路可 ``buf._eb._get_dispatch_recv_count = lambda args: S*K``
                补丁兼得免同步与精确（EB 实例钩子，非本 API 契约）
                行数（plan 均衡不变量）。

        Returns:
            (recv_x, recv_w, handle)——EB 4 元组的 meta 位丢弃（现有全部
            调用均弃置；如后续需要再扩展返回形态）。recv_x 为 [cap, hidden]
            接收区（do_cpu_sync=True 时 cap == 实际行数）；recv_w 为 [cap]
            fp32 路由权重（topk_weights=None 时为 None）；handle 供对应 combine
            使用（含反向各自的 handle，见 Usage）。
        """
        self._require_alive()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(
                f"dispatch 需要 MoonepPlan（buf.plan 返回），got {type(plan).__name__}"
            )
        if not isinstance(x, torch.Tensor) or tuple(x.shape) != (
            self._S,
            self._H,
        ):
            raise ValueError(
                f"x.shape={tuple(x.shape) if isinstance(x, torch.Tensor) else x}"
                f" 须为构造期 (S, H)=({self._S}, {self._H})"
            )
        if topk_weights is not None and tuple(topk_weights.shape) != (
            self._S,
            self._K,
        ):
            raise ValueError(
                f"topk_weights.shape={tuple(topk_weights.shape)} "
                f"!= 构造期 (S, K)=({self._S}, {self._K})"
            )
        recv_x, _meta, recv_w, handle = self._eb.dispatch(
            x,
            topk_idx=plan.vtopk.view(self._S, self._K),
            topk_weights=topk_weights.float() if topk_weights is not None else None,
            num_experts=2 * self._E,
            num_max_tokens_per_rank=self._nmtr,
            do_cpu_sync=do_cpu_sync,
        )
        return recv_x, recv_w, handle

    def combine(
        self,
        y: torch.Tensor,
        handle,
        *,
        topk_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """EB combine：专家输出按路由回收（前向加权 / 反向裸和同构）。

        Args:
            y: [cap, hidden]——**须填满容量** ``cap = 对应 dispatch 返回的
                recv_x.shape[0]``（host 侧 shape，零同步可得）；有效行不足
                容量时尾行零填充（裸和语义不受影响）。本封装不代做零填充
                （避免引入 host 同步与隐式状态），惯用法三行::

                    combine_in = torch.zeros(cap, H, dtype=y.dtype, device=y.device)
                    combine_in[:n] = y                     # n 为有效行数
                    out, out_w = buf.combine(combine_in, handle, ...)

            handle: 对应 dispatch 返回的 handle（前向/反向各自的 handle）。
            topk_weights: None = 裸和（dispatch⁻¹——反向梯度按 K 份归和回
                token 主序）；传 recv_w = 权重回收（前向——out_w 应恰等于
                本 rank 路由权重）。

        Returns:
            (out, out_w) 透传 EB：out [S, hidden]；out_w [S, K] fp32 权重
            回收（topk_weights=None 时为 None）。
        """
        self._require_alive()
        if not isinstance(y, torch.Tensor) or y.dim() != 2 or y.shape[1] != self._H:
            raise ValueError(
                f"y.shape={tuple(y.shape) if isinstance(y, torch.Tensor) else y}"
                f" 须为 [cap, H={self._H}]（cap 由对应 dispatch 的 recv 决定）"
            )
        out, out_w = self._eb.combine(y, handle, topk_weights=topk_weights)
        return out, out_w

    def barrier(self) -> None:
        """对称堆自旋 barrier：返回即全组本轮此前流上工作全部完成。"""
        sym_barrier(self._require_alive())

    # ------------------------------------------------------------------
    # 只读 property
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """本 rank 在 EP 子组内的 id。"""
        return self._rank

    @property
    def rank_size(self) -> int:
        """EP 子组 rank 数。"""
        return self._rank_size

    @property
    def epn(self) -> int:
        """每 rank 本地专家数（E // R）。"""
        return self._epn

    @property
    def Hp(self) -> int:
        """专家 FFN 中间维（构造期 positional Hp；W = H × Hp）。"""
        return self._Hp

    @property
    def weight_dtype(self) -> torch.dtype:
        """权重 dtype 定型值（构造期；bf16 / e4m3fn / e5m2）。"""
        return self._weight_dtype

    @property
    def layout_groups(self) -> int | None:
        """运行期组数（首次 prefetch/reduce 后定型；此前为 None——组数自适应）。"""
        ctx = self._require_alive()
        return ctx.layout_groups

    @property
    def pull_dual(self) -> int:
        """reduce pull 双缓冲流水开关构造参数；env 仅缺省。"""
        return self._pull_dual

    @property
    def pull_seg_k(self) -> int:
        """reduce pull 流水段 chunk 数构造参数。"""
        return self._pull_seg_k

    @property
    def slots(self) -> int:
        """多槽窗口池层数。"""
        return self._slots

    @property
    def grad_dtype(self) -> torch.dtype:
        """梯度槽 dtype（构造期定型）。"""
        return self._grad_dtype

    @property
    def H(self) -> int:
        """hidden size（= MoonEP 的 H；EB token 维/dispatch 载荷宽，构造期定型）。"""
        return self._H

    @property
    def nmtr(self) -> int:
        """EB per-rank 容量（= S，发送侧 per-peer 上界；接收侧扩展行由
        plan 均衡不变量保证恰 S·K，由 recv 分配承载而非 nmtr）。"""
        return self._nmtr

    @property
    def heap_required_bytes(self) -> int:
        """构造期布局需求（HEAP_BASE + reduce_region_end + LAYOUT_TAIL）。"""
        return self._heap_need

    @property
    def heap_config_mb(self) -> int:
        """注入的 hccl_buffer_size 配置值（MB）；实际可用 ≈ 2×（HCCL 机制）。"""
        return self._buffer_mb

    @property
    def heap_bytes(self) -> int:
        """HCCL 对称堆实测可用字节数（hccl_buffer_size(ctx)）。"""
        return hccl_buffer_size(self._require_alive())

    @property
    def destroyed(self) -> bool:
        """是否已显式销毁。"""
        return self._destroyed

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def destroy(self, *, release_group: bool = False, release_eb: bool = False) -> None:
        """显式释放逻辑资源（幂等；视图/plan 张量使用须先于本调用）。

        Args:
            release_group: 缺省 False 不调 dist.destroy_process_group(子组)
                ——子组与其对称堆留待 world teardown/进程退出回收。依据：
                实际训练中 buffer 为进程级单例（模型初始化创建、整个训练
                复用、进程退出由 world teardown 统一回收）——训练稳态根本
                不存在 mid-process 释放场景；串行多 buffer（测试/多阶段
                作业）是唯一需要提前释放的场景，destroy_process_group 对
                HCCL 子组的耗时未验证，缺省走零风险路径。True 为 opt-in：
                显式提前释放对称堆，供长进程内串行多 buffer 场景（耗时待实测）。
            release_eb: 缺省 False **不调用** ``self._eb.destroy()``，EB 引用
                在逻辑 destroy 后仍由本对象持有到进程退出（"池化保活 + 进程
                退出硬退回收"，conftest 实证路径）。依据一（平台代价）：EB
                Destroy 的 HcclBarrier 建 AICPU_TS 通道必等 ~120s/个——缺省
                路径若立即弃引用，GC 可能在会话中段触发 EB 析构造成分钟级
                停顿；依据二（释放不完备性）：C++ 共享池按 tag 绑定的物理
                内存与 channel 注册本就无反注册接口（Destroy 不释放，同
                group 重建时复用）——True 也只是部分回收，代价收益不对称。
                True 为 opt-in：显式 ``self._eb.destroy()`` 确定性语义回收
                （慢路径，串行多 buffer 场景使用；接受 ~120s 级耗时）。

        与 GPU destroy（VMM 资源确定性释放，秒级）的有意差异：根因是 EB
        Destroy 的平台行为（不可反注册 + 分钟级 barrier），非封装设计偏好；
        GPU 侧训练同样是进程级单例 + 退出回收，本缺省与之在训练稳态等价。
        """
        if self._destroyed:
            return
        self._destroyed = True
        # 释放自建通信域（注入形态不释放——调用方负责）
        if self._owns_group:
            dist.destroy_process_group(self._group)
        if getattr(self, "_owns_ep_group", False):
            dist.destroy_process_group(self._ep_group)
        self._ctx = None  # 后续调用经 _require_alive 以 RuntimeError 拒绝
        if release_eb:
            self._eb.destroy()
        if release_group:
            dist.destroy_process_group(self._group)
        # release_eb=False 时 self._eb 引用保留（池化保活——防 GC 中途触发
        # EB 析构的分钟级停顿）；release_eb=True 时已销毁，引用保留无害

    def __del__(self):
        # 只警告不调 dist/EB API（解释器收尾/GC 期调 dist 不安全；conftest
        # atexit 硬退同样跳过 GC）——与 GPU 缺省自动 destroy 的有意差异
        # （理由见 destroy；EB 的 GC 析构代价见 release_eb 说明）
        if getattr(self, "_destroyed", True):
            return
        warnings.warn(
            "MoonepBuffer 未 destroy() 即被回收：子组/对称堆与 EB 留待进程"
            "退出或 GC 回收（__del__ 内不调用 dist/EB API——GC/解释器收尾期"
            "不安全；EB 析构的硬退代价见 destroy 的 release_eb 说明）",
            ResourceWarning,
            stacklevel=2,
        )
