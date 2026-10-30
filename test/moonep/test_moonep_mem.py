# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_train_memstat.py——显存对账训练用例（2026-10-26 战役）。

目标形状（用户 2026-10-26 指令）：
  S=8192, K=16, H=7168（BF16 dispatch token 维）
  隐层 D=3584（latent）、F=3072（expert hidden）——每专家 3 个 3584×3072 矩阵
  E=224（4 卡 epn=56 / 8 卡 epn=28；与用户理论公式 *224/4、*224/8 一致）
  ⚠️ 用户文字写"4卡：140个专家"但理论公式按 224/4=56/卡——本用例默认按公式
     取 E=224；如确为 140，传 --E 140（此时每卡理论权重 = 35×63MB = 2.15GB）。

架构（latent MoE——两组理论值同时成立的唯一一致读法）：
  x[S,H] → EB dispatch（token 维 H=7168 展开，理论 1.75GB/rank）
  → recv[n,H] → 共享降维 Wd[D,H]（每 rank 复制，非专家权重）
  → h[n,D] → 逐组专家 FFN（gate/up [F,D] + down [D,F]，SiTU 激活）
  → y[n,D] → 共享升维 Wu[H,D] → 加权 → EB combine → out[S,H]
  反向镜像：dW 落 bf16 视图 [2*epn, W] → reduce_grad 归并 → SGD。

显存统计（MEMSTAT 标签行，grep 汇总）：
  - DSL 三算子（plan/prefetch_weight/reduce_grad）全分配点 + 对称堆布局
  - EB dispatch/combine 的 Python/C++ 分配点 + moe_ep 通信窗
  - 训练侧激活张量逐个打点 + 每步末设备占用快照
  - 开跑前打印终态理论值（发布区/容量链/梯度/共享+激活四口径）对照

运行（A5_8P，qrun-next 内）：
  torchrun --nproc_per_node=<4|8> test/moonep/test_moonep_mem.py [--steps 2]
  pytest 形态：torchrun --nproc_per_node=4 -m pytest test/moonep/test_moonep_mem.py -q
  （单进程 pytest 收集形态自动跳过——CI 安全）
"""

import argparse
import os

import pytest
import sys
import time

# 位置自适应：脚本形态（samples/moonep 内直跑）与测试形态（test/moonep
# 收集）双兼容——测试形态下 samples 模块在 ../../samples/moonep
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(os.path.dirname(_here), os.pardir, "samples", "moonep"))

import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    # 平级导入：ctx 工具（堆布局公式）+ 三算子 + 显存打点
    import mem_stat

    _HAS_TRAIN_DEPS = True
except Exception:  # CI 无 NPU/无实现时收集不报错
    _HAS_TRAIN_DEPS = False

# PUSH_BLOCKS 规避开关（8P 并发 flush 同 peer 触发 aivec 271——issue#1 已报作者；
# 单 block 串行 put 对 W=22MiB 行的吞吐影响可忽略：传输时间主导，35µs/条发射
# 开销占比 <1%）。MOONEP_PUSH_BLOCKS=1 启用。
if _HAS_TRAIN_DEPS and os.environ.get("MOONEP_PUSH_BLOCKS"):
    import prefetch_weight as _pw

    _pw.PUSH_BLOCKS = int(os.environ["MOONEP_PUSH_BLOCKS"])
    print(
        f"[MEMSTAT] op=train kind=note what=push_blocks_override "
        f"note=PUSH_BLOCKS={_pw.PUSH_BLOCKS}（8P 规避）",
        flush=True,
    )
if _HAS_TRAIN_DEPS:
    from moonep_context import (  # 堆基址/ctx 工厂/堆上界公式
        HEAP_BASE,
        create_moonep_context,
    )
    import moonep_plan  # 五件套规划（vtopk-only 快路径）
    import prefetch_weight  # 副本权重直推（对称窗视图）
    import reduce_grad  # 副本梯度归并（pull 协议 kernel 直累）

# SiTU 激活参数（沿 R6 demo 同源——激活选择不影响显存结论）
SITU_BETA, SITU_LB = 4.0, 25.0


# ---------------------------------------------------------------------------
# SiTU 前向/反向（解析式——R6 demo 同款）
# ---------------------------------------------------------------------------


def situ_fwd(merged, beta, lb):
    """SiTU 激活：gate,up = chunk(2)；up'=lb*tanh(up/lb)；
    out = beta*tanh(gate/beta)*sigmoid(gate)*up'"""
    gate, up = merged.chunk(2, dim=-1)
    u = lb * torch.tanh(up / lb)
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * u


def situ_bwd(g, merged, beta, lb):
    """SiTU 解析反向（g 为 dL/dout）——乘积法则展开。"""
    gate, up = merged.chunk(2, dim=-1)
    # 预激活中间量（前向各函数的局部导数因子）
    t = torch.tanh(gate / beta)
    s = torch.sigmoid(gate)
    tu = torch.tanh(up / lb)
    # gate 侧：乘积法则两因子（tanh' 与 sigmoid'）
    d_gate = g * (lb * tu) * ((1 - t * t) * s + beta * t * s * (1 - s))
    # up 侧：u = lb·tu 的导数 = 1-tu²
    d_up = g * beta * t * s * (1 - tu * tu)
    return torch.cat((d_gate, d_up), dim=-1)


# ---------------------------------------------------------------------------
# 路由与权重生成（CPU 种子——跨 rank 确定性）
# ---------------------------------------------------------------------------


def gen_routing(S, K, E, seed, dev):
    """均匀 topk + softmax 分数（int32——plan 入参契约）。"""
    g = torch.Generator().manual_seed(seed)
    # 均匀 logits → topk（CPU 生成保证跨 rank 确定性）
    logits = torch.randn(S, E, generator=g)
    topk = torch.topk(logits, K, dim=-1).indices.to(torch.int32).to(dev)
    # ★ 8P bring-up 诊断模式：轮转 topk（完全平坦路由——零复制需求）
    # plan R=8 副本槽溢出（作者 issue#1）的绕行——仅诊断用，不改数值语义链
    if os.environ.get("MOONEP_FLAT_ROUTING") == "1":
        topk = (torch.arange(S * K) % E).to(torch.int32).reshape(S, K).to(dev)
    # 分数 = 选中项 softmax（gather 回取后归一）
    scores = torch.softmax(
        torch.gather(logits, 1, topk.long().cpu()).float(), dim=-1
    ).to(dev)
    # tpe = 每 expert 实收 token 数（plan 的 counts 入参）
    tpe = (
        torch.bincount(topk.cpu().reshape(-1).long(), minlength=E)
        .to(torch.int32)
        .to(dev)
    )
    return topk, scores, tpe


def init_view_params(views, epn, D, F, seed_base, rank, dev):
    """参数直接建在视图 home 行（作者推荐零拷贝形态——权重存储=发布区）。

    生成序 w1→w3→w2（与 R6 demo 一致）；写入 views[g][:epn] 即写入对称堆
    home 行——real_weights 与视图 data_ptr 相同，prefetch 的 sync_transport
    自动跳过拷贝。
    """
    shapes = ((F, D), (F, D), (D, F))  # w1/w3 gate+up、w2 down
    for g, shape in enumerate(shapes):
        gen = torch.Generator().manual_seed(seed_base + 100 * g + rank)
        # CPU 生成后拷入视图（bf16 精度——与训练权重同 dtype）
        w = (torch.randn(epn, *shape, generator=gen) * 0.02).to(torch.bfloat16)
        views[g][:epn].view(epn, *shape).copy_(w.to(dev))


# ---------------------------------------------------------------------------
# 训练主体
# ---------------------------------------------------------------------------


def run(args):
    # 进程组 + 设备绑定（每 rank 一卡）
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"

    # 形状解包（E 默认 224——用户理论公式口径）
    S, K, H, D, F, E = args.S, args.K, args.H, args.D, args.F, args.E
    assert E % R == 0, f"E={E} 必须被 R={R} 整除"
    epn = E // R
    W = D * F  # 权重行宽（展平）——三个矩阵均为 D×F
    slot_views = None
    beta, lb = SITU_BETA, SITU_LB

    # ---- 理论值打印（终态实现口径——实测对照的锚点）----
    grad_elem = 2 if args.grad_dtype == "bf16" else 4
    cap = S * K  # do_cpu_sync=True 精确分配口径（plan 均衡保证实际到达 S×K）
    cap_tag = "S×K（do_cpu_sync 精确）"
    # 权重发布区（堆上 home 行 + 副本行，slots 双缓冲翻倍）
    theory_weights = args.slots * 3 * 2 * epn * W * 2
    # dispatch/combine 容量链：recv/反向 recv/combine_in/gx_in 四条并存
    theory_dcb = 4 * cap * H * 2
    # 梯度（g1/g2/g3，grad_dtype 口径）+ reduce scratch
    theory_grad = 3 * epn * W * grad_elem
    theory_scratch = R * W * grad_elem
    # 共享投影 Wd/Wu + 激活链（y_up/y/h/merged）
    theory_shared = 2 * D * H * 2
    theory_act = (S * K) * (H + 2 * D + 2 * F) * 2
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_weights,
        note=f"权重发布区 = slots({args.slots})×3×2×epn({epn})×{W}×2B = "
        f"{theory_weights / 2**30:.3f} GiB（home+副本行；ctx 创建实耗≈2×堆配置）",
    )
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_dcb,
        note=f"dispatch/combine 容量链 = 4×cap({cap})×{H}×2B = "
        f"{theory_dcb / 2**30:.3f} GiB（四条并存；cap={cap_tag}）",
    )
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_grad + theory_scratch,
        note=f"梯度 g1/g2/g3 = 3×epn({epn})×{W}×{grad_elem}B + scratch = "
        f"R×{W}×{grad_elem}B = {(theory_grad + theory_scratch) / 2**30:.3f} GiB",
    )
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_shared + theory_act,
        note=f"共享 Wd/Wu = 2×{D}×{H}×2B + 激活链 = (S×K)×(H+2D+2F)×2B = "
        f"{(theory_shared + theory_act) / 2**30:.3f} GiB",
    )

    t0 = time.perf_counter()
    # ---- ctx 子组（对称堆按发布区终点预留——v4 无独立槽区）----
    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    # v3：含独立梯度槽区——堆终点 = reduce_region_end（发布区+槽区+尾）
    from moonep_context import LAYOUT_TAIL_BYTES, reduce_region_end

    grad_elem = 2 if args.grad_dtype == "bf16" else 4
    heap_need = (
        HEAP_BASE
        + reduce_region_end(
            R,
            epn,
            W,
            num_groups=3,
            slots=args.slots,
            grad_bytes_per_elem=grad_elem,
        )
        + LAYOUT_TAIL_BYTES
    )
    # HCCL 2× 分配机制：实际可用 ≈ 2×hccl_buffer_size
    # 传 heap_need/2 + 2 MiB 余量 → 实际可用 = heap_need（精确匹配）
    # 4P 验证：losses golden + device 47.85→35.30 GiB（−12.55）
    # 8P 验证：losses golden + 37.29→32.13 GiB（−5.16）
    buffer_mb = int(heap_need * args.heap_factor // (2 * 1024**2)) + 2
    hccl_options = ProcessGroupHCCL.Options()
    # hccl_buffer_size 即对称堆预留（设备显存在子组创建时扣减）
    hccl_options.hccl_config = {"hccl_buffer_size": buffer_mb}
    # 预留配置打点（与 create 后的实测 heap 行对照——差值即 HCCL 对齐开销）
    # 打点：配置值 vs 布局需求（×1.2 余量——子组创建后对照实测 heap 行）
    mem_stat.log(
        # 配置值打点（×1.2 余量——子组创建后与实测 heap 行对照）
        "train",
        "heap",
        "ctx_hccl_buffer_config",
        bytes_=buffer_mb * 1024**2,
        note=f"布局需求 {heap_need} ({heap_need / 2**30:.3f} GiB) × "
        f"{args.heap_factor}——"
        f"prefetch 发布区+reduce staging 的总预留",
    )
    # 显存基线：子组创建（堆在此刻真实扣减设备显存）
    with mem_stat.delta("train", note="ctx 子组创建（对称堆预留）"):
        # 独立子组（use_local_synchronization 减少建组开销）+ 堆大小注入
        ctx_group = dist.new_group(
            backend="hccl",
            ranks=list(range(R)),
            use_local_synchronization=True,
            pg_options=hccl_options,
        )
        # ctx = 通信句柄 + 堆布局守卫（创建即校验 plan 区容量）
        ctx = create_moonep_context(group=ctx_group)
        # ★ 接线：--slots 须进堆布局（权重发布区与梯度槽区宽度一致——
        # 缺此行时槽区基址按 slots=1 计算，落在发布区 slot-1 段内被
        # prefetch 踩：slots=2 对拍 sender rank 梯度偏差 15-48% 实证）
        ctx.window_slots = args.slots

    # ---- EB（WORLD 组——与 ctx 通信域分离，防对称堆互踩）----
    from cann_ops_transformer.ops import ElasticBuffer

    with mem_stat.delta("train", note="ElasticBuffer 构造"):
        # nmtr=S（每 rank 发送上限——EB 契约须 ≥ S）；hidden=H=7168
        eb = ElasticBuffer(
            dist.group.WORLD, num_max_tokens_per_rank=S, hidden=H, num_topk=K
        )
    # ★ do_cpu_sync=True：EB 官方精确分配路径——dispatch 首调时 CPU 同步
    # 获取实际到达行数后精确分配 recv（无 worst-case 预留，替代 compact 补丁）
    mem_stat.log(
        "train",
        "note",
        "dispatch_do_cpu_sync",
        note=f"do_cpu_sync=True 精确分配（替代 compact monkey-patch；"
        f"worst-case 上界 = R×S×min(K,2E/R) = {R * S * min(K, 2 * E // R)} 行）",
    )
    mem_stat.log(
        "train",
        "note",
        "transport_init",
        note=f"{time.perf_counter() - t0:.1f}s "
        f"(R={R} S={S} K={K} E={E} epn={epn} W={W})",
    )

    # ---- 共享投影（每 rank 复制——latent MoE 的压缩/解压矩阵，非专家权重）----
    gsh = torch.Generator().manual_seed(7000 + rank)
    # Wd 降维 [D,H] / Wu 升维 [H,D]（同一 generator 流——顺序固定）
    Wd = (torch.randn(D, H, generator=gsh) * 0.02).to(torch.bfloat16).to(dev)  # 降维
    Wu = (torch.randn(H, D, generator=gsh) * 0.02).to(torch.bfloat16).to(dev)  # 升维
    mem_stat.log_tensor("train", "共享降维 Wd[D,H]（ replicated）", Wd)
    mem_stat.log_tensor("train", "共享升维 Wu[H,D]（ replicated）", Wu)

    # ---- 每步数据 ----
    def step_data(step):
        topk, scores, tpe = gen_routing(S, K, E, 5000 + 97 * step + rank, dev)
        g = torch.Generator().manual_seed(6000 + 31 * step + rank)
        x = torch.randn(S, H, generator=g).to(torch.bfloat16).to(dev)
        return x, topk, scores, tpe

    # 权重视图（首步 plan 后创建——create_weight_view 契约）
    views = None
    # 梯度槽视图（pull 模式——首步 plan 后创建）
    losses = []

    for step in range(args.steps):
        torch.npu.synchronize()
        ts = time.perf_counter()
        x, topk, scores, tpe = step_data(step)

        # ---- 前向：plan → dispatch → 降维 → 分组 FFN → 升维 → combine ----
        # plan：v2_clean 五件套（vtopk-only 快路径）
        _d, cu_seqlens, etc, _s, vtopk = moonep_plan.moonep_plan(ctx, tpe, topk)
        n = S * K  # plan 保证每 rank 恰收 S×K 行（负载均衡副本）
        # dispatch：EB 主线（token 维 H 展开；num_experts=2E 虚拟专家空间）
        # vtopk 展平→[S,K]（EB 入参形态）；权重 fp32（EB 契约）
        with mem_stat.delta("dispatch", note="首调含 CCL 窗注册；稳态≈recv 重分配"):
            recv, _meta, recv_w, handle = eb.dispatch(
                x,
                topk_idx=vtopk.view(S, K),
                topk_weights=scores.float(),
                num_experts=2 * E,
                num_max_tokens_per_rank=S,
                do_cpu_sync=True,
            )
        cap = recv.shape[0]  # vanilla 容量（R×nmtr×min(K,2E/R)——对照理论 S×K）

        # 权重视图 + 参数（零拷贝：参数即视图 home 行）
        if views is None:
            # 三组各一视图（gate/up/down——组 g 基址 = 发布区起 + g×整视图步长）
            # 三组各一视图（gate/up/down——组 g 基址 = 发布区起 + g×整视图步长）
            views = [
                prefetch_weight.create_weight_view(
                    # slot=0 单槽（层间流水双槽传 --slots 2——发布区翻倍）
                    ctx,
                    epn,
                    W,
                    group=g,
                    groups=3,
                    slot=0,
                    slots=args.slots,
                )
                for g in range(3)
            ]
            # 参数随机初始化直接写入视图 home 行（零拷贝形态）
            init_view_params(views, epn, D, F, 3000, rank, dev)
        # 梯度槽视图（pull 协议 v3——独立于权重发布区，FFN 反向写 dW 落点）
        if slot_views is None:
            slot_views = [
                reduce_grad.create_grad_slot_view(
                    ctx,
                    epn,
                    W,
                    group=g,
                    groups=3,
                    grad_dtype=torch.bfloat16
                    if args.grad_dtype == "bf16"
                    else torch.float32,
                )
                for g in range(3)
            ]
            ctx.pull_slot_views = slot_views  # context 共享（reduce 接口零新增参数）

        # prefetch：副本权重直推（real=视图 home 行 → sync_transport 零拷贝跳过）
        prefetch_weight.prefetch_weight(
            ctx,
            etc,
            [v[:epn] for v in views],
            views,
            epn,
        )

        # cu D2H：唯一 host 同步点（分组边界）
        cu_list = cu_seqlens.cpu().tolist()
        assert cu_list[-1] == n, f"cu[-1]={cu_list[-1]} != n={n}"

        # 共享降维：recv[n,H] → h[n,D]（latent——后续 FFN 全在 D 维进行）
        h = recv[:n] @ Wd.t()
        mem_stat.log_tensor("train_act", "h[n,D] latent", h)
        # 逐组 FFN（朴素参考实现——生产换分组 GEMM）
        # FFN 输出与 SiTU 输入（反向复用免重算——R6 同款）
        y = torch.zeros(n, D, dtype=torch.bfloat16, device=dev)
        merged = torch.zeros(n, 2 * F, dtype=torch.bfloat16, device=dev)
        # 两块激活逐个打点（反向复用——计入训练激活占用）
        mem_stat.log_tensor("train_act", "y[n,D]", y)
        mem_stat.log_tensor("train_act", "merged[n,2F]", merged)
        prev = 0
        # VM group 循环：组 g 行区间 [prev,end)，home 段在前副本段在后
        for g, end in enumerate(cu_list):
            if end > prev:
                # 本组行段 [prev,end)——home 段 g<epn，副本段 g>=epn（同视图取行）
                seg = h[prev:end]
                w1g = views[0][g].view(F, D)  # gate [F,D]
                w3g = views[1][g].view(F, D)  # up   [F,D]
                w2g = views[2][g].view(D, F)  # down [D,F]
                # gate/up 投影拼 SiTU 输入，激活后 down 投影
                m = torch.cat((seg @ w1g.t(), seg @ w3g.t()), dim=-1)
                merged[prev:end] = m
                y[prev:end] = situ_fwd(m, beta, lb) @ w2g.t()
            prev = end
        # 共享升维 + 路由加权（combine 契约是裸和——权重在 combine 前乘）
        y_up = y @ Wu.t()
        mem_stat.log_tensor("train_act", "y_up[n,H]", y_up)
        # float 中转 3-kernel 路径（bf16 直乘实测回归 +11.5%——W1 教训）
        weighted = (y_up.float() * recv_w[:n].unsqueeze(-1)).to(torch.bfloat16)
        # combine 输入必须填满容量（尾行零填充不影响裸和）
        combine_in = torch.zeros(cap, H, dtype=torch.bfloat16, device=dev)
        mem_stat.log_tensor(
            "train_act",
            "combine_in[cap,H]",
            combine_in,
            note=f"cap={cap}（vanilla 容量，对照理论 S×K={n}）",
        )
        # 有效行填入 + 尾行零填充（裸和语义不受影响）
        combine_in[:n] = weighted
        with mem_stat.delta("combine", note="前向 combine"):
            out, _ = eb.combine(combine_in, handle, topk_weights=None)
        # 前向完成同步（计时切分点 + 确保后续读 loss 已落地）
        torch.npu.synchronize()
        tf = time.perf_counter()
        # 大临时即刻释放（combine 已同步——combine_in/weighted/y_up 不再需要；
        # 显存对账口径：释放后 empty_cache 归还 driver，快照更干净）
        del weighted, y_up, combine_in
        torch.npu.empty_cache()

        # 简单目标：把输出拉向 0（loss 仅报告——本用例主目的是显存对账）
        loss = out.float().pow(2).sum() / (S * H)

        # ---- 反向（手写——R6 demo 同构）----
        g_out = (2.0 * out.float() / (S * H)).to(torch.bfloat16)
        # combine 反向 = dispatch 复用 plan（不传权重——梯度无需再乘路由分）
        with mem_stat.delta("dispatch_bwd", note="反向 dispatch"):
            g_recv_h, _, _, _ = eb.dispatch(
                g_out,
                topk_idx=vtopk.view(S, K),
                topk_weights=None,
                num_experts=2 * E,
                num_max_tokens_per_rank=S,
                do_cpu_sync=True,
            )
        # dL/dy = dL/dy_up × w_k（前向加权的链式法则——乘前向的 recv_w）
        # float 中转保精度（bf16 广播乘法实测更慢——W1 教训同源）
        g_y_up = (g_recv_h[:n].float() * recv_w[:n].unsqueeze(-1)).to(torch.bfloat16)
        # g_recv_h（cap×H 大头）加权完毕即释放——后续 FFN 反向腾空间
        del g_recv_h
        g_y = g_y_up @ Wu  # [n, D]（升维反传回 latent）
        gWu = (g_y_up.t().float() @ y).to(torch.bfloat16)  # [H, D]
        # g_y_up/y 用完即释（gWu 已算）
        del g_y_up, y
        torch.npu.empty_cache()
        # dW 分配：pull 协议——[epn] 纯 home 段本地梯度 + 堆上梯度槽
        ddt = torch.bfloat16 if args.grad_dtype == "bf16" else torch.float32
        # 三组 [epn,...] 本地梯度（纯 home——reduce 后已是全量）
        g1 = torch.zeros(epn, F, D, dtype=ddt, device=dev)
        g3 = torch.zeros(epn, F, D, dtype=ddt, device=dev)
        g2 = torch.zeros(epn, D, F, dtype=ddt, device=dev)
        mem_stat.log_tensor("train_act", "g1[epn,F,D] pull", g1)
        mem_stat.log_tensor("train_act", "g3[epn,F,D] pull", g3)
        mem_stat.log_tensor("train_act", "g2[epn,D,F] pull", g2)
        # latent 梯度累加器（FFN 反向的输出）
        g_h = torch.zeros(n, D, dtype=torch.bfloat16, device=dev)
        prev = 0
        for g, end in enumerate(cu_list):
            if end > prev:
                seg = h[prev:end]
                m = merged[prev:end]
                g_seg = g_y[prev:end]
                w1g = views[0][g].view(F, D)
                w3g = views[1][g].view(F, D)
                w2g = views[2][g].view(D, F)
                hid = situ_fwd(m, beta, lb)
                g_m = g_seg @ w2g
                # dW2 贡献
                dwg2 = (g_seg.t() @ hid).to(ddt)
                g_mm = situ_bwd(g_m, m, beta, lb)
                g_h1, g_h3 = g_mm.chunk(2, dim=-1)
                dwg1 = (g_h1.t() @ seg).to(ddt)
                dwg3 = (g_h3.t() @ seg).to(ddt)
                # ★ pull 协议写入（E1）：home → 本地张量；副本 → 权重视图
                # 副本行 [epn, 2epn)（发布区远程可寻址，owner 远端读走）。
                # 写入须晚于本层权重读（w1g/w3g 在下一行才读——先算后写）
                if g < epn:
                    g1[g] += dwg1
                    g3[g] += dwg3
                    g2[g] += dwg2
                g_h[prev:end] = g_h1 @ w1g + g_h3 @ w3g
                if g >= epn:
                    b = g - epn
                    slot_views[0][b] = dwg1.reshape(-1)
                    slot_views[1][b] = dwg3.reshape(-1)
                    slot_views[2][b] = dwg2.reshape(-1)
            prev = end
        # h/merged 反向消费完毕——释放后再进 reduce
        del h, merged
        torch.npu.empty_cache()
        # reduce：副本梯度归并回 home（pull 协议——唯一实现）
        # ★ 功能验证（可选）：reduce 前用 all_reduce 建参考（torch 模式
        # 同款数学：各 rank 填自己贡献 → all_reduce → 取 home 段），
        # reduce 后对比。参考只读本地槽（reduce 前槽未被消费），无跨
        # rank 槽依赖，故可先于 reduce_grad 的 sym_barrier 执行。
        ref_home = None
        if args.verify:
            etc_v = etc.cpu()
            my_etc_v = etc_v[rank]
            src_v = [g1.view(epn, W), g3.view(epn, W), g2.view(epn, W)]
            ref_home = []
            for gi in range(3):
                full = torch.zeros(E, W, dtype=ddt, device=dev)
                full[rank * epn : (rank + 1) * epn] = src_v[gi]
                for b in range(epn):
                    ge = int(my_etc_v[b])
                    if ge >= 0:
                        full[ge] = slot_views[gi][b]
                dist.all_reduce(full)
                ref_home.append(full[rank * epn : (rank + 1) * epn].clone())
                del full
            torch.npu.empty_cache()
            mem_stat.log(
                "train_act",
                "note",
                "verify_ref",
                note="pull 功能验证：all_reduce 参考已建（3 组 home 段）",
            )
        if args.bench_reduce:
            # 纯 reduce_grad 基准：barrier+sync 后 Event 计时（去 host 开销）
            ev_s = torch.npu.Event(enable_timing=True)
            ev_e = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            dist.barrier()
            ev_s.record()
            reduce_grad.reduce_grad(
                ctx,
                etc,
                [g1.view(epn, W), g3.view(epn, W), g2.view(epn, W)],
                epn,
            )
            ev_e.record()
            torch.npu.synchronize()
            dist.barrier()
            ms = ev_s.elapsed_time(ev_e)
            # 路由命中计数（分析双峰分布归因——本 rank 拥有多少被副本的专家）
            hit_count = int((etc[rank] >= 0).sum().item())
            # 我需拉取的条目数 = 对端 etc 行中命中我 home 区间的条目总数
            my_home_lo = rank * epn
            my_home_hi = my_home_lo + epn
            pull_count = int(
                ((etc >= my_home_lo) & (etc < my_home_hi) & (etc >= 0)).sum().item()
            )
            print(
                f"[BENCH-REDUCE] step={step} {ms:.2f}ms "
                f"pull_entries={pull_count} my_replicas={hit_count}",
                flush=True,
            )
        else:
            reduce_grad.reduce_grad(
                ctx,
                etc,
                [g1.view(epn, W), g3.view(epn, W), g2.view(epn, W)],
                epn,
            )
        if ref_home is not None:
            all_ok = True
            for name, got, ref in zip(("g1", "g3", "g2"), (g1, g3, g2), ref_home):
                diff = (got.float() - ref.float()).abs()
                scale = ref.float().abs().max().clamp_min(1e-6)
                rel = (diff.max() / scale).item()
                ok = rel < 1e-2
                all_ok = all_ok and ok
                print(
                    f"[VERIFY:{name}] max_abs={diff.max().item():.3e} "
                    f"rel={rel:.3e} {'PASS' if ok else 'FAIL'}",
                    flush=True,
                )
            print(
                f"[VERIFY] pull vs all_reduce 参考："
                f"{'✅ PASS' if all_ok else '❌ FAIL'}",
                flush=True,
            )
            del ref_home
        # dispatch 反向 = combine（K 份梯度裸和回 token 序）；
        # 共享降维的反传：g_recv = g_h @ Wd，gWd 由 g_h 与 recv 外积
        g_recv = g_h @ Wd  # [n, H]（h=recv@Wdᵀ 的反向）
        gWd = (g_h.t().float() @ recv[:n]).to(torch.bfloat16)  # [D, H]
        # combine 输入填满容量（与 forward 同款零填充）
        gx_in = torch.zeros(cap, H, dtype=torch.bfloat16, device=dev)
        mem_stat.log_tensor("train_act", "gx_in[cap,H]", gx_in)
        gx_in[:n] = g_recv
        with mem_stat.delta("combine_bwd", note="反向 combine"):
            g_x, _ = eb.combine(gx_in, handle, topk_weights=None)

        # ---- SGD（手写——无 optimizer 状态）----
        with torch.no_grad():
            # 专家权重（视图 home 行）——reduce 后 g* 已是全量梯度
            p1 = views[0][:epn].view(epn, F, D)
            p3 = views[1][:epn].view(epn, F, D)
            p2 = views[2][:epn].view(epn, D, F)
            if args.mode == "pull":
                # pull：g* 已是 [epn] 全量（home + Σ 副本——reduce_pull 归并完）
                p1.copy_(p1 - args.lr * g1.to(torch.bfloat16))
                p3.copy_(p3 - args.lr * g3.to(torch.bfloat16))
                p2.copy_(p2 - args.lr * g2.to(torch.bfloat16))
            else:
                # push：dw* 是 [2epn]，reduce 后取 [:epn]（home 段）
                p1.copy_(p1 - args.lr * g1[:epn].to(torch.bfloat16))
                p3.copy_(p3 - args.lr * g3[:epn].to(torch.bfloat16))
                p2.copy_(p2 - args.lr * g2[:epn].to(torch.bfloat16))
            # 共享投影
            Wu -= args.lr * gWu
            Wd -= args.lr * gWd
        torch.npu.synchronize()
        tb = time.perf_counter()
        losses.append(loss.item())
        # 每步末设备占用快照（torch 池口径 + driver 口径双轨）
        # driver 口径（真实占用）+ torch 池口径（allocated/reserved）双轨
        free, total = torch.npu.mem_get_info()
        # 每步末快照：loss/耗时 + 双口径占用（对账表的时间轴锚点）
        mem_stat.log(
            # 快照行：对账表时间轴锚点（driver/torch 双口径）
            "train",
            "note",
            f"step{step}_snapshot",
            # 逐字段：loss/耗时 → driver 占用 → torch 池两口径
            note=f"loss={loss.item():.4f} fwd={tf - ts:.2f}s bwd={tb - tf:.2f}s | "
            f"device_used={(total - free) / 2**30:.2f} GiB / {total / 2**30:.0f} GiB | "
            f"torch_allocated={torch.npu.memory_allocated() / 2**30:.2f} GiB "
            f"reserved={torch.npu.memory_reserved() / 2**30:.2f} GiB",
        )

    # ★ 跨运行对拍导出（功能验证）：种子确定性下，pull 与 torch 两模式的
    # 终态梯度/权重应一致（差异仅在 reduce 协议的舍入路径）。
    # 背景：in-process --verify 的 all_reduce 在 EB URMA 混用下触发 A5 已知
    # EI0006（设计文档 P8 记载）——跨运行对拍为零集合通信的替代验证路径。
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        torch.save(
            {
                "g1": g1.detach().cpu(),
                "g3": g3.detach().cpu(),
                "g2": g2.detach().cpu(),
                # clone 规避 blob 视图 storage 尺寸错配（E1 小堆布局下
                # rank 偶发 torch.save enforce fail 104B）
                "w1": views[0][:epn].detach().clone().cpu(),
                "w3": views[1][:epn].detach().clone().cpu(),
                "w2": views[2][:epn].detach().clone().cpu(),
                "Wu": Wu.detach().cpu(),
                "Wd": Wd.detach().cpu(),
                "etc": etc.cpu(),
                "losses": list(losses),
            },
            f"{args.dump}/rank{rank}.pt",
        )
        print(f"[DUMP] rank{rank} -> {args.dump}/rank{rank}.pt", flush=True)

    # 收尾汇总（rank0）
    if rank == 0:
        # rank0 收尾行（shell 侧 grep 的完成标记）
        free, total = torch.npu.mem_get_info()
        print(
            f"TRAIN-MEMSTAT-DONE: steps={args.steps} "
            f"losses={[f'{v:.4f}' for v in losses]} "
            f"device_used={(total - free) / 2**30:.2f} GiB",
            flush=True,
        )
    dist.destroy_process_group()


def main(argv=None):
    # CLI：默认形状 = 用户指令值（E=224 按理论公式口径——见文件头说明）
    # 形状参数默认 = 用户指令值（E=224 按理论公式——见文件头 ⚠️ 说明）
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--K", type=int, default=16)
    # H=dispatch token 维；D/F=隐层两维（三矩阵均 D×F）
    ap.add_argument("--H", type=int, default=7168)
    ap.add_argument("--D", type=int, default=3584, help="latent 维（隐层 1）")
    ap.add_argument("--F", type=int, default=3072, help="expert hidden 维（隐层 2）")
    # E 默认 224（4 卡 epn=56 / 8 卡 epn=28——与理论公式一致）
    ap.add_argument(
        "--E",
        type=int,
        default=224,
        help="专家总数（默认按用户理论公式 224；4 卡 140 说传 140）",
    )
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument(
        "--bench-reduce",
        action="store_true",
        help="纯 reduce_grad 基准模式：跳过 GEMM/dispatch/combine，Event 计时 reduce",
    )
    # 堆预留系数：布局需求 × factor（+1MiB 页余量）。1.2 系拍脑袋值
    # 已被二分实测证伪——1.0（精确布局+页余量）实测通过（4P bf16：
    # TRAIN-DONE losses 正确 + 39.87 GiB；1.05 下 40.55、1.2 下
    # 42.62 GiB），HCCL 无隐藏对齐开销
    ap.add_argument("--heap-factor", type=float, default=1.0)
    # lr/steps：演示训练参数（本用例主目的为显存对账）
    ap.add_argument("--lr", type=float, default=1e-2)
    # 窗口槽数（层间流水双槽发布区翻倍——对照理论时注意）
    ap.add_argument(
        "--slots",
        type=int,
        default=1,
        help="权重窗口槽数（1=单槽；2=层间流水双槽——发布区翻倍）",
    )
    # dW dtype（A1 口径 bf16——staging 减半；fp32 对照口径）
    ap.add_argument(
        "--grad-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="dW 视图 dtype（bf16=A1 口径 staging 减半；fp32 翻倍）",
    )
    # reduce 协议：push（旧 v2_clean staging）或 pull（新——消除 R 倍 staging）
    # reduce 协议：仅 pull（v3 kernel 直累——旧 push staging 已移除）
    ap.add_argument(
        "--mode",
        choices=["pull"],
        default="pull",
        help="reduce_grad 协议（pull——唯一实现）",
    )
    # 功能验证（pull 模式专用）：all_reduce 参考对照（torch 模式同款数学）
    ap.add_argument(
        "--verify",
        action="store_true",
        help="pull 功能验证：reduce 前建 all_reduce 参考，"
        "reduce 后对比（会增加 ~[E,W] 瞬态——勿与显存测量同用；"
        "qrun 下 EB URMA+all_reduce 混用有 EI0006 已知风险，"
        "建议直跑或用 --dump 跨运行对拍替代）",
    )
    # 跨运行对拍导出：pull/torch 两模式分别 --dump，离线比较终态
    ap.add_argument(
        "--dump",
        metavar="DIR",
        default=None,
        help="导出终态梯度/权重/loss 到 DIR/rank{r}.pt"
        "（跨运行功能对拍——零集合通信的验证路径）",
    )
    args = ap.parse_args(argv)
    run(args)


# ---------------------------------------------------------------------------
# pytest 收集入口（test/moonep 约定）：CI 单进程 pytest 形态自动跳过，
# 真正执行需 torchrun + NPU（与 test_moonep_ep.py 的守卫同款思想）
# ---------------------------------------------------------------------------
_requires_train_env = pytest.mark.skipif(
    not _HAS_TRAIN_DEPS
    or "RANK" not in os.environ
    or "WORLD_SIZE" not in os.environ
    or os.environ["WORLD_SIZE"] == "1",
    reason="需 torch.distributed.run 多卡拉起（单进程 pytest 收集形态跳过）",
)


@_requires_train_env
def test_moonep_mem_train():
    """全链训练 2 步显存对账（plan→dispatch→FFN→combine→reduce_grad→SGD）。

    torchrun --nproc_per_node=4 -m pytest test/moonep/test_moonep_mem.py -q
    （或脚本形态：torchrun --nproc_per_node=4 test/moonep/test_moonep_mem.py）
    """
    rc = main(["--mode", "pull", "--steps", "2"])
    assert rc in (None, 0)


if __name__ == "__main__":
    main()
