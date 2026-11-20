# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_train_memstat.py——显存对账训练用例（Buffer 驱动迁移版）。

目标形状：
  S=8192, K=16, H=3584（token/专家 FFN 输入维——同参）、Hp=3072（中间维）
  每专家 3 个 H×Hp 矩阵
  E=224（4 卡 epn=56 / 8 卡 epn=28；理论公式 *224/4、*224/8 口径一致）
  如需 140 专家形态，传 --E 140（此时每卡理论权重 = 35×63MB = 2.15GB）。

架构（标准 MoE FFN 反向链——H 既是 token 维也是专家输入维）：
  x[S,H]（token 数据面不在本套件——token 行以本地桩注入：
  recv[n,H] 按 cu 分组序确定性生成，形状/分组与 dispatch 产物同构）
  → recv[n,H] → 逐组专家 FFN（gate/up [Hp,H] + down [H,Hp]，SiTU 激活）
  → y[n,H] → 加权 → loss
  反向镜像：dW 落 bf16 视图 [2*epn, W] → reduce_grad 归并 → SGD。

迁移说明（测试用例须基于已封装的接口调用）：
- 三算子（plan/prefetch_weight/reduce_grad）全部经
  Buffer 驱动：堆需求精算/ctx 定型/window_slots 接线/权重与梯度
  槽视图全部内化进构造与方法（HCCL 子组创建与组显存注入仍由测试侧
  ——通信域创建方——承担：数据面不占 HCCL 外部 buffer，注入引擎下限
  控制进程显存）；FFN 读写视图逻辑原样保留；
- --slots/--grad-dtype CLI 旋钮直接映射构造参数；堆需求
  打印改经 heap_required_bytes 只读 property 回读；
- MEMSTAT delta/log 打点全部保留（观测点，非接口调用）；
- MOONEP_PUSH_BLOCKS 覆盖开关已移除：实现读 _MAX_PUSH_BLOCKS
  经 resident_blocks() 按核数自适应，开关写入手已是无效死代码（8P aivec
  271 规避由核数自适应承担）。

显存统计（MEMSTAT 标签行，grep 汇总）：
  - Buffer 构造（HCCL 子组+workspace 预留+通信建连）delta 打点，
    堆配置经只读 property 回读；
  - DSL 三算子（plan/prefetch_weight/reduce_grad）全分配点 + workspace 布局
  - 训练侧激活张量逐个打点 + 每步末设备占用快照
  - 开跑前打印终态理论值（发布区/数据面行缓冲/梯度+副本区四口径）对照

运行（A5_8P，qrun-next 内）：
  torchrun --nproc_per_node=<4|8> test/moonep/test_moonep_mem.py [--steps 2]
  pytest 形态：torchrun --nproc_per_node=4 -m pytest test/moonep/test_moonep_mem.py -q
  （单进程 pytest 收集形态自动跳过——CI 安全）
"""

import argparse
import os
import sys
import time

import pytest

from conftest import _prefetch_api, _reduce_api  # noqa: E402

# 位置自适应：脚本形态（samples/moonep 内直跑）与测试形态（test/moonep
# 收集）双兼容——测试形态下 samples 模块在 ../../samples/moonep
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(_here)), os.pardir, "samples", "mc2", "moonep"
    ),
)

import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    # 平级导入：显存打点 mem_stat 已归位测试树（本目录 test/moonep，经
    # _here 插入的 sys.path 命中）；Buffer 含
    # ElasticBuffer 依赖——缺 whl 时收集不报错（skip 而非 error）
    import mem_stat
    from moonep_buffer import Buffer  # noqa: F401

    _HAS_TRAIN_DEPS = True
except Exception:  # CI 无 NPU/无实现时收集不报错
    _HAS_TRAIN_DEPS = False

# SiTU 激活参数（激活选择不影响显存结论）
SITU_BETA, SITU_LB = 4.0, 25.0


# ---------------------------------------------------------------------------
# SiTU 前向/反向（解析式）
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


def init_param_tensors(epn, H, Hp, seed_base, rank, dev):
    """参数张量（[epn, W] 二维——首次调用 prefetch 传入本体）。

    生成序 w1→w3→w2；首调后参数本体即视图 home 行
    （视图返回后直接读写，后续步传 home 行切片零拷贝直推）。
    """
    shapes = ((Hp, H), (Hp, H), (H, Hp))  # w1/w3 gate+up、w2 down
    reals = []
    for g, shape in enumerate(shapes):
        gen = torch.Generator().manual_seed(seed_base + 100 * g + rank)
        w = (torch.randn(epn, *shape, generator=gen) * 0.02).to(torch.bfloat16)
        reals.append(w.reshape(epn, -1).to(dev))
    return tuple(reals)


# ---------------------------------------------------------------------------
# 训练主体
# ---------------------------------------------------------------------------


def run(args):
    # 进程组 + 设备绑定（每 rank 一卡）——幂等（pytest/torchrun 可能已初始化）
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = f"npu:{rank}"

    # 形状解包（E 默认 224——理论公式口径）——须先解包再算容量
    S, K, H, Hp, E = args.S, args.K, args.H, args.Hp, args.E
    assert E % R == 0, f"E={E} 必须被 R={R} 整除"
    epn = E // R
    W = H * Hp  # 权重行宽（展平）——三个矩阵均为 H×Hp

    # moonep 子组：数据面走私有 workspace，不占通信域 HCCL 外部 buffer——
    # 注入引擎下限（2 MB，覆盖 comm 控制面/建连握手）封顶测试进程显存
    # （组显存配置由建组方即本测试自理；不注入即用 HCCL 缺省配置）
    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    g_opts = ProcessGroupHCCL.Options()
    g_opts.hccl_config = {"hccl_buffer_size": 2}
    dist.new_group(
        backend="hccl",
        ranks=list(range(R)),
        use_local_synchronization=True,
        pg_options=g_opts,
    )
    # EB 组：EB 数据面自建注册内存——建组建议值直取 EB 包静态方法
    # （当前为默认值）；纯 DSL 环境降级按下限注入
    try:
        from cann_ops_transformer.ops import ElasticBuffer

        eb_mb = ElasticBuffer.get_moe_ep_ccl_buffer_size(R, S, H, E, K)
    except (ImportError, ModuleNotFoundError):
        eb_mb = 2
    ep_opts = ProcessGroupHCCL.Options()
    ep_opts.hccl_config = {"hccl_buffer_size": eb_mb}
    dist.new_group(
        backend="hccl",
        ranks=list(range(R)),
        use_local_synchronization=True,
        pg_options=ep_opts,
    )
    beta, lb = SITU_BETA, SITU_LB

    # ---- 理论值打印（终态实现口径——实测对照的锚点；纯算术不依赖 buffer）----
    grad_elem = 2 if args.grad_dtype == "bf16" else 4
    cap = S * K  # 数据面行容量（plan 均衡保证分组行恰 S×K）
    cap_tag = "S×K（plan 均衡不变量）"
    # 权重发布区（堆上 home 行 + 副本行，slots 双缓冲翻倍）
    theory_weights = args.slots * 3 * 2 * epn * W * 2
    # 数据面行缓冲（本地桩口径：recv/反向 g_recv 两条并存）
    theory_dcb = 2 * cap * H * 2
    # 梯度（g1/g2/g3，grad_dtype 口径）+ reduce 副本区
    theory_grad = 3 * epn * W * grad_elem
    theory_scratch = epn * W * grad_elem
    # 激活链（y/merged）
    theory_act = (S * K) * (H + 2 * Hp) * 2
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_weights,
        note=f"权重发布区 = slots({args.slots})×3×2×epn({epn})×{W}×2B = "
        f"{theory_weights / 2**30:.3f} GiB（home+副本行；实耗=构造期精算 workspace）",
    )
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_dcb,
        note=f"数据面行缓冲 = 2×cap({cap})×{H}×2B = "
        f"{theory_dcb / 2**30:.3f} GiB（recv/g_recv 两条并存；cap={cap_tag}）",
    )
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_grad + theory_scratch,
        note=f"梯度 g1/g2/g3 = 3×epn({epn})×{W}×{grad_elem}B + 副本区 = "
        f"epn×{W}×{grad_elem}B = {(theory_grad + theory_scratch) / 2**30:.3f} GiB",
    )
    mem_stat.log(
        "theory",
        "note",
        "理论值（终态口径）",
        bytes_=theory_act,
        note=f"激活链 = (S×K)×(H+2Hp)×2B = {theory_act / 2**30:.3f} GiB",
    )

    t0 = time.perf_counter()
    # ---- Buffer 构造（ctx 子组 + workspace 预留 + 通信建连一体化）----
    # 旧 run 的手工段（heap_need 计算 → hccl_buffer_size 注入 → new_group →
    # ctx 创建 → window_slots 定型）全部内化于构造
    with mem_stat.delta("train", note="Buffer 构造（workspace 精算 + 通信建连）"):
        buf = Buffer(
            S,
            H,
            Hp,
            K,
            E,
            num_ep_ranks=dist.get_world_size(),
            group=dist.group.WORLD,
            slots=args.slots,
            grad_dtype=args.grad_dtype,
        )
    # 预留配置打点（构造期定型量经只读 property 回读——与子组创建后的
    # 实测 heap 行对照，差值即 HCCL 对齐开销）
    mem_stat.log(
        "train",
        "heap",
        "ctx_workspace_alloc",
        bytes_=buf.heap_bytes,
        note=f"布局需求 {buf.heap_required_bytes} "
        f"({buf.heap_required_bytes / 2**30:.3f} GiB)——"
        f"prefetch 发布区+梯度副本区的精确预留",
    )
    # 数据面行缓冲：本地桩口径（plan 均衡不变量 S*K 行——dispatch/combine
    # 不在本套件，行以确定性种子本地生成，形状/分组与真实产物同构）
    mem_stat.log(
        "train",
        "note",
        "data_plane_stub",
        note=f"本地行桩 S*K={S * K} 行/组（dispatch/combine 不在本 PR；"
        f"显存对账口径：行缓冲 [n,H]×2 条计理论值）",
    )
    mem_stat.log(
        "train",
        "note",
        "transport_init",
        note=f"{time.perf_counter() - t0:.1f}s (R={R} S={S} K={K} E={E} epn={epn} W={W})",
    )

    # ---- 每步数据 ----
    def step_data(step):
        topk, scores, tpe = gen_routing(S, K, E, 5000 + 97 * step + rank, dev)
        g = torch.Generator().manual_seed(6000 + 31 * step + rank)
        x = torch.randn(S, H, generator=g).to(torch.bfloat16).to(dev)
        return x, topk, scores, tpe

    # 权重视图（首步 prefetch 返回——视图创建契约）
    views = None  # (weights, scales) 二元组
    losses = []

    for step in range(args.steps):
        torch.npu.synchronize()
        ts = time.perf_counter()
        x, topk, scores, tpe = step_data(step)

        # ---- 前向：plan → 行桩注入 → 分组 FFN ----
        # plan：五件套（vtopk-only 快路径；NamedTuple 解包兼容）
        plan = buf._plan(tpe, topk)
        _dst_n, cu_seqlens, etc, _src_n, _vtopk = plan
        n = S * K  # plan 保证每 rank 恰收 S×K 行（负载均衡副本）
        # 数据面行桩：recv [n,H] 按 cu 分组序确定性
        # 生成（形状/分组与 dispatch 产物同构）；recv_w 为对应路由权重
        # （topk 展平序——与真实 dispatch 行序一致）
        with mem_stat.delta("data_plane", note="行桩注入（recv/recv_w 本地生成）"):
            gen = torch.Generator().manual_seed(7000 + 53 * step + rank)
            recv = torch.randn(n, H, generator=gen).to(torch.bfloat16).to(dev)
            recv_w = scores.reshape(-1).to(dev)
        cap = n  # 行容量（本地桩——恰 S*K）

        # 权重视图 + 参数：首步以参数张量为 real（视图按各组行宽创建并返回，
        # 参数拷入 home 行）；次步起零拷贝——参数更新直接写视图 home 行，
        # 传 home 行切片（data_ptr 同址 ⇒ sync 跳过）
        if views is None:
            reals = init_param_tensors(epn, H, Hp, 3000, rank, dev)
            views = _prefetch_api(buf, plan, local_weights=reals)
        else:
            _prefetch_api(buf, plan, local_weights=[w[:epn] for w in views[0]])

        # cu D2H：唯一 host 同步点（分组边界）
        cu_list = cu_seqlens.cpu().tolist()
        assert cu_list[-1] == n, f"cu[-1]={cu_list[-1]} != n={n}"

        # 逐组 FFN（朴素参考实现——生产换分组 GEMM）
        # FFN 输出与 SiTU 输入（反向复用免重算）
        y = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        merged = torch.zeros(n, 2 * Hp, dtype=torch.bfloat16, device=dev)
        # 两块激活逐个打点（反向复用——计入训练激活占用）
        mem_stat.log_tensor("train_act", "y[n,H]", y)
        mem_stat.log_tensor("train_act", "merged[n,2Hp]", merged)
        prev = 0
        # VM group 循环：组 g 行区间 [prev,end)，home 段在前副本段在后
        for g, end in enumerate(cu_list):
            if end > prev:
                # 本组行段 [prev,end)——home 段 g<epn，副本段 g>=epn（同视图取行）
                seg = recv[prev:end]
                w1g = views[0][0][g].view(Hp, H)  # gate [Hp,H]
                w3g = views[0][1][g].view(Hp, H)  # up   [Hp,H]
                w2g = views[0][2][g].view(H, Hp)  # down [H,Hp]
                # gate/up 投影拼 SiTU 输入，激活后 down 投影
                m = torch.cat((seg @ w1g.t(), seg @ w3g.t()), dim=-1)
                merged[prev:end] = m
                y[prev:end] = situ_fwd(m, beta, lb) @ w2g.t()
            prev = end
        # 路由加权（数据面桩口径：直接对 y 加权——combine 归并不在本 PR）
        # float 中转 3-kernel 路径（bf16 直乘性能回退，实测结论）
        out = (y.float() * recv_w[:n].unsqueeze(-1)).to(torch.bfloat16)
        # 前向完成同步（计时切分点 + 确保后续读 loss 已落地）
        torch.npu.synchronize()
        tf = time.perf_counter()
        # 大临时即刻释放（out 已同步——y 后续反向还要用，保留；
        # 显存对账口径：释放后 empty_cache 归还 driver，快照更干净）
        torch.npu.empty_cache()

        # 简单目标：把输出拉向 0（loss 仅报告——本用例主目的是显存对账）
        loss = out.float().pow(2).sum() / (S * H)

        # ---- 反向（手写链式；数据面桩——g_recv 不由 g_out 派生）----
        # 数据面行桩（反向）：g_recv [n,H] 确定性生成
        # （形状/分组与前向 recv 同构）
        with mem_stat.delta("data_plane_bwd", note="反向行桩注入（g_recv 本地生成）"):
            gen_b = torch.Generator().manual_seed(8000 + 53 * step + rank)
            g_recv_h = torch.randn(n, H, generator=gen_b).to(torch.bfloat16).to(dev)
        # dL/dy = dL/dy_out × w_k（前向加权的链式法则——乘前向的 recv_w）
        # float 中转保精度（bf16 广播乘法实测更慢）
        g_y = (g_recv_h[:n].float() * recv_w[:n].unsqueeze(-1)).to(torch.bfloat16)
        # g_recv_h（n×H 大头）加权完毕即释放——后续 FFN 反向腾空间
        del g_recv_h
        # y 用完即释
        del y
        torch.npu.empty_cache()
        # dW 分配：full-dW 形态 [2*epn,...]（前 epn 行 home 段、后 epn 行
        # 副本段——GMM 反传原生形态；副本入槽由 reduce_grad 内部完成）
        ddt = torch.bfloat16 if args.grad_dtype == "bf16" else torch.float32
        g1 = torch.zeros(2 * epn, Hp, H, dtype=ddt, device=dev)
        g3 = torch.zeros(2 * epn, Hp, H, dtype=ddt, device=dev)
        g2 = torch.zeros(2 * epn, H, Hp, dtype=ddt, device=dev)
        mem_stat.log_tensor("train_act", "g1[2epn,Hp,H] full-dW", g1)
        mem_stat.log_tensor("train_act", "g3[2epn,Hp,H] full-dW", g3)
        mem_stat.log_tensor("train_act", "g2[2epn,H,Hp] full-dW", g2)
        # 输入梯度累加器（FFN 反向的输出）
        g_h = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        prev = 0
        for g, end in enumerate(cu_list):
            if end > prev:
                seg = recv[prev:end]
                m = merged[prev:end]
                g_seg = g_y[prev:end]
                w1g = views[0][0][g].view(Hp, H)
                w3g = views[0][1][g].view(Hp, H)
                w2g = views[0][2][g].view(H, Hp)
                hid = situ_fwd(m, beta, lb)
                g_m = g_seg @ w2g
                # dW2 贡献
                dwg2 = (g_seg.t() @ hid).to(ddt)
                g_mm = situ_bwd(g_m, m, beta, lb)
                g_h1, g_h3 = g_mm.chunk(2, dim=-1)
                dwg1 = (g_h1.t() @ seg).to(ddt)
                dwg3 = (g_h3.t() @ seg).to(ddt)
                # ★ full-dW 写入：home 行 [0,epn) 累加种子；副本行 [epn,2epn)
                # 直写 dW（reduce 内部入槽——owner 远端读对称堆槽归并）
                if g < epn:
                    g1[g] += dwg1
                    g3[g] += dwg3
                    g2[g] += dwg2
                g_h[prev:end] = g_h1 @ w1g + g_h3 @ w3g
                if g >= epn:
                    b = g - epn
                    g1[epn + b] = dwg1
                    g3[epn + b] = dwg3
                    g2[epn + b] = dwg2
            prev = end
        # merged 反向消费完毕——释放后再进 reduce
        del merged
        torch.npu.empty_cache()
        # reduce：副本梯度归并回 home（pull 协议——唯一实现；内部自动
        # view(epn, W) + dtype/组数守卫）
        # ★ 功能验证（可选）：reduce 前用 all_reduce 建参考（torch 模式
        # 同款数学：各 rank 填自己贡献 → all_reduce → 取 home 段），
        # reduce 后对比。参考只读本地槽（reduce 前槽未被消费），无跨
        # rank 槽依赖，故可先于 reduce_grad 的 sym_barrier 执行。
        ref_home = None
        if args.verify:
            etc_v = etc.cpu()
            my_etc_v = etc_v[rank]
            src_v = [
                g1[:epn].reshape(epn, W),
                g3[:epn].reshape(epn, W),
                g2[:epn].reshape(epn, W),
            ]
            ref_home = []
            src_full = [
                g1.reshape(2 * epn, W),
                g3.reshape(2 * epn, W),
                g2.reshape(2 * epn, W),
            ]
            for gi in range(3):
                full = torch.zeros(E, W, dtype=ddt, device=dev)
                full[rank * epn : (rank + 1) * epn] = src_v[gi]
                for b in range(epn):
                    ge = int(my_etc_v[b])
                    if ge >= 0:
                        full[ge] = src_full[gi][epn + b]
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
            _reduce_api(buf, plan, local_grads=[g1, g3, g2])
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
            _reduce_api(buf, plan, local_grads=[g1, g3, g2])
        if ref_home is not None:
            all_ok = True
            for name, got, ref in zip(
                ("g1", "g3", "g2"), (g1[:epn], g3[:epn], g2[:epn]), ref_home
            ):
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
                f"[VERIFY] pull vs all_reduce 参考：{'PASS' if all_ok else 'FAIL'}",
                flush=True,
            )
            del ref_home
        # （dispatch 反向/combine 均不在本 PR——g_h 的 token 序回收由数据面
        #  承接，本用例显存对账不消费该输出）
        del g_h
        torch.npu.empty_cache()

        # ---- SGD（手写——无 optimizer 状态）----
        with torch.no_grad():
            # 专家权重（视图 home 行）——reduce 后 g* 已是全量梯度
            p1 = views[0][0][:epn].view(epn, Hp, H)
            p3 = views[0][1][:epn].view(epn, Hp, H)
            p2 = views[0][2][:epn].view(epn, H, Hp)
            # full-dW：归并结果在 g*[:epn]（home 段——reduce 后即全量）
            p1.copy_(p1 - args.lr * g1[:epn].to(torch.bfloat16))
            p3.copy_(p3 - args.lr * g3[:epn].to(torch.bfloat16))
            p2.copy_(p2 - args.lr * g2[:epn].to(torch.bfloat16))
        torch.npu.synchronize()
        tb = time.perf_counter()
        losses.append(loss.item())
        # 每步末设备占用快照（torch 池口径 + driver 口径双轨）
        # driver 口径（真实占用）+ torch 池口径（allocated/reserved）双轨
        free, total = torch.npu.mem_get_info()
        # 每步末快照：loss/耗时 + 双口径占用（对账表的时间轴锚点）
        mem_stat.log(
            "train",
            "note",
            f"step{step}_snapshot",
            note=f"loss={loss.item():.4f} fwd={tf - ts:.2f}s bwd={tb - tf:.2f}s | "
            f"device_used={(total - free) / 230:.2f} GiB / {total / 230:.0f} GiB | "
            f"torch_allocated={torch.npu.memory_allocated() / 2**30:.2f} GiB "
            f"reserved={torch.npu.memory_reserved() / 2**30:.2f} GiB",
        )

    # ★ 跨运行对拍导出（功能验证）：种子确定性下，pull 与 torch 两模式的
    # 终态梯度/权重应一致（差异仅在 reduce 协议的舍入路径）。
    # 背景：in-process --verify 的 all_reduce 在 EB 与 MoonEP 通信混用下触发 A5 已知
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
                "w1": views[0][0][:epn].detach().clone().cpu(),
                "w3": views[0][1][:epn].detach().clone().cpu(),
                "w2": views[0][2][:epn].detach().clone().cpu(),
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
    # 逻辑资源释放（幂等；缺省不销毁子组/EB——池化保活到进程退出，conftest
    # 硬退跳过 GC 析构的分钟级停顿；视图/plan 张量的使用须先于本调用）
    buf.destroy()
    dist.destroy_process_group()


def main(argv=None):
    # CLI：默认形状 = 用户指令值（E=224 按理论公式口径——见文件头说明）
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--K", type=int, default=16)
    # H=token/专家 FFN 输入维；Hp=中间维（三矩阵均 H×Hp）
    ap.add_argument("--H", type=int, default=3584)
    ap.add_argument("--Hp", type=int, default=3072, help="专家 FFN 中间维")
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
    # 堆不做粗放预留：布局精确 + 双尾部 + 2MB 桶
    # 余量已保证 heap_bytes ≥ heap_need（历史 1.0/1.05/1.2 对照实测：1.2
    # 拍脑袋值被二分证伪，1.0 通过——HCCL 无隐藏对齐开销，结论留档于此）
    # lr/steps：演示训练参数（本用例主目的为显存对账）
    ap.add_argument("--lr", type=float, default=1e-2)
    # 窗口槽数（层间流水双槽发布区翻倍——对照理论时注意；映射构造参数 slots）
    ap.add_argument(
        "--slots",
        type=int,
        default=1,
        help="权重窗口槽数（1=单槽；2=层间流水双槽——发布区翻倍）",
    )
    # dW dtype（A1 口径 bf16——staging 减半；fp32 对照口径；映射构造参数
    # grad_dtype——构造期定型槽区布局与 reduce 期梯度 dtype）
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
        "qrun 下 EB 通信+all_reduce 混用有 EI0006 已知风险，"
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
