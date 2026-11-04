# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MEMSTAT 设备显存统计。

对张量分配、对称堆区段和算子调用的显存变化输出统一格式的记录：

  [MEMSTAT] op=<算子> kind=<类别> what=<对象> [shape=..] [dtype=..] bytes=N (MiB)

记录类型包括 alloc、heap、ccl、delta 和 note。delta 在同步后读取
mem_get_info；heap 表示预分配对称堆中的逻辑区段。设置 MEMSTAT=1 开启输出。"""

import os

import torch

# 总开关：默认关闭；打点须显式 MEMSTAT=1 开启
ENABLED = os.environ.get("MEMSTAT", "0") != "0"

_TAG = "[MEMSTAT]"


def _fmt(bytes_):
    """字节值的人类可读后缀（MiB 为主——显存对账的惯用单位）。"""
    if bytes_ is None:
        return "?"
    return f"{bytes_} ({bytes_ / 2**20:.2f} MiB)"


def _enabled():
    # 运行期重读环境变量（测试可 monkeypatch）
    return ENABLED and os.environ.get("MEMSTAT", "1") != "0"


def log(op, kind, what, *, shape=None, dtype=None, bytes_=None, note=""):
    """单条显存事件打印（分配/堆段/窗/备注——统一格式便于 grep 汇总）。"""
    if not _enabled():
        return
    # 固定字段序：op/kind/what 在前，可选字段（shape/dtype/bytes/note）在后
    parts = [f"{_TAG} op={op} kind={kind} what={what}"]
    # 形状与 dtype（列表统一成紧凑形式）
    if shape is not None:
        parts.append(f"shape={list(shape)}")
    if dtype is not None:
        # torch dtype 去掉 "torch." 前缀（紧凑输出）
        d = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
        parts.append(f"dtype={d}")
    # 字节数 + MiB 双形式（对账表直接可读）
    if bytes_ is not None:
        parts.append(f"bytes={_fmt(bytes_)}")
    if note:
        parts.append(f"note={note}")
    print(" ".join(parts), flush=True)


def tensor_bytes(t):
    """torch 张量的字节数（视图按自身 shape×itemsize——统计口径一致）。"""
    if t is None:
        return None
    return t.numel() * t.element_size()


def log_tensor(op, what, t, *, note=""):
    """打印一个刚分配的 torch 张量（None 安全——可选输出如权重跳过）。"""
    if not _enabled() or t is None:
        return
    log(
        op,
        "alloc",
        what,
        shape=tuple(t.shape),
        dtype=t.dtype,
        bytes_=tensor_bytes(t),
        note=note,
    )


def free_bytes():
    """当前设备空闲显存（mem_get_info 反映 driver 真实占用，非 torch 池口径）。"""
    if not torch.npu.is_available():
        return None
    free, _total = torch.npu.mem_get_info()
    return free


class delta:
    """算子调用范围的设备显存差测量（上下文管理器）。

    用法：
        with mem_stat.delta("prefetch_weight", note="step0"):
            prefetch_weight(...)
    进出各取一次 mem_get_info，结束打印差值（正=净增）。首调含编译/窗注册
    的一次性分配；稳态调用差值≈0（缓存池复用）——两态都打，标注 step 序。
    """

    def __init__(self, op, note=""):
        # op/note 仅用于打印标注（无设备副作用）
        self.op = op
        self.note = note

    def __enter__(self):
        # 进入点快照空闲显存（None = 无 NPU——__exit__ 里安全跳过）
        self.before = free_bytes()
        return self

    # （__exit__ 见下——差值 = before - after，正数即净增占用）

    def __exit__(self, *exc):
        # 退出点快照 + 差值打印（异常路径也打印——便于定位失败前后的占用）
        after = free_bytes()
        if not _enabled() or self.before is None or after is None:
            return False
        # 差值 = before - after（正 = 净增占用；负 = 释放）
        d = self.before - after
        sign = "+" if d >= 0 else ""
        # 带符号原值 + 绝对值双形式（grep 汇总两边都好读）
        log(
            self.op,
            "delta",
            "device_free_delta",
            bytes_=abs(d),
            note=f"{sign}{d} B{' ' + self.note if self.note else ''}",
        )
        return False
