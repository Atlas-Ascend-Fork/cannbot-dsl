<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Tiling 结果采集

通过统一开关 `CANNBOTDSL_DUMP_ASCENDC=1` 同时归档 ASC 源码、输入快照和运行时 tiling 结果。
设置 `CANNBOTDSL_DUMP_DIR` 指定输出目录，未设置时使用当前目录下的 `.dump`。
无需算子注册回调或提供专用 plan，MatMul、Attention 和普通 DSL kernel 使用同一条采集路径。

```bash
export CANNBOTDSL_DUMP_ASCENDC=1
export CANNBOTDSL_DUMP_DIR=/absolute/path/dsl_dump
python example.py
```

关闭开关时不创建 tiling writer，也不复制参数或写入文件。配置在 callable 创建或加载时读取，
改变配置后应重新创建或加载对象。旧的 `CANNBOTDSL_DUMP_TILING` 不再生效。

## 使用前准备

安装包和算子产物需包含通用编译层 tiling 元数据与诊断 v3 入口。旧产物开启导出后提示缺少入口时，应重新编译。先加载 CANN 环境，再运行实际算子程序；该功能无需注册算子专用回调。

## 输出内容

每次 kernel 提交前写入一个 JSON 和一个同名 TXT 摘要，描述同一份分块和分核结果。
文件存在不代表设备已经执行成功。

| 字段 | 含义 |
| --- | --- |
| `operator` / `kernel` | Host 入口和 kernel 名称 |
| `block_dim` / `launch` | 逻辑 block 数、启动类型、三维 block/thread 配置及动态 UB 字节数 |
| `arguments` | Tensor 名称、用途、shape、dtype、布局，以及可解码的标量值 |
| `tiling.tiles` | 按 Tensor 和几何信息归并的分块、父范围、块数、最后一块大小 |
| `tiling.loops` | 循环边界、嵌套与条件，以及可证明的逐 block 工作量 |
| `tiling.buffers.rings` | RingBuffer 深度、槽布局及重复声明数量 |
| `tiling.buffers.allocations` | 按空间、容量和对齐归并的分配，保留每次分配的规划偏移 |
| `tiling.buffers.planned_address_span_bytes` | 每种空间的最大规划结束地址，复用空间不重复相加 |
| `tiling.notes` / `unavailable` | 未解析、采集截断或元数据缺失等限制，仅有问题时出现 |

输出不包含 `version`、`stage`、时间戳、进程线程号、流地址、产物标识、ABI 排列和原始参数字节。
`host_bytes_hex` 属于内部参数解码信息，不属于 tiling 结果。
Tensor 指针和独立指针参数的地址均不输出，也不读取其指向的数据。
内部编译元数据仍有格式校验，该内部协议不作为用户 JSON 字段。

参数名来自 kernel 形参。嵌套参数直接显示元素路径，如 `tiles["sizes"][0]`；缺少名称时使用 `arg[0]`，
多值参数使用 `.component[0]` 区分。Tensor 的 `role` 来自编译器内存访问分析，取值为
`input`、`output`、`inout`、`unused` 或 `unknown`，不按变量名猜测。 <!-- codespell:ignore inout -->
物理布局与逻辑布局相同时不重复显示。标量整数保留完整精度，非有限浮点数使用
`nan`、`inf`、`-inf` 字符串，保持合法 JSON。未知类型保留不可解码原因，不输出原始字节。

## 分块、尾块与分核

例如输入长度为 35、每块 8 个元素、启动 3 个逻辑 block：

```python
for block in range(cb.get_block_idx(), (x.shape[0] + 7) // 8, cb.get_block_num()):
    tile = cb.tile_slice(x, (8,), (block,))
```

关键输出如下，省略参数和缓冲信息：

```json
{
  "block_dim": 3,
  "tiling": {
    "tiles": [
      {
        "tensor": "x",
        "memory_space": "gm",
        "dtype": "i64",
        "parent_shape": [35],
        "nominal_shape": [8],
        "last_tile_shape": [3],
        "tile_counts": [5]
      }
    ],
    "loops": [
      {
        "index": "i0",
        "mapping": "grid_stride",
        "lower": "block_id",
        "upper": 5,
        "step": 3,
        "block_ranges": [
          {"blocks": [0, 1], "lower": "block_id", "upper_exclusive": 5, "step": 3, "iterations": 2},
          {"blocks": [2, 2], "lower": 2, "upper_exclusive": 5, "step": 3, "iterations": 1}
        ]
      }
    ]
  }
}
```

循环上界不包含端点，`blocks` 的两个端点均包含在范围内。上例 block 0 处理块 0、3，
block 1 处理块 1、4，block 2 处理块 2。只有连续 block 的边界符合相同线性规律、步长与
迭代次数一致时才归并；不能证明时保留不同范围。
`iterations` 是一次进入该循环的次数，不乘外层循环次数；`within` 保留外层循环和分支条件。
未解析条件不会被当成真，不会生成猜测的每核工作量。逻辑 block 不是物理 NPU 卡号或执行轨迹。

已解析的值直接输出为数值。循环索引使用 `i0` 等名称，block 索引使用 `block_id`。
其他符号表达式以函数表示，如 `addi_64(i0, 1)`；运算名称和位宽保留有符号、截断和溢出语义，
不能随意按数学整数重写。设备数据、循环携带值或暂不支持的操作显示为 `unknown`，并说明原因。
表达式图、节点编号及完整 IR 类型不再展开。

`tiles` 汇总几何声明，不是实际执行的 tile 列表或每次访问的坐标轨迹。
父范围、名义块、尾块或 Tensor 不同的记录不会归并；重复声明以 `occurrences` 表示。
该计数不是运行次数，也不意味着不同分支同时执行。一般 View 窗口和内部访问轨迹不单独展开。
尾块由父范围和名义块推导，不能求值时明确保留未知状态。

## 缓冲规划

RingBuffer 相同槽布局归并为一项，并保留槽数量；不同深度仍是不同记录。
同规格 RingBuffer 用 `occurrences` 表示声明数量，不能误作深度。
槽布局与内存规划属于不同编译阶段，不根据内部编号强行建立一一对应关系。

分配按空间、容量和对齐归并，`offsets` 保留所有规划偏移，包括地址复用时的重复偏移。
未规划分配用 `unplanned_count` 表示，手动绑定空间单独列出。
`planned_address_span_bytes` 是编译器规划的地址范围，不是硬件总容量或实测峰值。

## 文件关联及执行路径

```text
dsl_dump/tiling/
  process-<PID>-<session>/
    <operator>-<identity_hash>/
      call-00000001/
        0000-<kernel>.json
        0000-<kernel>.txt
        0001-<kernel>.json
        0001-<kernel>.txt
```

进程、callable 实例、编译变体和调用序号由目录隔离，文件名前缀表示本次调用中的提交序号。
这些信息不在 JSON 重复。写入成功后终端显示 JSON 和摘要路径。

支持直接 JIT、`cb.compile()` 和 `cb.load()`。编译时无需开启 dump，产物始终保留采集元数据。
源码归档编译沿用既有缓存绕过策略；诊断缓存二进制可先保存产物，再开启开关通过 `cb.load()` 加载。
Graph capture 执行 Host 提交时生成记录，单独 replay 不重复执行 Host tiling，也不重复写文件。
独立 OpKit/ACLNN C++ runtime 不经过此入口，不在本开关范围内。

## 限制与故障处理

- 采集来自通用编译层和最终下发参数，不要求算子提供 plan，不推测调度器名称、候选或实际执行轨迹。
- 每阶段最多采集 16384 个表达式节点，最多枚举 512 个逻辑 block。达到限制时显示原因，仍保留循环公式。
- 元数据缺失、损坏或复用未经验证的 ASC 源码时，保留参数和不可用原因，不沿用其他调用的解析值。
- 缺少诊断 v3 入口的旧产物开启 dump 后提示重新编译；关闭开关仍可正常运行。
- 开启后有参数解析、序列化和文件 I/O 开销；统一开关启用的输入快照也有设备到 CPU 的复制开销。
- 文件采用临时文件加原子替换。先写 JSON，再写 TXT，摘要失败时已生成的 JSON 保留。
- 普通写入失败输出 `tiling_dump failed; kernel execution continues` 类诊断，停止本次调用的后续写入，
  kernel 继续执行，下次调用重试。用户中断和 `SystemExit` 在提交前传播，阻止当前和后续提交。
- 现有 AIC 异常诊断保持独立，其内部 ABI 数据不属于本功能的公开 tiling 输出。

`CANNBOTDSL_TILING_DUMP_DIR` 属于 OpKit 编译调试，不能与本功能的运行时结果开关混用。
