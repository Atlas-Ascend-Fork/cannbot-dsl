<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AIC / AIV 异常诊断与 Python 源码定位

DSL 在 NPU 执行时默认启用 AIC / AIV 异常监测。发生异常后保存 kernel、host 参数、PC 和寄存器，并生成终端日志及 JSON / LOG 报告。异常回调还会尝试保存当次输入 Tensor，供排查和准备单算子复现数据。当前不采集运行时 Python 调用栈。

## 使用前准备

安装包需包含 AIC / AIV 诊断、异常输入保存及 Python 来源映射能力。Python 定位需要带行号和来源映射的编译产物；旧产物需重新编译。异常输入保存还要求产物包含 Tensor 参数布局。

## 配置方法

监测无需额外开关。若需要在报告中查看生成的 AscendC 代码行，应在编译算子包或在线编译前设置：

```bash
export CANNBOTDSL_AIC_ERROR_DEBUG=1
export CANNBOTDSL_AIC_ERROR_DEBUG_DIR="$PWD/aic-reports"
python your_compile_or_jit_script.py
```

`CANNBOTDSL_AIC_ERROR_DEBUG` 默认关闭，设为 `1` 才添加 `-lineinfo`；它控制编译行号信息和 Python 来源映射，不控制异常监测。运行时打开该开关不能为已编译的旧包补充行号，需要重新编译。

`CANNBOTDSL_AIC_ERROR_DEBUG_DIR` 控制诊断产物和异常报告目录，默认是当前目录下的 `.dump/aic_error`。建议使用可写、运行期间不会被清理的绝对路径；配置后再启动进程。

在异常回调时自动尝试读取输入 Tensor，见[异常输入 Tensor 保存](#异常输入-tensor-保存)；若需要执行前的原始输入，见[输入 Tensor 快照](input-tensor-dump.md)。

## 输出文件

```text
aic-reports/
  build-*/                    # 诊断所需源码、manifest 和二进制归档
  artifact-index/              # 编译产物索引
  process-<PID>-*/
    events.jsonl               # 原始异常事件
    events.jsonl.registers.jsonl
    snapshot-<sequence>.json    # 回调现场及恢复报告所需信息
    error-*.native.log         # 回调阶段生成的基础日志
    error-*.json               # 结构化诊断报告
    error-*.log                # 可读诊断报告
    inputs-<sequence>.json     # 异常时的逐参数复制结果
    inputs-<sequence>-arg-<index>.bin
    inputs-<sequence>-arg-<index>.pt
    inputs-<sequence>.metadata.json
```

上图列出可能存在的诊断材料，不表示正常执行一定会创建所有目录。材料的持久化时机取决于安装版本；采用异常时落盘的版本在捕获异常后才创建相应材料。

不同进程分别保存到 `process-<PID>-*`，同一事件通过 `sequence` 对应。二进制也可能在加载阶段归档到 `loaded-*` 目录；分析或搬迁时应保留整个诊断根目录。

## 阅读报告

先看标准 `error-*.log`，再按需要查看 JSON：

| 字段 | 阅读方法 |
| --- | --- |
| PID、进程信息、Callback time | 确认哪个进程、何时收到异常；多进程任务结合 rank 判断先后 |
| Task binding | `matched` 表示关联到精确提交记录；`pending` 表示尚未匹配；`ambiguous` 表示存在歧义 |
| Kernel / Runtime kernel | 前者来自关联的 DSL 提交，后者来自 CANN 异常接口；只有函数名不能证明已关联到当次参数 |
| Arguments | 查看当次 host 参数及可解析的 Tensor 地址、shape、dtype 等描述 |
| Cores / Core | 查看异常核、原始寄存器、startPC/currentPC 及换算后的 ELF 地址 |
| Source | 查看 PC 对应的生成 AscendC 文件、行号和上下文；内联调用位置见 Source inline chain |
| Python DSL source | 查看对应的 Python DSL 位置和编译时源码快照；内联调用位置见 DSL inlined from |
| Unavailable | 说明该项信息缺失的具体原因，例如缺少 lineinfo、无法唯一选择入口或符号化工具不可用 |

`Source` 展示生成的 AscendC 代码，不是原始 Python DSL 行号，也不直接判定算法错误根因。源码来自编译时嵌入的 manifest，因此构建机上的 `source_path` 在运行机上不存在，不妨碍读取已保存的源码。行号解析还需要匹配的设备二进制和 CANN 符号化工具。

`Python DSL source` 在有效映射存在时补充原始 Python 位置；字段含义、缓存与 AOT 用法见下文的[Python 源码定位](#python-源码定位)。

混合 AIC / AIV kernel 按异常核类型选择对应入口。入口或 PC 归属不能唯一确定时，报告说明原因，不把候选代码当作确定的报错行。

## 只有基础日志或报告缺失时

异常回调会向 stderr 打印 `[CANNBotDSL][exceptionCallback]` 日志，无需打开 lineinfo 开关。按同一 PID 和 `sequence` 检查：

1. `stage=captured`：确认回调已收到异常，并查看 `binding`、参数数量和寄存器数量。
2. `stage=checkpoint`：检查 `snapshot`、`native_log` 和 `notify`。`notify=no_snapshot` 表示快照保存失败，不是没有捕获异常。
3. `checkpoint_write` / `checkpoint_directory`：检查失败操作、路径和 errno。目录被删除时会尝试重建并重试，仍失败时尽力向 stderr 输出原始现场。
4. `binding_lookup` / `binding_registration`：关联失败时检查接口状态及提交身份。`binding_candidate` 只供排查，不能作为已匹配的 kernel 或参数。

输入流与实际任务流不同时，`launch_identity` / `binding_identity` 会记录两者及查询状态。匹配仍要求 device、实际 stream、task 和提交线程一致，不通过忽略 stream 来猜测。

回调先保存快照，独立 CPU 报告进程再解码；提交进程退出后仍可生成标准报告。若报告进程也被结束，可用现有快照恢复，无需重跑算子：

```bash
python -m cannbotdsl.core.diag.aic_error_recovery /path/to/process-PID-directory
```

恢复失败时查看 `snapshot-*.recovery-error`。只有 `events.jsonl` 不能说明标准报告已经生成；只有 `.native.log` 时应检查快照和恢复状态。

## 诊断范围与限制

- 诊断不会为正常调用额外同步 stream 或 device，也不读取 plog；报告解码在异常路径执行。
- task 记录、参数和事件队列有容量上限，截断或缺失会显式标记。不要用其他提交的参数补齐未匹配事件。
- Graph 中保留的是捕获提交时的参数；外部更新节点或 replay 不会重新执行 Python 侧记录。关联依赖运行时提供的任务身份接口。
- 进程内 device 是逻辑编号；多进程分析应结合 PID/rank 和时间，不把后续通信超时自动判为首发故障。
- 快照完成前进程被强制终止、目录不可写或存储故障时，无法保证报告完整。分析前保留日志、快照和对应二进制。

## 异常输入 Tensor 保存

该功能在收到 AIC / AIV 异常回调后，读取精确关联 kernel 的输入 Tensor，保存原始字节，再由独立 CPU 报告进程生成 `.pt` 和参数说明。正常执行不复制 Tensor、不写输入文件；不采集 Python 调用栈。

### 保存配置

异常输入保存始终启用，无需配置开关。启动运行进程前可配置日志目录：

```bash
export CANNBOTDSL_AIC_ERROR_DEBUG_DIR=/absolute/path/aic-reports
python run_operator.py
```

异常输入保存不要求打开 `CANNBOTDSL_DUMP_ASCENDC`，也不改变后者的行为。若需要 AscendC 行号，在编译算子包或在线编译前另设 `CANNBOTDSL_AIC_ERROR_DEBUG=1`；Tensor 读取本身不依赖 lineinfo。

算子二进制必须带有 Tensor 参数布局 manifest；已有此布局的包无需为异常输入保存重新编译。旧产物缺少布局时会记录 `tensor_schema_unavailable`，不会猜测设备地址和长度。

### 输出与读取

文件放在同一异常的 `process-<PID>-*` 目录中，用 `sequence` 对应 `snapshot-<sequence>.json` 和 `error-*.json/log`：

| 文件 | 内容 |
| --- | --- |
| `inputs-<sequence>.json` | 原生回调的逐参数复制结果、地址、字节数和错误码 |
| `inputs-<sequence>-arg-<index>.bin` | 回调中已读出的设备存储字节；即使原进程立即退出仍可保留 |
| `inputs-<sequence>-arg-<index>.pt` | 报告进程生成的 CPU Tensor |
| `inputs-<sequence>.metadata.json` | 参数顺序、标量原始字节/可解码值、shape、strides、dtype、format、保存状态和文件名 |

例如读取第一个输入：

```python
import json
from pathlib import Path
import torch

process = Path("/absolute/path/aic-reports/process-12345-example")
metadata = json.loads((process / "inputs-1.metadata.json").read_text())
for arg in metadata["arguments"]:
    if arg.get("status") == "saved_pt":
        tensor = torch.load(
            process / arg["pt_file"], map_location="cpu", weights_only=True
        )
        print(
            arg["index"],
            arg["shape"],
            arg["dtype"],
            arg["format"],
            arg["pt_representation"],
        )
        print(tensor)
    elif arg.get("kind") != "tensor":
        print(
            "scalar or pointer:",
            arg["index"],
            arg.get("value"),
            arg.get("host_bytes_hex"),
        )
```

`pt_representation=logical_tensor` 表示 ND 逻辑 Tensor，保存 shape 和非负 strides，可用于准备单算子输入。`physical_tensor` 表示私有格式的物理布局，不能直接当成逻辑 ND 输入；需结合原算子的格式要求重建。低位宽或未映射到 PyTorch dtype 的数据保存为 `raw_storage_bytes`（uint8），原始 dtype/format 仍在 metadata 中，不伪装成已还原的逻辑 Tensor。

读取后仍需使用原算子的代码、编译配置及调用约定。动态标量按原参数顺序传入；别名关系可依据原地址判断，但不会自动将分别保存的 `.pt` 恢复为共享存储。

### 数据含义与限制

保存的是**异常回调时仍能读取的值**，不是执行前快照。仅保存指针不会延长 Tensor 生命周期；原地修改、其他 stream 写入、释放或内存池复用，都可能使数值不同于原始输入。metadata 固定标记 `original_inputs_guaranteed=false`，不能仅凭文件存在认定一定可复现。若必须保证原始输入，应使用调用前的[输入 Tensor 快照](input-tensor-dump.md)。

此实现读取编译器标记为 `input`、`inout` 或 `unknown` 的 Tensor 参数；纯 `output` / `unused` 参数仅保留描述。 <!-- codespell:ignore inout -->标量保留 host ABI 字节并尽可能解码。无类型指针、workspace 以及 kernel 内部 UB/L1 数据不会当作 Tensor 猜测读取。

只有 `Task binding=matched` 才读取当次参数。`pending`、`ambiguous`、参数截断、布局缺失、负 stride、跨度越界和设备复制失败都会保留具体状态。地址范围检查仅能确认仍处于某个可查询的设备分配范围，不能证明仍属于原 Tensor。设备错误后的复制能力取决于运行时和设备状态。

回调先保存基础快照和原生日志，再进行 D2H 复制；复制失败不丢弃原诊断。原进程退出后，报告进程只能转换已保存的字节，不能重新读取原进程设备内存。若进程在复制结束前被强杀，或存储不可写，仍可能缺少数据。报告环境没有 PyTorch 或发生非存储类转换失败时，保留 `.bin` 与原因，可修复环境后离线恢复报告。

异常输入采集没有人为的总字节数或单 Tensor 大小限额；旧的 `CANNBOTDSL_AIC_ERROR_DUMP_INPUTS_MAX_BYTES` 不再生效。原生 D2H 使用 1 MiB 分块缓冲，报告进程通过内存映射读取 `.bin`，避免先把大文件整体复制到多份 CPU 缓冲。仍保留地址范围、shape/stride 和整数溢出校验。此功能不额外执行 stream/device synchronize，不重置设备；D2H 本身是阻塞调用。

原始字节、`.pt` 或 metadata 的写入、关闭、发布失败时（包括空间不足 `ENOSPC` 和配额用尽 `EDQUOT`），清理**当次异常**已经保存的输入 `.bin`、`.pt`、临时文件及过期 metadata，并停止把该次输入标为可用。基础快照、诊断日志及其它异常的数据保留。释放空间后尽力写入 `storage_failed`、错误码和 `cleanup_complete`；无法删除的文件会记录清理失败原因。设备读取失败等非存储错误不会删除其它已成功读取的参数。

## Python 源码定位

先查看异常报告中的 `Source`，确认生成的 AscendC 位置，再查看 `Python DSL source`。例如：

```text
  Python DSL source (compile-time statement origin; mapped):
    kernel.py:12:12 in read_value
    >    12 |     return x[0]
    DSL inlined from kernel.py:20:45 in copy_kernel
```

示例中的 Python 位置指向 `read_value` 内的访问表达式，`DSL inlined from` 表示编译时记录的内联调用位置。具体文件和行列随用户程序变化；来源位置用于追溯操作，不直接判定错误根因。

同一信息保存在 `error-*.log` 和 `error-*.json` 中。JSON 的 `kernels[].python_source` 包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `status` | 映射结果，见下表 |
| `reason` | 来源不可用或不完整时的原因 |
| `locations` | 已确认或候选的来源位置 |
| `filename`、`function`、`line`、`column` | 编译时的文件、函数和行列；行列从 1 开始 |
| `context` | 编译时保存的源码上下文 |
| `callers` | 已记录的内联调用位置，可包含嵌套调用 |
| `source_sha256` | 源码快照摘要，用于区分不同版本 |

除前三项外，其余字段位于 `locations` 的每个位置对象中。

| `status` | 阅读方法 |
| --- | --- |
| `mapped` | 存在一个完整且已确认的来源 |
| `partial` | 已保留有效来源，但部分来源或调用关系缺失 |
| `ambiguous` | 存在多个候选，不能确定唯一位置 |
| `unavailable` | 未取得有效映射，结合 `reason` 查看原因 |

报告目录由 `CANNBOTDSL_AIC_ERROR_DEBUG_DIR` 指定，默认 `.dump/aic_error`，不随 `CANNBOTDSL_DUMP_DIR` 改变。Python 定位信息随现有报告保存，不单独生成一套报告目录。

### 缓存与 AOT

带有来源映射的产物通过内存缓存、磁盘缓存或 AOT 保存加载后，仍保留对应的编译时快照。源码文件后续变化不会改写快照；旧产物没有映射时，报告保留已有 AscendC 信息并说明 Python 来源不可用。

- Python 定位需要先取得可靠的 PC → AscendC 位置；纯 SIMT 等场景也受这一条件约束。
- 来源丢失、快照损坏或候选不唯一时，报告保留可确认的信息，不根据相邻代码猜测位置。
- 内联调用关系来自编译过程，不是运行时 Python 提交栈；报告中的提交栈状态仍为 `disabled by design`。

## 与其他 dump 功能的关系

| 功能 | 触发方式 | 产物与目录 |
| --- | --- | --- |
| 本文的 DSL 异常输入保存 | AIC / AIV 异常回调，默认启用 | `.bin`、`.pt` 和 metadata，使用 `CANNBOTDSL_AIC_ERROR_DEBUG_DIR`。 |
| [执行前 Tensor 快照](input-tensor-dump.md) | `CANNBOTDSL_DUMP_ASCENDC=1`，Host 调用前 | CPU `.pt` 和 metadata，使用 `CANNBOTDSL_DUMP_DIR/inputs/`。 |
| [CANN 异常 Tensor 导出](../api/diag/exception_dump.md) | 支持该能力的版本配置 CANN 异常 dump 场景 | `.custom.<timestamp>`，使用 CANN 返回目录，交给配套工具解析。 |

三者的保存时机、格式和目录各自独立。需要原始输入时提前开启执行前快照；异常回调保存的数据不保证仍为执行前原值。编译配置见[编译选项与产物观察](compiler-options.md)。
