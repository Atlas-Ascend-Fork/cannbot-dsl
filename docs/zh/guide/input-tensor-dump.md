<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# 输入 Tensor 快照

在算子执行前，将调用方传入的 Tensor 保存为 CPU `.pt` 文件，并在 `metadata.json` 中记录原始 shape、dtype 和 format，用于检查输入和复现精度问题。

[异常输入 Tensor 保存](aic-error-debug.md#异常输入-tensor-保存)在异常回调时读取数据；本功能保存调用前快照，不需要先发生 AIC / AIV 错误。

## 开启方法

使用包含输入快照能力的 CANNBotDSL、PyTorch 和 TorchNPU 环境。输入快照与 AscendC 源码归档、tiling 结果共用开关，默认关闭。在启动编译或运行进程前设置：

```bash
export CANNBOTDSL_DUMP_ASCENDC=1
export CANNBOTDSL_DUMP_DIR="$PWD/operator-dump"
python your_program.py
```

将 `your_program.py` 替换为实际算子程序。只设置输出目录不会开启快照。配置在可调用对象创建或加载时读取；修改后需重新创建或加载对象，建议重启运行进程。

| 配置 | 说明 |
| --- | --- |
| `CANNBOTDSL_DUMP_ASCENDC` | 设为 `1` 开启源码归档、输入快照和 tiling 导出；设为 `0` 或不设置时关闭。 |
| `CANNBOTDSL_DUMP_DIR` | 普通 dump 根目录，默认当前目录下的 `.dump`；输入快照使用其 `inputs/` 子目录。 |
| `CANNBOTDSL_DUMP_INPUT_NAMES` | 可选，按 Python 参数名筛选，多个名称以逗号分隔。 |

`CANNBOTDSL_AIC_ERROR_DEBUG` 另行控制编译行号和 Python 来源映射，不控制本功能或默认启用的异常监测。

## 保存范围与文件组织

```text
operator-dump/inputs/<入口名>-<算子身份哈希>/
  process-<PID>/
    call-<时间戳>-<唯一后缀>/
      000-x-32x64-float16.pt
      001-y-32x64-float16.pt
      metadata.json
```

不同算子身份、进程及调用分别保存，避免覆盖。`.pt` 与 metadata 使用临时文件加原子替换发布。

默认保存所有传入的 Tensor 参数，包括 TensorList 展开后的元素、输出缓冲区和原地更新缓冲区。保存的是调用前数据；输出缓冲区若尚未初始化，其快照不代表有效计算结果。

仅保存特定参数时：

```bash
export CANNBOTDSL_DUMP_INPUT_NAMES=x,xs
```

`xs` 选择整个 TensorList，`xs[1]` 只选择该元素。未匹配名称记录在 `unmatched_parameters` 中并告警。

## 读取数据和元信息

将 `directory` 替换为实际 `call-*` 目录：

```python
import json
from pathlib import Path
import torch

directory = Path("/absolute/path/to/call-directory")
metadata = json.loads((directory / "metadata.json").read_text())
for item in metadata["tensors"]:
    print(item["name"], item.get("shape"), item.get("dtype"), item.get("format"))
    if item["status"] == "saved":
        value = torch.load(
            directory / item["file"], map_location="cpu", weights_only=True
        )
        print(value.shape, value.dtype)
    else:
        print(item.get("error"))
```

| 内容 | 阅读方法 |
| --- | --- |
| `shape`、`dtype`、`stride`、`storage_offset`、`device`、`format` | 原始 Tensor 描述；NPU format 记录格式 ID 及可获取的名称，CPU Tensor 记录 `torch.layout`。 |
| `saved_stride`、`saved_layout` | 保存后的 CPU 数据布局，可能不同于原始 NPU 布局。 |
| Tensor 的 `status=saved` | 该项数据已保存；其他状态结合 `error` 判断。 |
| 顶层 `status=complete` / `partial` | 前者表示选中 Tensor 及格式信息均保存成功；后者表示部分数据、格式或参数选择存在问题。 |
| PID、TID、rank、local_rank | 用于区分并发调用与进程，结合算子身份和目录关联。 |

`.pt` 保存逻辑数值。NPU 私有格式通过独立的 detached 值转换为 ND 再复制到 CPU，不修改原始 Tensor。加载后的 CPU Tensor 不具有原 NPU 私有存储布局；复现时按 metadata 和算子的输入要求重建格式。格式查询失败会记录不可用原因。

## 执行路径与限制

- 仅加载已有算子包也可保存输入；没有重新编译时不会补生成未保留的 AscendC 源码。源码归档和每次 kernel 提交前的 tiling 导出分别见[源码归档](ascendc-dump.md)与[Tiling 结果](tiling-dump.md)。
- 每次调用包含设备到 CPU 的阻塞复制和磁盘写入，会增加 Host 耗时并影响异步流水。测量性能时关闭本功能。
- NPU Graph capture 期间不执行 CPU 复制或同步，记录 `partial` 元数据和跳过原因。Graph replay 不重复执行 Python 侧快照逻辑；需要输入快照时，在 capture 之外保存。
- 当前支持 `torch.Tensor`。其他 provider、格式查询或文件写入失败会记录原因并告警，保留算子原本的结果或异常。目录不可写时 metadata 也可能无法落盘，应查看 stderr。
- 关闭时不执行本功能的元信息查询、数据复制和文件写入。普通 dump 与 AIC 诊断目录独立。
