<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AICPU Tensor 参数与 Profiling

独立 AICPU kernel 的 `interface` 模式支持直接传入 NPU `torch.Tensor`。
框架在下发时上报实际 shape、dtype、storage format 和输入输出角色，任务类型为 `AI_CPU`。
采集仍由外部工具启动；元数据上报不读取 Tensor 内容，也不额外同步设备。

## 使用前准备

使用包含 AICPU Tensor profiling 的安装包，并重新编译 `interface` 动态库。此路径已在 Ascend 950PR、CANN 9.2 验证。

接口定义见 [@aicpu_kernel](../api/aicpu/programming/aicpu-kernel.md)、[launch](../api/aicpu/runtime/launch.md) 和 [ProfileSpec](../api/diag/profiling.md#aicpu-接口配置)。

## 运行与验证

先加载 CANN 环境，并使用包含 CANNBotDSL、PyTorch 和 TorchNPU 的 Python 环境。
下面的示例面向 `dav-3510`，保存为 `aicpu_copy.py` 后运行：

```python
import torch
import torch_npu  # noqa: F401
from cannbotdsl.aicpu import I32, U32, aicpu_kernel, current_raw_stream, gm_in, gm_out
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel


class Args:
    x: gm_in(I32)
    out: gm_out(I32)
    size: U32


@aicpu_kernel
def copy(a: Args):
    for i in range(a.size):
        a.out[i] = a.x[i]
    return 0


torch.npu.set_device(0)
device = torch.npu.current_device()
compiled = compile_aicpu_kernel(
    copy, workdir="./copy_build", launch_mode="interface", npu_arch="dav-3510"
)
x = torch.arange(32, dtype=torch.int32, device=f"npu:{device}")
out = torch.zeros_like(x)
compiled.launch(current_raw_stream(device), x=x, out=out, size=x.numel())
torch.npu.synchronize()
torch.testing.assert_close(out.cpu(), x.cpu())
print("AICPU copy passed")
```

执行 `python aicpu_copy.py`，预期打印 `AICPU copy passed`。

需要在 profiling 中解析并上报输入输出 shape 时，应像上例一样直接传入 Tensor 对象，
例如 `x=x, out=out`，而不是 `x=x.data_ptr(), out=out.data_ptr()`。
裸指针只携带内存地址，不包含 shape 信息，框架无法从地址解析 Tensor 的实际形状。

`gm_in` 字段默认作为输入，`gm_out` 字段默认作为输出，名称和算子类型默认使用函数名。
示例中的同步用于等待执行结果并校验数值，不是上报所需操作。

## 自定义名称与参数角色

将上例的 `copy` 定义替换为以下代码，其他代码保持一致：

```python
import cannbotdsl as cbd


@aicpu_kernel(
    profile=cbd.ProfileSpec(
        name="copy.custom",
        op_type="Copy",
        inputs=("x",),
        outputs=("out",),
    )
)
def copy(a: Args):
    for i in range(a.size):
        a.out[i] = a.x[i]
    return 0
```

角色列表引用 `Args` 的 GM 字段名，例如 `x`，不是结构体形参名 `a`。
显式角色列表替代默认推导；只填写 `name` 或 `op_type` 时不报告 Tensor 列表。
放入 `inouts` 的字段同时作为输入和输出上报。

## 采集与查看

```bash
msprof --application="python aicpu_copy.py" --task-time=on --ai-core=on --output="./profile"
```

在 `op_summary_*.csv` 中查看对应 `AI_CPU` 任务的名称、类型，以及 Input/Output 的
Shapes、Data Types、Formats 字段。上述默认示例上报 shape `32`、dtype `INT32`、format `ND`。
报告字段的展示由采集工具版本决定，通用说明见 [Profiling Tensor 信息上报](../api/diag/profiling.md)。

## 常见问题

- Tensor 必须在同一 NPU 上且连续，dtype 与 GM 元素类型一致；调用前将该设备设为当前设备。
- `stream_handle` 必须来自该设备当前 TorchNPU stream 的 `current_raw_stream()`。
  使用非默认 stream 时，在对应的 `torch.npu.stream(stream)` 上下文中获取句柄并调用。
- Tensor 描述最多包含 8 个维度，维度值须能用 uint32 表示。属性查询失败或描述无法表示时，在下发前报错。
- 裸指针调用仍有效，但裸指针没有 Tensor 描述；显式角色选择的 GM 字段必须传入 Tensor。
- 旧动态库仍可执行，获得 Tensor 上报能力需要重新编译。
  `legacy` 参数块地址模式保持原有行为，不自动解析参数块中的 Tensor。

AICPU 编程入口见 [AI CPU 与调度计划](aicpu.md)。
