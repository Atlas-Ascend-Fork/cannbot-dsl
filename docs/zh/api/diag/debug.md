<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# debug（硬件流水消融）

## 产品支持情况

- Ascend 950PR：已验证。其他产品未验证。
- 需要包含 `cannbotdsl.debug` 流水消融能力的 CANNBotDSL 安装包。

## 功能说明

`cannbotdsl.debug` 是编译期装饰器，在 kernel 或设备端 jit 作用域内屏蔽选定流水的数据指令，用于性能分析。循环、分配、标量控制和显式同步继续保留。

消融后的计算结果可能失真。先验证原始算子精度，再对比性能变体；耗时差受流水重叠、争用与编译优化影响，不能直接解释为某条流水的独立耗时。

## 函数原型

```python
def debug(options: str): ...
```

通过 `import cannbotdsl as cbd` 后调用 `cbd.debug(...)`；也可使用 `from cannbotdsl.utils.pipe_ablation import debug`。`cannbotdsl.utils.debug` 是原有日志接口，不是本装饰器。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `options` | 输入 | 编译期 Python 字符串。使用 `"disablepipes='mte2,mte3'"` 屏蔽选定流水，或 `"enablepipes='mte2'"` 仅保留选定流水的数据指令。两种选项互斥，无默认值。 |

选项名称为小写 `disablepipes` 或 `enablepipes`，右侧必须是字符串字面量。选择项用逗号分隔，大小写、首尾空白和重复项会规范化；空列表、空项、未知选项或无效流水组合会报错，不执行任意 Python 表达式。

| 流水选择项 | 说明 |
| --- | --- |
| `MTE1`、`MTE2`、`MTE3`、`M`、`FIXPIPE`、`V` | 选择对应的数据流水。 |
| `AIC.MTE1`、`AIC.MTE2`、`AIC.MTE3`、`AIC.M`、`AIC.FIXPIPE` | 限定 AIC 核。 |
| `AIV.MTE2`、`AIV.MTE3`、`AIV.V` | 限定 AIV 核；`AIV.M` 等其他组合无效。 |
| `cube`、`vector` | 分别是 `M`、`V` 的别名，支持核类型前缀。`cube` 不表示屏蔽整个 AIC 核。 |

不带前缀的选择项覆盖所有有效核类型，例如 `MTE2` 同时选择 AIC 的 GM→L1 和 AIV 的 GM→UB。`V` 按完整 SIMD VF 调用筛选。合法选择项在 kernel 中不存在时，报告命中数为零。

## 返回值说明

返回接收目标对象的装饰器。应用于已有 kernel launcher、kernel 类 launcher 或 `JitFunction` 时，返回附带配置的独立启动对象，原启动对象保留原配置；应用于供 kernel 类识别的 Python 方法声明时，在原函数上记录配置并返回该函数。

`cbd.debug(options)(existing_kernel)` 可用于创建独立性能变体，不表示执行 kernel，也不返回性能测量结果。

## 约束说明

- 装饰器放在 `@cbd.kernel` 或 `@cbd.jit` 上方，支持 kernel 函数、实例方法、kernel 类和设备端 jit helper。同一目标不能叠加多个流水配置。
- kernel 类配置作为入口方法的默认值，方法自己的配置完整覆盖类默认值。
- 内层 jit 配置完整覆盖外层；未配置的下层 jit 继承当前配置，返回后恢复调用者配置。正常兄弟调用保留原行为。
- `@host` 不支持流水配置，已配置的 jit 不能从 Host 域调用。
- 编译器拒绝失真值影响地址、长度、mask、控制流、配置或同步参数的情况，也可能保守拒绝无法证明安全的别名或未知副作用。
- 独立 SIMT、SIMT VF、while/轮询、间接 gather/scatter、设备原子访问、SSBUF、通信、调试输出和未知目标操作不支持。共享 VF 的作用域若要求不同的 V 去留，也会报错。
- 不支持与普通 Native 源码导出、包收集、`CANNBOTDSL_NATIVE_BINARY_MODE=require`、自定义 `CANNBOTDSL_CANNBOTIR_PIPELINE` 或 `CANNBOTDSL_REUSE_ASCENDC=1` 组合。
- 保留显式同步；M/FIXPIPE 分开保留时，编译器修正可证明的 `unit_flag`，无法证明的动态 flag 会拒绝编译。跨独立程序、多 stream 或通信的历史依赖不在静态分析范围内。

## 异常

| 情况 | 行为 |
| --- | --- |
| `options` 不是字符串，或目标对象类型不支持 | 抛出 `TypeError`。 |
| 配置语法错误、选项冲突、无效选择项或重复装饰 | 抛出 `ValueError`。 |
| 编译模式不兼容 | 在编译前检查时报错。 |
| 后端无法证明消融安全或目标操作不支持 | 编译失败，通过 `RuntimeError` 消息给出具体 MLIR 诊断。 |

## 调用示例

### 使用前准备

安装包需包含 `cannbotdsl.debug` 流水消融能力；旧包没有该接口时需升级。下例使用 Ascend 950PR、CANN 9.2 以及包含 PyTorch、TorchNPU 的环境。

```bash
export CANNBOTDSL_AUTO_BUFFER_SYNC=1
export CANNBOTDSL_NATIVE_BINARY_MODE=off
```

调试开关在启动进程前设置。旧的 `CANNBOTDSL_DROP_PIPES` 和 `CANNBOTDSL_KEEP_PIPES` 不再生效。

### 对比正常路径与消融路径

将以下完整示例保存为 `pipe_ablation.py`。默认路径验证 copy 数值；`--drop` 创建 MTE2/MTE3 消融变体，只检查其报告和运行完成情况。

```python
import argparse
import json

import torch
import torch_npu  # noqa: F401
import cannbotdsl as cbd


@cbd.kernel
def copy_kernel(x: cbd.Tensor, y: cbd.Tensor):
    local = cbd.RingBuffer(
        cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((32, 32)), depth=1
    ).next_write()
    cbd.mem_copy(local, x)
    cbd.mem_copy(y, local)


@cbd.host
def normal(x: cbd.Tensor, y: cbd.Tensor):
    copy_kernel[1](x, y)


def ablated_entry():
    @cbd.debug("disablepipes='MTE2,MTE3'")
    @cbd.kernel
    def measured(x: cbd.Tensor, y: cbd.Tensor):
        local = cbd.RingBuffer(
            cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((32, 32)), depth=1
        ).next_write()
        cbd.mem_copy(local, x)
        cbd.mem_copy(y, local)

    @cbd.host
    def run(x: cbd.Tensor, y: cbd.Tensor):
        measured[1](x, y)

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(0)
    spec = cbd.TensorSpec((32, 32), cbd.dtypes.float32)
    entry = ablated_entry() if args.drop else normal
    with cbd.compile(entry, spec, spec) as program:
        x = torch.randn(32, 32, device="npu:0")
        y = torch.zeros_like(x)
        program(x, y)
        torch.npu.synchronize()
        if args.drop:
            assert program.pipe_ablation is not None
            print(json.dumps(program.pipe_ablation, indent=2))
        else:
            torch.testing.assert_close(y, x)
            assert program.pipe_ablation is None
            print("normal copy passed")


if __name__ == "__main__":
    main()
```

依次执行：

```bash
python pipe_ablation.py
python pipe_ablation.py --drop
```

正常路径预期打印 `normal copy passed`；消融路径打印含 `kernels`、作用域和静态指令信息的 JSON。消融后的 `y` 不作为正常 copy 的输出使用。若要仅保留指定流水，将配置替换为 `enablepipes='MTE2'`，重新编译后对比报告。

### 局部 jit 消融

只分析某个 helper 时，将 `@cbd.debug(...)` 放在该 helper 的 `@cbd.jit` 上方。内层配置完整覆盖外层，返回后恢复；未配置的下层 helper 继承当前配置。

```python
import cannbotdsl as cbd


@cbd.jit
def copy_part(x, out):
    local = cbd.RingBuffer(
        cbd.MemLoc.UB, x.dtype, cbd.make_layout((32, 32)), depth=1
    ).next_write()
    cbd.mem_copy(local, x)
    cbd.mem_copy(out, local)


@cbd.debug("disablepipes='mte2,mte3'")
@cbd.jit
def measured_part(x, out):
    copy_part(x, out)


@cbd.kernel
def mixed(x: cbd.Tensor, out: cbd.Tensor, reference: cbd.Tensor):
    measured_part(x, out)
    copy_part(x, reference)
```

此片段用于说明作用域：`out` 属于消融路径，兄弟调用写入的 `reference` 仍是正常 copy。将其接入 Host 入口后，应校验 `reference`，不能因为启用消融而取消其他正常路径的精度检查。

## 报告与持久化

编译后的 `program.pipe_ablation` 返回静态报告，普通程序返回 `None`。配置参与缓存身份，通过 `program.save()` / `cannbotdsl.load()` 后仍保留性能变体标记与报告。

| 字段 | 阅读方法 |
| --- | --- |
| `kernels` | 每个逻辑 kernel 的模式、选择项、拆分函数和静态操作计数。 |
| `scope_issue_counts` | jit 作用域分组；共享 VF 只计一次，不按祖先作用域重复累计。 |
| `unit_flag_adjustments` | M/FIXPIPE 握手调整及其对外层作用域的影响。 |
| `inter_kernel` | 静态 Host launch 与可能的数据依赖，包括正常消费者。 |
| `static_operations` | 已配置 kernel 的静态操作总数，不表示运行时执行次数。 |

开启 `CANNBOTDSL_DUMP_MLIR=1` 或 `CANNBOTDSL_DUMP_ASCENDC=1` 时，导出目录还会生成 `<function>.pipe_ablation_report.json`。先用导出结果确认选择范围；正式测量前关闭普通 dump，避免源码生成、输入快照和 tiling 文件写入影响计时。

## 测量与结果解释

保持 shape、设备、预热次数和采集方式一致，按[性能测量规范](../../guide/performance.md#测量规范)分别采集正常和消融变体。

- 合法选择项命中数为零时，先确认 kernel 是否使用该流水；不要只根据名称认定消融生效。
- 消融会改变流水重叠、争用、M/FIXPIPE 握手及后端优化，前后时间差不能直接作为某条流水的独立耗时。
- 框架屏蔽已识别的数据指令；控制、同步及编译器生成的固定工作仍可能使用相关流水，应结合生成代码和实际指标判断。
- 编译失败时先阅读具体诊断，核对失真数据是否参与地址或控制流，以及是否存在本页“约束说明”列出的不支持操作。不能跳过安全检查来运行变体。
