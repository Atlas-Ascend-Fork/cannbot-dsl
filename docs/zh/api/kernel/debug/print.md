---
title: print
api_name: print
category: debugging
api_group: kernel
layer: tensor
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `print`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

在 Kernel 中打印 Tensor 或标量，用于调试，打印到当前进程的标准输出
`stdout`。

除文本输出外，也可以通过 `cannbotdsl.core.diag.debug.get_debug_prints()` 获取保存在
Host 内存中的结构化记录。

## 函数原型

```python
def print(
    value: Tensor | Scalar,
    *,
    label: str | None = None,
    mode: str = "ring",
) -> None: ...
```

除 `value` 外，其他参数均须使用关键字传入。

推荐使用 `import cannbotdsl as cb` 后调用 `cb.print(...)`，避免与 Python 内置 `print` 混淆。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `value` | `Tensor` 或 DSL 标量 | 无 | 要打印的 Tensor 或单个数值标量。Tensor 元素可先通过点索引取得标量，例如 `tensor[(0,)]`。 |
| `label` | `str` 或 `None` | `None` | 记录标签。省略时自动生成。 |
| `mode` | `str` | `"ring"` | 记录超过 64 次时的保留方式：`"ring"` 保留最近 64 次，`"stop"` 保留最早 64 次。 |

## 支持范围

| 输入 | 支持范围 |
| --- | --- |
| Tensor | UB、L1、L0C Tensor |
| 标量 | Kernel 局部标量，以及从 GM 或 UB Tensor 点索引得到的标量 |

标量支持 bool、8/16/32/64 位整数以及 float16、bfloat16、float32。

## 返回值说明

返回 `None`。

Host 侧可使用以下辅助函数管理结构化记录：

```python
from cannbotdsl.core.diag.debug import clear_debug_prints, get_debug_prints

clear_debug_prints()   # 清空此前记录
records = get_debug_prints()
```

每条记录包含标签、dtype、shape、存储位置、执行序号、block/subblock 信息。

## 使用约束

- 必须在 `@kernel` 函数中调用，不能直接打印 Python 数值或 `torch.Tensor`。
- Tensor 只支持 UB、L1 和 L0C。
- Channel 不能直接打印。普通同核 Channel 应先通过 `produce()` 或 `consume()`
  取得 Tensor，再对该 Tensor 调用 `print`。
- Tensor 按物理存储顺序采样，最多 1024 字节。非连续视图不会自动压紧，NZ/ZN 数据
  也不会自动重排为 ND。较大的 Tensor 只显示可采样的前部数据。
- Tensor 视图可以使用运行时偏移；无法静态确定偏移时，框架会采用保守的完整采样范围。
- 每个打印点在每个 block/subblock 上独立保留 64 条记录。`mode="stop"` 只停止保存
  新数据，不会停止 Kernel 执行。
- 调试打印会增加设备内存占用并触发 Host 回读和同步，不能用于性能测量。

## 调用示例

以下示例将 16 个 `float32` 元素搬入 UB，同时打印整个 UB Tensor 和 GM 中的第一个
标量，并在 Host 侧校验捕获结果。运行环境需安装 cannbotdsl、CANN、PyTorch 和
torch_npu，并具有受支持的 NPU。

```python
import cannbotdsl as cb
import torch
import torch_npu

from cannbotdsl.core.diag.debug import clear_debug_prints, get_debug_prints

@cb.kernel
def debug_kernel(src: cb.Tensor):
    tmp = cb.Channel(
        cb.MemLoc.UB,
        shape=(16,),
        dtype=src.dtype,
        depth=1,
    )

    slot = tmp.produce()
    cb.mem_copy(slot, src)
    slot = tmp.consume()
    cb.print(slot, label="ub_data")
    cb.print(src[(0,)], label="first_value")

@cb.host
def run_debug_kernel(src: cb.Tensor):
    debug_kernel[1](src)

def main():
    clear_debug_prints()
    src = torch.arange(16, dtype=torch.float32, device="npu:0")

    run_debug_kernel(src)
    torch.npu.synchronize()

    records = get_debug_prints()
    tensor_records = [record for record in records if record["label"] == "ub_data"]
    scalar_records = [record for record in records if record["label"] == "first_value"]
    assert tensor_records and scalar_records

    expected = torch.arange(16, dtype=torch.float32)
    for record in tensor_records:
        torch.testing.assert_close(record["data"].reshape(-1), expected, rtol=0, atol=0)
    for record in scalar_records:
        assert record["value"] == 0.0

if __name__ == "__main__":
    main()
```

### 预期输出

运行时会先输出 `ub_data` 和 `first_value` 的调试记录。记录头中的源码位置、block、
subblock 和计数取决于实际编译与执行；省略这些可变字段后的关键输出如下：

```text
[cannbotdsl.core.diag.debug] ... "ub_data" ... shape=(16,) ...
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15.])
[cannbotdsl.core.diag.debug] ... "first_value" ... shape=() ...
0.0
```
