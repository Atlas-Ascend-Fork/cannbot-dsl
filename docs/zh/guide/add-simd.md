# Add 算子样例（SIMD）

## 任务与目标

本节介绍如何使用 CANNBotDSL SIMD 编程方式实现 Add 算子，并通过测试用例验证结果。通过本节学习，可以了解如何分配 UB 存储、完成 GM 与 UB 之间的数据搬运，以及使用矢量寄存器接口实现逐元素加法。

本示例固定处理两个 Shape 为 `(64,)`、数据类型为 float32 的 Tensor，使用一个 Vector Core 完成逐元素加法。输入数据先从 GM 搬入 UB，再通过 SIMD 寄存器接口完成计算，最后将结果搬回 GM。

## 算子设计规格

| name | shape | data type | format |
| --- | --- | --- | --- |
| src0（输入） | (64,) | float32 | 连续 ND |
| src1（输入） | (64,) | float32 | 连续 ND |
| dst（输出） | (64,) | float32 | 连续 ND |

数学表达式：

$$
\mathrm{dst}[i]=\mathrm{src0}[i]+\mathrm{src1}[i],\quad 0\le i<64.
$$

使用的主要接口：

- 存储定义与分配：`RingBuffer`、`make_layout`、`MemLoc.UB`。
- 数据搬运：`mem_copy`。
- SIMD 计算：`vf`、`update_mask`、`vload`、`vadd`、`vstore`。
- Kernel 定义与启动：`kernel`、`host`。

输入须为同一 NPU 上的连续 Tensor，Shape 均为 `(64,)`，dtype 均为 float32；不支持广播和自动求导。

## 导入 CANNBotDSL 模块

导入 PyTorch 和 torch_npu，用于构造 NPU Tensor 和验证结果；导入 CANNBotDSL 的存储、搬运、寄存器计算及启动接口。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import RingBuffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import update_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc
```

## 核心代码逻辑

### 1. 定义 UB 存储

在 Kernel 中为两个输入和一个输出分别申请 UB 存储，每个槽位包含 64 个 float32 元素，占用 256 字节。三个 RingBuffer 合计使用 768 字节，`depth=1` 表示每路仅有一个槽位，适用于本例的单块计算。

```python
buf0 = RingBuffer(MemLoc.UB, src0.dtype, make_layout((64,)), depth=1)
buf1 = RingBuffer(MemLoc.UB, src1.dtype, make_layout((64,)), depth=1)
out = RingBuffer(MemLoc.UB, dst.dtype, make_layout((64,)), depth=1)
```

### 2. 实现 Add Kernel 函数

通过 `@kernel` 定义设备函数。计算依次完成输入搬入、读取输入槽位、SIMD 相加、写入输出槽位和结果搬出。

```python
@kernel
def _reg_add_kernel(src0, src1, dst):
    buf0 = RingBuffer(MemLoc.UB, src0.dtype, make_layout((64,)), depth=1)
    buf1 = RingBuffer(MemLoc.UB, src1.dtype, make_layout((64,)), depth=1)
    out = RingBuffer(MemLoc.UB, dst.dtype, make_layout((64,)), depth=1)

    # 1. GM -> UB
    mem_copy(buf0.next_write(), src0)
    mem_copy(buf1.next_write(), src1)

    # 2. 获取本次计算使用的输入、输出槽位
    in0 = buf0.next_read()
    in1 = buf1.next_read()
    res = out.next_write()

    # 3. UB -> 寄存器 -> 加法 -> UB
    with vf(mode="simd"):
        mask, _ = update_mask(64, 32)
        acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, acc, mask)

    # 4. UB -> GM
    mem_copy(dst, out.next_read())
```

`next_write()` 和 `next_read()` 分别获取写入和读取的槽位；它们在 VF 外调用。框架根据 Tensor 的实际访问关系生成搬运与计算之间的同步。

`vf(mode="simd")` 定义 SIMD 计算区域。`update_mask(64, 32)` 生成覆盖 64 个 32 位元素的掩码；`vload` 从 UB 加载到寄存器，`vadd` 对各 lane 的对应元素相加，`vstore` 将结果写回 UB。这里的偏移量 0 以元素为单位。

### 3. 定义 Host 启动入口

使用 `@host` 定义启动函数，并通过 `[1]` 指定一个 block。Kernel 的输出由 `dst` 参数写回。

```python
@host
def run(src0, src1, dst):
    _reg_add_kernel[1](src0, src1, dst)
```

## 测试用例

构造两个长度为 64 的输入，调用 Host 入口后同步设备，并与 PyTorch 加法结果比较。

```python
def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 + src1).cpu())
    print("reg compute example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")


if __name__ == "__main__":
    main()
```

输入分别为 `0, 1, ..., 63` 和 `1, 2, ..., 64`，预期输出为 `1, 3, ..., 127`。

## 完整代码

文件名：`add_simd.py`。

<!-- runnable-example:start -->
```python
import torch
import torch_npu  # noqa: F401  # 向 PyTorch 注册 Ascend NPU 后端

from cannbotdsl import RingBuffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import update_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc


@kernel
def _reg_add_kernel(src0, src1, dst):
    buf0 = RingBuffer(MemLoc.UB, src0.dtype, make_layout((64,)), depth=1)
    buf1 = RingBuffer(MemLoc.UB, src1.dtype, make_layout((64,)), depth=1)
    out = RingBuffer(MemLoc.UB, dst.dtype, make_layout((64,)), depth=1)

    # GM 数据搬运至 UB
    mem_copy(buf0.next_write(), src0)
    mem_copy(buf1.next_write(), src1)

    in0 = buf0.next_read()
    in1 = buf1.next_read()
    res = out.next_write()
    # Vector Function：更新 mask、搬入、计算、搬出
    with vf(mode="simd"):
        mask, _ = update_mask(64, 32)
        acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, acc, mask)

    # UB 数据搬运至 GM
    mem_copy(dst, out.next_read())


@host
def run(src0, src1, dst):
    _reg_add_kernel[1](src0, src1, dst)


def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 + src1).cpu())
    print("reg compute example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")


if __name__ == "__main__":
    main()
```
<!-- runnable-example:end -->

## 编译与执行

运行环境为 Ascend 950，需安装 CANNBotDSL、PyTorch 和 torch_npu。在代码文件所在目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python add_simd.py
```

首次执行时，框架编译并启动 Kernel。运行成功后输出：

```text
reg compute example passed
first=1.0000, last=127.0000
```
