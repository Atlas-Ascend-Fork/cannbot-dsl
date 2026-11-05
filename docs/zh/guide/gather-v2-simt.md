# GatherV2 算子样例（SIMT）

## 任务与目标

本节介绍如何使用 CANNBotDSL SIMT 编程方式实现 GatherV2，并通过测试用例验证结果。通过本节学习，可以了解如何获取线程编号、将输出元素分配给线程，以及使用标量索引和条件分支完成间接访存。

本示例固定处理 Shape 为 `(4, 8)`、数据类型为 float32 的输入 Tensor，根据 6 个索引沿第 0 维选择输入行，生成 Shape 为 `(6, 8)` 的输出 Tensor。每个有效线程负责一个输出元素，直接访问 GM，并处理负索引和越界索引。

## 算子设计规格

| name | shape | data type | format |
| --- | --- | --- | --- |
| x（输入） | (4, 8) | float32 | 连续 ND |
| indices（索引） | (6,) | int32 | 连续 ND |
| y（输出） | (6, 8) | float32 | 连续 ND |

固定 `axis=0`、`batch_dims=0`，开启负索引。令 `s = indices[g]`；如果 `s < 0`，先将其加上输入行数 4。数学表达式为：

$$
y[g,i]=\begin{cases}
x[s,i], & 0\le s<4,\\
0, & \text{其他情况}.
\end{cases}
$$

负索引只归一化一次，归一化后仍越界则写零。例如 `-1` 选择最后一行，`-5` 对应全零输出行。

使用的主要接口：

- SIMT 计算区域：`vf(mode="simt")`。
- 线程与 block 索引：`simt.block_idx`、`simt.thread_idx`、`simt.thread_dim`。
- Kernel 定义与启动：`kernel`、`host`、`dim3`。

## 导入 CANNBotDSL 模块

导入 PyTorch 和 torch_npu，用于构造数据、执行 NPU 计算和验证结果；导入 CANNBotDSL 的 SIMT 及启动接口。

```python
import torch
import torch_npu  # noqa: F401
from cannbotdsl import dim3, host, kernel, simt, vf
```

## 核心代码逻辑

### 1. 定义线程与输出元素的映射

输出包含 `6 × 8 = 48` 个元素。启动两个 block，每个 block 使用 32 个线程。使用 block 编号、线程数和线程编号计算全局输出偏移：

```python
offset = simt.block_idx()[0] * simt.thread_dim()[0] + simt.thread_idx()[0]
```

仅 `offset < 48` 的线程执行访存。对有效线程，通过除法和取余得到输出的行、列坐标：

```python
row = offset // 8
column = offset % 8
```

每个有效 offset 对应唯一的输出位置。重复索引只会导致读取同一输入行，不会产生输出写冲突。

### 2. 实现 GatherV2 Kernel 函数

通过 `@kernel` 定义设备函数，在 `vf(mode="simt")` 区域内计算线程坐标、读取行索引、归一化负索引并复制数据。

```python
@kernel
def gather_kernel(x, indices, y):
    with vf(mode="simt"):
        offset = simt.block_idx()[0] * simt.thread_dim()[0] + simt.thread_idx()[0]
        if offset < 48:
            row = offset // 8
            column = offset % 8
            selected = indices[row]
            # 负索引只加一次轴长。
            if selected < 0:
                selected += 4
            if selected >= 0 and selected < 4:
                y[row, column] = x[selected, column]
            else:
                y[row, column] = 0
```

先检查输出范围，再读取 `indices[row]`；先检查输入行号是否合法，再读取 `x[selected, column]`。这样可避免无效线程或非法索引产生越界访问。

线程之间没有共享中间结果，每个线程独立写入一个输出元素，因此不需要线程屏障或原子操作。

### 3. 定义 Host 启动入口

使用 `@host` 定义启动函数，通过两个 `dim3` 参数分别指定 block 数与每个 block 的线程数。

```python
@host
def run(x, indices, y):
    gather_kernel[dim3(2), dim3(32)](x, indices, y)
```

`vf(mode="simt")` 使用 Host 启动时指定的线程配置，每个 block 中的线程编号范围为 0～31。

## 测试用例

构造递增输入和包含重复、负数及越界值的索引，调用 Kernel 后与显式构造的 CPU 结果比较。

```python
def main():
    torch.npu.set_device(0)
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    indices = torch.tensor([2, 0, -1, 4, -5, 2], dtype=torch.int32)
    y = torch.empty((6, 8), dtype=torch.float32, device="npu:0")
    run(x.npu(), indices.npu(), y)
    torch.npu.synchronize()
    golden = torch.stack([x[2], x[0], x[3], torch.zeros(8), torch.zeros(8), x[2]])
    torch.testing.assert_close(y.cpu(), golden, rtol=0, atol=0)
    print("SIMT GatherV2 example passed")


if __name__ == "__main__":
    main()
```

索引 `2、0、-1` 分别选择第 2、0、3 行；索引 `4、-5` 越界，对应输出行填零；最后一个索引 `2` 再次选择第 2 行。GatherV2 只复制数据，因此本例采用零容差比较。

## 完整代码

文件名：`gather_v2_simt.py`。

<!-- runnable-example:start -->
```python
"""Fixed-shape SIMT GatherV2 tutorial: axis=0, batch_dims=0."""
import torch
import torch_npu  # noqa: F401
from cannbotdsl import dim3, host, kernel, simt, vf


@kernel
def gather_kernel(x, indices, y):
    with vf(mode="simt"):
        offset = simt.block_idx()[0] * simt.thread_dim()[0] + simt.thread_idx()[0]
        if offset < 48:
            row = offset // 8
            column = offset % 8
            selected = indices[row]
            # 本例开启负索引：只加一次轴长，仍越界则写零。
            if selected < 0:
                selected += 4
            if selected >= 0 and selected < 4:
                y[row, column] = x[selected, column]
            else:
                y[row, column] = 0


@host
def run(x, indices, y):
    gather_kernel[dim3(2), dim3(32)](x, indices, y)


def main():
    torch.npu.set_device(0)
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    indices = torch.tensor([2, 0, -1, 4, -5, 2], dtype=torch.int32)
    y = torch.empty((6, 8), dtype=torch.float32, device="npu:0")
    run(x.npu(), indices.npu(), y)
    torch.npu.synchronize()
    golden = torch.stack([x[2], x[0], x[3], torch.zeros(8), torch.zeros(8), x[2]])
    torch.testing.assert_close(y.cpu(), golden, rtol=0, atol=0)
    print("SIMT GatherV2 example passed")


if __name__ == "__main__":
    main()
```
<!-- runnable-example:end -->

## 编译与执行

运行环境为 Ascend 950，需安装 CANNBotDSL、PyTorch 和 torch_npu。在代码文件所在目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python gather_v2_simt.py
```

首次执行时，框架编译并启动 Kernel。运行成功后输出：

```text
SIMT GatherV2 example passed
```
