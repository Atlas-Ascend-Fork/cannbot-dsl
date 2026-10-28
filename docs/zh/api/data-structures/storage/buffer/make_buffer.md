# make_buffer

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将从片上存储空间派生的静态 Tensor 绑定为可使用的片上存储资源，保留输入 Tensor 的元素类型、布局和地址。

## 函数原型

```python
make_buffer(view: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `view` | 输入 | 待绑定的片上 Tensor。必填，须来自对应层级的[片上存储空间](../on-chip-memory.md)或其子视图。`view.shape` 的各维长度和 `view.physical_stride` 的各项均须为静态正整数。 |

## 返回值说明

返回与传入 Tensor 共享存储、元素类型及布局的 Tensor。

## 约束说明

- 支持 `MemLoc.UB`、`MemLoc.L1`、`MemLoc.L0A`、`MemLoc.L0B`、`MemLoc.L0C`、`MemLoc.BIAS`、`MemLoc.SSBUF`、`MemLoc.FBUF`。
- 视图的物理访问范围必须位于对应存储空间内，并满足目标层级的地址对齐要求。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    root = cbd.UB.view(262144)
    region = root[0:128,]
    t = cbd.make_buffer(region).view(cbd.dtypes.float32).view(4, 8)
    y[0] = t.shape[0]
    y[1] = t.shape[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(2, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y.cpu().tolist())
```

输入：

```text
UB 手工区域=[0, 128) 字节，目标 dtype=float32, shape=(4, 8)
```

输出：

```text
shape: [4, 8]
```
