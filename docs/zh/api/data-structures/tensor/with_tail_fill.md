# Tensor.with_tail_fill

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

为 `axes` 指定的末尾轴组合设置填充要求，将该轴组有效数据末尾不足 32 字节整数倍的部分按 `mode` 和 `value` 指定的方式填充。本接口不扩展或重新分配存储，也不立即写入填充值。

调用本接口的 Tensor 中，`axes` 指定的区域内元素须连续存放，且区域末尾须预留足够空间，用于将该区域的字节数补齐到 32 字节的整数倍。

例如，普通 ND Tensor 按行填充时：

- 原有布局：`dtype=float16`、`shape=(2, 10)`、`stride=(16, 1)`。每行 10 个有效元素连续存放，两行起点相隔 16 个元素，因此每行末尾留出 6 个填充位置。
- 存储容量：从该 Tensor 起点算起，须已预留至少 64 字节，包含两行有效数据和两行尾部填充空间。
- 设置 `axes=(-1,), mode="constant", value=0`：为每行末尾的 6 个位置指定填充值 `0`，`shape` 和 `stride` 不变。

若原有 `stride=(10, 1)`，两行之间没有填充空间，不能直接按上述方式填充。

## 函数原型

```python
with_tail_fill(self, *, mode='constant', value=None, axes=(-1,)) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `mode` | 输入 | 填充方式。`"constant"` 为默认值，使用 `value`；`"first"` 使用对应组的首个元素。 |
| `value` | 输入 | 常量填充值。`mode="constant"` 时必须指定 Python 整数、浮点数或 Kernel 运行时标量；`mode="first"` 时使用默认值 `None`。 |
| `axes` | 输入 | 参与尾部填充的轴编号序列。默认值为 `(-1,)`，表示最后一轴；负数轴编号从末轴向前计数。 |

## 返回值说明

返回带尾部填充要求的新 Tensor 视图，逻辑形状和元素类型不变。

## 约束说明

- 原 Tensor 已是组合视图时，设置尾部填充后的结果仍为组合视图。
- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- `axes` 必须是以最后一轴结尾的非空连续轴序列，例如二维 Tensor 的 `(1,)` 或 `(0, 1)`。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(x: cbd.Tensor, y: cbd.Tensor):
    full = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float16, cbd.make_layout((2, 16)))
    current = full.bare().view(
        cbd.make_layout((2, 10), stride=(16, 1)),
        physical_layout=cbd.make_layout((2, 16)),
    )
    cbd.mem_copy(current.with_tail_fill(value=-7.0), x)
    cbd.mem_copy(y, full.flatten())


@cbd.host
def run(x, y):
    example_kernel[1](x, y)


x = torch.arange(20, dtype=torch.float16, device="npu").reshape(2, 10)
y = torch.empty(32, dtype=torch.float16, device="npu")
run(x, y)
torch.npu.synchronize()
print(y.cpu().reshape(2, 16).tolist())
```

输入：

```text
x=arange(20).reshape(2, 10), dtype=float16
逻辑 shape=(2, 10)，物理 shape=(2, 16)，尾部填充值=-7.0
```

输出：

```text
[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0]]
```
