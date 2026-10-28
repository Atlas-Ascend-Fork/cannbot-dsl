# Tensor.unflatten

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将当前 Tensor 的 `dim` 轴按 `sizes` 指定的长度拆分为多个轴，返回共享原存储的新视图。拆分后各轴长度的乘积等于原轴长度，本操作不搬运或重排实际数据。

## 函数原型

```python
unflatten(self, dim, sizes) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `dim` | 输入 | <p>必填。需要拆分的逻辑轴，类型为 <code>int</code>。</p>  |
| `sizes` | 输入 | <p>必填。拆分后各轴的长度，可以是以下类型：</p> <ul> <li><code>tuple[int, ...]</code>：指定各维的静态正整数长度。</li> <li><code>list[int]</code>：与 tuple 形式含义相同。</li> <li><code>含运行时整数的 tuple 或 list</code>：指定运行时才能确定的正整数长度。</li> </ul> <p>序列不能为空，各长度必须为正整数，乘积必须等于 <code>self.shape[dim]</code>。</p>  |

## 返回值说明

返回新 Tensor，逻辑形状将指定轴拆分为给定长度，`rank` 随新形状确定；`dtype`、`memloc` 不变，并共享原存储。

能用各轴固定跨度表示变换后的访问方式时，推导相应的逻辑布局和物理布局；否则返回组合视图，保留原物理布局，通过编译器维护的坐标映射访问原数据。

## 约束说明

- 当拆分后的形状无法通过原布局的逐轴跨度表示，或原 Tensor 为 NZ / ZN 布局、组合视图时，返回组合视图。普通 ND 布局能推导出目标逐轴跨度时，不受此限制。
- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- 当前 Tensor 的逻辑形状必须非空，且不能包含嵌套 tuple。
- `dim` 必须为 Python 整数，范围为 `-self.rank ≤ dim < self.rank`。负数轴编号从末轴向前计数，`-1` 表示最后一轴。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.unflatten(1, (2, 4))
    y[0] = u.shape[0]
    y[1] = u.shape[1]
    y[2] = u.shape[2]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(3, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y.cpu().tolist())
```

输入：

```text
原 shape=(4, 8), dtype=float32
t.unflatten(1, (2, 4))
```

输出：

```text
shape: [4, 2, 4]
```
