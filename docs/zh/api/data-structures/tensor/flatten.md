# Tensor.flatten

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将当前 Tensor 中从 `start_dim` 到 `end_dim` 的逻辑轴合并为一个轴，新轴长度等于这些轴长度的乘积。返回的新 Tensor 视图与当前对象共享存储，不搬运或重排实际数据。

## 函数原型

```python
flatten(self, start_dim=0, end_dim=-1) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `start_dim` | 输入 | <p>合并轴的起始位置，类型为 <code>int</code>。默认值为 0（第一个轴）。</p>  |
| `end_dim` | 输入 | <p>合并轴的结束位置（含该轴），类型为 <code>int</code>。默认值为 -1（最后一个轴）。</p>  |

## 返回值说明

返回新 Tensor，逻辑形状将指定连续轴合并，`rank` 随新形状确定；`dtype`、`memloc` 不变，并共享原存储。

能用各轴固定跨度表示变换后的访问方式时，推导相应的逻辑布局和物理布局；否则返回组合视图，保留原物理布局，通过编译器维护的坐标映射访问原数据。

起止轴相同时，直接返回原 Tensor。

## 约束说明

- 当合并后的轴无法用固定跨度表示，或原 Tensor 为 NZ / ZN 布局、组合视图时，返回组合视图。连续 ND 布局展平后仍可用固定跨度表示，不受此限制。
- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- 当前 Tensor 的逻辑形状必须非空，且不能包含嵌套 tuple。
- `end_dim` 指定的轴必须与 `start_dim` 指定的轴相同，或位于其后。
- `start_dim/end_dim` 必须为 Python 整数，范围为 `-self.rank ≤ start_dim/end_dim < self.rank`。负数轴编号从末轴向前计数，`-1` 表示最后一轴。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.flatten(0, 1)
    y[0] = u.shape[0]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(1, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y.cpu().tolist())
```

输入：

```text
原 shape=(4, 8), dtype=float32
t.flatten(0, 1)
```

输出：

```text
shape: [32]
```
