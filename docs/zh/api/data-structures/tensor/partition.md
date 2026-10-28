# Tensor.partition

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据参数 `dim` 指定的轴，为当前 Tensor 设置两路搬运分区规则，返回带分区标记的 Tensor 视图。分区规则记录在编译过程生成的内部表示中，不作为 Tensor 的公开属性。指定 `part_id` 时，返回视图表示选中的分区；省略时，返回视图保留完整形状，并携带两路分区的规则。

在以下搬运路径中使用分区搬运时，对相应的 Tensor 调用 `partition()`，并将返回视图作为该位置的搬运参数：

- **GM → UB**：对 GM 中的源 Tensor 分区，返回视图作为搬运源。
- **UB → GM/L1**：对 GM 或 L1 中的目标 Tensor 分区，返回视图作为搬运目标。
- **L0C → UB**：对 L0C 中的源 Tensor 调用 `partition()`，省略 `part_id`，返回保留完整形状、带分区标记的视图，作为搬运源。

## 函数原型

```python
partition(self, *, dim, part_id=None, alignment=1) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `dim` | 输入 | <p>必填。两路搬运分区采用的逻辑轴，类型为 Python 整数。可选值：</p> <ul> <li><code>0</code>：沿 M 轴分区。先将长度的一半向上取整，再向上补齐到 <code>alignment</code> 的整数倍，作为切分位置。第一块取前面部分，第二块取剩余部分。</li> <li><code>1</code>：沿 N 轴分区。先将总长度向上补齐到 32 的整数倍，再除以 2，作为切分位置。此时 <code>alignment</code> 必须为 <code>1</code>。</li> </ul>  |
| `part_id` | 输入 | <p>选取的两路分区编号，可以是以下类型：</p> <ul> <li><code>None</code>：默认值，不单独选择某一路；L0C → UB 必须采用此值。</li> <li><code>int</code>：静态编号，只能为 0（第一路）或 1（第二路）。</li> <li><code>受支持的运行时整数</code>：动态选择 0 或 1。</li> </ul>   |
| `alignment` | 输入 | <p>M 轴分区的切分对齐粒度，类型为 <code>int</code>。Python 正整数，默认 1。</p> <ul> <li><code>dim=0</code>：按指定粒度切分 M。</li> <li><code>dim=1</code>：只能为 1，N 分区使用硬件 32 列边界。</li> </ul>  |

## 返回值说明

返回带分区信息的新 Tensor。指定 `part_id` 时，逻辑形状中 `dim` 轴长度为所选分区的实际长度；省略时逻辑形状不变。保留原 Tensor 的 `dtype`、`rank`、存储层级和物理布局；普通 ND 布局保留原逻辑跨度。

## 约束说明

- 原 Tensor 已是组合视图时，分区后的结果仍为组合视图。
- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- 仅支持逻辑形状为非嵌套二维形状的 Tensor。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.partition(dim=0, part_id=0)
    v = t.partition(dim=0, part_id=1)
    y[0] = u.shape[0]
    y[1] = u.shape[1]
    y[2] = v.shape[0]
    y[3] = v.shape[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(4, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("第一分区 shape:", y[:2].cpu().tolist())
print("第二分区 shape:", y[2:].cpu().tolist())
```

输入：

```text
shape=(4, 8), dim=0, part_id=0、1
```

输出：

```text
第一分区 shape: [2, 8]
第二分区 shape: [2, 8]
```
