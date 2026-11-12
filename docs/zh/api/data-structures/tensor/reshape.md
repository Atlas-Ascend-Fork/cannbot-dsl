# Tensor.reshape

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在保持当前 Tensor 的逻辑元素顺序和元素总数不变的前提下，返回采用参数 `shape` 指定形状的新视图。新视图与当前 Tensor 共享存储，不搬运或重排实际数据。

## 函数原型

```python
reshape(self, *shape) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `shape` | 输入 | <p>必填。新旧逻辑元素总数必须相同，变换后的逻辑形状可以是以下类型：</p> <ul> <li><code>多个 int 位置参数</code>：逐维提供静态正整数长度。</li> <li><code>tuple[int, ...]</code>：指定各维的静态正整数长度。</li> <li><code>list[int]</code>：与 tuple 形式含义相同。</li> <li><code>含运行时整数值的 tuple 或 list</code>：指定包含动态长度的目标形状，也可将各维长度分别作为位置参数传入。</li> </ul>  <p>最多允许一个维度设为 <code>-1</code>，根据元素总数推导该维长度。原形状及其他目标维度必须为静态正整数，且推导结果必须为正整数。</p>   |

## 返回值说明

返回新 Tensor，逻辑形状改为目标形状，`rank` 随新形状确定；`dtype`、`memloc` 不变，并共享原存储。

返回视图的`stride`和`data_format`读取规则见[stride](./properties/stride.md#形状变换后的跨度)。

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.reshape(8, 4)
    y[0] = u.shape[0]
    y[1] = u.shape[1]


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
原 shape=(4, 8), dtype=float32
t.reshape(8, 4)
```

输出：

```text
shape: [8, 4]
```
