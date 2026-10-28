# Tensor.view

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据目标形状或目标元素类型，返回与当前 Tensor 共享存储的 Tensor 视图。指定目标形状时，只改变 Tensor 的形状，元素总数和元素类型不变；指定目标元素类型时，使用新类型读取同一段存储中的字节，但不进行数据类型转换。通过返回视图写入数据时，会修改同一份底层存储。

## 函数原型

```python
view(self, *shape, size=None, dtype=None) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| <span id="shape-可选值"></span>`shape` | 输入 | <p>形状视图的目标逻辑尺寸，可以是以下类型：</p> <ul> <li><code>多个整数位置参数</code>：逐维给出静态或运行时长度。</li> <li><code>tuple 或 list</code>：使用一个序列给出完整目标形状。</li> </ul> <p><strong>维度取值：</strong></p> <ul> <li>正整数：静态长度。</li> <li>运行时整数值：指定动态长度，须为正整数。</li> <li>-1：最多一个，源形状及其他目标维度须静态且为正，元素总数须可整除。</li> </ul> <p>目标形状非空。</p>  |
| `size` | 输入 | 通过关键字指定目标形状，与使用位置参数 `shape` 指定效果相同，形状取值要求和约束相同。默认值为 `None`，表示未指定。 |
| <span id="dtype-可选值"></span>`dtype` | 输入 | <p>按原始字节重新解释的目标元素类型，可以是以下类型：</p><ul><li><code>None</code>：默认值，使用形状视图，不改变元素类型。</li><li><code>DType</code>：使用 <code>cannbotdsl.dtypes.*</code> 指定目标类型。</li></ul><p>也可通过单个位置参数指定，例如 <code>t.view(cbd.dtypes.int16)</code>，此时不能同时指定形状。</p><p>支持以下类型：<code>dtypes.float16</code>、<code>dtypes.float32</code>、<code>dtypes.float64</code>、<code>dtypes.bfloat16</code>、<code>dtypes.float8_e4m3fn</code>、<code>dtypes.float8_e5m2</code>、<code>dtypes.float8_e8m0</code>、<code>dtypes.hifloat8</code>、<code>dtypes.int8</code>、<code>dtypes.int16</code>、<code>dtypes.int32</code>、<code>dtypes.int64</code>、<code>dtypes.uint8</code>、<code>dtypes.uint16</code>、<code>dtypes.uint32</code>、<code>dtypes.uint64</code>。原 Tensor 的元素类型也须满足该要求。具体类型说明见<a href="../data-types.md">预定义数据类型</a>。</p><ul><li>原 Tensor 的 <code>self.stride[-1]</code> 必须为 <code>1</code>。</li><li>目标类型位宽更大时，<code>self.shape[-1]</code> 和 <code>self.stride[:-1]</code> 中的各项须能被目标位宽与原位宽的比值整除。</li></ul><p>返回视图按位宽比例调整末维长度和相关跨度，不进行数值类型转换。</p> |

## 返回值说明

返回与原 Tensor 共享存储的 Tensor，原对象不变。

- **shape/size 模式**：`shape` 改为目标形状，`rank` 随之确定，逻辑和物理布局按目标形状的连续跨度生成；`dtype`、`memloc` 不变。调用者须保证按该布局访问原存储是有效的。
- **dtype 模式**：`dtype` 改为目标类型，末维长度和非末维跨度按源/目标位宽比例调整，生成对应物理布局。目标类型与原类型相同时，返回原 Tensor。

## 约束说明

- `shape`、`size`、`dtype` 三种指定方式互斥，一次只能选择一种。
- 仅支持逻辑格式和物理格式均为 ND 的 Tensor。
- 原 Tensor 的逻辑布局和物理布局均不能包含嵌套 tuple，二者维数必须相同且大于 0。
- 支持 GM、UB、L1、L0A、L0B、L0C、BIAS、SSBUF 和 FBUF。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.view(8, 4)
    v = t.view(size=(8, 4))
    w = t.view(dtype=cbd.dtypes.int16)
    y[0] = u.shape[0]
    y[1] = u.shape[1]
    y[2] = v.shape[0]
    y[3] = v.shape[1]
    y[4] = w.shape[0]
    y[5] = w.shape[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(6, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("位置参数 shape:", y[:2].cpu().tolist())
print("size 参数 shape:", y[2:4].cpu().tolist())
print("dtype 参数 shape:", y[4:].cpu().tolist())
```

输入：

```text
shape=(4, 8), dtype=float32
位置形状=(8, 4)，size=(8, 4)，目标 dtype=int16
```

输出：

```text
位置参数 shape: [8, 4]
size 参数 shape: [8, 4]
dtype 参数 shape: [4, 16]
```
