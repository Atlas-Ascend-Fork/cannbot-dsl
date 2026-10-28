# Tensor.extent

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取当前 Tensor 某个逻辑叶维的长度。嵌套形状按从左到右的顺序展开后编号。

## 函数原型

```python
extent(self, axis) -> Int64
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `axis` | 输入 | 叶维编号。必填，为 Python 整数；负数从最后一个叶维向前计数。 |

## 返回值说明

返回该叶维长度的 Kernel 整数标量。

## 约束说明

- `axis` 必须位于展开后的叶维编号范围内。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    y[0] = t.extent(0)
    y[1] = t.extent(-1)


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(2, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print(y.cpu().tolist())
```

输入：

```text
shape=(4, 8), axis=0、-1
```

输出：

```text
[4, 8]
```
