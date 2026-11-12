# Tensor.ptr

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取Tensor指定元素的指针，可传递给需要元素指针的标量访存或SIMT原子操作接口。获取指针本身不读取或写入数据。

## 函数原型

```python
ptr(self, coord)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `coord` | 输入 | 元素坐标，结构须与Tensor的逻辑形状匹配。 |

## 返回值说明

返回指向指定元素的指针。

## 约束说明

- 支持GM和UB中的Tensor；具体可使用的存储位置由接收该指针的接口决定。
- 静态坐标须在对应维度的范围内；运行时坐标的有效性由调用方保证。

## 调用示例

以下示例获取输出Tensor的元素指针，将0写入该元素并检查结果。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def write_index(out: cbd.Tensor):
    index = cbd.get_block_idx()
    cbd.scalar.vec_store_bypass(out.ptr(index), index)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [0]
```
