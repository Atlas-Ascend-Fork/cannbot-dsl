# simt.bitcast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

保留value的存储位模式，按dtype重新解释，不进行数值转换或舍入。

## 函数原型

```python
simt.bitcast(value, *, dtype) -> 目标类型标量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 带明确数据类型的标量；本页类型范围为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`，不接受裸Python数值字面量。 |
| dtype | 输入 | 目标标量数据类型，必须与源类型具有相同存储位宽；类型组合见约束说明。 |

## 返回值说明

返回目标dtype的位重解释结果；同类型重解释可被消除。

## 约束说明

源和目标必须都是标量。同一行中的类型具有相同位宽，可用于本页的位模式重解释；不同位宽之间不能互转。

| 存储位宽 | 源、目标类型 |
| --- | --- |
| 8位 | `int8`、`uint8` |
| 16位 | `int16`、`uint16`、`float16`、`bfloat16` |
| 32位 | `int32`、`uint32`、`float32` |
| 64位 | `int64`、`uint64` |

- 不在标量和短向量之间转换；短向量使用`vbitcast`。
- 不进行数值转换或舍入；相同位模式在不同类型中可能表示不同数值。
- 上表不包含FP8、FP4或其他类型，不能仅凭标称位宽推断这些类型在当前设备上可用。

## 调用示例

将`float32`的存储位模式解释为`int32`，并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        dst[0] = simt.bitcast(src[0], dtype=dtypes.int32)


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1.0, 2.0], device='npu', dtype=torch.float32)
    dst = torch.zeros(2, device='npu', dtype=torch.int32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [1065353216, 0]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1065353216, 0]
```
