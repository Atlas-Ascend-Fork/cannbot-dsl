# simt.vbitcast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

保留value的存储位模式，按dtype重新解释，不进行数值转换或舍入。

## 函数原型

```python
simt.vbitcast(value, *, dtype) -> 短向量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 带明确类型的短向量；类型范围包括整数`int8/uint8`、`int16/uint16`、`int32/uint32`、`int64/uint64`的2、3、4分量类型，`float32x2/x3/x4`、`float16x2`、`bfloat16x2`、`fp8x2_e4m3fn`、`fp8x2_e5m2`、`hifloat8x2`。 |
| dtype | 输入 | 目标短向量数据类型，取值范围与`value`相同；源、目标总存储位宽必须一致，不要求元素类型或分量数相同。 |

## 返回值说明

返回目标dtype的位重解释结果；同类型重解释可被消除。

## 约束说明

- 源和目标必须都是短向量；不能与普通标量互转。
- 源、目标的总存储位宽必须相同。例如`float16x2`与`int16x2`均为32位，`float32x2`与`int32x2`均为64位；`float16x2`与`float32x2`位宽不同，不能直接重解释。
- 不进行逐分量数值转换，也不执行舍入。

## 调用示例

重解释短向量的存储位模式，并打印各分量。

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
        x = simt.make_short_vector(src[0], src[1])
        bits = simt.vbitcast(x, dtype=dtypes.int32x2)
        dst[0] = bits.x
        dst[1] = bits.y


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1.0, 2.0], device='npu', dtype=torch.float32)
    dst = torch.zeros(2, device='npu', dtype=torch.int32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [1065353216, 1073741824]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1065353216, 1073741824]
```
