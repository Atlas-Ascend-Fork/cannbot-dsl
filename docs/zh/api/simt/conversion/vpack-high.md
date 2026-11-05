# simt.vpack_high

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

单输入时复制x的高分量到结果的两个分量；双输入时依次取x、y的高分量组成短向量。指定dtype=dtypes.float32的单输入形式返回高分量的float32数值。

## 函数原型

```python
simt.vpack_high(x, y=None, *, dtype=None) -> 短向量或 float32 标量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | float16x2或bfloat16x2短向量。 |
| y | 输入 | 可选同dtype短向量，默认None。 |
| dtype | 输入 | 默认None返回短向量；仅单输入形式可指定dtypes.float32返回标量。 |

## 返回值说明

dtype=None时返回与x同dtype的短向量；dtype=dtypes.float32时返回float32标量。

## 约束说明

- 本页使用`float16x2`或`bfloat16x2`，不扩展到整数或三、四分量短向量。
- 双输入必须同dtype，第一个分量x为低位，第二个分量y为高位；本接口不执行任意字节拼接。
- 单分量提取形式使用`vpack_high(x, dtype=dtypes.float32)`，不要同时传入`y`。

## 调用示例

提取短向量的高分量，并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor, old: Tensor):
    with vf(mode='simt'):
        x = simt.make_short_vector(src[0], src[1])
        y = simt.make_short_vector(src[2], src[3])
        z = simt.vpack_high(x, y)
        dst[0] = z.x
        dst[1] = z.y
        r = simt.vpack_high(x)
        old[0] = r.x
        old[1] = r.y


@host
def run(src, dst, old):
    demo[dim3(1), dim3(1)](src, dst, old)


def main():
    src = torch.tensor([1, 2, 3, 4], device='npu', dtype=torch.float16)
    dst = torch.zeros(2, device='npu', dtype=torch.float16)
    old = torch.zeros(2, device='npu', dtype=torch.float16)
    run(src, dst, old)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [2, 4]
    assert old.cpu().tolist() == [2, 2]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[2.0, 4.0]
```
