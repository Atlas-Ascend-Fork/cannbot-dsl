# simt.make_short_vector

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按输入顺序构造短向量，分量依次为x、y、z、w。结果dtype由分量类型及数量推导。

## 函数原型

```python
simt.make_short_vector(*values) -> 短向量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| values | 输入 | 2、3或4个标量分量，元素类型为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`或`float32`。所有带类型的分量必须同类型；类型与分量数的对应关系见约束说明。 |

## 返回值说明

返回推导出的短向量，可通过x、y、z、w访问存在的分量。

## 约束说明

- 第一个分量带有明确数据类型时，以该类型作为分量类型；否则，输入中含Python `float`字面量时推导为`float32`，不含时推导为`int32`。所有带类型的分量必须与推导类型一致，字面量必须能匹配该类型。

- 实际分量数必须有已定义dtype。
- int8/16/32/64、uint8/16/32/64和float32有2/3/4分量；float16、bfloat16仅有2分量。不存在float16x3或bfloat16x4构造。
- 构造出的短向量能否用于数学、原子或访存操作，仍取决于对应接口支持的数据类型。

## 调用示例

将两个标量组成短向量，并打印其分量。

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
        dst[0] = x.x
        dst[1] = x.y


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1.5, 2.5, 3.5, 4.5], device='npu', dtype=torch.float32)
    dst = torch.zeros(2, device='npu', dtype=torch.float32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [1.5, 2.5]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1.5, 2.5]
```
