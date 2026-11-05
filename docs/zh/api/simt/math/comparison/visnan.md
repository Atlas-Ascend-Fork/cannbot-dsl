# simt.visnan

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

逐分量判断双分量短向量x是否为NaN；NaN分量映射为1.0，其余分量映射为0.0。

## 函数原型

```python
simt.visnan(x) -> dtypes.float16x2 | dtypes.bfloat16x2
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |

## 返回值说明

返回与x同dtype的短向量，包含两个浮点0.0/1.0；不是整对bool，也不是整数掩码。通过`.x`、`.y`分别访问两个结果分量。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.visnan`，校验并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def compute(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        x = simt.make_short_vector(src[0], src[1])
        y = simt.make_short_vector(src[2], src[3])
        a = simt.visnan(x)
        b = simt.visnan(y)
        dst[0] = a.x
        dst[1] = a.y
        dst[2] = b.x
        dst[3] = b.y


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)
def main():
    src = torch.tensor([float('nan'), 1.0, float('inf'), float('nan')], dtype=torch.float16, device='npu')
    dst = torch.empty((4,), dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([1, 0, 0, 1], dtype=torch.float16)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1.0, 0.0, 0.0, 1.0]
```
