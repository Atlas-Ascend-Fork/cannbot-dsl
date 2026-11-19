# simt.brev

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将输入的全部位逐位反转。`uint32` 的第 i 位变为第 31−i 位，`uint64` 的第 i 位变为第 63−i 位。

## 函数原型

```python
simt.brev(x) -> uint32 | uint64
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `x` | 输入 | 待处理的输入值，`uint32` 或 `uint64` 标量。 |

## 返回值说明

返回反转后的值，数据类型与输入相同。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.brev`，校验并打印结果。

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
        dst[0] = simt.brev(src[0])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1], dtype=torch.uint64, device='npu')
    dst = torch.empty((1,), dtype=torch.uint64, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [0x8000000000000000]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[9223372036854775808]
```
