# simt.block_idx

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前逻辑块在本次Kernel启动中的三维坐标`(x, y, z)`，各维编号从0开始。

## 函数原型

```python
simt.block_idx() -> tuple
```

## 参数说明

无

## 返回值说明

返回包含三个整数标量的元组`(x, y, z)`。`x`、`y`、`z`分别表示当前逻辑块在本次Kernel启动的三个方向上的编号。

每个分量的类型为`dtypes.int32`。使用`[0]`、`[1]`、`[2]`获取对应分量。

## 约束说明

无

## 调用示例

启动两个逻辑块，每块包含1个线程，读取并打印每个逻辑块的坐标。

```python
import torch
import torch_npu
from cannbotdsl import dim3, host, simt
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def example_kernel(dst: Tensor):
    with vf(mode="simt"):
        x, y, z = simt.block_idx()
        dst[x, 0] = x
        dst[x, 1] = y
        dst[x, 2] = z


@host
def run(dst):
    example_kernel[dim3(2), dim3(1)](dst)


def main():
    dst = torch.empty((2, 3), dtype=torch.int32, device="npu")
    run(dst)
    torch.npu.synchronize()
    result = dst.cpu().tolist()
    assert result == [[0, 0, 0], [1, 0, 0]]
    print(result)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[[0, 0, 0], [1, 0, 0]]
```
