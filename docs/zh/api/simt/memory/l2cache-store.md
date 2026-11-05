# simt.l2cache_store

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将value写入ptr指向的GM地址，并缓存到L2 Cache，不填入Data Cache。

## 函数原型

```python
simt.l2cache_store(ptr, value) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ptr | 输出 | 由`tensor.ptr(...)`获取的GM指针，元素类型为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`。 |
| value | 输入 | 待写标量，类型与ptr元素一致。 |

## 返回值说明

无返回值。结果写入ptr指向的内存。

## 约束说明

- UB和通用地址空间指针被拒绝。
- ptr地址按元素类型自然对齐并处于有效内存范围。
- L2/GM的一致性不等于各核Data Cache已失效。与普通缓存访问混用且数据被其他核或原子更新后，读取方需要适当DCCI维护；写入与标志传递还应配合同步及内存栅栏。

## 调用示例

经L2 Cache写入一个元素，并打印写入结果。

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
        simt.l2cache_store(dst.ptr(0), src[0])


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3, 7], device='npu', dtype=torch.int32)
    dst = torch.zeros(2, device='npu', dtype=torch.int32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [3, 0]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3, 0]
```
