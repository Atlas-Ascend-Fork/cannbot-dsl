# vreinterpret
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将寄存器按等宽dtype重新解释比特模式，不改变任何比特。

## 函数原型

```python
vreinterpret(src: RawVReg, dtype) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `dtype` | 输入 | 指定返回寄存器值的元素数据类型，元素位宽须与`src`一致。 |

## 返回值说明

返回按`dtype`指定的元素数据类型重新解释后的寄存器值，类型为`RawVReg`，原始比特保持不变。

## 约束说明

- 源与目标dtype位宽必须一致，否则编译期报错。
- 本接口需在VF作用域内调用。

## 调用示例

以下示例演示`vreinterpret`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vreinterpret, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vreinterpret_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        r = vreinterpret(vload(in0, 0), dtypes.float32)
        vstore(res, 0, r, mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vreinterpret_kernel[1](src, dst)

def main():
    src = torch.arange(1, 65, dtype=torch.int32, device="npu:0")
    dst = torch.empty(64, dtype=torch.float32, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    bits = dst.cpu().view(torch.int32)
    torch.testing.assert_close(bits, src.cpu())
    print("vreinterpret example passed")
    print(f"first={int(bits[0])}, last={int(bits[-1])}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vreinterpret example passed
first=1, last=64
```
