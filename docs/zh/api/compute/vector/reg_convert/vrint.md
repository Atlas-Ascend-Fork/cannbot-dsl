# vrint
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将`src`中的浮点数元素按照RINT（四舍六入五成双）舍入模式舍入到整数值，结果仍保持原浮点数据类型。接口通过函数返回值返回结果。舍入规则等价于C标准库`rint`在舍入方向设置为FE_TONEAREST时的语义：舍入到最接近的整数；当输入值恰好位于两个整数中间时，舍入到偶数（如1.5 -> 2.0、2.5 -> 2.0、-2.5 -> -2.0），整数值保持不变。未被`mask`筛选的元素置零。

## 函数原型

```python
vrint(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源矢量数据寄存器，元素类型支持`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`。 |
| `mask` | 输入 | 执行掩码。对应位置为1时执行舍入，为0时输出元素置零。除全有效掩码外，`elem_bits`须与`src`的元素位宽一致。 |

## 返回值说明

返回保存舍入结果的`RawVReg`，元素类型与`src`一致，仍为浮点类型；不修改`src`。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。
- `dtypes.float16`和`dtypes.bfloat16`支持饱和模式，`dtypes.float32`只支持不饱和模式。

## 调用示例

以下示例对包含正负中点值的64个浮点元素执行中点取偶数舍入，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vrint, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vrint_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    # 将完整输入搬至UB，再在VF内对寄存器元素舍入。
    mem_copy(buf, src)
    with vf(mode="simd"):
        mask = full_mask(elem_bits=32)
        result = vrint(vload(buf, 0), mask=mask)
        vstore(out, 0, result, mask)
    mem_copy(dst, out)

@host
def run(src, dst):
    _vrint_kernel[1](src, dst)

def main():
    values = [-3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 1.4, -1.4]
    src = torch.tensor(values * 8, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src)
    run(src, dst)
    torch.npu.synchronize()

    expected = torch.tensor([-4., -2., -2., 2., 2., 4., 1., -1.] * 8)
    torch.testing.assert_close(dst.cpu(), expected)
    print("vrint example passed")
    print(dst.cpu()[:8].tolist())

if __name__ == "__main__":
    main()
```

预期结果：

```text
vrint example passed
[-4.0, -2.0, -2.0, 2.0, 2.0, 4.0, 1.0, -1.0]
```
