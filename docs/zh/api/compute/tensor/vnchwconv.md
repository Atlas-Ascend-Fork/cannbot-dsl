# vnchwconv

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在UB中按位模式重排数据块，可用于矩阵转置和卷积权重预处理。先用`mem_copy`或计算接口准备UB源数据，调用本接口后再读取目标中的结果。本接口实际写入目标存储，不做数值转换；`Tensor.permute`仅改变视图的轴顺序，不搬动数据。

令 `src[r,i,j]`、`dst[r,i,j]` 表示第 r 次重复、第 i 块、第 j 个元素，每块 32 字节，i 为 0～15。16 位类型完成 16×16 转置：

$$\mathrm{dst}[r,j,i]=\mathrm{src}[r,i,j].$$

32 位类型每块 8 个元素，j 为 0～7：

$$\mathrm{dst}[r,2j+\lfloor i/8\rfloor,i\bmod8]=\mathrm{src}[r,i,j].$$

8 位类型令 s、d 分别为 `src_high_half`、`dst_high_half` 的 0/1 值，i、j 为 0～15：

$$\mathrm{dst}[r,j,16d+i]=\mathrm{src}[r,i,16s+j].$$

8 位目标块未选中的另一半保留原值。

## 函数原型

```python
vnchwconv(
    dst: Tensor, src: Tensor, *,
    dst_high_half: bool = False, src_high_half: bool = False,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dst | 输出 | UB 目标视图，与 src 类型相同。形状为 `(R,16,B)` 或单次的 `(16,B)`；B=32/元素字节数，R 为 0～255。8 位模式只覆盖选中的半块。每块内部连续、起点 32 字节对齐，存储窗口覆盖完整块。 |
| src | 输入 | UB 源视图，支持框架的 8/16/32 位整数及浮点类型。R 与 dst 一致；16 块可由嵌套轴表达且块起点不必等距。重复步长在 layout 中以元素为单位，换算为字节后须为非负 32 的倍数，最多 65535 个块。目标重复步长遵守同样要求。 |
| dst_high_half | 输入 | 8 位类型：False 写低 16 字节，True 写高 16 字节，支持运行时 DSL 布尔值。16/32 位必须使用常量 False。 |
| src_high_half | 输入 | 8 位类型：False 读低 16 字节，True 读高 16 字节，支持运行时 DSL 布尔值。16/32 位必须使用常量 False。 |

## 返回值说明

无返回值。结果写入 `dst`，不会分配新的存储；R=0 不访问源或目标。

## 约束说明

- 在 kernel 内、VF 区域外调用，不能放入 `with vf(...)` 或 VF 函数。
- 源与目标的有效访问区域不能重叠，不能当作原地转置使用。
- 运行时视图也须满足不越界、不重叠和参数表中的对齐条件。默认自动同步处理与前后搬运的依赖；手动同步模式须由调用方保证生产者写完再读。

## 调用示例

以下完整示例使用三次 16×16 块转置实现 16×48 矩阵的转置，并按 fp16 位模式验证结果。

<!-- example: vnchwconv_example.py -->

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import torch
import torch_npu
import cannbotdsl as cbd
from cannbotdsl import MemLoc, dtypes, make_layout

@cbd.kernel
def transpose(x: cbd.Tensor, y: cbd.Tensor):
    src = cbd.Buffer(MemLoc.UB, dtypes.float16, make_layout((16, 48)))
    dst = cbd.Buffer(MemLoc.UB, dtypes.float16, make_layout((48, 16)))
    cbd.mem_copy(src, x)
    s = cbd.from_tensor(src, layout=make_layout((3, 16, 16), stride=(16, 48, 1)))
    d = cbd.from_tensor(dst, layout=make_layout((3, 16, 16), stride=(256, 16, 1)))
    cbd.vnchwconv(d, s)
    # Each Vector sub-core has its own UB; only one writes the shared GM output.
    if cbd.get_subblock_id() == 0:
        cbd.mem_copy(y, dst)

@cbd.host
def run(x: cbd.Tensor, y: cbd.Tensor):
    transpose[1](x, y)

def main():
    assert torch.npu.is_available(), "NPU is required"
    x = torch.arange(16 * 48).reshape(16, 48).half()
    y = torch.empty((48, 16), dtype=torch.float16, device="npu")
    run(x.npu(), y)
    torch.npu.synchronize()
    assert torch.equal(y.cpu().contiguous().view(torch.int16), x.T.contiguous().view(torch.int16))
    print("vnchwconv example passed")

if __name__ == "__main__":
    main()
```
<!-- end-example -->

预期结果：输出 `vnchwconv example passed`。
