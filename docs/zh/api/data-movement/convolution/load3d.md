# load3d

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从 L1 中按 NC1HWC0 排布的单个样本特征图中提取卷积窗口，将每个输出位置所需的输入值展开为矩阵的一行，再把指定的矩阵区域搬入 L0A 或 L0B，供 `matmul` 使用。

窗口按 `stride` 移动，按 `dilation` 取点；超出特征图边界且位于声明填充范围内的位置使用 `padding` 指定的值。相邻窗口重叠时，同一输入值会出现在展开矩阵的不同行中。

![Load3D 按卷积窗口展开特征图](../../figures/conv2d_im2col_windows.png)

图中红色、黄色窗口分别展开为第一行、第二行，Ci=4 仅为简化示意。此图描述逻辑展开关系，写入 L0A/L0B 时仍须采用相应的分形排布。

使用时，通过 `feature` 描述源特征图及四周填充范围，通过 `filter`、`stride`、`dilation` 描述窗口取数规则，通过 `window` 选择展开矩阵的起点和大小；`repeat`、`destination` 分别控制重复搬运和目标存放位置。`window` 的范围始终按转置前的矩阵描述：写入 L0A 时可通过 `transpose` 选择是否转置，写入 L0B 时始终转置。

令 `H=feature.height`、`W=feature.width`，`(F_h,F_w)=filter`、`(S_h,S_w)=stride`、`(D_h,D_w)=dilation`，则：

$$
H_o=\left\lfloor\frac{H+\mathrm{feature.pad\_top}+\mathrm{feature.pad\_bottom}-D_h(F_h-1)-1}{S_h}\right\rfloor+1
$$

$$
W_o=\left\lfloor\frac{W+\mathrm{feature.pad\_left}+\mathrm{feature.pad\_right}-D_w(F_w-1)-1}{S_w}\right\rfloor+1
$$

展开矩阵的 M 轴表示输出空间位置，共 `H_o*W_o` 项；K 轴表示卷积核位置及输入通道，共 `feature.channels*F_h*F_w` 项。`window` 在转置前选择这两个轴的范围。

## 函数原型

```python
load3d(
    dst: Tensor,
    src: Tensor,
    *,
    feature: Load3DFeatureMap,
    window: Load3DWindow,
    filter: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    transpose: bool = False,
    config_bank: str | int = "a",
    repeat: Load3DRepeat = Load3DRepeat(),
    destination: Load3DDestination | None = None,
    padding: Load3DPadding = Load3DPadding.value(0),
) -> None
```

运行时确定的几何参数也须满足相同的取值范围和容量要求。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dst | 输出 | L0A 或 L0B 目标 `Tensor` 。<br>必填，无默认值。类型为 `Tensor`，起点须按 512B 对齐，存储须覆盖目标偏移及全部重复写入；未写入区域保持原值。 |
| src | 输入 | 源 L1 `Tensor`，NC1HWC0 排布。<br>必填，无默认值。类型为 `Tensor`，起点须按 32B 对齐；元素类型须与 `dst` 相同，支持 `int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`float16`、`bfloat16`、`float32`、`float8_e4m3fn`、`float8_e5m2`、`hifloat8`。 |
| feature | 输入 | 描述源特征图的高、宽、通道数及四周填充范围，见 [Load3DFeatureMap](../../data-structures/convolution/load3d-feature-map.md)。 |
| window | 输入 | 指定转置前展开矩阵的起点及搬运范围，见 [Load3DWindow](../../data-structures/convolution/load3d-window.md)。 |
| filter | 输入 | 指定卷积核的高、宽，顺序为 `(filter_h, filter_w)`。<br>默认 `(1, 1)`。支持包含 2 个整数的 tuple/list，每项为 0～511；0 表示空搬运。 |
| stride | 输入 | 指定窗口在高、宽方向每次移动的距离，顺序为 `(stride_h, stride_w)`。<br>默认 `(1, 1)`。支持包含 2 个整数的 tuple/list，每项为 1～63。 |
| dilation | 输入 | 指定窗口内部相邻采样点在高、宽方向的距离，顺序为 `(dilation_h, dilation_w)`；1 表示连续取点。<br>默认 `(1, 1)`。支持包含 2 个整数的 tuple/list，每项为 1～255。 |
| transpose | 输入 | 指定写入 L0A 前是否转置展开矩阵；写入 L0B 时始终转置，不受此参数控制。<br>默认 `False`。支持布尔值。 |
| config_bank | 输入 | 选择本次使用的配置槽，与目标位于 L0A 还是 L0B 无关。<br>默认 `"a"`。支持字符串 `"a"`、`"b"` 或整数选择值 `0`、`1`，分别对应配置槽 A、B。 |
| repeat | 输入 | 控制重复搬运次数及源数据推进方式，见 [Load3DRepeat](../../data-structures/convolution/load3d-repeat.md)。 |
| destination | 输入 | 控制目标分形块的间距及写入起点，见 [Load3DDestination](../../data-structures/convolution/load3d-destination.md)。 |
| padding | 输入 | 指定特征图边界填充位置使用的单元素值，见 [Load3DPadding](../../data-structures/convolution/load3d-padding.md)。 |

## 返回值说明

无返回值。展开结果写入 `dst`，不创建新的 Tensor 或分配存储。

## 约束说明

- 在 kernel 内使用，源特征图先初始化，再展开，再由 `matmul` 等消费者读取目标。使用默认自动同步时保持 Buffer/Tensor 的读写关系，不通过无关裸地址绕过资源依赖。
- 同一批次的全部源通道平面必须在 L1 中按声明的 NC1HWC0 排列。多个 batch 由调用方分别选择源视图，不把 batch 合并到 `feature.channels`。
- 非空展开时，filter 与 dilation 必须能落在补边后的特征图几何范围内。运行时几何同样必须满足参数表中的范围及容量要求。

## 调用示例

以下示例展开 L1 特征图，通过单位矩阵乘法读回结果并校验。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def kernel(src: cb.Tensor, eye: cb.Tensor, out: cb.Tensor):
    fmap = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((256,)), data_format="nd")
    identity = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((16, 16)))
    a = cb.Buffer(cb.MemLoc.L0A, cb.dtypes.float16, cb.make_layout((16, 16)))
    b = cb.Buffer(cb.MemLoc.L0B, cb.dtypes.float16, cb.make_layout((16, 16)))
    c = cb.Buffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((16, 16)))
    cb.mem_copy(fmap, src)
    cb.mem_copy(identity, eye)
    cb.load3d(a, fmap,
        feature=cb.Load3DFeatureMap(height=4, width=4, channels=16),
        window=cb.Load3DWindow(m_extent=16, k_extent=16),
        destination=cb.Load3DDestination(k_stride_fractals=1),
        repeat=cb.Load3DRepeat(count=1), padding=cb.Load3DPadding.value(0))
    cb.mem_copy(b, identity)
    cb.matmul(c, a, b, init=True)
    cb.mem_copy(out, c)

@cb.host
def launch(src: cb.Tensor, eye: cb.Tensor, out: cb.Tensor):
    kernel[1](src, eye, out)

def main():
    matrix = torch.arange(256, dtype=torch.float16).reshape(16, 16)
    # One C1 plane: (C1, H, W, C0) = (1, 4, 4, 16).
    src = matrix.flatten().to("npu:0")
    eye = torch.eye(16, dtype=torch.float16, device="npu:0")
    out = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    launch(src, eye, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), matrix.float(), rtol=0, atol=0)
    print("load3d example passed")

if __name__ == "__main__":
    main()
```

预期结果：校验通过，输出 `load3d example passed`。
