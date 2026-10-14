---
title: make_buffer
api_name: make_buffer
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `make_buffer(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

把一个由 `dsl.UB.view()` 根视图派生出的静态 UB Tensor 视图绑定为硬件 Buffer
资源。该操作复用原视图的存储根和地址，只增加 Buffer 资源身份，不会生成新的
片上分配，也不会复制数据。

## 函数原型

```python
make_buffer(view: Tensor) -> Tensor
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `view` | `Tensor` | 是 | 无 | 待绑定的静态 UB 视图。必须可沿受支持的静态视图链追溯到 `dsl.UB.view()` 创建的根分配。其 shape 和 stride 的每一项都必须是严格大于 0 的 Python int。 |

## 返回值说明

返回绑定了 Buffer 资源身份的 UB `Tensor`。返回 Tensor 与输入 view 共享存储根、
地址和已有数据。

## 约束说明

- view 的 shape 和 stride 必须全部为正的静态 Python int；动态维或动态 stride
  不允许绑定。
- 使用显式 UB 空间时，后续 UB Buffer 必须通过 `make_buffer()` 创建。

## 调用示例

```python
import cannbotdsl as cbd
from torch import as_tensor as from_torch_npu
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def copy_with_explicit_ub(x: cbd.Tensor, y: cbd.Tensor):
    ub_root = cbd.UB.view(256 * 1024)

    # 取根视图开头 64 字节。
    slot_view = ub_root.reinterpret(
        shape=(64,),
        offset=0,
    )
    tmp = cbd.make_buffer(slot_view)

    cbd.mem_copy(tmp, x)
    cbd.mem_copy(y, tmp)


@cbd.host
def run(x, y):
    copy_with_explicit_ub[1](x, y)


x = torch.arange(64, device="npu:0").byte()
y = torch.zeros_like(x)
run(from_torch_npu(x), from_torch_npu(y))
torch.npu.synchronize()
torch.testing.assert_close(y, x)
print("output:", y.cpu().tolist())
```

### 示例输入

```text
x: Host 侧绑定的 NPU Tensor(shape=(64,))
   内容 [0, 1, 2, ..., 63]
y: Host 侧绑定的 NPU Tensor(shape=(64,))
   初始内容全为 0

slot_view:
  来自 dsl.UB.view() 根视图
  offset = 0 字节
  shape = (64,)
  覆盖字节数 = 64
```

### 预期输出

```text
output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
```
