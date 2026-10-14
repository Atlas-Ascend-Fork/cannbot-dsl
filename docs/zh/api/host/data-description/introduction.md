# Host 参数与数据描述

本目录介绍 Host 侧用于描述输入数据的类型。调用 `cannbotdsl.compile(...)`
进行提前编译时，还没有可供程序读取的真实 NPU 数据，因此需要使用
`TensorSpec`、`TensorListSpec` 等类型告诉编译器：将来传入的 Tensor 是什么
shape、dtype 和 stride，或者 Tensor 列表有多少个元素。

这些 Spec 只记录输入应满足的条件，不保存 Tensor 数据。编译完成后执行程序时，
仍然需要传入真实的 NPU `Tensor` 或 Tensor 列表。

## 接口一览

| 文档 | 接口 | 作用 |
| --- | --- | --- |
| [Dim](./dim.md) | `Dim` | 为 Tensor shape、stride 或 Tensor 列表长度声明动态值及其范围约束。 |
| [TensorSpec](./tensor_spec.md) | `TensorSpec` | 提前编译单个 Tensor 参数时，告诉编译器它的 shape、dtype、stride 和存储格式。 |
| [TensorListSpec](./tensor_list_spec.md) | `TensorListSpec` | 提前编译 Tensor 列表参数时，告诉编译器列表长度，以及每个 Tensor 的类型和形状规则。 |

## 使用要点

- `TensorSpec` 对应设备侧 `Tensor` 形参，`TensorListSpec` 对应设备侧
  `TensorList` 形参。
- 两种规格都只传给 `cannbotdsl.compile(...)`；直接调用 `@host` 函数时应传真实 NPU Tensor
  （例如 torch_npu 创建的 `torch.Tensor`），或由真实 NPU Tensor 组成的
  `list`/`tuple`。
- `Dim` 用于描述动态 shape、动态 stride 或动态列表长度及其取值边界。
- 规格对象是不可变的纯 Python 描述，不触发数据分配、数据搬运或 Kernel 执行。
