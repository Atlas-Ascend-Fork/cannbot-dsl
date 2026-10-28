# CopyOptions

## 类型说明

`cannbotdsl.ops.memcpy.CopyOptions` 是不可变的数据搬运配置类型，由 [copy_options](copy-options.md) 创建，传给 `mem_copy(dst, src, options=...)` 使用。不直接调用其构造函数，也不修改内部字段。

配置选项的取值、默认行为和适用方向统一见 [copy_options 的参数说明](copy-options.md#参数说明)。派生配置只覆盖显式传入的选项，原对象保持不变。

## 公开属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `engine` | `str` | 只读。返回显式选择的搬运方式 `"auto"` 或 `"nddma"`；未指定或显式传入 `None` 时返回 `"auto"`。 |
| `channel_split` | `ChannelSplitMode` | 只读。返回通道拆分的配置模式，不表示读取了硬件实际执行状态。 |

其他搬运控制项通过 `copy_options` 设置或派生，不提供同名的公开可写属性。

## ChannelSplitMode

`cannbotdsl.ChannelSplitMode` 是 `channel_split` 属性的返回类型。

| 枚举成员 | 含义 |
| --- | --- |
| `AUTO` | 自动选择，对应配置值 `None`。 |
| `ON` | 启用通道拆分，对应配置值 `True`。 |
| `OFF` | 禁用通道拆分，对应配置值 `False`。 |

`copy_options(channel_split=...)` 接收的是 `bool` 或 `None`，不能把这些枚举成员作为该参数传入。
