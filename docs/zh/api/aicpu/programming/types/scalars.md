# 标量类型

AI CPU 接口提供四个标量类型常量，用于声明参数类中的整数参数，也可以作为 GM 字段的元素类型。

## 类型列表

| 类型 | 符号 | 取值范围 |
| --- | --- | --- |
| `I32` | 32位有符号整数 | `-2**31`至`2**31 - 1` |
| `I64` | 64位有符号整数 | `-2**63`至`2**63 - 1` |
| `U32` | 32位无符号整数 | `0`至`2**32 - 1` |
| `U64` | 64位无符号整数 | `0`至`2**64 - 1` |

## 声明标量参数

```python
from cannbotdsl.aicpu import I32, I64

class MetadataArgs:
    task_count: I32
    offset: I64
```

调用时分别传入Python整数：

```python
metadata_kernel.run_host(task_count=8, offset=0)
compiled.launch(stream, task_count=8, offset=0)
```

## 作为GM元素类型

标量类型也用于声明GM字段的元素类型：

```python
from cannbotdsl.aicpu import I32, gm_in, gm_out

class MetadataArgs:
    lengths: gm_in(I32)
    metadata: gm_out(I32)
```

GM缓冲区的存储类型必须与字段声明一致。例如，`gm_in(I32)`对应32位有符号整数缓冲区，不能传入其他元素类型的缓冲区。

## 选择类型

- 计数、长度、索引等非负值可以根据数据范围选择`I32`或`I64`。
- 需要表达完整无符号范围时选择`U32`或`U64`。
- GM输入、输出和标量字段应使用与调用方缓冲区或整数值一致的类型。
