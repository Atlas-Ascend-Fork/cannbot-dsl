# GM输入字段

`gm_in(T)`用于在参数类中声明只读GM输入字段。函数体可以按元素读取该字段，但不能写入。

## 声明方式

```python
from cannbotdsl.aicpu import I32, gm_in

class MetadataArgs:
    lengths: gm_in(I32)
```

`T`只能是`I32`、`I64`、`U32`或`U64`。

## 访问方式

GM输入字段不携带shape或长度信息，函数需要的长度应作为独立标量字段传入：

```python
from cannbotdsl.aicpu import I64, gm_in

class MetadataArgs:
    lengths: gm_in(I64)
    count: I64

@aicpu_kernel
def read_lengths(args: MetadataArgs):
    for i in range(args.count):
        value = args.lengths[i]
    return 0
```

索引以元素为单位。调用方必须保证函数访问的索引位于输入缓冲区范围内。

## 与GM输出字段的区别

`gm_in(T)`只能读取，不能写入；需要写回结果时，使用[GM输出字段](./gm-out.md)。
