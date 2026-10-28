# GM输出字段

`gm_out(T)`用于在参数类中声明可读写GM输出字段。调用方预先分配输出缓冲区，函数体通过元素索引写入结果。

## 声明方式

```python
from cannbotdsl.aicpu import I32, gm_out

class MetadataArgs:
    metadata: gm_out(I32)
```

`T`只能是`I32`、`I64`、`U32`或`U64`。

## 访问方式

GM输出字段不携带shape或长度信息，函数需要的长度应作为独立标量字段传入：

```python
from cannbotdsl.aicpu import I64, gm_out

class MetadataArgs:
    metadata: gm_out(I64)
    count: I64

@aicpu_kernel
def write_metadata(args: MetadataArgs):
    for i in range(args.count):
        args.metadata[i] = i
    return 0
```

当函数先读取输出字段原值再更新时，调用方必须在启动前初始化这些元素。

## 与GM输入字段的区别

`gm_out(T)`允许读写；只需要读取输入时，使用[GM输入字段](./gm-in.md)。完整参数类示例见[参数声明](./parameters.md)。
