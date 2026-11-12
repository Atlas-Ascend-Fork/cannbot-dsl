# @kernel

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

`@kernel`用于声明在NPU AI Core上执行的核函数，也可以装饰类，以组织多个核函数方法和共享配置。核函数通过Tensor参数写回计算结果，返回值应为`None`。

## 定义与启动核函数

在函数定义前添加`@cbd.kernel`，然后在[@host](./host.md)函数中启动。以下示例由一个逻辑Block将自己的索引写入输出Tensor。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.jit
def current_index():
    return cbd.get_block_idx()


@cbd.kernel
def write_index(out: cbd.Tensor):
    index = current_index()
    cbd.scalar.vec_store_bypass(out.ptr(index), index)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [0]
```

上述启动语句中：

- 方括号内的`1`表示本次启动1个逻辑Block。该Block的索引为`0`，可在核函数中通过`cbd.get_block_idx()`获取。
- 圆括号内的`out`是调用方创建的输出Tensor，对应核函数`write_index`的`out`参数。本例中的Tensor位于NPU上，包含1个`int64`元素；核函数将Block索引写入该元素，因此执行后输出为`[0]`。

支持的启动配置见[Kernel启动配置](./launch.md)。启动语句须直接写在Host函数体中。

## 配置性能分析信息

需要在性能分析工具中区分任务时，可使用`profile`配置名称、类型及Tensor参数角色。未配置时，框架使用实际入口函数名补齐名称和类型。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.jit
def current_index():
    return cbd.get_block_idx()


@cbd.kernel(profile=cbd.ProfileSpec(name="write_index", outputs=("out",)))
def write_index(out: cbd.Tensor):
    index = current_index()
    cbd.scalar.vec_store_bypass(out.ptr(index), index)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [0]
```

完整配置项及展示规则见[Profiling名称与类型](../debug/profiling.md)。

## 配置调度模式

`@cbd.kernel(schedmode=0)`使用普通调度模式，`@cbd.kernel(schedmode=1)`使用batch调度模式。默认值为`0`；该配置也可与`profile`同时使用。

## 定义Kernel类

对类使用`@kernel`，可将编译期配置通过构造参数传入，将运行时数据通过入口方法传入。以下示例将`offset`设为`7`，由`__call__`方法写入输出Tensor。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
class WriteIndex:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, out: cbd.Tensor):
        index = cbd.get_block_idx()
        cbd.scalar.vec_store_bypass(out.ptr(index), index + self.offset)


@cbd.host
def run(out):
    WriteIndex(7)[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [7]
```
