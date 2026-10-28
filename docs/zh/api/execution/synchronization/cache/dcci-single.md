# dcci_single
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将本核DCache中指定GM地址所在缓存行的脏数据写回GM，并使该缓存行失效。失效后的下一次标量读取会重新从GM加载数据。

## 函数原型

```python
dcci_single(point: Any) -> None
```

`Any`来自`typing`。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `point` | 输入 | 必选，无默认值；必须直接使用GM [Tensor](../../../data-structures/tensor/overview.md)的`tensor[index]`表达式指定地址，只支持`int32`或`uint32`元素。地址无需按缓存行边界对齐。 |


## 返回值说明

无

## 约束说明

- 本接口不替代核间同步，也不使其他核的缓存失效；跨核读取更新后的数据时，仍需保证同步和缓存一致性。具体参见[核间同步能力概述](../resource-management/inter-core-sync-overview.md)和[缓存控制接口汇总](./overview.md#缓存控制接口汇总)。

## 调用示例

以下示例在写入一个GM元素后清理并失效其所在缓存行，并在Host侧校验写入值。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import dcci_single

@kernel
def _kernel(output):
    # 写入GM元素后，清理并失效其所在缓存行。
    output[0] = 7
    dcci_single(output[0])

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.zeros(32, dtype=torch.int32, device="npu")
    run(output)
    torch.npu.synchronize()
    value = int(output.cpu()[0])
    assert value == 7
    print(f"output[0]: {value}")
    print("dcci_single example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
output[0]: 7
dcci_single example passed
```
