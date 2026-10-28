# dcci_entire_atomic
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将本核数据缓存（DCache）中由标量原子操作访问的GM缓存行写回GM，并使其失效。

## 函数原型

```python
dcci_entire_atomic() -> None
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- 本接口不替代核间同步，也不使其他核的缓存失效；跨核读取更新后的数据时，仍需保证同步和缓存一致性。具体参见[核间同步能力概述](../resource-management/inter-core-sync-overview.md)和[缓存控制接口汇总](./overview.md#缓存控制接口汇总)。

## 调用示例

以下示例对计数器执行原子加后发布缓存更新，并在Host侧校验计数结果。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import atomic_add
from cannbotdsl.ops.sync import dcci_entire_atomic

@kernel
def _kernel(counter):
    atomic_add(counter.ptr(0), 1)
    dcci_entire_atomic()

@host
def run(counter):
    _kernel[1](counter)

def main():
    counter = torch.zeros(1, dtype=torch.int32, device="npu")
    run(counter)
    torch.npu.synchronize()
    value = int(counter.cpu()[0])
    assert value == 1
    print(f"counter: {value}")
    print("dcci_entire_atomic example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
counter: 1
dcci_entire_atomic example passed
```
