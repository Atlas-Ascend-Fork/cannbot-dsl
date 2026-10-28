# dcci_entire_out
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将本核DCache中GM地址对应的脏数据写回GM，并使GM地址对应的所有有效缓存行失效。适用于多处普通标量GM写入后，统一写回修改。

## 函数原型

```python
dcci_entire_out() -> None
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- 本接口不替代核间同步，也不使其他核的缓存失效；跨核读取更新后的数据时，仍需保证同步和缓存一致性。具体参见[核间同步能力概述](../resource-management/inter-core-sync-overview.md)和[缓存控制接口汇总](./overview.md#缓存控制接口汇总)。

## 调用示例

以下示例写入两个GM位置后统一发布普通标量写入，并在Host侧校验结果。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import dcci_entire_out

@kernel
def _kernel(output):
    # 写入多个GM位置后，统一清理并失效GM缓存行。
    output[0] = 7
    output[32] = 9
    dcci_entire_out()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.zeros(64, dtype=torch.int32, device="npu")
    run(output)
    torch.npu.synchronize()
    values = output.cpu()[[0, 32]].tolist()
    assert values == [7, 9]
    print(f"output values: {values}")
    print("dcci_entire_out example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
output values: [7, 9]
dcci_entire_out example passed
```
