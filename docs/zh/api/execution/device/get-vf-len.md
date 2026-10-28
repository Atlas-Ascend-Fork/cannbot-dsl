# get_vf_len
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取矢量数据寄存器位宽VL（Vector Length）的大小。

## 函数原型

```python
get_vf_len() -> Int64
```

## 参数说明

无。

## 返回值说明

返回VL的大小，单位为字节，返回类型为 `Int64`。Ascend 950PR&950DT系列产品的矢量数据寄存器位宽为256字节，本接口返回 `256`。

## 约束说明

无。

## 调用示例

以下示例读取并打印向量寄存器长度，单位为字节。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_vf_len

@kernel
def _kernel(output):
    output[0] = get_vf_len()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.empty(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    vf_len = int(output.cpu()[0])
    print(f"vf_len: {vf_len}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vf_len: 256
```
