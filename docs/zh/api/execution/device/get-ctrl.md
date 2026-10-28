# get_ctrl
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取当前核的CTRL控制寄存器值。

## 函数原型

```python
get_ctrl() -> Int64
```

## 参数说明

无。

## 返回值说明

返回CTRL控制寄存器的完整值，类型为 `Int64`。

## 约束说明

无。

## 调用示例

以下示例读取CTRL值并写回GM，在Host侧以十六进制打印结果。

```python
import cannbotdsl as cbd
import torch
import torch_npu


@cbd.kernel
def query_kernel(output: cbd.Tensor):
    output[0] = cbd.get_ctrl()


@cbd.host
def run(output: cbd.Tensor):
    query_kernel[1](output)


def main():
    output = torch.zeros(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    ctrl = int(output.cpu()[0])
    print(f"ctrl: 0x{ctrl & ((1 << 64) - 1):016x}")


if __name__ == "__main__":
    main()
```

预期结果：

```text
ctrl: 0x<16位十六进制数>
```
