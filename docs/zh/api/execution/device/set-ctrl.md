# set_ctrl
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

设置当前核的CTRL控制寄存器。

## 函数原型

```python
set_ctrl(value: int) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 写入CTRL控制寄存器的完整值，支持Python整数或 `Int64` 标量。 |

## 返回值说明

无返回值。写入值用于更新当前核的CTRL控制寄存器。

## 约束说明

无。

## 调用示例

以下示例读取当前CTRL值并原样写回，再读取并打印写入前后的值。

```python
import cannbotdsl as cbd
import torch
import torch_npu


@cbd.kernel
def write_kernel(output: cbd.Tensor):
    current = cbd.get_ctrl()
    output[0] = current
    cbd.set_ctrl(current)
    output[1] = cbd.get_ctrl()


@cbd.host
def run(output: cbd.Tensor):
    write_kernel[1](output)


def main():
    output = torch.zeros(2, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    before, after = output.cpu().tolist()
    print(f"before: 0x{before & ((1 << 64) - 1):016x}")
    print(f"after:  0x{after & ((1 << 64) - 1):016x}")


if __name__ == "__main__":
    main()
```

预期结果：

```text
before: 0x<16位十六进制数>
after:  0x<与before相同的16位十六进制数>
```
