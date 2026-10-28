# get_system_cycle
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取当前系统cycle计数器，返回64位整数类型的当前计数值。该计数器只读，复位值为 `0`，并随系统时钟持续递增，用于记录硬件运行以来累计的cycle数。该接口常用于性能统计、耗时测量和执行时序判断，在AIC和AIV上均可调用，返回值含义相同。

## 函数原型

```python
get_system_cycle() -> Int64
```

## 参数说明

无。

## 返回值说明

返回当前系统时钟周期计数，返回类型为 `Int64`。

## 约束说明

- 读取cycle计数与其他计算、搬运流水相互独立。测量这些流水上的操作时，需要先使用对应同步接口确保被测操作完成。
- 针对Ascend 950PR&950DT系列产品，若换算成时间需要按照1 GHz的频率，时间单位为us，换算公式为：`time = (cycle数 / 1000) us`。

## 调用示例

以下示例读取标量求和前后的系统周期计数，并打印求和结果与周期差。测量范围包含标量读取与累加，周期差随实际运行情况变化。

```python
import torch
import torch_npu
from cannbotdsl import host, kernel, get_system_cycle

@kernel
def measure(src, output):
    start = get_system_cycle()
    total = 0
    for i in range(src.shape[0]):
        total += src[i]
    end = get_system_cycle()
    output[0] = start
    output[1] = end
    output[2] = total

@host
def run(src, output):
    measure[1](src, output)

def main():
    src = torch.arange(128, dtype=torch.int64, device="npu")
    output = torch.empty(3, dtype=torch.int64, device="npu")
    run(src, output)
    torch.npu.synchronize()
    start, end, total = output.cpu().tolist()
    elapsed_cycles = end - start
    print(f"sum: {total}")
    print(f"elapsed_cycles: {elapsed_cycles}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
sum: 8128
elapsed_cycles: <正整数>
```
