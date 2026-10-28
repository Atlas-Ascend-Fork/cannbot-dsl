# is_aic
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

判断当前编译通道是否为Cube（AIC）。

## 函数原型

```python
is_aic() -> Boolean
```

## 参数说明

无。

## 返回值说明

返回 `Boolean` 值。AIC编译通道返回 `True`，AIV或MIX编译通道返回 `False`。

## 约束说明

无。

## 调用示例

以下示例将is_aic的判断结果写回GM，并在Host侧打印。

```python
import cannbotdsl as cbd
import torch
import torch_npu


@cbd.kernel
def query_kernel(output: cbd.Tensor):
    if cbd.is_aic():
        output[0] = 1
    else:
        output[0] = 0


@cbd.host
def run(output: cbd.Tensor):
    query_kernel[1](output)


def main():
    output = torch.zeros(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    print("is_aic:", bool(output.cpu()[0].item()))


if __name__ == "__main__":
    main()
```

预期结果：

```text
is_aic: <True或False>
```
