# range_constexpr

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

范围规则与Python `range`相同；包含它的循环在编译期展开，不保留运行时循环。展开次数会增加生成代码的体积，适合较小且固定的迭代范围。

## 函数原型

```python
range_constexpr(*args: int) -> range
```

支持以下调用形式：`range_constexpr(stop)`、`range_constexpr(start, stop)`和`range_constexpr(start, stop, step)`。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `*args` | 输入 | 一至三个编译期已知的Python整数位置参数，依次对应`stop`、`start, stop`或`start, stop, step`，单位为循环计数。`start`默认为0，`stop`为循环终点（不包含），无默认值，`step`默认为1；范围规则与Python `range`相同。 |

## 返回值说明

返回Python `range`对象。

## 约束说明

- 循环中的`break`、`continue`遵守编译期执行的Python循环语义。

## 调用示例

以下示例使用编译期循环计算0至3的和。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def constexpr_range_kernel(out: cbd.Tensor):
    total = 0
    for i in cbd.range_constexpr(4):
        total = total + i
    out[0] = total


@cbd.host
def run(out):
    constexpr_range_kernel[1](out)


def main():
    out = torch.zeros(1, dtype=torch.int64, device="npu")
    run(out)
    result = out.cpu().tolist()
    assert result == [6]
    print(result)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[6]
```
