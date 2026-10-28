# target_version
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`target_version()`用于标记调用方提供的目标版本判断结果，在编译期选择分支或控制`while`循环。接口本身不会查询或比较硬件版本。

## 函数原型

```python
target_version(value: Any) -> bool
```

`Any`来自`typing`。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 必选，无默认值。必须是编译期已知且可转换为`bool`的Python值，不能依赖Tensor读取值、Kernel标量形参或其他运行时结果。版本判断结果由调用方提供。 |

## 返回值说明

返回Python `bool`，值为`bool(value)`。当作为`if`或`while`的完整条件时，条件判断在编译期完成，只选择命中的代码路径。

## 约束说明

无

## 调用示例

以下示例由调用方查询目标架构并计算版本判断结果，通过`target_version()`选择分支，在Host侧校验输出。

```python
import torch
import torch_npu  # noqa: F401

import cannbotdsl as cbd


TARGET_ARCH = cbd.get_platform_info().npu_arch
IS_DAV_3510 = TARGET_ARCH == "dav-3510"


@cbd.kernel
def target_version_kernel(out: cbd.Tensor):
    # 使用调用方提供的版本判断结果选择编译期分支。
    if cbd.target_version(IS_DAV_3510):
        out[0] = 950
    else:
        out[0] = 0


@cbd.host
def run(out):
    target_version_kernel[1](out)


def main():
    out = torch.zeros((1,), dtype=torch.int32, device="npu")
    run(out)
    torch.npu.synchronize()
    expected = [950 if IS_DAV_3510 else 0]
    assert out.cpu().tolist() == expected
    print("target:", TARGET_ARCH)
    print("output:", out.cpu().tolist())


if __name__ == "__main__":
    main()
```

当`TARGET_ARCH = dav-3510`时，预期结果：

```text
target: dav-3510
output: [950]
```
