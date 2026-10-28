# OpKit ACLNN 构建

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`opkit.build` 查找带有 `@opkit.aclnn` 声明的入口，生成 ACLNN 工程所需的设备二进制和 C++ 源码。生成目录是后续 C++/CMake 构建的输入，不是已经安装好的最终库文件。

## 函数原型

```python
opkit.build(*, collectors, target, output, operators=None, pattern="*.py", l0=False, l2=False)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `collectors` | 输入 | 包含 `@opkit.aclnn` 入口的模块或目录。 |
| `target` | 输入 | 目标设备型号。 |
| `output` | 输入 | 生成 ACLNN 构建输入的目录。 |
| `operators` | 输入 | 可选的算子名称筛选列表。 |
| `pattern` | 输入 | 扫描 Python 文件的匹配模式。 |
| `l0` | 输入 | 是否生成 L0 相关 ACLNN 实现，默认关闭。 |
| `l2` | 输入 | 是否生成 L2 相关 ACLNN 实现，默认关闭。 |

## 返回值说明

返回生成的 ACLNN 构建目录。

## 约束说明

- `collectors` 中至少要有一个 `@opkit.aclnn` 入口。
- `opkit.build` 生成的是设备二进制和 C++ 构建输入；最终库文件的名称、安装目录和 CMake 规则由用户工程决定。

## 调用示例

下面的示例生成 ACLNN 构建输入：

```python
import opkit

output = opkit.build(collectors="my_project", target="dav-3510",
                     output="build/aclnn")
print(output)
```
