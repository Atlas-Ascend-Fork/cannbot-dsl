# aot.register_directory

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`register_directory` 将 `aot.build` 生成的 Native 产物目录加入当前进程的查找路径，使后续 Host 调用可以复用其中已发布的编译结果。

## 函数原型

```python
cannbotdsl.aot.register_directory(path) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `path` | 输入 | 包含 Native 产物清单的目录路径，可以是字符串或 PathLike。 |

## 返回值说明

无返回值。

## 约束说明

- `path` 必须指向完整的 Native 产物目录，不能只指向单个二进制文件。
- 注册操作不会执行编译；资源在后续查找时按需读取。

## 调用示例

```python
from pathlib import Path
from cannbotdsl import aot

aot.register_directory(Path(__file__).parent / "_native")
```
