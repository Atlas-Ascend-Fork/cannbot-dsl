# get_mem_size
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

查询编译目标架构对应的片上存储总容量。

## 函数原型

```python
get_mem_size(mem_type: str) -> int
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `mem_type` | 输入 | 片上存储类型，不区分大小写。可选值为 `"ub"`、`"l1"`、`"l0a"`、`"l0b"`、`"l0c"`、`"bt"`、`"fb0"`。 |

## 返回值说明

返回指定片上存储的总容量，类型为 `int`，单位为字节，不表示当前剩余可用空间。容量由编译目标架构决定，不要求当前存在可用设备；不支持查询的目标架构返回 `0`。

## 约束说明

无。

## 调用示例

以下示例查询并打印UB容量。

```python
import cannbotdsl as cbd


def main():
    ub_size = cbd.get_mem_size("ub")
    print(f"ub_size: {ub_size} bytes")


if __name__ == "__main__":
    main()
```

预期结果：

```text
ub_size: <UB容量> bytes
```
