---
title: get_mem_size
api_name: get_mem_size
category: platform-profiling
api_group: host
layer: platform
call_context: host
execution_unit: host
status: experimental
since: 待追溯
---

# `get_mem_size`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

在 Host 侧查询指定片上存储的容量。

## 函数原型

```python
def get_mem_size(mem_type: str) -> int: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- | --- |
| `mem_type` | 输入 | `str` | 是 | 存储类型，不区分大小写。可选值为 `ub`（Unified Buffer）、`l1`、`l0a`、`l0b`、`l0c`、`bt`、`fb0`。 |

## 返回值说明

返回指定存储的容量，单位为字节。无法获取平台信息时返回 `0`。

## 约束说明

无。

## 调用示例

```python
import cannbotdsl as cb

ub_size = cb.get_mem_size("ub")
assert isinstance(ub_size, int)
assert ub_size >= 0
print(f"ub_size: {ub_size} bytes")
print("get_mem_size example passed")
```

### 预期结果

```text
ub_size: <UB 容量> bytes
get_mem_size example passed
```
