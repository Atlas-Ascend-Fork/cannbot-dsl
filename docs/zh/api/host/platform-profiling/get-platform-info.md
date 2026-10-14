---
title: get_platform_info
api_name: get_platform_info
category: platform-profiling
api_group: host
layer: platform
call_context: host
execution_unit: host
status: experimental
since: 待追溯
---

# `get_platform_info`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

在 Host 侧查询当前 NPU 的型号和可用核数。NPU 执行流（stream）可以限制本次任务最多使用多少个核；如果指定的stream设置了该限制，接口返回的核数不会超过限制值。例如，设备共有 28 个核，而stream最多允许使用 14 个核时，`core_num` 返回 `14`。

## 函数原型

```python
def get_platform_info(
    force_refresh: bool = False,
    stream: torch.npu.Stream | None = None,
) -> PlatformInfo: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `force_refresh` | 输入 | `bool` | 否 | `False` | 是否重新查询硬件信息。 |
| `stream` | 输入 | `torch.npu.Stream` 或 `None` | 否 | `None` | 指定要查询的 NPU 执行流。接口返回该执行流可使用的核数。取值为 `None` 时，查询当前 PyTorch NPU 执行流；如果无法获取当前执行流，则返回设备的硬件核数。 |

## 返回值说明

返回 `PlatformInfo` 对象，包含以下公开属性。

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `soc_version` | `str` | NPU 架构版本，例如 `DAV_3510`。 |
| `short_soc_version` | `str` | 芯片型号简称，例如 `Ascend950`。 |
| `npu_arch` | `str` | Bisheng 编译器的 `--npu-arch` 参数值，例如 `dav-3510`。 |
| `core_num` | `int` | 设备核数与执行流核数上限中的较小值。 |
| `cube_core_num` | `int` | 设备 Cube Core 数量与执行流 Cube Core 上限中的较小值。 |
| `vector_core_num` | `int` | 设备 Vector Core 数量与执行流 Vector Core 上限中的较小值。 |
| `ai_cpu_num` | `int` | AI CPU 数量。 |
| `available` | `bool` | 是否成功获取平台信息。 |

## 约束说明

无。

## 调用示例

```python
import cannbotdsl as cb

info = cb.get_platform_info()
assert isinstance(info, cb.PlatformInfo)
assert isinstance(info.available, bool)
assert info.core_num >= 0

print(f"available: {info.available}")
print(f"soc_version: {info.soc_version}")
print(f"short_soc_version: {info.short_soc_version}")
print(f"npu_arch: {info.npu_arch}")
print(
    "core counts: "
    f"total={info.core_num}, "
    f"cube={info.cube_core_num}, "
    f"vector={info.vector_core_num}, "
    f"ai_cpu={info.ai_cpu_num}"
)
print("get_platform_info example passed")
```

### 预期结果

以下是成功获取平台信息时的示例输出，具体型号和核数以实际设备及执行流配置为准。

```text
available: True
soc_version: <芯片架构版本>
short_soc_version: <芯片型号简称>
npu_arch: <编译架构名>
core counts: total=<AI Core 数量>, cube=<Cube Core 数量>, vector=<Vector Core 数量>, ai_cpu=<AI CPU 数量>
get_platform_info example passed
```
