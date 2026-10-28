# get_platform_info
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

查询NPU型号和可用核数。若执行流设置了核数上限，返回的核数不会超过该上限。

## 函数原型

```python
get_platform_info(
    force_refresh: bool = False,
    stream: torch.npu.Stream | int | None = None,
) -> PlatformInfo
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `force_refresh` | 输入 | 是否重新查询硬件信息。`True` 表示重新查询，`False` 表示优先使用缓存，默认值为 `False`。 |
| `stream` | 输入 | 指定查询核数时使用的NPU执行流，支持 `torch.npu.Stream` 对象或整数形式的流句柄。默认值为 `None`，使用当前PyTorch NPU执行流。 |

## 返回值说明

返回一个包含NPU平台信息的 `PlatformInfo` 对象。

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `soc_version` | `str` | NPU架构版本。 |
| `short_soc_version` | `str` | 芯片型号简称。 |
| `npu_arch` | `str` | NPU编译目标架构。 |
| `core_num` | `int` | 可用AI Core数量。 |
| `cube_core_num` | `int` | 可用Cube Core数量。 |
| `vector_core_num` | `int` | 可用Vector Core数量。 |
| `ai_cpu_num` | `int` | AI CPU数量。 |
| `available` | `bool` | 是否成功获取平台信息。 |

无法加载CANN平台查询库时，型号及架构字符串为空、核数为 `0`，`available` 为 `False`。

## 约束说明

无。

## 调用示例

以下示例查询并打印NPU平台信息。

```python
import cannbotdsl as cbd


def main():
    info = cbd.get_platform_info()

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


if __name__ == "__main__":
    main()
```

预期结果：

以下是成功获取平台信息时的示例输出，具体型号和核数以实际设备及执行流配置为准。

```text
available: True
soc_version: <芯片架构版本>
short_soc_version: <芯片型号简称>
npu_arch: <编译架构名>
core counts: total=<AI Core数量>, cube=<Cube Core数量>, vector=<Vector Core数量>, ai_cpu=<AI CPU数量>
```
