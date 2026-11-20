# log 统一错误模板

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

提供形状、类型、属性、配置及文件错误的统一日志模板。每次调用先记录ERROR日志，再上报对应的结构化错误消息。框架为每个接口绑定EZ错误码，调用时无需传入，也不支持自定义。

| 内部编码 | 接口及位置参数 | 用途 |
| --- | --- | --- |
| EZ0001 | [`log.error_with_invalid_input_shape(entity_name, index, incorrect_shape, correct_shape)`](#error_with_invalid_input_shape) | 输入形状不符合要求 |
| EZ0002 | [`log.error_with_invalid_attr(entity_name, attr_name, incorrect_val, correct_val)`](#error_with_invalid_attr) | 算子属性值不合法 |
| EZ0003 | [`log.error_with_invalid_attr_size(entity_name, attr_name, incorrect_size, correct_size)`](#error_with_invalid_attr_size) | 算子属性大小不合法 |
| EZ0004 | [`log.error_with_invalid_input(entity_name, param_name)`](#error_with_invalid_input) | 必需输入或参数为空 |
| EZ0005 | [`log.error_with_invalid_input_shapesize(entity_name, index, incorrect_size, correct_size)`](#error_with_invalid_input_shapesize) | 输入 shape size 不合法 |
| EZ0006 | [`log.error_with_invalid_input_format(entity_name, param_name, data_format, expected_format_list)`](#error_with_invalid_input_format) | 输入格式不在允许集合中 |
| EZ0007 | [`log.error_with_invalid_input_dtype(entity_name, param_name, data_dtype, expected_dtype_list)`](#error_with_invalid_input_dtype) | 输入 dtype 不在允许集合中 |
| EZ0008 | [`log.error_for_invalid_shape(entity_name, param_name, incorrect_shape, correct_shape)`](#error_for_invalid_shape) | 指定参数形状不合法 |
| EZ0009 | [`log.error_for_invalid_shape_with_reason(entity_name, param_name, incorrect_shape, reason)`](#error_for_invalid_shape_with_reason) | 指定参数形状不合法并说明原因 |
| EZ0010 | [`log.error_for_invalid_shapes_with_reason(entity_name, param_names, incorrect_shapes, reason)`](#error_for_invalid_shapes_with_reason) | 多个参数形状关系不合法 |
| EZ0011 | [`log.error_for_invalid_shapedim(entity_name, param_name, incorrect_dim, correct_dim)`](#error_for_invalid_shapedim) | 指定参数 shape dim 不合法 |
| EZ0012 | [`log.error_for_invalid_shapedim_with_reason(entity_name, param_name, incorrect_dim, reason)`](#error_for_invalid_shapedim_with_reason) | 指定参数 shape dim 不合法并说明原因 |
| EZ0013 | [`log.error_for_invalid_shapedims_with_reason(entity_name, param_names, incorrect_dims, reason)`](#error_for_invalid_shapedims_with_reason) | 多个参数 shape dim 不合法 |
| EZ0014 | [`log.error_for_invalid_shapesize(entity_name, param_name, incorrect_size, correct_size)`](#error_for_invalid_shapesize) | 指定参数 shape size 不合法 |
| EZ0015 | [`log.error_for_invalid_shapesize_with_reason(entity_name, param_name, incorrect_size, reason)`](#error_for_invalid_shapesize_with_reason) | 指定参数 shape size 不合法并说明原因 |
| EZ0016 | [`log.error_for_invalid_shapesizes_with_reason(entity_name, param_names, incorrect_sizes, reason)`](#error_for_invalid_shapesizes_with_reason) | 多个参数 shape size 不合法 |
| EZ0017 | [`log.error_for_invalid_format(entity_name, param_name, incorrect_format, correct_format)`](#error_for_invalid_format) | 指定参数格式不合法 |
| EZ0018 | [`log.error_for_invalid_formats_with_reason(entity_name, param_names, incorrect_formats, reason)`](#error_for_invalid_formats_with_reason) | 多个参数格式不合法 |
| EZ0019 | [`log.error_for_invalid_dtype(entity_name, param_name, incorrect_dtype, correct_dtype)`](#error_for_invalid_dtype) | 指定参数 dtype 不合法 |
| EZ0020 | [`log.error_for_invalid_dtype_with_reason(entity_name, param_name, incorrect_dtype, reason)`](#error_for_invalid_dtype_with_reason) | 指定参数 dtype 不合法并说明原因 |
| EZ0021 | [`log.error_for_invalid_dtypes_with_reason(entity_name, param_names, incorrect_dtypes, reason)`](#error_for_invalid_dtypes_with_reason) | 多个参数 dtype 不合法 |
| EZ0022 | [`log.error_for_invalid_tensornum(entity_name, param_name, incorrect_num, correct_num)`](#error_for_invalid_tensornum) | tensor 列表中的 tensor 数量不合法 |
| EZ0023 | [`log.error_for_invalid_tensornums_with_reason(entity_name, param_names, incorrect_nums, reason)`](#error_for_invalid_tensornums_with_reason) | 多个 tensor 列表的 tensor 数量不合法 |
| EZ0024 | [`log.error_for_invalid_value(entity_name, param_name, incorrect_value, correct_value)`](#error_for_invalid_value) | 参数值不符合期望 |
| EZ0025 | [`log.error_for_invalid_listsize(entity_name, param_name, incorrect_size, correct_size)`](#error_for_invalid_listsize) | 列表大小不合法 |
| EZ0026 | [`log.error_for_invalid_value_with_reason(entity_name, param_name, incorrect_value, reason)`](#error_for_invalid_value_with_reason) | 参数值不合法并说明原因 |
| EZ0027 | [`log.error_for_invalid_values_with_reason(entity_name, param_names, incorrect_values, reason)`](#error_for_invalid_values_with_reason) | 多个参数值不合法 |
| EZ0028 | [`log.error_for_invalid_stride(entity_name, param_name, incorrect_stride, correct_stride)`](#error_for_invalid_stride) | tensor 步长不合法 |
| EZ0029 | [`log.error_for_file_path(op_name, file_path, reason)`](#error_for_file_path) | 文件路径不合法 |
| EZ0030 | [`log.error_for_file_open(op_name, file_path, reason)`](#error_for_file_open) | 文件打开失败 |
| EZ0031 | [`log.error_for_file_parse(op_name, file_name, reason)`](#error_for_file_parse) | 文件解析失败 |
| EZ0032 | [`log.error_for_invalid_config(op_name, config_file, item, value, expect)`](#error_for_invalid_config) | 配置项值不符合期望 |
| EZ0033 | [`log.error_for_invalid_config_with_reason(op_name, config_file, item, value, reason)`](#error_for_invalid_config_with_reason) | 配置项值不合法并说明原因 |
| EZ0034 | [`log.error_for_invalid_configs_with_reason(op_name, config_file, items, values, reason)`](#error_for_invalid_configs_with_reason) | 多个配置项值不合法 |
| EZ0035 | [`log.error_for_invalid_format_with_reason(entity_name, param_name, incorrect_format, reason)`](#error_for_invalid_format_with_reason) | 指定参数格式不合法并说明原因 |
| EZ0036 | [`log.error_for_invalid_graph_node(node_name, reason)`](#error_for_invalid_graph_node) | 输入图节点不合法 |
| EZ0037 | [`log.error_for_invalid_argument_with_reason(entity_name, param_name, reason)`](#error_for_invalid_argument_with_reason) | 通用参数不合法并说明原因 |
| EZ0038 | [`log.error_for_invalid_listsize_with_reason(entity_name, param_name, incorrect_size, reason)`](#error_for_invalid_listsize_with_reason) | 列表元素数量不合法并说明原因 |

## error_with_invalid_input_shape

输入形状不符合要求，对应错误码EZ0001。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_input_shape(entity_name, index, incorrect_shape, correct_shape, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `index` | 输入 | 输入序号；框架原样记录，不进行起始下标转换 |
| `incorrect_shape` | 输入 | 实际形状 |
| `correct_shape` | 输入 | 期望或允许的形状 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_with_invalid_attr

算子属性值不合法，对应错误码EZ0002。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_attr(entity_name, attr_name, incorrect_val, correct_val, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `attr_name` | 输入 | 属性名称 |
| `incorrect_val` | 输入 | 实际的错误属性值 |
| `correct_val` | 输入 | 期望或允许的属性值 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_with_invalid_attr_size

算子属性大小不合法，对应错误码EZ0003。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_attr_size(entity_name, attr_name, incorrect_size, correct_size, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `attr_name` | 输入 | 属性名称 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与 incorrect_size 使用相同度量 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_with_invalid_input

必需输入或参数为空，对应错误码EZ0004。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_input(entity_name, param_name, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_with_invalid_input_shapesize

输入 shape size 不合法，对应错误码EZ0005。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_input_shapesize(entity_name, index, incorrect_size, correct_size, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `index` | 输入 | 输入序号；框架原样记录，不进行起始下标转换 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与 incorrect_size 使用相同度量 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_with_invalid_input_format

输入格式不在允许集合中，对应错误码EZ0006。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_input_format(entity_name, param_name, data_format, expected_format_list, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `data_format` | 输入 | 实际输入数据格式 |
| `expected_format_list` | 输入 | 允许的数据格式集合或说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_with_invalid_input_dtype

输入 dtype 不在允许集合中，对应错误码EZ0007。

### 函数原型

```python
cannbotdsl.log.error_with_invalid_input_dtype(entity_name, param_name, data_dtype, expected_dtype_list, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `data_dtype` | 输入 | 实际输入数据类型 |
| `expected_dtype_list` | 输入 | 允许的数据类型集合或说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shape

指定参数形状不合法，对应错误码EZ0008。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shape(entity_name, param_name, incorrect_shape, correct_shape, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_shape` | 输入 | 实际形状 |
| `correct_shape` | 输入 | 期望或允许的形状 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shape_with_reason

指定参数形状不合法并说明原因，对应错误码EZ0009。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shape_with_reason(entity_name, param_name, incorrect_shape, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_shape` | 输入 | 实际形状 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapes_with_reason

多个参数形状关系不合法，对应错误码EZ0010。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapes_with_reason(entity_name, param_names, incorrect_shapes, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_shapes` | 输入 | 多个实际形状 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapedim

指定参数 shape dim 不合法，对应错误码EZ0011。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapedim(entity_name, param_name, incorrect_dim, correct_dim, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_dim` | 输入 | 实际形状维度值 |
| `correct_dim` | 输入 | 期望或允许的形状维度值 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapedim_with_reason

指定参数 shape dim 不合法并说明原因，对应错误码EZ0012。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapedim_with_reason(entity_name, param_name, incorrect_dim, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_dim` | 输入 | 实际形状维度值 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapedims_with_reason

多个参数 shape dim 不合法，对应错误码EZ0013。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapedims_with_reason(entity_name, param_names, incorrect_dims, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_dims` | 输入 | 多个实际形状维度值 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapesize

指定参数 shape size 不合法，对应错误码EZ0014。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapesize(entity_name, param_name, incorrect_size, correct_size, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与 incorrect_size 使用相同度量 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapesize_with_reason

指定参数 shape size 不合法并说明原因，对应错误码EZ0015。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapesize_with_reason(entity_name, param_name, incorrect_size, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_shapesizes_with_reason

多个参数 shape size 不合法，对应错误码EZ0016。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_shapesizes_with_reason(entity_name, param_names, incorrect_sizes, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_sizes` | 输入 | 多个实际大小 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_format

指定参数格式不合法，对应错误码EZ0017。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_format(entity_name, param_name, incorrect_format, correct_format, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_format` | 输入 | 实际数据格式 |
| `correct_format` | 输入 | 期望或允许的数据格式 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_formats_with_reason

多个参数格式不合法，对应错误码EZ0018。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_formats_with_reason(entity_name, param_names, incorrect_formats, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_formats` | 输入 | 多个实际数据格式 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_dtype

指定参数 dtype 不合法，对应错误码EZ0019。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_dtype(entity_name, param_name, incorrect_dtype, correct_dtype, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_dtype` | 输入 | 实际数据类型 |
| `correct_dtype` | 输入 | 期望或允许的数据类型 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_dtype_with_reason

指定参数 dtype 不合法并说明原因，对应错误码EZ0020。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_dtype_with_reason(entity_name, param_name, incorrect_dtype, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_dtype` | 输入 | 实际数据类型 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_dtypes_with_reason

多个参数 dtype 不合法，对应错误码EZ0021。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_dtypes_with_reason(entity_name, param_names, incorrect_dtypes, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_dtypes` | 输入 | 多个实际数据类型 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_tensornum

tensor 列表中的 tensor 数量不合法，对应错误码EZ0022。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_tensornum(entity_name, param_name, incorrect_num, correct_num, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_num` | 输入 | 实际 tensor 数量 |
| `correct_num` | 输入 | 期望或允许的 tensor 数量 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_tensornums_with_reason

多个 tensor 列表的 tensor 数量不合法，对应错误码EZ0023。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_tensornums_with_reason(entity_name, param_names, incorrect_nums, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_nums` | 输入 | 多个实际 tensor 数量 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_value

参数值不符合期望，对应错误码EZ0024。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_value(entity_name, param_name, incorrect_value, correct_value, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_value` | 输入 | 实际参数值 |
| `correct_value` | 输入 | 期望值或允许范围 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_listsize

列表大小不合法，对应错误码EZ0025。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_listsize(entity_name, param_name, incorrect_size, correct_size, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与 incorrect_size 使用相同度量 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_value_with_reason

参数值不合法并说明原因，对应错误码EZ0026。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_value_with_reason(entity_name, param_name, incorrect_value, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_value` | 输入 | 实际参数值 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_values_with_reason

多个参数值不合法，对应错误码EZ0027。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_values_with_reason(entity_name, param_names, incorrect_values, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_values` | 输入 | 多个实际参数值 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_stride

tensor 步长不合法，对应错误码EZ0028。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_stride(entity_name, param_name, incorrect_stride, correct_stride, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_stride` | 输入 | 实际步长 |
| `correct_stride` | 输入 | 期望或允许的步长 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_file_path

文件路径不合法，对应错误码EZ0029。

### 函数原型

```python
cannbotdsl.log.error_for_file_path(op_name, file_path, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志 OpName |
| `file_path` | 输入 | 出错文件路径 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_file_open

文件打开失败，对应错误码EZ0030。

### 函数原型

```python
cannbotdsl.log.error_for_file_open(op_name, file_path, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志 OpName |
| `file_path` | 输入 | 出错文件路径 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_file_parse

文件解析失败，对应错误码EZ0031。

### 函数原型

```python
cannbotdsl.log.error_for_file_parse(op_name, file_name, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志 OpName |
| `file_name` | 输入 | 解析失败的文件名 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_config

配置项值不符合期望，对应错误码EZ0032。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_config(op_name, config_file, item, value, expect, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志 OpName |
| `config_file` | 输入 | 配置文件名或路径 |
| `item` | 输入 | 配置项名称 |
| `value` | 输入 | 实际配置项值 |
| `expect` | 输入 | 期望的配置值或约束 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_config_with_reason

配置项值不合法并说明原因，对应错误码EZ0033。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_config_with_reason(op_name, config_file, item, value, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志 OpName |
| `config_file` | 输入 | 配置文件名或路径 |
| `item` | 输入 | 配置项名称 |
| `value` | 输入 | 实际配置项值 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_configs_with_reason

多个配置项值不合法，对应错误码EZ0034。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_configs_with_reason(op_name, config_file, items, values, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志 OpName |
| `config_file` | 输入 | 配置文件名或路径 |
| `items` | 输入 | 多个配置项名称 |
| `values` | 输入 | 多个实际配置项值 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_format_with_reason

指定参数格式不合法并说明原因，对应错误码EZ0035。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_format_with_reason(entity_name, param_name, incorrect_format, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_format` | 输入 | 实际数据格式 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_graph_node

输入图节点不合法，对应错误码EZ0036。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_graph_node(node_name, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `node_name` | 输入 | 被诊断的图节点名称 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_argument_with_reason

通用参数不合法并说明原因，对应错误码EZ0037。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_argument_with_reason(entity_name, param_name, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## error_for_invalid_listsize_with_reason

列表元素数量不合法并说明原因，对应错误码EZ0038。

### 函数原型

```python
cannbotdsl.log.error_for_invalid_listsize_with_reason(entity_name, param_name, incorrect_size, reason, *, tag="") -> None
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的 op_name 字段 |
| `param_name` | 输入 | 单个输入、输出或参数名称 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `reason` | 输入 | 错误原因或不满足的约束说明 |
| `tag` | 输入 | 仅限关键字传入的编译期字符串，默认空串，不含NUL；用于附加诊断标签。 |

## 通用参数说明

`entity_name`用于描述出错的算子或实体。日志前缀中的OpName取自当前DSL函数，可以与`entity_name`不同。文件和配置类接口的`op_name`用于兼容原宏，不写入模板正文，也不改变日志前缀。

`index`按传入值记录，不转换下标起点。形状、维度、大小和数量的含义及单位由算子定义，接口不计算或检查这些值。

字段支持静态字符串、整数、浮点数及list/tuple序列。序列可包含动态数值，每层最多128项，嵌套深度受限。字符串按原样输出，百分号无需转义，但不能包含NUL。参数仅用于描述错误，不会触发文件读写或形状、大小、维度计算。

## 返回值说明

返回`None`。

## 约束说明

- 仅支持在`@host`、`@kernel`及相应执行域的`@jit`中调用。在普通Python代码中直接调用会报错。
- 错误日志仅记录错误，不检查输入、不抛出异常、不停止Kernel，也不返回失败状态。算子需通过分支处理非法输入，避免继续执行导致越界。
- Kernel外层标量控制流支持设备日志。SIMD VF内部、SIMT线程级日志及SIMT算子外层组合的支持情况尚未明确，请勿用于每个lane或thread的并发打印。
- 设备日志要求在编译期确定启动block数。每个日志调用点在每个core的缓冲槽中保留最近64条记录，每条最多包含128个动态标量；旧记录被覆盖时会输出提示。Host同步回读后，按调用点和core输出日志，不保证跨核顺序。时间戳为Host输出日志的时间。启用设备日志会增加回读和同步开销。

结构化错误后端默认使用`liberror_manager.so`，并尝试从`$ASCEND_HOME_PATH/lib64/liberror_manager.so`加载。也可在首次调用前，通过`CANNBOTDSL_ERROR_MESSAGE_LIBRARY`指定库的绝对路径。库或所需符号不可用、或上报失败时，改为输出带`[ERROR_MESSAGE][EZxxxx]`前缀的文本。此前的ERROR日志仍独立输出。

部分Torch/SDK组合不允许覆盖预注册的EZ0022模板，此时该模板可能改用文本输出。ERROR日志写入plog并不表示结构化错误上报成功。

## 调用示例

以下示例展示正常执行和参数错误两种情况。`count=-1`时，调用`error_for_invalid_value`记录参数错误，并跳过Kernel调用，输出Tensor保持为零。使用其他模板时，按对应原型传入错误描述，并保留处理非法输入的分支。

错误日志包含`LogDemo`、`count`、`-1`和`nonnegative`。原生结构化后端使用EZ0024错误码，并上报SDK最近错误；上报失败时，输出带`[ERROR_MESSAGE][EZ0024]`前缀的文本。日志配置及plog查看方式见[通用日志](./log.md#日志输出配置)。

运行前，按[日志配置](./log.md#日志输出配置)加载环境。

```python
import torch
import torch_npu  # noqa: F401
import cannbotdsl as cbd


@cbd.kernel
def copy_one(x: cbd.Tensor, y: cbd.Tensor):
    cbd.log.info("copy begins", tag="copy")
    y[0, 0] = x[0, 0]


@cbd.host
def entry(x: cbd.Tensor, y: cbd.Tensor, count: int = 1):
    if count < 0:
        cbd.log.error_for_invalid_value("LogDemo", "count", count, "nonnegative")
        # 通过分支跳过非法输入对应的计算。
    else:
        cbd.log.debug("count=%d", count)
        cbd.log.info("host begins")
        cbd.log.warn("demonstration warning")
        cbd.log.error("demonstration error")
        cbd.log.error_without_report("additional diagnostic")
        copy_one[1](x, y)
        cbd.log.event("host completed")


def main():
    torch.npu.set_device(0)
    x = torch.tensor([[3.0]], device="npu", dtype=torch.float32)
    y = torch.zeros_like(x)
    spec = cbd.TensorSpec((1, 1), cbd.dtypes.float32)
    program = cbd.compile(entry, spec, spec)
    try:
        program(x, y, 1)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), x.cpu())
        y.zero_()
        program(x, y, -1)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), torch.zeros((1, 1)))
    finally:
        program.close()
    print("log example completed")


if __name__ == "__main__":
    main()
```

示例运行结束后输出：

```text
log example completed
```
