# log统一错误模板

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

提供形状、类型、属性、配置及文件错误的统一模板，在普通Python和DSL中使用相同接口。EZ错误码由框架绑定，调用方无需传入或自定义。

### 错误码索引

| 错误码 | 模板名 | 场景 |
| --- | --- | --- |
| EZ0001 | [error_with_invalid_input_shape](#error_with_invalid_input_shape) | 输入形状不符合要求。 |
| EZ0002 | [error_with_invalid_attr](#error_with_invalid_attr) | 算子属性值不合法。 |
| EZ0003 | [error_with_invalid_attr_size](#error_with_invalid_attr_size) | 算子属性大小不合法。 |
| EZ0004 | [error_with_invalid_input](#error_with_invalid_input) | 必需输入或参数为空。 |
| EZ0005 | [error_with_invalid_input_shapesize](#error_with_invalid_input_shapesize) | 输入shape size不合法。 |
| EZ0006 | [error_with_invalid_input_format](#error_with_invalid_input_format) | 输入格式不在允许集合中。 |
| EZ0007 | [error_with_invalid_input_dtype](#error_with_invalid_input_dtype) | 输入dtype不在允许集合中。 |
| EZ0008 | [error_for_invalid_shape](#error_for_invalid_shape) | 指定参数形状不合法。 |
| EZ0009 | [error_for_invalid_shape_with_reason](#error_for_invalid_shape_with_reason) | 指定参数形状不合法并说明原因。 |
| EZ0010 | [error_for_invalid_shapes_with_reason](#error_for_invalid_shapes_with_reason) | 多个参数形状关系不合法。 |
| EZ0011 | [error_for_invalid_shapedim](#error_for_invalid_shapedim) | 指定参数shape dim不合法。 |
| EZ0012 | [error_for_invalid_shapedim_with_reason](#error_for_invalid_shapedim_with_reason) | 指定参数shape dim不合法并说明原因。 |
| EZ0013 | [error_for_invalid_shapedims_with_reason](#error_for_invalid_shapedims_with_reason) | 多个参数shape dim不合法。 |
| EZ0014 | [error_for_invalid_shapesize](#error_for_invalid_shapesize) | 指定参数shape size不合法。 |
| EZ0015 | [error_for_invalid_shapesize_with_reason](#error_for_invalid_shapesize_with_reason) | 指定参数shape size不合法并说明原因。 |
| EZ0016 | [error_for_invalid_shapesizes_with_reason](#error_for_invalid_shapesizes_with_reason) | 多个参数shape size不合法。 |
| EZ0017 | [error_for_invalid_format](#error_for_invalid_format) | 指定参数格式不合法。 |
| EZ0035 | [error_for_invalid_format_with_reason](#error_for_invalid_format_with_reason) | 指定参数格式不合法并说明原因。 |
| EZ0018 | [error_for_invalid_formats_with_reason](#error_for_invalid_formats_with_reason) | 多个参数格式不合法。 |
| EZ0019 | [error_for_invalid_dtype](#error_for_invalid_dtype) | 指定参数dtype不合法。 |
| EZ0020 | [error_for_invalid_dtype_with_reason](#error_for_invalid_dtype_with_reason) | 指定参数dtype不合法并说明原因。 |
| EZ0021 | [error_for_invalid_dtypes_with_reason](#error_for_invalid_dtypes_with_reason) | 多个参数dtype不合法。 |
| EZ0022 | [error_for_invalid_tensornum](#error_for_invalid_tensornum) | tensor列表中的tensor数量不合法。 |
| EZ0023 | [error_for_invalid_tensornums_with_reason](#error_for_invalid_tensornums_with_reason) | 多个tensor列表的tensor数量不合法。 |
| EZ0024 | [error_for_invalid_value](#error_for_invalid_value) | 参数值不符合期望。 |
| EZ0025 | [error_for_invalid_listsize](#error_for_invalid_listsize) | 列表大小不合法。 |
| EZ0026 | [error_for_invalid_value_with_reason](#error_for_invalid_value_with_reason) | 参数值不合法并说明原因。 |
| EZ0027 | [error_for_invalid_values_with_reason](#error_for_invalid_values_with_reason) | 多个参数值不合法。 |
| EZ0028 | [error_for_invalid_stride](#error_for_invalid_stride) | tensor步长不合法。 |
| EZ0029 | [error_for_file_path](#error_for_file_path) | 文件路径不合法。 |
| EZ0030 | [error_for_file_open](#error_for_file_open) | 文件打开失败。 |
| EZ0031 | [error_for_file_parse](#error_for_file_parse) | 文件解析失败。 |
| EZ0032 | [error_for_invalid_config](#error_for_invalid_config) | 配置项值不符合期望。 |
| EZ0033 | [error_for_invalid_config_with_reason](#error_for_invalid_config_with_reason) | 配置项值不合法并说明原因。 |
| EZ0034 | [error_for_invalid_configs_with_reason](#error_for_invalid_configs_with_reason) | 多个配置项值不合法。 |
| EZ0036 | [error_for_invalid_graph_node](#error_for_invalid_graph_node) | 输入图节点不合法。 |
| EZ0037 | [error_for_invalid_argument_with_reason](#error_for_invalid_argument_with_reason) | 通用参数不合法并说明原因。 |
| EZ0038 | [error_for_invalid_listsize_with_reason](#error_for_invalid_listsize_with_reason) | 列表元素数量不合法并说明原因。 |

## 导入方式

导入方式与[通用日志](./log.md#导入方式)相同，支持通过`cbd.log.xxx`或`log.xxx`调用错误模板接口。

## 函数原型

以下签名仅用于说明支持的参数类型，`cbd.ScalarValue`指[DSL数值标量](../data-structures/data-types.md#标量值)，`list`和`tuple`的元素规则见参数说明；普通Python不接受DSL标量对象。

```python
cbd.log.error_with_invalid_input_shape(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    index: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_shape: str | int | float | cbd.ScalarValue | list | tuple,
    correct_shape: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_with_invalid_attr(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    attr_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_val: str | int | float | cbd.ScalarValue | list | tuple,
    correct_val: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_with_invalid_attr_size(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    attr_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_size: str | int | float | cbd.ScalarValue | list | tuple,
    correct_size: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_with_invalid_input(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_with_invalid_input_shapesize(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    index: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_size: str | int | float | cbd.ScalarValue | list | tuple,
    correct_size: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_with_invalid_input_format(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    data_format: str | int | float | cbd.ScalarValue | list | tuple,
    expected_format_list: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_with_invalid_input_dtype(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    data_dtype: str | int | float | cbd.ScalarValue | list | tuple,
    expected_dtype_list: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shape(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_shape: str | int | float | cbd.ScalarValue | list | tuple,
    correct_shape: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shape_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_shape: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapes_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_shapes: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapedim(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_dim: str | int | float | cbd.ScalarValue | list | tuple,
    correct_dim: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapedim_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_dim: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapedims_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_dims: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapesize(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_size: str | int | float | cbd.ScalarValue | list | tuple,
    correct_size: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapesize_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_size: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_shapesizes_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_sizes: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_format(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_format: str | int | float | cbd.ScalarValue | list | tuple,
    correct_format: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_format_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_format: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_formats_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_formats: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_dtype(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_dtype: str | int | float | cbd.ScalarValue | list | tuple,
    correct_dtype: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_dtype_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_dtype: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_dtypes_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_dtypes: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_tensornum(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_num: str | int | float | cbd.ScalarValue | list | tuple,
    correct_num: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_tensornums_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_nums: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_value(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_value: str | int | float | cbd.ScalarValue | list | tuple,
    correct_value: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_listsize(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_size: str | int | float | cbd.ScalarValue | list | tuple,
    correct_size: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_value_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_value: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_values_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_names: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_values: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_stride(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_stride: str | int | float | cbd.ScalarValue | list | tuple,
    correct_stride: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_file_path(
    op_name: str | int | float | cbd.ScalarValue | list | tuple,
    file_path: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_file_open(
    op_name: str | int | float | cbd.ScalarValue | list | tuple,
    file_path: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_file_parse(
    op_name: str | int | float | cbd.ScalarValue | list | tuple,
    file_name: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_config(
    op_name: str | int | float | cbd.ScalarValue | list | tuple,
    config_file: str | int | float | cbd.ScalarValue | list | tuple,
    item: str | int | float | cbd.ScalarValue | list | tuple,
    value: str | int | float | cbd.ScalarValue | list | tuple,
    expect: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_config_with_reason(
    op_name: str | int | float | cbd.ScalarValue | list | tuple,
    config_file: str | int | float | cbd.ScalarValue | list | tuple,
    item: str | int | float | cbd.ScalarValue | list | tuple,
    value: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_configs_with_reason(
    op_name: str | int | float | cbd.ScalarValue | list | tuple,
    config_file: str | int | float | cbd.ScalarValue | list | tuple,
    items: str | int | float | cbd.ScalarValue | list | tuple,
    values: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_graph_node(
    node_name: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_argument_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None

cbd.log.error_for_invalid_listsize_with_reason(
    entity_name: str | int | float | cbd.ScalarValue | list | tuple,
    param_name: str | int | float | cbd.ScalarValue | list | tuple,
    incorrect_size: str | int | float | cbd.ScalarValue | list | tuple,
    reason: str | int | float | cbd.ScalarValue | list | tuple,
    *,
    tag: str = "",
) -> None
```

## 参数说明

全部接口的参数均为输入参数，不会修改所传对象。`entity_name`写入模板的`op_name`字段；前缀`OpName`来自调用函数，二者可以不同。文件/配置类接口的`op_name`保留原接口调用形式，不进入模板正文，也不改变前缀。

字段支持`str`、`int`、`float`、`list`和`tuple`；DSL中还可包含运行时数值标量。列表和元组每层最多128项，最多嵌套8层，元素递归遵循同一规则；设备侧每条日志的动态标量合计最多128个。字符串不能含NUL，字段中的百分号按原样输出，无需写成`%%`。浮点字段按`%g`转换，整数范围和DSL标量类型要求与[通用日志](./log.md#参数说明)一致。

序列统一显示为方括号，例如`(2, 1)`显示为`[2, 1]`；模板原有括号不会去重。`error_with_invalid_input_shape`模板还会包一层方括号，因此传元组时显示`[[2, 1]]`，若需单层显示可传字符串`"2, 1"`。`index`原样记录，不转换起始下标。形状、维度、大小和数量的含义及单位由算子定义，接口不自动求值或检查这些业务条件。

### 通用参数

下表统一定义公共参数的含义；各接口参数表仍列出全部参数，参数顺序和类型以函数原型为准。字段类型及格式化规则见本节开头说明。

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 被诊断的算子或实体名称；写入模板中的`op_name`字段。 |
| `reason` | 输入 | 错误原因，写入模板正文。 |
| `param_name` | 输入 | 被诊断的参数名称。 |
| `tag` | 输入 | 仅限关键字传入的`str`，默认空串，不能含NUL；DSL中须为编译期常量，用于附加诊断标签。 |

<a id="error_with_invalid_input_shape"></a>

### error_with_invalid_input_shape

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `index` | 输入 | 输入序号；框架原样记录，不进行起始下标转换 |
| `incorrect_shape` | 输入 | 实际形状 |
| `correct_shape` | 输入 | 期望或允许的形状 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_with_invalid_attr"></a>

### error_with_invalid_attr

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `attr_name` | 输入 | 属性名称 |
| `incorrect_val` | 输入 | 实际的错误属性值 |
| `correct_val` | 输入 | 期望或允许的属性值 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_with_invalid_attr_size"></a>

### error_with_invalid_attr_size

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `attr_name` | 输入 | 属性名称 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与incorrect_size使用相同度量 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_with_invalid_input"></a>

### error_with_invalid_input

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_with_invalid_input_shapesize"></a>

### error_with_invalid_input_shapesize

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `index` | 输入 | 输入序号；框架原样记录，不进行起始下标转换 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与incorrect_size使用相同度量 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_with_invalid_input_format"></a>

### error_with_invalid_input_format

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `data_format` | 输入 | 实际输入数据格式 |
| `expected_format_list` | 输入 | 允许的数据格式集合或说明 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_with_invalid_input_dtype"></a>

### error_with_invalid_input_dtype

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `data_dtype` | 输入 | 实际输入数据类型 |
| `expected_dtype_list` | 输入 | 允许的数据类型集合或说明 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shape"></a>

### error_for_invalid_shape

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_shape` | 输入 | 实际形状 |
| `correct_shape` | 输入 | 期望或允许的形状 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shape_with_reason"></a>

### error_for_invalid_shape_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_shape` | 输入 | 实际形状 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapes_with_reason"></a>

### error_for_invalid_shapes_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_shapes` | 输入 | 多个实际形状 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapedim"></a>

### error_for_invalid_shapedim

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_dim` | 输入 | 实际形状维度值 |
| `correct_dim` | 输入 | 期望或允许的形状维度值 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapedim_with_reason"></a>

### error_for_invalid_shapedim_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_dim` | 输入 | 实际形状维度值 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapedims_with_reason"></a>

### error_for_invalid_shapedims_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_dims` | 输入 | 多个实际形状维度值 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapesize"></a>

### error_for_invalid_shapesize

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与incorrect_size使用相同度量 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapesize_with_reason"></a>

### error_for_invalid_shapesize_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_shapesizes_with_reason"></a>

### error_for_invalid_shapesizes_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_sizes` | 输入 | 多个实际大小 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_format"></a>

### error_for_invalid_format

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_format` | 输入 | 实际数据格式 |
| `correct_format` | 输入 | 期望或允许的数据格式 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_format_with_reason"></a>

### error_for_invalid_format_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_format` | 输入 | 实际数据格式 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_formats_with_reason"></a>

### error_for_invalid_formats_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_formats` | 输入 | 多个实际数据格式 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_dtype"></a>

### error_for_invalid_dtype

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_dtype` | 输入 | 实际数据类型 |
| `correct_dtype` | 输入 | 期望或允许的数据类型 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_dtype_with_reason"></a>

### error_for_invalid_dtype_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_dtype` | 输入 | 实际数据类型 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_dtypes_with_reason"></a>

### error_for_invalid_dtypes_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_dtypes` | 输入 | 多个实际数据类型 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_tensornum"></a>

### error_for_invalid_tensornum

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_num` | 输入 | 实际tensor数量 |
| `correct_num` | 输入 | 期望或允许的tensor数量 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_tensornums_with_reason"></a>

### error_for_invalid_tensornums_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_nums` | 输入 | 多个实际tensor数量 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_value"></a>

### error_for_invalid_value

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_value` | 输入 | 实际参数值 |
| `correct_value` | 输入 | 期望值或允许范围 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_listsize"></a>

### error_for_invalid_listsize

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `correct_size` | 输入 | 期望或允许的大小，与incorrect_size使用相同度量 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_value_with_reason"></a>

### error_for_invalid_value_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_value` | 输入 | 实际参数值 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_values_with_reason"></a>

### error_for_invalid_values_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_names` | 输入 | 多个输入、输出或参数名称 |
| `incorrect_values` | 输入 | 多个实际参数值 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_stride"></a>

### error_for_invalid_stride

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_stride` | 输入 | 实际步长 |
| `correct_stride` | 输入 | 期望或允许的步长 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_file_path"></a>

### error_for_file_path

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志OpName |
| `file_path` | 输入 | 出错文件路径 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_file_open"></a>

### error_for_file_open

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志OpName |
| `file_path` | 输入 | 出错文件路径 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_file_parse"></a>

### error_for_file_parse

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志OpName |
| `file_name` | 输入 | 解析失败的文件名 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_config"></a>

### error_for_invalid_config

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志OpName |
| `config_file` | 输入 | 配置文件名或路径 |
| `item` | 输入 | 配置项名称 |
| `value` | 输入 | 实际配置项值 |
| `expect` | 输入 | 期望的配置值或约束 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_config_with_reason"></a>

### error_for_invalid_config_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志OpName |
| `config_file` | 输入 | 配置文件名或路径 |
| `item` | 输入 | 配置项名称 |
| `value` | 输入 | 实际配置项值 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_configs_with_reason"></a>

### error_for_invalid_configs_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `op_name` | 输入 | 算子名称；为对齐原宏保留该参数，文件和配置类模板正文不使用它，也不会覆盖自动推导的日志OpName |
| `config_file` | 输入 | 配置文件名或路径 |
| `items` | 输入 | 多个配置项名称 |
| `values` | 输入 | 多个实际配置项值 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_graph_node"></a>

### error_for_invalid_graph_node

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `node_name` | 输入 | 被诊断的图节点名称 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_argument_with_reason"></a>

### error_for_invalid_argument_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

<a id="error_for_invalid_listsize_with_reason"></a>

### error_for_invalid_listsize_with_reason

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `entity_name` | 输入 | 见[通用参数](#通用参数)。 |
| `param_name` | 输入 | 见[通用参数](#通用参数)。 |
| `incorrect_size` | 输入 | 实际大小；具体表示属性元素数、shape size、列表元素数等，由接口场景决定 |
| `reason` | 输入 | 见[通用参数](#通用参数)。 |
| `tag` | 输入 | 见[通用参数](#通用参数)。 |

## 返回值说明

返回`None`。接口不返回SDK上报结果或算子执行状态。

## 约束说明

- 普通Python、`@host`、`@kernel`及相应域的`@jit`使用相同接口；分流取决于当前DSL编译上下文。在tracing期间执行的辅助函数仍属于DSL。执行时机、SIMD/SIMT支持边界、设备槽位和容量限制见[通用日志约束](./log.md#约束说明)。
- 调用方须检查业务条件，在非法条件分支中调用错误模板，并自行跳过计算、返回或在普通Python中抛出异常。接口负责记录错误诊断，不检查输入、不读写所描述的文件，也不会抛出业务异常或停止Kernel。
- 所有路径每次模板调用只产生一条ERROR，日志级别、终端和plog配置与[通用日志](./log.md#日志输出配置)一致；关闭ERROR打印不等于关闭DSL的结构化错误上报。

- DSL结构化上报后端不可用或上报失败时，ERROR日志仍按[通用日志配置](./log.md#日志输出配置)处理，不额外追加`[ERROR_MESSAGE][EZxxxx]`文本。运行时保留最近一次模板的诊断状态；出现ERROR日志不代表结构化上报成功。

## 调用示例

以下完整示例同时演示普通Python和Host的输入形状错误日志，并运行一个单元素拷贝Kernel；`rows=1`执行计算，`rows=2`记录错误并跳过Kernel，最后校验输出不变。`rows`在此示例中是算子自行检查的参数，接口不会根据Tensor自动判错。保存为`error_message_example.py`，按[日志配置](./log.md#日志输出配置)加载环境后运行`python error_message_example.py`。

```python
import torch
import torch_npu  # noqa: F401
import cannbotdsl as cbd


@cbd.kernel
def copy_one(x: cbd.Tensor, y: cbd.Tensor):
    y[0, 0] = x[0, 0]


@cbd.host
def entry(x: cbd.Tensor, y: cbd.Tensor, rows: int = 1):
    if rows != 1:
        cbd.log.error_with_invalid_input_shape(
            "ShapeDemo", 0, (rows, 1), (1, 1), tag="validation"
        )
    else:
        copy_one[1](x, y)


def main():
    # 普通 Python 也可使用同一接口；此处只记录诊断，不抛业务异常。
    cbd.log.error_with_invalid_input_shape(
        "ShapeDemo", 0, (2, 1), (1, 1), tag="validation"
    )
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
        program(x, y, 2)
        torch.npu.synchronize()
        # 非法参数的分支没有启动 Kernel，因此输出保持为零。
        torch.testing.assert_close(y.cpu(), torch.zeros((1, 1)))
    finally:
        program.close()
    print("error_message example passed")


if __name__ == "__main__":
    main()
```

预期结果：除日志外，程序稳定输出：

```text
error_message example passed
```

两次错误诊断的正文均为`The 0th input of ShapeDemo has incorrect shape [[2, 1]]. It should be [[1, 1]].`；普通Python前缀角色为`@python`，DSL为`@host`，每次各一条ERROR。普通Python记录的EZ码为`EZ0001`；正常DSL结构化路径上报同一码。日志前缀中的时间、进程/线程ID、源码位置及函数名使用实际值。
