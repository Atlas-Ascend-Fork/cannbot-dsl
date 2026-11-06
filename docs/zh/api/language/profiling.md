# Profiling 名称与类型

## 产品支持情况

`ProfileSpec` 的名称配置和默认补齐在 Host 编译阶段完成。

## 功能说明

`@kernel` 为每次 Kernel 启动提供 profiling 名称 `name` 和逻辑类型 `op_type`，
用于性能分析工具中的任务识别。用户可以通过 `cb.ProfileSpec` 分别覆盖这两个字段；
未指定的字段由框架根据实际入口函数名自动补齐。

`op_type` 是逻辑分类字符串，框架不会根据计算内容自动判断 MatMul、Softmax 等
算法类型。它与硬件任务类型 `Task Type` 不同，后者由实际生成的 AIC、AIV 或 MIX
Kernel 决定。

## 函数原型

```python
cb.ProfileSpec(
    name: str | None = None,
    op_type: str | None = None,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    inouts: tuple[str, ...] = (),
) -> cb.ProfileSpec
```

通过 `import cannbotdsl as cb` 导入后，将配置传入 `@cb.kernel(profile=cb.ProfileSpec(...))`：

```python
@cb.kernel(
    profile=cb.ProfileSpec(
        name="copy",
        op_type="Copy",
        inputs=("x",),
        outputs=("out",),
    )
)
def copy_kernel(x: cb.Tensor, out: cb.Tensor):
    out[0] = x[0]
```

该配置也可用于 `@kernel` 装饰的类。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `name` | 输入 | profiling 任务名称。允许为 `None` 或不含 NUL（`\0`）的非空字符串。默认值 `None`表示未指定，由框架使用实际入口函数名补齐。 |
| `op_type` | 输入 | profiling 逻辑类型。允许为 `None` 或不含 NUL（`\0`）的非空字符串。默认值 `None`表示未指定，由框架使用实际入口函数名补齐。 |
| `inputs` | 输入 | 作为输入上报的 Tensor 参数名元组，默认为 `()`，例如 `("x",)`。 |
| `outputs` | 输入 | 作为输出上报的 Tensor 参数名元组，默认为 `()`，例如 `("out",)`。 |
| `inouts` | 输入 | 同时作为输入和输出上报的 Tensor 参数名元组，默认为 `()`。同一参数会分别出现在输入和输出描述中。 |

### 默认补齐与显式覆盖

下表中的“入口函数名”是框架用于本次启动的实际入口名。

| 声明方式 | 最终 `name` | 最终 `op_type` |
| --- | --- | --- |
| 普通函数 `@kernel def copy_kernel(...)` | `copy_kernel` | `copy_kernel` |
| `@kernel class CopyKernel`，启动 `__call__` | `CopyKernel___call__` | `CopyKernel___call__` |
| `@kernel class CopyKernel`，启动 `run` | `CopyKernel_run` | `CopyKernel_run` |
| 普通类 `Holder` 中，仅将方法 `run` 标为 `@kernel` | `run` | `run` |
| `ProfileSpec(name="copy")` | `copy` | 入口函数名 |
| `ProfileSpec(op_type="Copy")` | 入口函数名 | `Copy` |
| `ProfileSpec(name="copy", op_type="Copy")` | `copy` | `Copy` |
| `ProfileSpec()`，或 `name=None, op_type=None` | 入口函数名 | 入口函数名 |

普通函数使用其函数名，不自动添加模块名或类的限定名。整个类被 `@kernel` 装饰时，
入口名由类名与实际启动的方法名拼接：

```text
入口函数名 = 类名 + "_" + 方法名

CopyKernel + "_" + "__call__" = CopyKernel___call__
CopyKernel + "_" + "run"      = CopyKernel_run
```

因此，`CopyKernel___call__` 中类名与 `call` 之间有三个下划线。

### 一个类的多个入口

同一个 `@kernel` 类可以同时定义 `__call__`、`run` 等入口。
每次启动由调用表达式选择执行的方法，profiling 名称根据本次启动的入口及配置确定。
例如，文中的 `CopyKernel` 同时定义这两个方法时，可以这样启动：

```python
@cb.host
def demo(x: cb.Tensor, out: cb.Tensor):
    op = CopyKernel()       # 创建启动句柄，此时不执行 Kernel
    op[1](x, out)           # 启动 __call__ 入口
    op[1].run(x, out)       # 启动 run 入口
```

这段 Host 代码会启动两次 Kernel，分别执行 `__call__` 和 `run`。
未配置 `ProfileSpec` 时，`name` 和 `op_type` 都按各自入口名自动补齐。
如果类装饰器统一配置了
`ProfileSpec(name="copy", op_type="Copy", inputs=("x",), outputs=("out",))`，
则两个入口都使用 `name="copy"`、`op_type="Copy"` 上报：

| 启动入口 | 未配置时的 `name / op_type` | 类上统一配置后的 `name / op_type` |
| --- | --- | --- |
| `op[1](x, out)`，执行 `__call__` | `CopyKernel___call__ / CopyKernel___call__` | `copy / Copy` |
| `op[1].run(x, out)`，执行 `run` | `CopyKernel_run / CopyKernel_run` | `copy / Copy` |

统一名称后，本例仍是两次独立的 Kernel 启动和两条目标性能记录，只是记录中的名称和
类型相同。名称相同不会把两个入口合并，也不会改变实际执行的方法；执行入口仍由
`op[1](...)` 或 `op[1].run(...)` 决定。

需要从性能记录的名称直接区分入口时，可以保留默认名称。如果希望使用自定义名称，
可以将两个入口拆成独立的 Kernel 类，分别配置不同的 `ProfileSpec`，例如
`name="copy_default"` 和 `name="copy_run"`，便能按这两个名称区分记录。

### 名称配置与 Tensor 角色

`inputs`、`outputs` 和 `inouts` 中填写的是 Kernel 的形参名，
例如 `def copy_kernel(x, out)` 对应 `"x"` 和 `"out"`。
shape、dtype、format 等属性来自本次启动的实际 Tensor 描述，不从名称字符串中推导。

一个 Kernel 可以有多个输入或输出，把需要上报的所有形参名放入对应元组即可。
例如，两个输入 `x`、`y` 和一个输出 `out` 可以这样配置：

```python
import cannbotdsl as cb


@cb.kernel(
    profile=cb.ProfileSpec(
        name="add",
        op_type="Add",
        inputs=("x", "y"),
        outputs=("out",),
    )
)
def add_kernel(x: cb.Tensor, y: cb.Tensor, out: cb.Tensor):
    out[0] = x[0] + y[0]
```

框架会分别获取 `x`、`y` 和 `out` 本次实际 Tensor 的 shape、dtype、format 等属性。
多个输出同样可以写成 `outputs=("out", "aux")`，其中 `out` 和 `aux` 都必须是
Kernel 的形参名；读写参数也可以通过 `inouts` 元组列出多个名称。

显式配置时，只上报所列参数的 Tensor 信息。例如只写 `inputs=("x",)`，
`y` 仍会正常传入并参与计算，但不会出现在输入 Tensor 的 profiling 描述中。

显式配置的 Tensor 角色
未提供 `ProfileSpec` 时，框架根据 Kernel 的读写行为推导默认 Tensor 角色。
只要提供了 `ProfileSpec`，角色就按配置处理，未声明的参数不参与 Tensor 信息上报。

例如，`ProfileSpec(name="copy")` 和 `ProfileSpec()` 都可以得到正确的名称，
但它们的 Tensor 上报列表为空。需要同时查看输入输出属性时，应显式填写
`inputs`、`outputs` 或 `inouts`。

### 矩阵和张量参数的类型要求

需要将矩阵或张量的 shape、dtype、format 等属性作为 profiling 信息上报时，
`@kernel` 的对应输入和输出参数应使用 `cb.Tensor` 类型，并以 Tensor 方式传递，
例如 `def matrix_kernel(x: cb.Tensor, out: cb.Tensor)`。框架从 Tensor 描述中获取
这些属性，不能仅凭参数名或数据地址推测矩阵的维数和各维大小。

如果 Kernel 接口只接收裸指针，即使该地址来自一个 Tensor，指针参数本身也没有保留
完整的 Tensor 描述；指针指向的元素类型不足以确定 shape 和布局。仅在 `ProfileSpec`
中填写参数名，不会把裸指针补成 Tensor，也不会恢复缺失的矩阵形状。

- 未提供 `ProfileSpec` 时，裸指针参数不参与默认 Tensor 角色推导，也不会自动生成
  对应的完整 Tensor 属性上报信息。
- 显式把裸指针或标量参数列入 `inputs`、`outputs`、`inouts` 时，编译阶段会进行类型
  校验并报错；这些角色组要求参数能转换为可上报的 Tensor 描述。

这一要求针对 Tensor 属性上报。`name` 和 `op_type` 的默认补齐与显式配置独立生效，
不通过指针或 Tensor 的 shape 决定名称。

## 返回值说明

返回一个不可变的 `cb.ProfileSpec` 配置对象，用于 `@cb.kernel(profile=...)`。
构造配置对象本身不启动 Kernel，也不启动性能采集。

## 约束说明

- `profile` 必须是 `ProfileSpec` 对象或 `None`；传入字典、字符串等其他类型会报错。
- `name` 和 `op_type` 的合法回退值是 `None`；空字符串、含 NUL 的字符串及非字符串值
  会抛出 `ValueError`。
- Tensor 角色配置必须使用参数名元组，各名称必须存在于 Kernel 签名中，并对应可上报的
  Tensor 参数。同一名称不能在角色组内或跨角色组重复；读写同一 Tensor 时使用 `inouts`。
  需要上报矩阵或张量的 shape 等属性时，对应输入和输出应使用 `cb.Tensor`，详见
  [矩阵和张量参数的类型要求](#矩阵和张量参数的类型要求)。
- 默认名称不保证全局唯一。不同模块的同名函数、不同编译变体都可以上报同一个名称；
  区分性能数据时，应结合输入规格、进程和任务信息，或显式指定可区分的 `name`。
- 名称在编译时写入产物。修改配置后需要重新编译算子；保存、加载已有产物会沿用其中的
  名称，不会根据后续修改的 Python 声明自动改名。

### Profiling 名称与执行符号

Profiling 名称用于任务展示和逻辑分类，实际执行仍由编译产物中的 Kernel 符号确定。

| 项目 | 生成规则 | 示例 |
| --- | --- | --- |
| profiling 名称 | 自定义 `name`，或入口函数名 | `copy`、`CopyKernel___call__` |
| profiling 类型 | 自定义 `op_type`，或入口函数名 | `Copy`、`CopyKernel___call__` |
| Kernel 调用符号 | `"kernel_" + 入口函数名 + "_" + 编号` | `kernel_CopyKernel___call___0` |

`ProfileSpec(name="copy")` 设置的是性能采集时显示的名称，不会修改生成的 AscendC 函数名。

例如，框架为 `CopyKernel.__call__` 生成的设备入口函数名为
`kernel_CopyKernel___call___0`，那么生成的 Host 代码仍通过这个函数名启动 Kernel：

```cpp
// 示意：启动参数和函数参数按实际生成代码填写
kernel_CopyKernel___call___0<<<block_dim, ..., stream>>>(...);
```

性能采集时，这次执行可以显示为 `copy`；执行时，Host 代码调用的仍是上述设备入口函数。
修改 `ProfileSpec.name` 不会改变执行哪个 Kernel。

函数名末尾的 `_0` 是框架在生成代码时分配的编号，用于区分当前 Host 编译中登记的
不同 Kernel 实现。编号从 `0` 开始，新增实现时分配下一个编号，复用已有实现时保留
原编号。不同入口即使代码相似，也可能因函数身份不同生成不同实现。
编号在编译阶段确定，运行时反复调用同一个 Kernel，编号不会变化。

每次独立的 Host 编译都从 `0` 开始编号；如果直接使用缓存产物，就沿用产物中已经生成的
函数名。编号与设备卡数、调用次数无关。

### 在性能工具中查看

使用当前框架上报路径重新编译、采集后，msprof summary 的 `Op Name` 和 `OP Type`
分别展示解析后的 `name` 和 `op_type`。MindStudio 中应查看对应的算子名称和类型字段；
Kernel 符号字段用于查看实际执行入口，可以继续保留带参数类型编码的 C++ 符号。

这套配置不会自动启动 msprof。旧框架或旧预编译算子可能仍以设备符号展示，需要使用
当前框架重新编译并重新采集；已经生成的历史性能数据不会因此改变。

## 调用示例

以下示例复制一个 `float32` 元素，展示普通函数、类的 `__call__`、类的 `run`
以及自定义名称四种入口，并在 Host 侧校验结果。运行环境需安装 cannbotdsl、
CANN、PyTorch 和 torch_npu，并配置可用的 NPU。

```python
import cannbotdsl as cb
import torch
import torch_npu


@cb.kernel
def copy_kernel(x: cb.Tensor, out: cb.Tensor):
    out[0] = x[0]


@cb.kernel
class CopyKernel:
    def __call__(self, x: cb.Tensor, out: cb.Tensor):
        out[0] = x[0]

    def run(self, x: cb.Tensor, out: cb.Tensor):
        out[0] = x[0]


@cb.kernel(
    profile=cb.ProfileSpec(
        name="copy",
        op_type="Copy",
        inputs=("x",),
        outputs=("out",),
    )
)
class NamedCopyKernel:
    def __call__(self, x: cb.Tensor, out: cb.Tensor):
        out[0] = x[0]


@cb.host
def demo(x, out_fn, out_call, out_run, out_named):
    copy_kernel[1](x, out_fn)
    op = CopyKernel()
    op[1](x, out_call)
    op[1].run(x, out_run)
    NamedCopyKernel()[1](x, out_named)


def main():
    torch.npu.set_device(0)
    x = torch.tensor([7.0], dtype=torch.float32, device="npu:0")
    outputs = [torch.empty_like(x) for _ in range(4)]
    demo(x, *outputs)
    torch.npu.synchronize()
    for out in outputs:
        torch.testing.assert_close(out.cpu(), x.cpu())
    print("profiling naming example passed")


if __name__ == "__main__":
    main()
```

预期输出：

```text
profiling naming example passed
```

将示例保存为 `profiling_names.py` 后，可在已配置好 CANN 和当前框架的环境中采集：

```bash
msprof --application="python profiling_names.py" \
       --output=./profiling_names_data \
       --task-time=on --ai-core=on
```

生成的 `mindstudio_profiler_output/op_summary_*.csv` 中，这四个 DSL 任务的名称字段
应分别为下表所示；其他任务及列省略。

| Kernel 入口 | Op Name | OP Type |
| --- | --- | --- |
| `copy_kernel` | `copy_kernel` | `copy_kernel` |
| `CopyKernel.__call__` | `CopyKernel___call__` | `CopyKernel___call__` |
| `CopyKernel.run` | `CopyKernel_run` | `CopyKernel_run` |
| `NamedCopyKernel.__call__` | `copy` | `Copy` |

相关接口：[装饰器](./decorators.md#kernel)、[Kernel 启动配置](./launch.md)。
