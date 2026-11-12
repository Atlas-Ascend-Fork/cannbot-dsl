# SIMT Kernel启动配置

## 功能说明

纯SIMT Kernel通过两个`dim3`值配置逻辑块（block）数量和每个逻辑块中的线程数量。启动语句写在`@host`函数体中。

## 启动形式

```python
import cannbotdsl as cbd


@cbd.host
def run(x, y, out):
    simt_kernel[cbd.dim3(2), cbd.dim3(128)](x, y, out)
```

第一个`dim3`表示逻辑块配置，第二个表示每个逻辑块的线程配置。上例启动2个逻辑块，每个逻辑块包含128个线程；`x`、`y`、`out`为核函数的实参。

第三个位置可指定动态UB缓冲区大小`dyn_ub_buf`，省略时为0：

```python
simt_kernel[cbd.dim3(2), cbd.dim3(128), dyn_ub_buf](x, y, out)
```

## dim3配置

`dim3`由CANNBot-DSL提供，通过`cbd.dim3(x=1, y=1, z=1)`创建三维配置，各维默认值为1。

- 用于逻辑块配置时，逻辑块总数为`x * y * z`。
- 用于线程配置时，每个逻辑块中的线程总数为`x * y * z`，不能超过1024。
- 三个维度都必须是正的32位无符号整数。
- `dyn_ub_buf`必须是非负整数。

## 相关接口

- [Kernel启动配置](../language/launch.md)：通用启动规则及整数block配置。
- [矢量函数作用域](../compute/vector/overview.md)：声明SIMT计算区域。
- [block_idx](./thread/block-idx.md)、[grid_dim](./thread/grid-dim.md)：获取逻辑块坐标和数量。
- [thread_idx](./thread/thread-idx.md)、[thread_dim](./thread/thread-dim.md)：获取线程坐标和每个逻辑块中的线程数量。
