# GroupSync

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`GroupSync`使用GM工作区协调多个生产者和消费者按轮次复用槽位。生产者使用`acquire`和`commit`，消费者使用`wait`和`release`；各方法在[接口说明](#接口)中列出。

## 函数原型

```python
GroupSync(workspace, *, depth=None, producers=None, consumers=None)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `workspace` | 输入/输出 | GM中的工作区Tensor，调用前须清零。 |
| `depth` | 输入 | 槽位数量，必须显式指定为编译期正整数。 |
| `producers` | 输入 | 每轮生产者贡献次数，必须显式指定为编译期正整数。 |
| `consumers` | 输入 | 每轮消费者贡献次数，必须显式指定为编译期正整数。 |

## 返回值说明

返回`GroupSync`对象。成功获取的凭据具有`slot`和`round`属性，分别表示槽位编号和轮次。

## 约束说明

- 工作区起始地址须按64字节对齐，容量至少为`128 * depth`字节。工作区首次使用前须清零，不能同时用于相互独立的同步任务。在所有使用该工作区的任务执行结束前，不得释放或复用这块内存。
- 接口只能在设备标量执行上下文使用，不能在VF内调用。
- `q`为非负的轮次编号。使用同一槽位的各参与者须按一致的轮次推进，并保证计数不溢出。
- 凭据只能由对应的`GroupSync`对象按生产者或消费者角色使用一次，不能作为循环状态或独立函数的ABI参数传递。
- 使用非阻塞形式时，仅在`ready`为真时使用凭据；失败后可再次调用检查。
- 接口不会自动完成有效数据的内存屏障或缓存维护；调用方须根据数据访问方式保证可见性和执行顺序。

## 接口

| 接口 | 说明 |
| --- | --- |
| [acquire](./group-sync/acquire.md) | 生产者获取槽位凭据。 |
| [commit](./group-sync/commit.md) | 提交生产完成通知。 |
| [wait](./group-sync/wait.md) | 消费者等待生产完成并获取凭据。 |
| [release](./group-sync/release.md) | 提交消费完成通知。 |

## 调用示例

以下示例使用非阻塞形式完成一轮生产和消费，并在Host侧检查两次操作均已就绪。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def kernel(workspace: cbd.Tensor, result: cbd.Tensor):
    sync = cbd.GroupSync(workspace, depth=1, producers=1, consumers=1)
    ready, ticket = sync.acquire(0, blocking=False)
    cbd.scalar.vec_store_bypass(result.ptr(0), cbd.dtypes.int32(ready))
    if ready:
        cbd.data_barrier(scope="gm")
        sync.commit(ticket)
        cbd.data_barrier(scope="gm")
    ready, ticket = sync.wait(0, blocking=False)
    cbd.scalar.vec_store_bypass(result.ptr(1), cbd.dtypes.int32(ready))
    if ready:
        cbd.data_barrier(scope="gm")
        sync.release(ticket)
        cbd.data_barrier(scope="gm")


@cbd.host
def run(workspace, result):
    kernel[1](workspace, result)


workspace = torch.zeros((16,), dtype=torch.int64, device="npu")
result = torch.full((2,), -1, dtype=torch.int32, device="npu")
run(workspace, result)
assert result.cpu().tolist() == [1, 1]
```
