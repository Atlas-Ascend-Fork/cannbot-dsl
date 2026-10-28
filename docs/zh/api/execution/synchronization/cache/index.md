# 缓存控制

AI Core的标量计算单元直接读写全局内存（Global Memory，GM）时，数据可能暂存在每个核内部的数据缓存（Data Cache，DCache）中。DCache以缓存行（Cache Line）为最小管理单位；将缓存行“写回”表示把修改同步到GM，将其“失效”表示下次访问时重新从GM读取。

这些接口用于保证DCache与GM中的数据一致。通过直接内存访问（Direct Memory Access，DMA）在GM与片上存储之间搬运数据时不经过DCache，通常不需要调用这些接口。

> 这些接口不会由框架根据GM访问自动插入。使用Scalar直接读写GM时，调用方必须根据数据共享关系显式选择`dcci_*`、`dci`或bypass接口。

- **[概述](./overview.md)**

## 接口

| 接口 | 说明 |
| --- | --- |
| [`dcci_single`](./dcci-single.md) | 写回并失效指定GM地址所在的一个缓存行。 |
| [`dcci_entire_out`](./dcci-entire-out.md) | 写回脏数据，并使所有GM缓存行失效。 |
| [`dcci_entire_atomic`](./dcci-entire-atomic.md) | 写回并失效标量原子操作访问的GM数据。 |
| [`dci`](./dci.md) | 失效本核的整个数据缓存，并丢弃尚未写回的修改。 |
