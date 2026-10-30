# 系统变量


系统接口用于在设备Kernel中查询逻辑核索引、物理核、子核、系统周期和状态信息，判断编译通道，以及读写CTRL控制寄存器。

## 接口

| 接口 | 说明 |
| --- | --- |
| [get_block_idx](./get-block-idx.md) | 获取当前执行Kernel的逻辑核索引。 |
| [get_block_num](./get-block-num.md) | 获取本次Kernel启动配置的逻辑核数。 |
| [get_subblock_id](./get-subblock-id.md) | 获取当前子核索引。 |
| [get_subblock_dim](./get-subblock-dim.md) | 获取当前类型的子核数量。 |
| [get_core_id](./get-core-id.md) | 获取当前物理核ID。 |
| [get_system_cycle](./get-system-cycle.md) | 读取系统时钟周期计数。 |
| [get_status](./get-status.md) | 读取状态寄存器。 |
| [get_vf_len](./get-vf-len.md) | 查询向量寄存器长度，单位为字节。 |
| [get_squeeze_status](./get-squeeze-status.md) | 获取squeeze有效数据长度。 |
| [is_aic](./is-aic.md) | 判断当前编译通道是否为Cube（AIC）。 |
| [is_aiv](./is-aiv.md) | 判断当前编译通道是否为Vector（AIV）。 |
| [get_ctrl](./spr/get-ctrl.md) | 读取CTRL控制寄存器。 |
| [set_ctrl](./spr/set-ctrl.md) | 设置CTRL控制寄存器。 |
