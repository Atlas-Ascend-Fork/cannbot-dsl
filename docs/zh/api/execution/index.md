# 设备执行

本目录介绍设备执行上下文、流水类型和当前公开的同步接口。调试接口单独归入[调试](../debug/index.md)。

## 目录

- [设备执行上下文](./device/index.md)
- [流水类型](./pipeline.md)
- [已公开的同步接口](./synchronization/index.md)

CANNBot-DSL 会自动处理大多数核内数据依赖；需要显式控制设备顺序或跨核协作时，使用同步接口。
