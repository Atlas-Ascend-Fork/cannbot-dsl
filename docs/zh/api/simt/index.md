# SIMT接口

本节收录242个公开SIMT函数入口（含6个兼容别名）及`RoundingMode`枚举，按DSL的实际功能组织目录。各接口页提供原型、参数、返回值、约束及完整调用示例。

接口在`with vf(mode="simt"):`执行域使用，纯SIMT Kernel的启动方式参见[SIMT Kernel启动配置](./launch.md)。短向量是单线程中的多分量值，不等同于Warp内多个线程的数据；相关术语见[术语表](../../guide/glossary.md)。


- [SIMT Kernel启动配置](./launch.md)
- [短向量数据类型](./data-types.md)
- [线程信息](./thread/index.md)
- [同步与内存栅栏](./synchronization/index.md)
- [原子操作](./atomic/index.md)
- [Warp函数](./warp/index.md)
- [数学函数](./math/index.md)
- [类型转换](./conversion/index.md)
- [访存与缓存控制](./memory/index.md)
