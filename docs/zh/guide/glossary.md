# 术语表

昇腾硬件与基础术语参见[昇腾术语表](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)。以下记录 CANNBot-DSL 文档中使用的概念。

| 术语/缩略语 | 含义 |
| --- | --- |
| GM（Global Memory） | 设备全局内存，可供 Kernel 读取输入或写入输出。 |
| UB（Unified Buffer） | Vector 计算使用的片上统一缓冲区。 |
| L1 | Cube 数据搬运与计算使用的片上存储层级。 |
| L0A、L0B、L0C | 分别用于 Cube 左操作数、右操作数和累加结果的片上存储空间。 |
| BIAS | 为 Cube 计算存放偏置数据的片上存储空间。 |
| Vector 子核 | 与 Cube 核配合工作的 Vector 执行子核；搬运目标编号为 0 或 1。 |
| ND | 普通多维数据布局，具体地址映射由形状和步长描述，不等同于一定连续。 |
| NZ、ZN | 面向 Cube 数据访问的分块物理布局；二者的矩阵轴与块内轴组织不同。应按搬运目标声明对应布局。 |
| C0 | 分块物理布局的块内列维长度，以元素个数计；其值取决于数据类型和所声明的物理布局。 |
| DMA | 在存储空间之间传输数据的数据搬运方式。 |
| NDDMA | 按多个维度的长度与步长进行数据搬运的方式；本文档中的 `copy_options(engine="nddma")` 将其显式用于 GM→UB。 |
| 提前编译（Ahead-of-Time Compilation，AOT） | 使用参数规格生成可调用的编译产物，执行时再传入实际数据。 |
| 参数规格（Spec） | 描述参数应满足的类型、形状或结构等条件的 Host 对象，本身不保存 Tensor 数据。 |
| 动态维度（Dim） | 运行时确定的长度，通过名称、上下界和整除倍数描述；可用于 Tensor shape、stride 或列表长度。 |
| 维度表达式 | 由 Dim 和整数通过算术运算构造的长度或步长关系。运行时先绑定 Dim，再求表达式的值。 |
| TensorList | DSL 的 Tensor 列表参数类型；Host 侧传入真实 Tensor 的 list 或 tuple，提前编译时使用 TensorListSpec 描述。 |
| 逻辑核（block） | Kernel启动时配置的逻辑执行单元，通过逻辑核索引划分数据；在Mix 1:2模式下，每个逻辑核组合包含一个Cube Core和两个Vector Core。 |
| 子核（subblock） | 一个逻辑核组合中参与执行的同类核心实例，可通过子核索引区分。 |
