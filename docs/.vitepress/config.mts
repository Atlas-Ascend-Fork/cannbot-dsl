import { defineConfig } from 'vitepress'

const docsVersion = process.env.DOCS_VERSION || 'next'

const guideSidebar = [
  { text: '开始使用', link: '/zh/install/' },
  { text: '项目介绍', link: '/zh/guide/project/overview' },
  { text: '编程模型', link: '/zh/guide/programming-model/overview' },
  { text: '仓库结构', link: '/zh/guide/project/repository' },
  { text: '样例导航', link: '/zh/project/examples/' }
]

const programmingModelSidebar = [
  { text: '编程模型概览', link: '/zh/guide/programming-model/overview' },
  {
    text: '语言机制',
    collapsed: false,
    items: [
      { text: '简介', link: '/zh/guide/programming-model/introduction' },
      { text: '硬件与执行模型', link: '/zh/guide/programming-model/hardware-model' },
      { text: '代码生成', link: '/zh/guide/programming-model/code-generation' },
      { text: '类型系统与宿主语言边界', link: '/zh/guide/programming-model/type-system' },
      { text: '控制流', link: '/zh/guide/programming-model/control-flow' }
    ]
  },
  {
    text: '数据与硬件',
    collapsed: false,
    items: [
      { text: '数据、布局与切块', link: '/zh/guide/programming-model/data-and-layout' },
      { text: '片上存储、流水与 GM 协作', link: '/zh/guide/programming-model/onchip-memory' },
      { text: '矢量寄存器与 lane 模型', link: '/zh/guide/programming-model/vector-registers' },
      { text: '三类计算单元', link: '/zh/guide/programming-model/compute' },
      { text: '数据类型与量化', link: '/zh/guide/programming-model/data-types' },
      { text: '同步、Cache 与跨核交接', link: '/zh/guide/programming-model/synchronization' }
    ]
  },
  {
    text: '动手实践',
    collapsed: false,
    items: [
      { text: '写出第一个算子', link: '/zh/guide/programming-model/first-operator' },
      { text: '第一个 Cube 算子与第一个 Mix 算子', link: '/zh/guide/programming-model/cube-and-mix' },
      { text: 'AI CPU 与调度计划', link: '/zh/guide/programming-model/aicpu' },
      { text: '融合算子的设计方法论', link: '/zh/guide/programming-model/fusion-design' }
    ]
  },
  {
    text: '工程化与附录',
    collapsed: false,
    items: [
      { text: 'JIT 参数与编译缓存', link: '/zh/guide/programming-model/jit-arguments' },
      { text: 'torch 接口与 stream 语义', link: '/zh/guide/programming-model/torch-interop' },
      { text: '编译选项与产物观察', link: '/zh/guide/programming-model/compiler-options' },
      { text: '高性能算子编写指南', link: '/zh/guide/programming-model/performance' },
      { text: '调试与精度验证', link: '/zh/guide/programming-model/debugging' },
      { text: 'AOT 与 Native 算子包', link: '/zh/guide/programming-model/aot-packaging' },
      { text: '当前限制、迁移与常见问题', link: '/zh/guide/programming-model/limitations' },
      { text: '概念对照、命名速查与术语表', link: '/zh/guide/programming-model/appendix' }
    ]
  }
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'CANNBot-DSL',
  description: '面向 Ascend NPU 的算子开发项目文档',
  base: process.env.DOCS_BASE || '/',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    math: true
  },
  head: [
    ['meta', { name: 'theme-color', content: '#2f5bea' }],
    ['link', { rel: 'icon', href: '/logo.svg', type: 'image/svg+xml' }]
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'CANNBot-DSL',
    nav: [
      { text: '语言 / Language', items: [
        { text: '简体中文', link: '/zh/' },
        { text: 'English', link: '/en/' }
      ] },
      { text: '开始使用', link: '/zh/install/' },
      { text: '编程模型', link: '/zh/guide/programming-model/overview' },
      { text: '样例', link: '/zh/project/examples/' },
      { text: 'API 文档', link: '/zh/api/' },
      { text: '参与贡献', link: '/zh/project/community/contributing' },
      { text: '关于', items: [
        { text: '文档范围', link: '/zh/project/about/scope' },
        { text: '发布与部署', link: '/zh/project/about/deployment' },
        { text: '文档版本策略', link: '/zh/project/about/versioning' }
      ] }
    ],
    sidebar: {
      '/zh/guide/programming-model/': programmingModelSidebar,
      '/zh/install/': [{ text: '开始使用', collapsed: true, items: guideSidebar }],
      '/zh/guide/project/': [{ text: '项目指南', collapsed: true, items: guideSidebar }],
      '/zh/project/examples/': [{ text: '样例', collapsed: true, items: [
        { text: '样例导航', link: '/zh/project/examples/' },
        { text: '运行测试', link: '/zh/project/examples/testing' }
      ] }],
      '/zh/api/': [{
        text: 'API 文档',
        collapsed: false,
        link: '/zh/api/',
        items: [
          { text: '装饰器', link: '/zh/api/decorators' },
          { text: 'Host API', link: '/zh/api/host/', collapsed: true, items: [
            { text: '简介', link: '/zh/api/host/introduction' },
            { text: 'Host 参数与数据描述', link: '/zh/api/host/data-description/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/host/data-description/introduction' },
              { text: 'Dim', link: '/zh/api/host/data-description/dim' },
              { text: 'TensorSpec', link: '/zh/api/host/data-description/tensor_spec' },
              { text: 'TensorListSpec', link: '/zh/api/host/data-description/tensor_list_spec' }
            ] },
            { text: 'Host API：平台信息', link: '/zh/api/host/platform-profiling/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/host/platform-profiling/introduction' },
              { text: 'get_platform_info', link: '/zh/api/host/platform-profiling/get-platform-info' },
              { text: 'get_mem_size', link: '/zh/api/host/platform-profiling/get-mem-size' }
            ] }
          ] },
          { text: 'Kernel API', link: '/zh/api/kernel/', collapsed: false, items: [
            { text: '简介', link: '/zh/api/kernel/introduction' },
            { text: 'Kernel API：基础操作', link: '/zh/api/kernel/base/', collapsed: false, items: [
              { text: '简介', link: '/zh/api/kernel/base/introduction' },
              { text: 'tile_slice', link: '/zh/api/kernel/base/tile-slice' }
            ] },
            { text: '类型与相关接口', link: '/zh/api/kernel/types-and-views/', collapsed: true, items: [
              { text: '类型与相关接口说明', link: '/zh/api/kernel/types-and-views/types-and-related-interfaces' },
              { text: 'Shape 类型', link: '/zh/api/kernel/types-and-views/shape' },
              { text: 'Stride 类型', link: '/zh/api/kernel/types-and-views/stride' },
              { text: 'Layout 类型', link: '/zh/api/kernel/types-and-views/layout' },
              { text: 'Tiler 类型', link: '/zh/api/kernel/types-and-views/tiler' },
              { text: 'Coord 类型', link: '/zh/api/kernel/types-and-views/coord' },
              { text: 'Tensor 类型', link: '/zh/api/kernel/types-and-views/tensor' },
              { text: 'Buffer 类型', link: '/zh/api/kernel/types-and-views/buffer' },
              { text: 'Channel 类型', link: '/zh/api/kernel/types-and-views/channel' },
              { text: 'DelayLineGroup 类型', link: '/zh/api/kernel/types-and-views/delay-line-group' },
              { text: 'make_tiler', link: '/zh/api/kernel/types-and-views/make_tiler' },
              { text: 'make_buffer', link: '/zh/api/kernel/types-and-views/make_buffer' },
              { text: 'make_channel', link: '/zh/api/kernel/types-and-views/make_channel' },
              { text: 'channel_rewind', link: '/zh/api/kernel/types-and-views/channel_rewind' },
              { text: 'idx2crd', link: '/zh/api/kernel/types-and-views/idx2crd' },
              { text: 'ceil_div', link: '/zh/api/kernel/types-and-views/ceil_div' }
            ] },
            { text: '控制流', link: '/zh/api/kernel/control-flow/', collapsed: true, items: [
              { text: '控制流说明', link: '/zh/api/kernel/control-flow/control-flow' },
              { text: '判断条件', link: '/zh/api/kernel/control-flow/predicate-expressions' },
              { text: '值选择', link: '/zh/api/kernel/control-flow/value-selection' },
              { text: 'if', link: '/zh/api/kernel/control-flow/if' },
              { text: 'for', link: '/zh/api/kernel/control-flow/for' },
              { text: 'while', link: '/zh/api/kernel/control-flow/while' },
              { text: '编译期控制', link: '/zh/api/kernel/control-flow/compile-time-control' }
            ] },
            { text: 'Kernel API：数据搬运', link: '/zh/api/kernel/data-movement/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/data-movement/introduction' },
              { text: 'mem_copy', link: '/zh/api/kernel/data-movement/mem-copy' },
              { text: 'make_copy_engine', link: '/zh/api/kernel/data-movement/make-copy-engine' }
            ] },
            { text: 'Reg矢量计算', link: '/zh/api/kernel/reg_compute/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/reg_compute/introduction' },
              { text: '关键特性说明', link: '/zh/api/kernel/reg_compute/key-features' },
              { text: 'Reg数据搬入', link: '/zh/api/kernel/reg_compute/load/', collapsed: true, items: [
                { text: '简介', link: '/zh/api/kernel/reg_compute/load/introduction' },
                { text: 'vload_broadcast', link: '/zh/api/kernel/reg_compute/load/vload-broadcast' },
                { text: 'vload_deinterleave', link: '/zh/api/kernel/reg_compute/load/vload-deinterleave' },
                { text: 'vload_downsample', link: '/zh/api/kernel/reg_compute/load/vload-downsample' },
                { text: 'vload_strided', link: '/zh/api/kernel/reg_compute/load/vload-strided' },
                { text: 'vload_unalign_init', link: '/zh/api/kernel/reg_compute/load/vload-unalign-init' },
                { text: 'vload_unalign', link: '/zh/api/kernel/reg_compute/load/vload-unalign' },
                { text: 'vload_unpack', link: '/zh/api/kernel/reg_compute/load/vload-unpack' },
                { text: 'vload_upsample', link: '/zh/api/kernel/reg_compute/load/vload-upsample' },
                { text: 'vload', link: '/zh/api/kernel/reg_compute/load/vload' },
                { text: 'vmask_load', link: '/zh/api/kernel/reg_compute/load/vmask-load' }
              ] },
              { text: '基础算术', link: '/zh/api/kernel/reg_compute/reg_arith/', collapsed: true, items: [
                { text: 'vabs', link: '/zh/api/kernel/reg_compute/reg_arith/vabs' },
                { text: 'vadd', link: '/zh/api/kernel/reg_compute/reg_arith/vadd' },
                { text: 'vaddc', link: '/zh/api/kernel/reg_compute/reg_arith/vaddc' },
                { text: 'vaddco', link: '/zh/api/kernel/reg_compute/reg_arith/vaddco' },
                { text: 'vadds', link: '/zh/api/kernel/reg_compute/reg_arith/vadds' },
                { text: 'vdiv', link: '/zh/api/kernel/reg_compute/reg_arith/vdiv' },
                { text: 'vexp', link: '/zh/api/kernel/reg_compute/reg_arith/vexp' },
                { text: 'vlog', link: '/zh/api/kernel/reg_compute/reg_arith/vlog' },
                { text: 'vmax', link: '/zh/api/kernel/reg_compute/reg_arith/vmax' },
                { text: 'vmaxs', link: '/zh/api/kernel/reg_compute/reg_arith/vmaxs' },
                { text: 'vmin', link: '/zh/api/kernel/reg_compute/reg_arith/vmin' },
                { text: 'vmins', link: '/zh/api/kernel/reg_compute/reg_arith/vmins' },
                { text: 'vmul', link: '/zh/api/kernel/reg_compute/reg_arith/vmul' },
                { text: 'vmull', link: '/zh/api/kernel/reg_compute/reg_arith/vmull' },
                { text: 'vmuls', link: '/zh/api/kernel/reg_compute/reg_arith/vmuls' },
                { text: 'vneg', link: '/zh/api/kernel/reg_compute/reg_arith/vneg' },
                { text: 'vsqrt', link: '/zh/api/kernel/reg_compute/reg_arith/vsqrt' },
                { text: 'vsub', link: '/zh/api/kernel/reg_compute/reg_arith/vsub' },
                { text: 'vsubbo', link: '/zh/api/kernel/reg_compute/reg_arith/vsubbo' },
                { text: 'vsubc', link: '/zh/api/kernel/reg_compute/reg_arith/vsubc' }
              ] },
              { text: '广播操作', link: '/zh/api/kernel/reg_compute/reg_broadcast/', collapsed: true, items: [
                { text: 'vdup', link: '/zh/api/kernel/reg_compute/reg_broadcast/vdup' },
                { text: 'vdups', link: '/zh/api/kernel/reg_compute/reg_broadcast/vdups' }
              ] },
              { text: '比较计算', link: '/zh/api/kernel/reg_compute/reg_compare/', collapsed: true, items: [
                { text: 'veq', link: '/zh/api/kernel/reg_compute/reg_compare/veq' },
                { text: 'veqs', link: '/zh/api/kernel/reg_compute/reg_compare/veqs' },
                { text: 'vge', link: '/zh/api/kernel/reg_compute/reg_compare/vge' },
                { text: 'vges', link: '/zh/api/kernel/reg_compute/reg_compare/vges' },
                { text: 'vgt', link: '/zh/api/kernel/reg_compute/reg_compare/vgt' },
                { text: 'vgts', link: '/zh/api/kernel/reg_compute/reg_compare/vgts' },
                { text: 'vle', link: '/zh/api/kernel/reg_compute/reg_compare/vle' },
                { text: 'vles', link: '/zh/api/kernel/reg_compute/reg_compare/vles' },
                { text: 'vlt', link: '/zh/api/kernel/reg_compute/reg_compare/vlt' },
                { text: 'vlts', link: '/zh/api/kernel/reg_compute/reg_compare/vlts' },
                { text: 'vne', link: '/zh/api/kernel/reg_compute/reg_compare/vne' },
                { text: 'vnes', link: '/zh/api/kernel/reg_compute/reg_compare/vnes' }
              ] },
              { text: '类型转换', link: '/zh/api/kernel/reg_compute/reg_convert/', collapsed: true, items: [
                { text: 'RoundingMode', link: '/zh/api/kernel/reg_compute/reg_convert/roundingmode' },
                { text: 'vcast', link: '/zh/api/kernel/reg_compute/reg_convert/vcast' },
                { text: 'vceil', link: '/zh/api/kernel/reg_compute/reg_convert/vceil' },
                { text: 'vfloor', link: '/zh/api/kernel/reg_compute/reg_convert/vfloor' },
                { text: 'vreinterpret_lanes', link: '/zh/api/kernel/reg_compute/reg_convert/vreinterpret-lanes' },
                { text: 'vreinterpret', link: '/zh/api/kernel/reg_compute/reg_convert/vreinterpret' },
                { text: 'vtrunc', link: '/zh/api/kernel/reg_compute/reg_convert/vtrunc' }
              ] },
              { text: '复合计算', link: '/zh/api/kernel/reg_compute/reg_fused/', collapsed: true, items: [
                { text: 'vabs_sub', link: '/zh/api/kernel/reg_compute/reg_fused/vabs-sub' },
                { text: 'vaxpy', link: '/zh/api/kernel/reg_compute/reg_fused/vaxpy' },
                { text: 'vcast_exp_sub', link: '/zh/api/kernel/reg_compute/reg_fused/vcast-exp-sub' },
                { text: 'vexp_sub', link: '/zh/api/kernel/reg_compute/reg_fused/vexp-sub' },
                { text: 'vleakyrelu', link: '/zh/api/kernel/reg_compute/reg_fused/vleakyrelu' },
                { text: 'vmadd', link: '/zh/api/kernel/reg_compute/reg_fused/vmadd' },
                { text: 'vmula', link: '/zh/api/kernel/reg_compute/reg_fused/vmula' },
                { text: 'vprelu', link: '/zh/api/kernel/reg_compute/reg_fused/vprelu' },
                { text: 'vrelu', link: '/zh/api/kernel/reg_compute/reg_fused/vrelu' }
              ] },
              { text: '聚合操作', link: '/zh/api/kernel/reg_compute/reg_gather/', collapsed: true, items: [
                { text: 'vgather_reg', link: '/zh/api/kernel/reg_compute/reg_gather/vgather-reg' }
              ] },
              { text: '直方图', link: '/zh/api/kernel/reg_compute/reg_histogram/', collapsed: true, items: [
                { text: 'vhistogram_accumulate', link: '/zh/api/kernel/reg_compute/reg_histogram/vhistogram-accumulate' },
                { text: 'vhistogram_frequency', link: '/zh/api/kernel/reg_compute/reg_histogram/vhistogram-frequency' }
              ] },
              { text: '索引操作', link: '/zh/api/kernel/reg_compute/reg_index/', collapsed: true, items: [
                { text: 'varange', link: '/zh/api/kernel/reg_compute/reg_index/varange' }
              ] },
              { text: '逻辑计算', link: '/zh/api/kernel/reg_compute/reg_logic/', collapsed: true, items: [
                { text: 'mask_and', link: '/zh/api/kernel/reg_compute/reg_logic/mask-and' },
                { text: 'mask_not', link: '/zh/api/kernel/reg_compute/reg_logic/mask-not' },
                { text: 'mask_or', link: '/zh/api/kernel/reg_compute/reg_logic/mask-or' },
                { text: 'mask_xor', link: '/zh/api/kernel/reg_compute/reg_logic/mask-xor' },
                { text: 'vbitwise_and', link: '/zh/api/kernel/reg_compute/reg_logic/vbitwise-and' },
                { text: 'vbitwise_or', link: '/zh/api/kernel/reg_compute/reg_logic/vbitwise-or' },
                { text: 'vbitwise_xor', link: '/zh/api/kernel/reg_compute/reg_logic/vbitwise-xor' },
                { text: 'vmask', link: '/zh/api/kernel/reg_compute/reg_logic/vmask' },
                { text: 'vnot', link: '/zh/api/kernel/reg_compute/reg_logic/vnot' },
                { text: 'vshl', link: '/zh/api/kernel/reg_compute/reg_logic/vshl' },
                { text: 'vshr', link: '/zh/api/kernel/reg_compute/reg_logic/vshr' }
              ] },
              { text: '掩码寄存器操作', link: '/zh/api/kernel/reg_compute/reg_mask/', collapsed: true, items: [
                { text: 'create_mask', link: '/zh/api/kernel/reg_compute/reg_mask/create-mask' },
                { text: 'full_mask', link: '/zh/api/kernel/reg_compute/reg_mask/full-mask' },
                { text: 'mask_counter', link: '/zh/api/kernel/reg_compute/reg_mask/mask-counter' },
                { text: 'update_mask', link: '/zh/api/kernel/reg_compute/reg_mask/update-mask' }
              ] },
              { text: '排布变换', link: '/zh/api/kernel/reg_compute/reg_permute_sel/', collapsed: true, items: [
                { text: 'vcompress', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vcompress' },
                { text: 'vdeinterleave', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vdeinterleave' },
                { text: 'vinterleave', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vinterleave' },
                { text: 'vmerge', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vmerge' },
                { text: 'vpack', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vpack' },
                { text: 'vselect', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vselect' },
                { text: 'vsqueeze_and_storeunalign_finalize', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vsqueeze-and-storeunalign-finalize' },
                { text: 'vsqueeze_and_storeunalign_init', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vsqueeze-and-storeunalign-init' },
                { text: 'vsqueeze_and_storeunalign', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vsqueeze-and-storeunalign' },
                { text: 'vsqueeze', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vsqueeze' },
                { text: 'vstore_unalign_begin', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vstore-unalign-begin' },
                { text: 'vstore_unalign', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vstore-unalign' },
                { text: 'vunpack', link: '/zh/api/kernel/reg_compute/reg_permute_sel/vunpack' }
              ] },
              { text: '归约计算', link: '/zh/api/kernel/reg_compute/reg_reduce/', collapsed: true, items: [
                { text: 'vpair_reduce_sum', link: '/zh/api/kernel/reg_compute/reg_reduce/vpair-reduce-sum' },
                { text: 'vreduce_max_datablock', link: '/zh/api/kernel/reg_compute/reg_reduce/vreduce-max-datablock' },
                { text: 'vreduce_max', link: '/zh/api/kernel/reg_compute/reg_reduce/vreduce-max' },
                { text: 'vreduce_min_datablock', link: '/zh/api/kernel/reg_compute/reg_reduce/vreduce-min-datablock' },
                { text: 'vreduce_min', link: '/zh/api/kernel/reg_compute/reg_reduce/vreduce-min' },
                { text: 'vreduce_sum_datablock', link: '/zh/api/kernel/reg_compute/reg_reduce/vreduce-sum-datablock' },
                { text: 'vreduce_sum', link: '/zh/api/kernel/reg_compute/reg_reduce/vreduce-sum' }
              ] },
              { text: '同步控制', link: '/zh/api/kernel/reg_compute/reg_sync/', collapsed: true, items: [
                { text: 'vmem_bar', link: '/zh/api/kernel/reg_compute/reg_sync/vmem-bar' }
              ] },
              { text: 'Reg离散搬出', link: '/zh/api/kernel/reg_compute/scatter/', collapsed: true, items: [
                { text: 'vscatter', link: '/zh/api/kernel/reg_compute/scatter/vscatter' }
              ] },
              { text: 'Reg数据搬出', link: '/zh/api/kernel/reg_compute/store/', collapsed: true, items: [
                { text: '简介', link: '/zh/api/kernel/reg_compute/store/introduction' },
                { text: 'vmask_store', link: '/zh/api/kernel/reg_compute/store/vmask-store' },
                { text: 'vstore_first', link: '/zh/api/kernel/reg_compute/store/vstore-first' },
                { text: 'vstore_interleave', link: '/zh/api/kernel/reg_compute/store/vstore-interleave' },
                { text: 'vstore_mask', link: '/zh/api/kernel/reg_compute/store/vstore-mask' },
                { text: 'vstore_pack', link: '/zh/api/kernel/reg_compute/store/vstore-pack' },
                { text: 'vstore_strided', link: '/zh/api/kernel/reg_compute/store/vstore-strided' },
                { text: 'vstore_unalign_post', link: '/zh/api/kernel/reg_compute/store/vstore-unalign-post' },
                { text: 'vstore', link: '/zh/api/kernel/reg_compute/store/vstore' },
                { text: 'vstorealign_squeeze_status', link: '/zh/api/kernel/reg_compute/store/vstorealign-squeeze-status' }
              ] },
              { text: 'Reg聚合搬入', link: '/zh/api/kernel/reg_compute/ub_gather/', collapsed: true, items: [
                { text: 'vgather_datablock', link: '/zh/api/kernel/reg_compute/ub_gather/vgather-datablock' },
                { text: 'vgather', link: '/zh/api/kernel/reg_compute/ub_gather/vgather' }
              ] }
            ] },
            { text: '标量计算', link: '/zh/api/kernel/scalar_compute/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/scalar_compute/introduction' },
              { text: '位运算', link: '/zh/api/kernel/scalar_compute/scalar_bit/', collapsed: true, items: [
                { text: 'clear_nthbit', link: '/zh/api/kernel/scalar_compute/scalar_bit/clear-nthbit' },
                { text: 'clz', link: '/zh/api/kernel/scalar_compute/scalar_bit/clz' },
                { text: 'ffs', link: '/zh/api/kernel/scalar_compute/scalar_bit/ffs' },
                { text: 'ffz', link: '/zh/api/kernel/scalar_compute/scalar_bit/ffz' },
                { text: 'scalar.popc', link: '/zh/api/kernel/scalar_compute/scalar_bit/scalar-popc' },
                { text: 'set_nthbit', link: '/zh/api/kernel/scalar_compute/scalar_bit/set-nthbit' },
                { text: 'sflbits', link: '/zh/api/kernel/scalar_compute/scalar_bit/sflbits' },
                { text: 'zero_bits_cnt', link: '/zh/api/kernel/scalar_compute/scalar_bit/zero-bits-cnt' }
              ] },
              { text: '类型转换', link: '/zh/api/kernel/scalar_compute/scalar_convert/', collapsed: true, items: [
                { text: 'cast', link: '/zh/api/kernel/scalar_compute/scalar_convert/cast' },
                { text: 'RoundingMode', link: '/zh/api/kernel/scalar_compute/scalar_convert/roundingmode' }
              ] },
              { text: '数据搬入', link: '/zh/api/kernel/scalar_compute/scalar_load/', collapsed: true, items: [
                { text: 'load_bypass', link: '/zh/api/kernel/scalar_compute/scalar_load/load-bypass' }
              ] },
              { text: '数据搬出', link: '/zh/api/kernel/scalar_compute/scalar_store/', collapsed: true, items: [
                { text: 'cube_store_bypass', link: '/zh/api/kernel/scalar_compute/scalar_store/cube-store-bypass' },
                { text: 'vec_store_bypass', link: '/zh/api/kernel/scalar_compute/scalar_store/vec-store-bypass' }
              ] }
            ] },
            { text: '原子操作', link: '/zh/api/kernel/atomic/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/atomic/introduction' },
              { text: '关键特性说明', link: '/zh/api/kernel/atomic/key-features' },
              { text: '标量原子操作', link: '/zh/api/kernel/atomic/scalar_atomic/', collapsed: true, items: [
                { text: 'atomic_add', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-add' },
                { text: 'atomic_and', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-and' },
                { text: 'atomic_cas', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-cas' },
                { text: 'atomic_dec', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-dec' },
                { text: 'atomic_exch', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-exch' },
                { text: 'atomic_inc', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-inc' },
                { text: 'atomic_max', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-max' },
                { text: 'atomic_min', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-min' },
                { text: 'atomic_or', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-or' },
                { text: 'atomic_sub', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-sub' },
                { text: 'atomic_xor', link: '/zh/api/kernel/atomic/scalar_atomic/atomic-xor' }
              ] }
            ] },
            { text: '矩阵计算', link: '/zh/api/kernel/cube_compute/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/cube_compute/introduction' },
              { text: '矩阵计算单元', link: '/zh/api/kernel/cube_compute/unit' },
              { text: '计算流程', link: '/zh/api/kernel/cube_compute/flow' },
              { text: '背景与核心概念', link: '/zh/api/kernel/cube_compute/fractal-intro' },
              { text: '关键分形格式', link: '/zh/api/kernel/cube_compute/fractal-formats' },
              { text: '矩阵计算关键特性说明', link: '/zh/api/kernel/cube_compute/key-features', collapsed: true, items: [
                { text: 'GEMV', link: '/zh/api/kernel/cube_compute/key-features-gemv' },
                { text: 'HF32', link: '/zh/api/kernel/cube_compute/key-features-hf32' },
                { text: 'UnitFlag', link: '/zh/api/kernel/cube_compute/key-features-unit-flag' }
              ] },
              { text: 'enable_fp8', link: '/zh/api/kernel/cube_compute/enable-fp8' },
              { text: 'enable_hf32_trans', link: '/zh/api/kernel/cube_compute/enable-hf32-trans' },
              { text: 'enable_hf32', link: '/zh/api/kernel/cube_compute/enable-hf32' },
              { text: 'enable_hif8', link: '/zh/api/kernel/cube_compute/enable-hif8' },
              { text: 'matmul', link: '/zh/api/kernel/cube_compute/matmul' },
              { text: 'set_fp32_mode', link: '/zh/api/kernel/cube_compute/set-fp32-mode' },
              { text: 'set_hf32_round_mode', link: '/zh/api/kernel/cube_compute/set-hf32-round-mode' },
              { text: 'set_mmad_direction', link: '/zh/api/kernel/cube_compute/set-mmad-direction' }
            ] },
            { text: 'Kernel API：系统变量访问', link: '/zh/api/kernel/system/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/system/introduction' },
              { text: 'get_block_idx', link: '/zh/api/kernel/system/get-block-idx' },
              { text: 'get_block_num', link: '/zh/api/kernel/system/get-block-num' },
              { text: 'get_subblock_id', link: '/zh/api/kernel/system/get-subblock-id' },
              { text: 'get_subblock_dim', link: '/zh/api/kernel/system/get-subblock-dim' },
              { text: 'get_core_id', link: '/zh/api/kernel/system/get-core-id' },
              { text: 'get_system_cycle', link: '/zh/api/kernel/system/get-system-cycle' },
              { text: 'get_status', link: '/zh/api/kernel/system/get-status' },
              { text: 'get_vf_len', link: '/zh/api/kernel/system/get-vf-len' },
              { text: 'get_squeeze_status', link: '/zh/api/kernel/system/get-squeeze-status' }
            ] },
            { text: 'Kernel API：同步与缓存控制', link: '/zh/api/kernel/synchronization-cache/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/synchronization-cache/introduction' },
              { text: 'dcci_single', link: '/zh/api/kernel/synchronization-cache/dcci-single' },
              { text: 'dcci_entire_out', link: '/zh/api/kernel/synchronization-cache/dcci-entire-out' },
              { text: 'dcci_entire_atomic', link: '/zh/api/kernel/synchronization-cache/dcci-entire-atomic' },
              { text: 'dci', link: '/zh/api/kernel/synchronization-cache/dci' }
            ] },
            { text: 'Kernel API：同步管理', link: '/zh/api/kernel/resource-management/', collapsed: true, items: [
              { text: '简介', link: '/zh/api/kernel/resource-management/introduction' },
              { text: '系统同步能力概述', link: '/zh/api/kernel/resource-management/system-sync-overview' },
              { text: '核间同步能力概述', link: '/zh/api/kernel/resource-management/inter-core-sync-overview' },
              { text: '关键特性说明', link: '/zh/api/kernel/resource-management/key-features' },
              { text: 'cube_sync_block_arrive', link: '/zh/api/kernel/resource-management/cube-sync-block-arrive' },
              { text: 'cube_sync_block_wait', link: '/zh/api/kernel/resource-management/cube-sync-block-wait' },
              { text: 'cube_sync_intra_arrive', link: '/zh/api/kernel/resource-management/cube-sync-intra-arrive' },
              { text: 'cube_sync_intra_wait', link: '/zh/api/kernel/resource-management/cube-sync-intra-wait' },
              { text: 'global_sync_all', link: '/zh/api/kernel/resource-management/global-sync-all' },
              { text: 'vec_sync_block_arrive', link: '/zh/api/kernel/resource-management/vec-sync-block-arrive' },
              { text: 'vec_sync_block_wait', link: '/zh/api/kernel/resource-management/vec-sync-block-wait' },
              { text: 'vec_sync_intra_arrive', link: '/zh/api/kernel/resource-management/vec-sync-intra-arrive' },
              { text: 'vec_sync_intra_wait', link: '/zh/api/kernel/resource-management/vec-sync-intra-wait' }
            ] },
            { text: 'Debug API：调试接口', link: '/zh/api/kernel/debug/', collapsed: false, items: [
              { text: '简介', link: '/zh/api/kernel/debug/introduction' },
              { text: 'print', link: '/zh/api/kernel/debug/print' }
            ] }
          ] },
          { text: 'AI CPU API', link: '/zh/api/aicpu/', collapsed: true, items: [
            { text: '简介', link: '/zh/api/aicpu/introduction' }
          ] },
          { text: '通用接口文档模板', link: '/zh/api/api-template' },
        ]
      }],
      '/zh/project/community/': [{ text: '社区', collapsed: true, items: [
        { text: '参与贡献', link: '/zh/project/community/contributing' }
      ] }],
      '/zh/project/about/': [{ text: '关于', collapsed: true, items: [
        { text: '文档范围', link: '/zh/project/about/scope' },
        { text: '发布与部署', link: '/zh/project/about/deployment' },
        { text: '文档版本策略', link: '/zh/project/about/versioning' }
      ] }]
    },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到相关内容',
            resetButtonTitle: '清空搜索',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一页', next: '下一页' },
    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '切换主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    socialLinks: [
      { icon: { svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="m8 6-6 6 6 6m8-12 6 6-6 6m-2-15-4 18"/></svg>' }, link: 'https://gitcode.com/cann/cannbot-dsl', ariaLabel: 'GitCode 源码仓库' }
    ],
    footer: {
      message: `CANNBot-DSL 开源项目文档 · ${docsVersion}`,
      copyright: '基于 CANN Open Software License Agreement Version 2.0'
    }
  },
  sitemap: {
    hostname: 'https://cannbot-dsl.gitcode.com'
  }
})
