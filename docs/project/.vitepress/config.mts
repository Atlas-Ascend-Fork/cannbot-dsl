/*
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
  */
import { defineConfig } from 'vitepress'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const docsVersion = process.env.DOCS_VERSION || 'next'

// Keep directory entry URLs while using README files as the source entry points.
const rewriteReadmes = (id: string) => {
  if (id === 'README.md') return 'index.md'
  if (id.endsWith('/README.md')) return `${id.slice(0, -'README.md'.length)}index.md`
  return id
}

const guideSidebar = [
  { text: '源码构建', link: '/zh/install/source-build' },
  { text: '编程模型', link: '/zh/guide/overview' },
  { text: '仓库结构', link: '/zh/guide/repository' },
  { text: '样例导航', link: '/samples/' }
]

const programmingModelSidebar = [
  { text: '编程模型概览', link: '/zh/guide/overview' },
  {
    text: '语言机制',
    collapsed: false,
    items: [
      { text: '简介', link: '/zh/guide/introduction' },
      { text: '硬件与执行模型', link: '/zh/guide/hardware-model' },
      { text: '代码生成', link: '/zh/guide/code-generation' },
      { text: '类型系统与宿主语言边界', link: '/zh/guide/type-system' },
      { text: '控制流', link: '/zh/guide/control-flow' }
    ]
  },
  {
    text: '数据与硬件',
    collapsed: false,
    items: [
      { text: '数据、布局与切块', link: '/zh/guide/data-and-layout' },
      { text: '片上存储、流水与 GM 协作', link: '/zh/guide/onchip-memory' },
      { text: '矢量寄存器与 lane 模型', link: '/zh/guide/vector-registers' },
      { text: '三类计算单元', link: '/zh/guide/compute' },
      { text: '数据类型与量化', link: '/zh/guide/data-types' },
      { text: '同步、Cache 与跨核交接', link: '/zh/guide/synchronization' }
    ]
  },
  {
    text: '动手实践',
    collapsed: false,
    items: [
      { text: '写出第一个算子', link: '/zh/guide/first-operator' },
      { text: '第一个 Cube 算子与第一个 Mix 算子', link: '/zh/guide/cube-and-mix' },
      { text: 'AI CPU 与调度计划', link: '/zh/guide/aicpu' },
      { text: '融合算子的设计方法论', link: '/zh/guide/fusion-design' }
    ]
  },
  {
    text: '工程化与附录',
    collapsed: false,
    items: [
      { text: 'JIT 参数与编译缓存', link: '/zh/guide/jit-arguments' },
      { text: 'torch 接口与 stream 语义', link: '/zh/guide/torch-interop' },
      { text: '编译选项与产物观察', link: '/zh/guide/compiler-options' },
      { text: '高性能算子编写指南', link: '/zh/guide/performance' },
      { text: '调试与精度验证', link: '/zh/guide/debugging' },
      { text: 'AOT 与 Native 算子包', link: '/zh/guide/aot-packaging' },
      { text: '当前限制、迁移与常见问题', link: '/zh/guide/limitations' },
      { text: '概念对照、命名速查与术语表', link: '/zh/guide/appendix' }
    ]
  }
]

const apiSidebar = [
  {
    "text": "API 文档",
    "collapsed": false,
    "items": [
      {
        "text": "语言",
        "link": "/zh/api/language",
        "items": [
          {
            "text": "控制流",
            "link": "/zh/api/language/control-flow",
            "items": [
              {
                "text": "编译期控制",
                "link": "/zh/api/language/control-flow/compile-time-control",
                "items": [
                  {
                    "text": "const_expr",
                    "link": "/zh/api/language/control-flow/compile-time-control/const-expr"
                  },
                  {
                    "text": "target_version",
                    "link": "/zh/api/language/control-flow/compile-time-control/target-version"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "谓词表达式",
                "link": "/zh/api/language/control-flow/predicate-expressions",
                "items": [
                  {
                    "text": "all",
                    "link": "/zh/api/language/control-flow/predicate-expressions/all"
                  },
                  {
                    "text": "and",
                    "link": "/zh/api/language/control-flow/predicate-expressions/and"
                  },
                  {
                    "text": "any",
                    "link": "/zh/api/language/control-flow/predicate-expressions/any"
                  },
                  {
                    "text": "链式比较",
                    "link": "/zh/api/language/control-flow/predicate-expressions/chained-comparisons"
                  },
                  {
                    "text": "not",
                    "link": "/zh/api/language/control-flow/predicate-expressions/not"
                  },
                  {
                    "text": "or",
                    "link": "/zh/api/language/control-flow/predicate-expressions/or"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "值选择",
                "link": "/zh/api/language/control-flow/value-selection",
                "items": [
                  {
                    "text": "条件表达式",
                    "link": "/zh/api/language/control-flow/value-selection/conditional-expression"
                  },
                  {
                    "text": "max",
                    "link": "/zh/api/language/control-flow/value-selection/max"
                  },
                  {
                    "text": "min",
                    "link": "/zh/api/language/control-flow/value-selection/min"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "编译期控制",
                "link": "/zh/api/language/control-flow/compile-time-control"
              },
              {
                "text": "控制流",
                "link": "/zh/api/language/control-flow/control-flow"
              },
              {
                "text": "for",
                "link": "/zh/api/language/control-flow/for"
              },
              {
                "text": "if / elif / else",
                "link": "/zh/api/language/control-flow/if"
              },
              {
                "text": "判断条件",
                "link": "/zh/api/language/control-flow/predicate-expressions"
              },
              {
                "text": "值选择",
                "link": "/zh/api/language/control-flow/value-selection"
              },
              {
                "text": "while",
                "link": "/zh/api/language/control-flow/while"
              }
            ],
            "collapsed": true
          },
          {
            "text": "CANNBot-DSL 装饰器",
            "link": "/zh/api/language/decorators"
          },
          {
            "text": "Kernel 启动配置",
            "link": "/zh/api/language/launch"
          }
        ],
        "collapsed": true
      },
      {
        "text": "数据结构",
        "link": "/zh/api/data-structures",
        "items": [
          {
            "text": "Array 接口",
            "link": "/zh/api/data-structures/array",
            "items": [
              {
                "text": "Array 构造函数",
                "link": "/zh/api/data-structures/array/Array"
              },
              {
                "text": "Array 简介",
                "link": "/zh/api/data-structures/array/overview"
              }
            ],
            "collapsed": true
          },
          {
            "text": "数据类型",
            "link": "/zh/api/data-structures/data-types",
            "items": [
              {
                "text": "数据类型",
                "link": "/zh/api/data-structures/data-types"
              }
            ],
            "collapsed": true
          },
          {
            "text": "Layout 接口",
            "link": "/zh/api/data-structures/layout",
            "items": [
              {
                "text": "Coord",
                "link": "/zh/api/data-structures/layout/coord",
                "items": [
                  {
                    "text": "Coord",
                    "link": "/zh/api/data-structures/layout/coord"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Shape",
                "link": "/zh/api/data-structures/layout/shape",
                "items": [
                  {
                    "text": "Shape",
                    "link": "/zh/api/data-structures/layout/shape"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Stride",
                "link": "/zh/api/data-structures/layout/stride",
                "items": [
                  {
                    "text": "Stride",
                    "link": "/zh/api/data-structures/layout/stride"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Tiler 接口",
                "link": "/zh/api/data-structures/tiler",
                "items": [
                  {
                    "text": "make_tiler",
                    "link": "/zh/api/data-structures/tiler/make_tiler"
                  },
                  {
                    "text": "Tile / Tiler 简介",
                    "link": "/zh/api/data-structures/tiler/overview"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "make_layout",
                "link": "/zh/api/data-structures/layout/make_layout"
              },
              {
                "text": "Layout.offset",
                "link": "/zh/api/data-structures/layout/offset"
              },
              {
                "text": "Layout 简介",
                "link": "/zh/api/data-structures/layout/overview"
              }
            ],
            "collapsed": true
          },
          {
            "text": "片上存储与缓冲区",
            "link": "/zh/api/data-structures/storage",
            "items": [
              {
                "text": "Buffer 接口",
                "link": "/zh/api/data-structures/storage/buffer",
                "items": [
                  {
                    "text": "Buffer 构造函数",
                    "link": "/zh/api/data-structures/storage/buffer/Buffer"
                  },
                  {
                    "text": "make_buffer",
                    "link": "/zh/api/data-structures/storage/buffer/make_buffer"
                  },
                  {
                    "text": "Buffer 简介",
                    "link": "/zh/api/data-structures/storage/buffer/overview"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "RingBuffer 接口",
                "link": "/zh/api/data-structures/storage/ring-buffer",
                "items": [
                  {
                    "text": "make_ring_buffer",
                    "link": "/zh/api/data-structures/storage/ring-buffer/make_ring_buffer"
                  },
                  {
                    "text": "RingBuffer.next_read",
                    "link": "/zh/api/data-structures/storage/ring-buffer/next_read"
                  },
                  {
                    "text": "RingBuffer.next_write",
                    "link": "/zh/api/data-structures/storage/ring-buffer/next_write"
                  },
                  {
                    "text": "RingBuffer 简介",
                    "link": "/zh/api/data-structures/storage/ring-buffer/overview"
                  },
                  {
                    "text": "RingBuffer 构造函数",
                    "link": "/zh/api/data-structures/storage/ring-buffer/RingBuffer"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Storage 接口",
                "link": "/zh/api/data-structures/storage/storage",
                "items": [
                  {
                    "text": "Storage 简介",
                    "link": "/zh/api/data-structures/storage/storage/overview"
                  },
                  {
                    "text": "Storage.view",
                    "link": "/zh/api/data-structures/storage/storage/view"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "存储位置",
                "link": "/zh/api/data-structures/storage/memory-spaces"
              },
              {
                "text": "片上存储空间",
                "link": "/zh/api/data-structures/storage/on-chip-memory"
              }
            ],
            "collapsed": true
          },
          {
            "text": "Tensor 接口",
            "link": "/zh/api/data-structures/tensor",
            "items": [
              {
                "text": "Tensor.bare",
                "link": "/zh/api/data-structures/tensor/bare"
              },
              {
                "text": "Tensor.broadcast_to",
                "link": "/zh/api/data-structures/tensor/broadcast_to"
              },
              {
                "text": "Tensor.extent",
                "link": "/zh/api/data-structures/tensor/extent"
              },
              {
                "text": "Tensor.flatten",
                "link": "/zh/api/data-structures/tensor/flatten"
              },
              {
                "text": "Tensor 简介",
                "link": "/zh/api/data-structures/tensor/overview"
              },
              {
                "text": "Tensor.pad_to",
                "link": "/zh/api/data-structures/tensor/pad_to"
              },
              {
                "text": "Tensor.partition",
                "link": "/zh/api/data-structures/tensor/partition"
              },
              {
                "text": "Tensor.permute",
                "link": "/zh/api/data-structures/tensor/permute"
              },
              {
                "text": "Tensor.reinterpret",
                "link": "/zh/api/data-structures/tensor/reinterpret"
              },
              {
                "text": "Tensor.reshape",
                "link": "/zh/api/data-structures/tensor/reshape"
              },
              {
                "text": "Tensor.strided_slice",
                "link": "/zh/api/data-structures/tensor/strided_slice"
              },
              {
                "text": "Tensor.unflatten",
                "link": "/zh/api/data-structures/tensor/unflatten"
              },
              {
                "text": "Tensor.view",
                "link": "/zh/api/data-structures/tensor/view"
              },
              {
                "text": "Tensor.with_tail_fill",
                "link": "/zh/api/data-structures/tensor/with_tail_fill"
              }
            ],
            "collapsed": true
          }
        ],
        "collapsed": true
      },
      {
        "text": "计算",
        "link": "/zh/api/compute",
        "items": [
          {
            "text": "原子计算",
            "link": "/zh/api/compute/atomic",
            "items": [
              {
                "text": "标量原子操作",
                "link": "/zh/api/compute/atomic/scalar_atomic",
                "items": [
                  {
                    "text": "atomic_add",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-add"
                  },
                  {
                    "text": "atomic_and",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-and"
                  },
                  {
                    "text": "atomic_cas",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-cas"
                  },
                  {
                    "text": "atomic_dec",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-dec"
                  },
                  {
                    "text": "atomic_exch",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-exch"
                  },
                  {
                    "text": "atomic_inc",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-inc"
                  },
                  {
                    "text": "atomic_max",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-max"
                  },
                  {
                    "text": "atomic_min",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-min"
                  },
                  {
                    "text": "atomic_or",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-or"
                  },
                  {
                    "text": "atomic_sub",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-sub"
                  },
                  {
                    "text": "atomic_xor",
                    "link": "/zh/api/compute/atomic/scalar_atomic/atomic-xor"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "关键特性说明",
                "link": "/zh/api/compute/atomic/key-features"
              },
              {
                "text": "原子计算",
                "link": "/zh/api/compute/atomic/overview"
              }
            ],
            "collapsed": true
          },
          {
            "text": "矩阵计算",
            "link": "/zh/api/compute/matrix",
            "items": [
              {
                "text": "概述",
                "items": [
                  {
                    "text": "矩阵计算单元",
                    "link": "/zh/api/compute/matrix/unit"
                  },
                  {
                    "text": "普通矩阵计算流程",
                    "link": "/zh/api/compute/matrix/flow"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "矩阵计算分形介绍",
                "items": [
                  {
                    "text": "背景与核心概念",
                    "link": "/zh/api/compute/matrix/fractal-intro"
                  },
                  {
                    "text": "关键分形格式详解",
                    "link": "/zh/api/compute/matrix/fractal-formats"
                  },
                  {
                    "text": "辅助矩阵分形格式详解",
                    "link": "/zh/api/compute/matrix/aux-fractal-formats"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Mmad计算",
                "items": [
                  {
                    "text": "matmul（普通模式）",
                    "link": "/zh/api/compute/matrix/matmul"
                  },
                  {
                    "text": "matmul（MX模式）",
                    "link": "/zh/api/compute/matrix/matmul-mx"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Mmad计算关键特性说明",
                "link": "/zh/api/compute/matrix/key-features",
                "items": [
                  {
                    "text": "GEMV",
                    "link": "/zh/api/compute/matrix/key-features-gemv"
                  },
                  {
                    "text": "HF32",
                    "link": "/zh/api/compute/matrix/key-features-hf32"
                  },
                  {
                    "text": "UnitFlag",
                    "link": "/zh/api/compute/matrix/key-features-unit-flag"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Mmad计算辅助配置接口",
                "items": [
                  {
                    "text": "set_fp32_mode",
                    "link": "/zh/api/compute/matrix/set-fp32-mode"
                  },
                  {
                    "text": "enable_fp8",
                    "link": "/zh/api/compute/matrix/enable-fp8"
                  },
                  {
                    "text": "enable_hf32",
                    "link": "/zh/api/compute/matrix/enable-hf32"
                  },
                  {
                    "text": "enable_hif8",
                    "link": "/zh/api/compute/matrix/enable-hif8"
                  },
                  {
                    "text": "set_hf32_round_mode",
                    "link": "/zh/api/compute/matrix/set-hf32-round-mode"
                  },
                  {
                    "text": "set_mmad_direction",
                    "link": "/zh/api/compute/matrix/set-mmad-direction"
                  }
                ],
                "collapsed": true
              }
            ],
            "collapsed": true
          },
          {
            "text": "数据搬运",
            "link": "/zh/api/data-movement",
            "items": [
              {
                "text": "CopyOptions",
                "link": "/zh/api/data-movement/copy-options-type"
              },
              {
                "text": "copy_options",
                "link": "/zh/api/data-movement/copy-options"
              },
              {
                "text": "mem_copy",
                "link": "/zh/api/data-movement/mem-copy"
              }
            ],
            "collapsed": true
          },
          {
            "text": "标量计算",
            "link": "/zh/api/compute/scalar",
            "items": [
              {
                "text": "标量位操作",
                "link": "/zh/api/compute/scalar/scalar_bit",
                "items": [
                  {
                    "text": "clear_nthbit",
                    "link": "/zh/api/compute/scalar/scalar_bit/clear-nthbit"
                  },
                  {
                    "text": "clz",
                    "link": "/zh/api/compute/scalar/scalar_bit/clz"
                  },
                  {
                    "text": "ffs",
                    "link": "/zh/api/compute/scalar/scalar_bit/ffs"
                  },
                  {
                    "text": "ffz",
                    "link": "/zh/api/compute/scalar/scalar_bit/ffz"
                  },
                  {
                    "text": "scalar.popc",
                    "link": "/zh/api/compute/scalar/scalar_bit/scalar-popc"
                  },
                  {
                    "text": "set_nthbit",
                    "link": "/zh/api/compute/scalar/scalar_bit/set-nthbit"
                  },
                  {
                    "text": "sflbits",
                    "link": "/zh/api/compute/scalar/scalar_bit/sflbits"
                  },
                  {
                    "text": "zero_bits_cnt",
                    "link": "/zh/api/compute/scalar/scalar_bit/zero-bits-cnt"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "标量类型转换",
                "link": "/zh/api/compute/scalar/scalar_convert",
                "items": [
                  {
                    "text": "cast",
                    "link": "/zh/api/compute/scalar/scalar_convert/cast"
                  },
                  {
                    "text": "RoundingMode",
                    "link": "/zh/api/compute/scalar/scalar_convert/roundingmode"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "load_bypass",
                "link": "/zh/api/compute/scalar/scalar_load/load-bypass"
              },
              {
                "text": "标量数据搬运",
                "link": "/zh/api/compute/scalar/scalar_store",
                "items": [
                  {
                    "text": "cube_store_bypass",
                    "link": "/zh/api/compute/scalar/scalar_store/cube-store-bypass"
                  },
                  {
                    "text": "vec_store_bypass",
                    "link": "/zh/api/compute/scalar/scalar_store/vec-store-bypass"
                  }
                ],
                "collapsed": true
              }
            ],
            "collapsed": true
          },
          {
            "text": "张量计算",
            "link": "/zh/api/compute/tensor",
            "items": [
              {
                "text": "ceil_div",
                "link": "/zh/api/compute/tensor/ceil_div"
              },
              {
                "text": "idx2crd",
                "link": "/zh/api/compute/tensor/idx2crd"
              },
              {
                "text": "tile_slice",
                "link": "/zh/api/compute/tensor/tile-slice"
              }
            ],
            "collapsed": true
          },
          {
            "text": "Reg 矢量计算",
            "link": "/zh/api/compute/vector",
            "items": [
              {
                "text": "Reg数据搬入",
                "link": "/zh/api/compute/vector/load",
                "items": [
                  {
                    "text": "Reg数据搬入",
                    "link": "/zh/api/compute/vector/load/overview"
                  },
                  {
                    "text": "vload_broadcast",
                    "link": "/zh/api/compute/vector/load/vload-broadcast"
                  },
                  {
                    "text": "vload_deinterleave",
                    "link": "/zh/api/compute/vector/load/vload-deinterleave"
                  },
                  {
                    "text": "vload_downsample",
                    "link": "/zh/api/compute/vector/load/vload-downsample"
                  },
                  {
                    "text": "vload_strided",
                    "link": "/zh/api/compute/vector/load/vload-strided"
                  },
                  {
                    "text": "vload_unalign_init",
                    "link": "/zh/api/compute/vector/load/vload-unalign-init"
                  },
                  {
                    "text": "vload_unalign",
                    "link": "/zh/api/compute/vector/load/vload-unalign"
                  },
                  {
                    "text": "vload_unpack",
                    "link": "/zh/api/compute/vector/load/vload-unpack"
                  },
                  {
                    "text": "vload_upsample",
                    "link": "/zh/api/compute/vector/load/vload-upsample"
                  },
                  {
                    "text": "vload",
                    "link": "/zh/api/compute/vector/load/vload"
                  },
                  {
                    "text": "vmask_load",
                    "link": "/zh/api/compute/vector/load/vmask-load"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 算术计算",
                "link": "/zh/api/compute/vector/reg_arith",
                "items": [
                  {
                    "text": "vabs",
                    "link": "/zh/api/compute/vector/reg_arith/vabs"
                  },
                  {
                    "text": "vadd",
                    "link": "/zh/api/compute/vector/reg_arith/vadd"
                  },
                  {
                    "text": "vaddc",
                    "link": "/zh/api/compute/vector/reg_arith/vaddc"
                  },
                  {
                    "text": "vaddco",
                    "link": "/zh/api/compute/vector/reg_arith/vaddco"
                  },
                  {
                    "text": "vadds",
                    "link": "/zh/api/compute/vector/reg_arith/vadds"
                  },
                  {
                    "text": "vdiv",
                    "link": "/zh/api/compute/vector/reg_arith/vdiv"
                  },
                  {
                    "text": "vexp",
                    "link": "/zh/api/compute/vector/reg_arith/vexp"
                  },
                  {
                    "text": "vlog",
                    "link": "/zh/api/compute/vector/reg_arith/vlog"
                  },
                  {
                    "text": "vmax",
                    "link": "/zh/api/compute/vector/reg_arith/vmax"
                  },
                  {
                    "text": "vmaxs",
                    "link": "/zh/api/compute/vector/reg_arith/vmaxs"
                  },
                  {
                    "text": "vmin",
                    "link": "/zh/api/compute/vector/reg_arith/vmin"
                  },
                  {
                    "text": "vmins",
                    "link": "/zh/api/compute/vector/reg_arith/vmins"
                  },
                  {
                    "text": "vmul",
                    "link": "/zh/api/compute/vector/reg_arith/vmul"
                  },
                  {
                    "text": "vmull",
                    "link": "/zh/api/compute/vector/reg_arith/vmull"
                  },
                  {
                    "text": "vmuls",
                    "link": "/zh/api/compute/vector/reg_arith/vmuls"
                  },
                  {
                    "text": "vneg",
                    "link": "/zh/api/compute/vector/reg_arith/vneg"
                  },
                  {
                    "text": "vsqrt",
                    "link": "/zh/api/compute/vector/reg_arith/vsqrt"
                  },
                  {
                    "text": "vsub",
                    "link": "/zh/api/compute/vector/reg_arith/vsub"
                  },
                  {
                    "text": "vsubbo",
                    "link": "/zh/api/compute/vector/reg_arith/vsubbo"
                  },
                  {
                    "text": "vsubc",
                    "link": "/zh/api/compute/vector/reg_arith/vsubc"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 广播",
                "link": "/zh/api/compute/vector/reg_broadcast",
                "items": [
                  {
                    "text": "vdup",
                    "link": "/zh/api/compute/vector/reg_broadcast/vdup"
                  },
                  {
                    "text": "vdups",
                    "link": "/zh/api/compute/vector/reg_broadcast/vdups"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 比较",
                "link": "/zh/api/compute/vector/reg_compare",
                "items": [
                  {
                    "text": "veq",
                    "link": "/zh/api/compute/vector/reg_compare/veq"
                  },
                  {
                    "text": "veqs",
                    "link": "/zh/api/compute/vector/reg_compare/veqs"
                  },
                  {
                    "text": "vge",
                    "link": "/zh/api/compute/vector/reg_compare/vge"
                  },
                  {
                    "text": "vges",
                    "link": "/zh/api/compute/vector/reg_compare/vges"
                  },
                  {
                    "text": "vgt",
                    "link": "/zh/api/compute/vector/reg_compare/vgt"
                  },
                  {
                    "text": "vgts",
                    "link": "/zh/api/compute/vector/reg_compare/vgts"
                  },
                  {
                    "text": "vle",
                    "link": "/zh/api/compute/vector/reg_compare/vle"
                  },
                  {
                    "text": "vles",
                    "link": "/zh/api/compute/vector/reg_compare/vles"
                  },
                  {
                    "text": "vlt",
                    "link": "/zh/api/compute/vector/reg_compare/vlt"
                  },
                  {
                    "text": "vlts",
                    "link": "/zh/api/compute/vector/reg_compare/vlts"
                  },
                  {
                    "text": "vne",
                    "link": "/zh/api/compute/vector/reg_compare/vne"
                  },
                  {
                    "text": "vnes",
                    "link": "/zh/api/compute/vector/reg_compare/vnes"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 类型转换",
                "link": "/zh/api/compute/vector/reg_convert",
                "items": [
                  {
                    "text": "RoundingMode",
                    "link": "/zh/api/compute/vector/reg_convert/roundingmode"
                  },
                  {
                    "text": "vcast",
                    "link": "/zh/api/compute/vector/reg_convert/vcast"
                  },
                  {
                    "text": "vceil",
                    "link": "/zh/api/compute/vector/reg_convert/vceil"
                  },
                  {
                    "text": "vfloor",
                    "link": "/zh/api/compute/vector/reg_convert/vfloor"
                  },
                  {
                    "text": "vreinterpret_lanes",
                    "link": "/zh/api/compute/vector/reg_convert/vreinterpret-lanes"
                  },
                  {
                    "text": "vreinterpret",
                    "link": "/zh/api/compute/vector/reg_convert/vreinterpret"
                  },
                  {
                    "text": "vrint",
                    "link": "/zh/api/compute/vector/reg_convert/vrint"
                  },
                  {
                    "text": "vround",
                    "link": "/zh/api/compute/vector/reg_convert/vround"
                  },
                  {
                    "text": "vtrunc",
                    "link": "/zh/api/compute/vector/reg_convert/vtrunc"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 融合计算",
                "link": "/zh/api/compute/vector/reg_fused",
                "items": [
                  {
                    "text": "vabs_sub",
                    "link": "/zh/api/compute/vector/reg_fused/vabs-sub"
                  },
                  {
                    "text": "vaxpy",
                    "link": "/zh/api/compute/vector/reg_fused/vaxpy"
                  },
                  {
                    "text": "vcast_exp_sub",
                    "link": "/zh/api/compute/vector/reg_fused/vcast-exp-sub"
                  },
                  {
                    "text": "vexp_sub",
                    "link": "/zh/api/compute/vector/reg_fused/vexp-sub"
                  },
                  {
                    "text": "vleakyrelu",
                    "link": "/zh/api/compute/vector/reg_fused/vleakyrelu"
                  },
                  {
                    "text": "vmadd",
                    "link": "/zh/api/compute/vector/reg_fused/vmadd"
                  },
                  {
                    "text": "vmula",
                    "link": "/zh/api/compute/vector/reg_fused/vmula"
                  },
                  {
                    "text": "vprelu",
                    "link": "/zh/api/compute/vector/reg_fused/vprelu"
                  },
                  {
                    "text": "vrelu",
                    "link": "/zh/api/compute/vector/reg_fused/vrelu"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "vgather_reg",
                "link": "/zh/api/compute/vector/reg_gather/vgather-reg"
              },
              {
                "text": "Reg 直方图",
                "link": "/zh/api/compute/vector/reg_histogram",
                "items": [
                  {
                    "text": "vhistogram_accumulate",
                    "link": "/zh/api/compute/vector/reg_histogram/vhistogram-accumulate"
                  },
                  {
                    "text": "vhistogram_frequency",
                    "link": "/zh/api/compute/vector/reg_histogram/vhistogram-frequency"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "varange",
                "link": "/zh/api/compute/vector/reg_index/varange"
              },
              {
                "text": "Reg 逻辑运算",
                "link": "/zh/api/compute/vector/reg_logic",
                "items": [
                  {
                    "text": "mask_and",
                    "link": "/zh/api/compute/vector/reg_logic/mask-and"
                  },
                  {
                    "text": "mask_not",
                    "link": "/zh/api/compute/vector/reg_logic/mask-not"
                  },
                  {
                    "text": "mask_or",
                    "link": "/zh/api/compute/vector/reg_logic/mask-or"
                  },
                  {
                    "text": "mask_xor",
                    "link": "/zh/api/compute/vector/reg_logic/mask-xor"
                  },
                  {
                    "text": "vbitwise_and",
                    "link": "/zh/api/compute/vector/reg_logic/vbitwise-and"
                  },
                  {
                    "text": "vbitwise_or",
                    "link": "/zh/api/compute/vector/reg_logic/vbitwise-or"
                  },
                  {
                    "text": "vbitwise_xor",
                    "link": "/zh/api/compute/vector/reg_logic/vbitwise-xor"
                  },
                  {
                    "text": "vmask",
                    "link": "/zh/api/compute/vector/reg_logic/vmask"
                  },
                  {
                    "text": "vnot",
                    "link": "/zh/api/compute/vector/reg_logic/vnot"
                  },
                  {
                    "text": "vshl",
                    "link": "/zh/api/compute/vector/reg_logic/vshl"
                  },
                  {
                    "text": "vshr",
                    "link": "/zh/api/compute/vector/reg_logic/vshr"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 掩码操作",
                "link": "/zh/api/compute/vector/reg_mask",
                "items": [
                  {
                    "text": "create_mask",
                    "link": "/zh/api/compute/vector/reg_mask/create-mask"
                  },
                  {
                    "text": "full_mask",
                    "link": "/zh/api/compute/vector/reg_mask/full-mask"
                  },
                  {
                    "text": "mask_counter",
                    "link": "/zh/api/compute/vector/reg_mask/mask-counter"
                  },
                  {
                    "text": "update_mask",
                    "link": "/zh/api/compute/vector/reg_mask/update-mask"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 重排与选择",
                "link": "/zh/api/compute/vector/reg_permute_sel",
                "items": [
                  {
                    "text": "vcompress",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vcompress"
                  },
                  {
                    "text": "vdeinterleave",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vdeinterleave"
                  },
                  {
                    "text": "vinterleave",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vinterleave"
                  },
                  {
                    "text": "vmerge",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vmerge"
                  },
                  {
                    "text": "vpack",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vpack"
                  },
                  {
                    "text": "vselect",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vselect"
                  },
                  {
                    "text": "vsqueeze_and_storeunalign_finalize",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vsqueeze-and-storeunalign-finalize"
                  },
                  {
                    "text": "vsqueeze_and_storeunalign_init",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vsqueeze-and-storeunalign-init"
                  },
                  {
                    "text": "vsqueeze_and_storeunalign",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vsqueeze-and-storeunalign"
                  },
                  {
                    "text": "vsqueeze",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vsqueeze"
                  },
                  {
                    "text": "vstore_unalign_begin",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vstore-unalign-begin"
                  },
                  {
                    "text": "vstore_unalign",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vstore-unalign"
                  },
                  {
                    "text": "vunpack",
                    "link": "/zh/api/compute/vector/reg_permute_sel/vunpack"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Reg 归约计算",
                "link": "/zh/api/compute/vector/reg_reduce",
                "items": [
                  {
                    "text": "vpair_reduce_sum",
                    "link": "/zh/api/compute/vector/reg_reduce/vpair-reduce-sum"
                  },
                  {
                    "text": "vreduce_max_datablock",
                    "link": "/zh/api/compute/vector/reg_reduce/vreduce-max-datablock"
                  },
                  {
                    "text": "vreduce_max",
                    "link": "/zh/api/compute/vector/reg_reduce/vreduce-max"
                  },
                  {
                    "text": "vreduce_min_datablock",
                    "link": "/zh/api/compute/vector/reg_reduce/vreduce-min-datablock"
                  },
                  {
                    "text": "vreduce_min",
                    "link": "/zh/api/compute/vector/reg_reduce/vreduce-min"
                  },
                  {
                    "text": "vreduce_sum_datablock",
                    "link": "/zh/api/compute/vector/reg_reduce/vreduce-sum-datablock"
                  },
                  {
                    "text": "vreduce_sum",
                    "link": "/zh/api/compute/vector/reg_reduce/vreduce-sum"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "vmem_bar",
                "link": "/zh/api/compute/vector/reg_sync/vmem-bar"
              },
              {
                "text": "vscatter",
                "link": "/zh/api/compute/vector/scatter/vscatter"
              },
              {
                "text": "Reg数据搬出",
                "link": "/zh/api/compute/vector/store",
                "items": [
                  {
                    "text": "Reg数据搬出",
                    "link": "/zh/api/compute/vector/store/overview"
                  },
                  {
                    "text": "vmask_store",
                    "link": "/zh/api/compute/vector/store/vmask-store"
                  },
                  {
                    "text": "vstore_first",
                    "link": "/zh/api/compute/vector/store/vstore-first"
                  },
                  {
                    "text": "vstore_interleave",
                    "link": "/zh/api/compute/vector/store/vstore-interleave"
                  },
                  {
                    "text": "vstore_mask",
                    "link": "/zh/api/compute/vector/store/vstore-mask"
                  },
                  {
                    "text": "vstore_pack",
                    "link": "/zh/api/compute/vector/store/vstore-pack"
                  },
                  {
                    "text": "vstore_strided",
                    "link": "/zh/api/compute/vector/store/vstore-strided"
                  },
                  {
                    "text": "vstore_unalign_post",
                    "link": "/zh/api/compute/vector/store/vstore-unalign-post"
                  },
                  {
                    "text": "vstore",
                    "link": "/zh/api/compute/vector/store/vstore"
                  },
                  {
                    "text": "vstorealign_squeeze_status",
                    "link": "/zh/api/compute/vector/store/vstorealign-squeeze-status"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "UB Gather",
                "link": "/zh/api/compute/vector/ub_gather",
                "items": [
                  {
                    "text": "vgather_datablock",
                    "link": "/zh/api/compute/vector/ub_gather/vgather-datablock"
                  },
                  {
                    "text": "vgather",
                    "link": "/zh/api/compute/vector/ub_gather/vgather"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "关键特性说明",
                "link": "/zh/api/compute/vector/key-features"
              },
              {
                "text": "Reg 矢量计算",
                "link": "/zh/api/compute/vector/overview"
              }
            ],
            "collapsed": true
          }
        ],
        "collapsed": true
      },
      {
        "text": "设备执行",
        "link": "/zh/api/execution",
        "items": [
          {
            "text": "调试",
            "link": "/zh/api/debug",
            "items": [
              {
                "text": "print",
                "link": "/zh/api/debug/print"
              }
            ],
            "collapsed": true
          },
          {
            "text": "系统变量",
            "link": "/zh/api/execution/device",
            "items": [
              {
                "text": "特殊寄存器",
                "items": [
                  {
                    "text": "get_ctrl",
                    "link": "/zh/api/execution/device/spr/get-ctrl"
                  },
                  {
                    "text": "set_ctrl",
                    "link": "/zh/api/execution/device/spr/set-ctrl"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "get_block_idx",
                "link": "/zh/api/execution/device/get-block-idx"
              },
              {
                "text": "get_block_num",
                "link": "/zh/api/execution/device/get-block-num"
              },
              {
                "text": "get_core_id",
                "link": "/zh/api/execution/device/get-core-id"
              },
              {
                "text": "get_squeeze_status",
                "link": "/zh/api/execution/device/get-squeeze-status"
              },
              {
                "text": "get_status",
                "link": "/zh/api/execution/device/get-status"
              },
              {
                "text": "get_subblock_dim",
                "link": "/zh/api/execution/device/get-subblock-dim"
              },
              {
                "text": "get_subblock_id",
                "link": "/zh/api/execution/device/get-subblock-id"
              },
              {
                "text": "get_system_cycle",
                "link": "/zh/api/execution/device/get-system-cycle"
              },
              {
                "text": "get_vf_len",
                "link": "/zh/api/execution/device/get-vf-len"
              },
              {
                "text": "is_aic",
                "link": "/zh/api/execution/device/is-aic"
              },
              {
                "text": "is_aiv",
                "link": "/zh/api/execution/device/is-aiv"
              }
            ],
            "collapsed": true
          },
          {
            "text": "同步控制",
            "link": "/zh/api/execution/synchronization",
            "items": [
              {
                "text": "缓存控制",
                "link": "/zh/api/execution/synchronization/cache",
                "items": [
                  {
                    "text": "dcci_entire_atomic",
                    "link": "/zh/api/execution/synchronization/cache/dcci-entire-atomic"
                  },
                  {
                    "text": "dcci_entire_out",
                    "link": "/zh/api/execution/synchronization/cache/dcci-entire-out"
                  },
                  {
                    "text": "dcci_single",
                    "link": "/zh/api/execution/synchronization/cache/dcci-single"
                  },
                  {
                    "text": "dci",
                    "link": "/zh/api/execution/synchronization/cache/dci"
                  },
                  {
                    "text": "缓存控制概览",
                    "link": "/zh/api/execution/synchronization/cache/overview"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "核间同步",
                "link": "/zh/api/execution/synchronization/resource-management",
                "items": [
                  {
                    "text": "cube_sync_block_arrive",
                    "link": "/zh/api/execution/synchronization/resource-management/cube-sync-block-arrive"
                  },
                  {
                    "text": "cube_sync_block_wait",
                    "link": "/zh/api/execution/synchronization/resource-management/cube-sync-block-wait"
                  },
                  {
                    "text": "cube_sync_intra_arrive",
                    "link": "/zh/api/execution/synchronization/resource-management/cube-sync-intra-arrive"
                  },
                  {
                    "text": "cube_sync_intra_wait",
                    "link": "/zh/api/execution/synchronization/resource-management/cube-sync-intra-wait"
                  },
                  {
                    "text": "global_sync_all",
                    "link": "/zh/api/execution/synchronization/resource-management/global-sync-all"
                  },
                  {
                    "text": "核间同步能力概述",
                    "link": "/zh/api/execution/synchronization/resource-management/inter-core-sync-overview"
                  },
                  {
                    "text": "关键特性说明",
                    "link": "/zh/api/execution/synchronization/resource-management/key-features"
                  },
                  {
                    "text": "系统同步能力概述",
                    "link": "/zh/api/execution/synchronization/resource-management/system-sync-overview"
                  },
                  {
                    "text": "vec_sync_block_arrive",
                    "link": "/zh/api/execution/synchronization/resource-management/vec-sync-block-arrive"
                  },
                  {
                    "text": "vec_sync_block_wait",
                    "link": "/zh/api/execution/synchronization/resource-management/vec-sync-block-wait"
                  },
                  {
                    "text": "vec_sync_intra_arrive",
                    "link": "/zh/api/execution/synchronization/resource-management/vec-sync-intra-arrive"
                  },
                  {
                    "text": "vec_sync_intra_wait",
                    "link": "/zh/api/execution/synchronization/resource-management/vec-sync-intra-wait"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "onchip_rewind",
                "link": "/zh/api/execution/synchronization/onchip_rewind"
              }
            ],
            "collapsed": true
          },
          {
            "text": "PIPE",
            "link": "/zh/api/execution/pipeline"
          }
        ],
        "collapsed": true
      },
      {
        "text": "编译与发布",
        "link": "/zh/api/compilation",
        "items": [
          {
            "text": "AOT 编译与 Native 发布",
            "link": "/zh/api/compilation/aot",
            "items": [
              {
                "text": "AOT 编译缓存",
                "link": "/zh/api/compilation/aot/cache"
              },
              {
                "text": "aot.compile",
                "link": "/zh/api/compilation/aot/compile"
              },
              {
                "text": "Native 导出与构建",
                "link": "/zh/api/compilation/aot/publication"
              },
              {
                "text": "aot.register_directory",
                "link": "/zh/api/compilation/aot/runtime-registration"
              },
              {
                "text": "AOT 参数规格",
                "link": "/zh/api/compilation/aot/specs"
              },
              {
                "text": "Constexpr",
                "link": "/zh/api/compilation/aot/constexpr"
              }
            ],
            "collapsed": true
          },
          {
            "text": "Host 编译支持",
            "link": "/zh/api/compilation/host",
            "items": [
              {
                "text": "Host 平台信息",
                "link": "/zh/api/compilation/host/platform",
                "items": [
                  {
                    "text": "get_mem_size",
                    "link": "/zh/api/compilation/host/platform/get-mem-size"
                  },
                  {
                    "text": "get_platform_info",
                    "link": "/zh/api/compilation/host/platform/get-platform-info"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "Host 参数规格",
                "link": "/zh/api/compilation/host/specs",
                "items": [
                  {
                    "text": "Dim",
                    "link": "/zh/api/compilation/host/specs/dim"
                  },
                  {
                    "text": "StructSchema",
                    "link": "/zh/api/compilation/host/specs/struct_schema"
                  },
                  {
                    "text": "StructSpec",
                    "link": "/zh/api/compilation/host/specs/struct_spec"
                  },
                  {
                    "text": "TensorListSpec",
                    "link": "/zh/api/compilation/host/specs/tensor_list_spec"
                  },
                  {
                    "text": "TensorSpec",
                    "link": "/zh/api/compilation/host/specs/tensor_spec"
                  }
                ],
                "collapsed": true
              }
            ],
            "collapsed": true
          },
          {
            "text": "OpKit / ACLNN 发布",
            "link": "/zh/api/compilation/opkit",
            "items": [
              {
                "text": "ACLNN 构建",
                "link": "/zh/api/compilation/opkit/aclnn-build"
              },
              {
                "text": "Native 与 ACLNN 的选择",
                "link": "/zh/api/compilation/opkit/native-or-aclnn"
              }
            ],
            "collapsed": true
          }
        ],
        "collapsed": true
      },
      {
        "text": "AI CPU",
        "link": "/zh/api/aicpu",
        "items": [
          {
            "text": "compile_aicpu_kernel",
            "link": "/zh/api/aicpu/build/compile"
          },
          {
            "text": "AI CPU 编程",
            "link": "/zh/api/aicpu/programming",
            "items": [
              {
                "text": "AI CPU 参数类型",
                "link": "/zh/api/aicpu/programming/types",
                "items": [
                  {
                    "text": "GM输入字段",
                    "link": "/zh/api/aicpu/programming/types/gm-in"
                  },
                  {
                    "text": "GM输出字段",
                    "link": "/zh/api/aicpu/programming/types/gm-out"
                  },
                  {
                    "text": "参数声明",
                    "link": "/zh/api/aicpu/programming/types/parameters"
                  },
                  {
                    "text": "标量类型",
                    "link": "/zh/api/aicpu/programming/types/scalars"
                  }
                ],
                "collapsed": true
              },
              {
                "text": "@aicpu_kernel",
                "link": "/zh/api/aicpu/programming/aicpu-kernel"
              }
            ],
            "collapsed": true
          },
          {
            "text": "AI CPU 运行时",
            "link": "/zh/api/aicpu/runtime",
            "items": [
              {
                "text": "current_raw_stream",
                "link": "/zh/api/aicpu/runtime/current-raw-stream"
              },
              {
                "text": "CompiledAicpuKernel.launch",
                "link": "/zh/api/aicpu/runtime/launch"
              }
            ],
            "collapsed": true
          },
          {
            "text": "Kernel.run_host",
            "link": "/zh/api/aicpu/validation/run-host"
          }
        ],
        "collapsed": true
      },
      {
        "text": "通用接口文档模板",
        "link": "/zh/api/api-template"
      },
      {
        "text": "CANNBot-DSL API 参考",
        "link": "/zh/api/"
      }
    ]
  }
];

// Sidebar directories are navigation groups. Their entries are deliberately
// not clickable; only leaf pages navigate. Remove the duplicate flat page
// when a directory already exposes the same route as a group.
const cleanSidebar = (items: any[]): any[] => items.map((item) => {
  if (!Array.isArray(item.items) || item.items.length === 0) return item

  const rawChildren = item.items
  const directoryLinks = new Set(
    rawChildren
      .filter((child: any) => Array.isArray(child.items) && child.items.length > 0)
      .map((child: any) => child.link?.replace(/\/$/, ''))
      .filter(Boolean),
  )
  const parentLink = item.link?.replace(/\/$/, '')
  const children = rawChildren.flatMap((rawChild: any) => {
    const wasDirectory = Array.isArray(rawChild.items) && rawChild.items.length > 0
    const child = cleanSidebar([rawChild])[0]
    const childLink = child.link?.replace(/\/$/, '')
    // Remove only duplicate leaf entries from the original sidebar. A directory
    // with one page is intentionally collapsed to that page and must be kept.
    if (!wasDirectory && (childLink === parentLink || directoryLinks.has(childLink))) return []
    return [child]
  })

  if (children.length === 0 && item.link) return { text: item.text, link: item.link }

  return { ...item, items: children }
})

const cleanedApiSidebar = cleanSidebar(apiSidebar)

const normalizeApiSidebar = (sidebar: any[]) => {
  const root = sidebar[0]
  const items = root.items ?? []
  const find = (list: any[], suffix: string) => list.find((item: any) => item.link?.endsWith(suffix))
  const remove = (list: any[], item: any) => list.filter((candidate: any) => candidate !== item)
  const withCleanLabels = (item: any): any => {
    const cleaned = typeof item.text === 'string'
      ? item.text.replace(/ 接口$/, '').replace(/^操作$/, '计算').replace(/^张量操作$/, '张量计算').replace(/^原子操作$/, '原子计算')
      : item.text
    const overviewLabel = typeof item.link === 'string' && /\/overview\/?$/.test(item.link) && typeof cleaned === 'string' && !/简介|概览/.test(cleaned)
      ? `${cleaned}（概览）`
      : cleaned
    const conceptLabel = item.link?.endsWith('/control-flow/control-flow') && overviewLabel === '控制流'
      ? '控制流（概览）'
      : overviewLabel
    return {
      ...item,
      text: conceptLabel,
      ...(Array.isArray(item.items) ? { items: item.items.map(withCleanLabels) } : {}),
    }
  }
  const introRank = (text: any) => {
    if (typeof text !== 'string') return 1
    if (/简介|概览/.test(text)) return 0
    if (/背景与核心概念|概述/.test(text)) return 1
    if (/关键特性/.test(text)) return 2
    return 3
  }
  const orderForReading = (item: any): any => {
    if (!Array.isArray(item.items)) return item
    const children = item.items.map(orderForReading)
    // Matrix pages follow ASC's concepts, computation, features, then configuration order.
    if (item.link === '/zh/api/compute/matrix') return { ...item, items: children }
    return {
      ...item,
      items: children
        .map((child: any, index: number) => ({ child, index }))
        .sort((a: any, b: any) => introRank(a.child.text) - introRank(b.child.text) || a.index - b.index)
        .map((entry: any) => entry.child),
    }
  }
  const removeParentLinks = (item: any): any => {
    if (!Array.isArray(item.items) || item.items.length === 0) return item
    const { link: _link, ...group } = item
    return { ...group, items: item.items.map(removeParentLinks) }
  }

  const dataStructures = find(items, '/data-structures')
  if (dataStructures) {
    const dataItems = dataStructures.items ?? []
    const dataTypes = find(dataItems, '/data-types')
    const tensor = find(dataItems, '/tensor')
    const layout = find(dataItems, '/layout')
    const tiler = find(dataItems, '/tiler')
    const storage = find(dataItems, '/storage')
    const array = find(dataItems, '/array')
    if (layout) {
      const layoutItems = layout.items ?? []
      const coord = find(layoutItems, '/layout/coord')
      const shape = find(layoutItems, '/layout/shape')
      const stride = find(layoutItems, '/layout/stride')
      layout.items = [
        find(layoutItems, '/layout/overview'),
        coord,
        shape,
        stride,
        find(layoutItems, '/layout/make_layout'),
        find(layoutItems, '/layout/offset'),
      ].filter(Boolean)
    }
    dataStructures.items = [dataTypes, tensor, layout, tiler, storage, array].filter(Boolean)
  }

  const language = find(items, '/language')
  if (language) {
    const languageItems = language.items ?? []
    const controlFlow = find(languageItems, '/language/control-flow')
    if (controlFlow) {
      const controlItems = controlFlow.items ?? []
      controlFlow.items = [
        find(controlItems, '/language/control-flow/compile-time-control'),
        find(controlItems, '/language/control-flow/predicate-expressions'),
        find(controlItems, '/language/control-flow/value-selection'),
        find(controlItems, '/language/control-flow/control-flow'),
        find(controlItems, '/language/control-flow/for'),
        find(controlItems, '/language/control-flow/if'),
        find(controlItems, '/language/control-flow/while'),
      ].filter(Boolean)
    }
    language.items = [
      find(languageItems, '/language/decorators'),
      controlFlow,
      find(languageItems, '/language/launch'),
    ].filter(Boolean)
  }

  const compute = find(items, '/compute')
  let dataMovement: any
  if (compute) {
    dataMovement = find(compute.items ?? [], '/data-movement')
    if (dataMovement) compute.items = remove(compute.items, dataMovement)
    const computeItems = compute.items ?? []
    compute.items = [
      find(computeItems, '/compute/tensor'),
      find(computeItems, '/compute/scalar'),
      find(computeItems, '/compute/vector'),
      find(computeItems, '/compute/matrix'),
      find(computeItems, '/compute/atomic'),
    ].filter(Boolean)
  }

  const execution = find(items, '/execution')
  let debug: any
  if (execution) {
    debug = find(execution.items ?? [], '/debug')
    if (debug) execution.items = remove(execution.items, debug)
    execution.items = [
      find(execution.items ?? [], '/execution/device'),
      find(execution.items ?? [], '/execution/pipeline'),
      find(execution.items ?? [], '/execution/synchronization'),
    ].filter(Boolean)
  }

  const compilation = find(items, '/compilation')
  if (compilation) {
    const compilationItems = compilation.items ?? []
    const host = find(compilationItems, '/compilation/host')
    const aot = find(compilationItems, '/compilation/aot')
    const opkit = find(compilationItems, '/compilation/opkit')
    if (host) {
      const hostItems = host.items ?? []
      host.items = [
        find(hostItems, '/compilation/host/specs'),
        find(hostItems, '/compilation/host/platform'),
      ].filter(Boolean)
    }
    if (aot) {
      const aotItems = aot.items ?? []
      aot.items = [
        find(aotItems, '/compilation/aot/specs'),
        find(aotItems, '/compilation/aot/constexpr'),
        find(aotItems, '/compilation/aot/compile'),
        find(aotItems, '/compilation/aot/cache'),
        find(aotItems, '/compilation/aot/publication'),
        find(aotItems, '/compilation/aot/runtime-registration'),
      ].filter(Boolean)
    }
    compilation.items = [host, aot, opkit].filter(Boolean)
  }

  const aicpu = find(items, '/aicpu')
  if (aicpu) {
    const aicpuItems = aicpu.items ?? []
    aicpu.items = [
      find(aicpuItems, '/aicpu/programming'),
      find(aicpuItems, '/aicpu/build'),
      find(aicpuItems, '/aicpu/validation'),
      find(aicpuItems, '/aicpu/runtime'),
    ].filter(Boolean)
  }

  const ordered = [
    find(items, '/language'),
    dataStructures,
    dataMovement,
    compute,
    execution,
    debug,
    find(items, '/compilation'),
    find(items, '/aicpu'),
  ].filter(Boolean).map(withCleanLabels).map(orderForReading)
  return [{ ...removeParentLinks(root), items: ordered.map(removeParentLinks) }]
}

const normalizedApiSidebar = normalizeApiSidebar(cleanedApiSidebar)

export default defineConfig({
  srcDir: '..',
  // Repository and handbook README files document the tree; they are not site pages.
  srcExclude: ['README.md', 'zh/api/README.md', 'zh/guide/README.md'],
  lang: 'zh-CN',
  title: 'CANNBot-DSL',
  description: '面向 Ascend NPU 的算子开发项目文档',
  base: process.env.DOCS_BASE || '/',
  rewrites: rewriteReadmes,
  cleanUrls: false,
  // Repository samples and contribution guides live outside the VitePress source tree.
  ignoreDeadLinks: [/samples/, /test/, /CONTRIBUTION_DOC_en/, /CHANGELOG/, /LICENSE/],
  lastUpdated: true,
  markdown: {
    math: true
  },
  vite: {
    resolve: {
      alias: [
        { find: 'vue/server-renderer', replacement: path.join(projectRoot, 'node_modules/@vue/server-renderer') },
        { find: 'vue', replacement: path.join(projectRoot, 'node_modules/vue') }
      ]
    }
  },
  head: [
    ['meta', { name: 'theme-color', content: '#2f5bea' }],
    ['link', { rel: 'icon', href: '/logo.svg', type: 'image/svg+xml' }]
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'CANNBot-DSL',
    nav: [
      { text: '首页', link: '/zh/' },
      { text: '算子开发指南', items: [
        { text: '源码构建', link: '/zh/install/source-build' },
        { text: '编程模型', link: '/zh/guide/overview' },
        { text: '样例导航', link: 'https://gitcode.com/cann/cannbot-dsl/blob/master/samples/README.md' }
      ] },
      { text: 'CANNBot-DSL API', items: [
        { text: '语言', link: '/zh/api/language/' },
        { text: '数据结构', link: '/zh/api/data-structures/' },
        { text: '数据搬运', link: '/zh/api/data-movement/' },
        { text: '计算', link: '/zh/api/compute/' },
        { text: '设备执行', link: '/zh/api/execution/' },
        { text: '调试', link: '/zh/api/debug/' },
        { text: '编译与发布', link: '/zh/api/compilation/' },
        { text: 'AI CPU', link: '/zh/api/aicpu/' }
      ] },
      { text: '语言 / Language', items: [
        { text: '简体中文', link: '/zh/' },
        { text: 'English', link: '/en/' }
      ] }
    ],
    sidebar: {
      '/zh/install/': [{ text: '源码构建', collapsed: true, items: guideSidebar }],
      '/zh/guide/': [...guideSidebar, ...programmingModelSidebar],
      '/zh/api/': normalizedApiSidebar,
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
