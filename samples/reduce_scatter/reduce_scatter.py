# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""HIF8 ReduceScatter: layout policy, AIV kernel and torch-group launcher.

The low-level CANNIR extractors in address() bridge the missing public
Tensor-to-integer-address API in DSL 0.7.1 for runtime-length Hcomm copies.
"""

from dataclasses import asdict, dataclass
import threading
import json

import opkit
import cannbotdsl as cb
from cannbotdsl import dtypes as d
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import make_pointer
from cannbotdsl.ops.sync import (
    PIPE,
    _CoreRole,
    hcomm_init,
    hcomm_drain,
    write_nbi,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ops.reg import (
    vload,
    vstore,
    vadd,
    vmax,
    vmin,
    update_mask,
    create_mask,
    vcast,
    vload_unpack,
    vstore_pack,
    vreinterpret_lanes,
)
from cannbotdsl.ops.reg.reg_defs import UnpackMode, PackMode


from .comm_utils import MAX_RANKS, _MeshCommunication

SMALL_BYTES = 512 * 1024
HIF8_COPY_BYTES = 190 * 1024
HIF8_REDUCE_ELEMENTS = 17664
DTYPE_BYTES = {"hifloat8": 1}
REDUCTIONS = ("sum", "max", "min")


def align_up(n, alignment=512):
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Layout:
    """Byte pitches, as in TemplateDataParams; counts are elements per rank."""

    count: int
    ranks: int
    itemsize: int
    repeats: int = 1
    input_slice_stride: int | None = None
    output_slice_stride: int | None = None
    input_repeat_stride: int | None = None
    output_repeat_stride: int | None = None

    def __post_init__(self):
        for name in ("count", "ranks", "itemsize", "repeats"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "count" else 1):
                raise ValueError(f"invalid {name}: {value}")
        if self.ranks > MAX_RANKS:
            raise ValueError(f"ranks must be <= {MAX_RANKS}")
        defaults = {
            "input_slice_stride": self.count * self.itemsize,
            "output_slice_stride": self.count * self.itemsize,
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        defaults = {
            "input_repeat_stride": self.input_slice_stride * self.ranks,
            "output_repeat_stride": self.output_slice_stride,
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        for name in (
            "input_slice_stride",
            "output_slice_stride",
            "input_repeat_stride",
            "output_repeat_stride",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value % self.itemsize:
                raise ValueError(
                    f"{name} must be a nonnegative whole-element byte stride"
                )
        if self.input_slice_stride < self.count * self.itemsize:
            raise ValueError("input rank slices overlap")
        if self.output_slice_stride < self.count * self.itemsize:
            raise ValueError("output slice stride is smaller than count")
        if self.repeats > 1:
            if (
                self.input_repeat_stride
                < (self.ranks - 1) * self.input_slice_stride
                + self.count * self.itemsize
            ):
                raise ValueError("input repeats overlap")
            if self.output_repeat_stride < self.count * self.itemsize:
                raise ValueError("output repeats overlap")
        if max(self.input_elements, self.output_elements) * self.itemsize >= 2**63:
            raise ValueError("address span exceeds signed int64")

    @property
    def input_elements(self):
        if self.count == 0:
            return 0
        return (
            (self.repeats - 1) * self.input_repeat_stride
            + (self.ranks - 1) * self.input_slice_stride
        ) // self.itemsize + self.count

    @property
    def output_elements(self):
        if self.count == 0:
            return 0
        return (
            self.repeats - 1
        ) * self.output_repeat_stride // self.itemsize + self.count


@dataclass(frozen=True)
class Plan:
    layout: Layout
    dtype: str = "hifloat8"
    reduction: str = "sum"
    num_blocks_limit: int = 32
    buffer_bytes: int | None = None
    tile_elements: int | None = None

    def __post_init__(self):
        if self.buffer_bytes is None:
            object.__setattr__(self, "buffer_bytes", 128 * 1024 * 1024)
        if self.tile_elements is None:
            object.__setattr__(self, "tile_elements", HIF8_REDUCE_ELEMENTS)
        if (
            self.dtype not in DTYPE_BYTES
            or DTYPE_BYTES[self.dtype] != self.layout.itemsize
        ):
            raise ValueError("unsupported dtype or mismatched element size")
        if self.reduction not in REDUCTIONS:
            raise ValueError(
                "AIV supports sum/max/min; PROD is not supported by the source selector"
            )
        if (
            type(self.num_blocks_limit) is not int
            or not 1 <= self.num_blocks_limit <= 1024
        ):
            raise ValueError("num_blocks_limit must be in [1,1024]")
        tile_limit = HIF8_REDUCE_ELEMENTS
        tile_alignment = 32
        if (
            type(self.tile_elements) is not int
            or not 256 <= self.tile_elements <= tile_limit
            or self.tile_elements % tile_alignment
        ):
            raise ValueError(
                f"tile_elements must be a multiple of {tile_alignment} in [256,{tile_limit}]"
            )
        if (
            type(self.buffer_bytes) is not int
            or self.buffer_bytes <= self.metadata_bytes
        ):
            raise ValueError("communication buffer is too small for metadata")
        if self.chunk_capacity < 1:
            raise ValueError("communication buffer is too small for one aligned chunk")

    @property
    def blocks(self):
        # CalNumBlocks uses inputSliceStride, not total input bytes.
        if self.layout.input_slice_stride < SMALL_BYTES:
            return min(self.num_blocks_limit, 2 * self.layout.ranks)
        return self.num_blocks_limit

    @property
    def branch(self):
        return "hif8"

    @property
    def flag_rows(self):
        # Preserve the flag layout: ready, reserved, entry and completion rows.
        return 4 * self.layout.ranks

    @property
    def metadata_bytes(self):
        # Preserve the allocated metadata footprint; flags use a separate window.
        return align_up(512 + 2 * self.flag_rows * 32 + self.blocks * 32)

    @property
    def banks(self):
        return 1

    @property
    def chunk_capacity(self):
        # Native HIF8 reserves 2 * ranks rows and aligns slices to 128 B.
        # Flags/epoch use a separate allocation and do not shrink data slices.
        return self.buffer_bytes // (2 * self.layout.ranks) // 128 * 128

    @property
    def row_elements(self):
        return (
            align_up(
                min(max(1, self.layout.count), self.chunk_capacity)
                * self.layout.itemsize,
                32,
            )
            // self.layout.itemsize
        )

    @property
    def chunks(self):
        return (self.layout.count + self.chunk_capacity - 1) // self.chunk_capacity

    @property
    def data_bytes(self):
        return 2 * self.layout.ranks * self.row_elements * self.layout.itemsize


@cb.datastruct(c_name="RsMeshComm")
class RsMeshComm:
    rank_id: d.int32
    rank_size: d.int32
    channel_handles: cb.CArray[d.int64, MAX_RANKS]
    comm_buffer_addrs: cb.CArray[d.int64, MAX_RANKS]
    # 0: direct mapped UBMEM, 1: URMA. Self is always direct.
    remote_transport: cb.CArray[d.int32, MAX_RANKS]
    # Direct UB mappings for HIF8 flags; never URMA RMA addresses.
    flag_buffer_addrs: cb.CArray[d.int64, MAX_RANKS]


def address(tensor):
    from cannbotdsl._mlir.dialects import cannir
    from cannbotdsl.core.ir_transport import unwrap_operand
    from cannbotdsl.types._numeric import Int64

    return Int64(cannir.extract_address(cannir.extract_pointer(unwrap_operand(tensor))))


def gm(addr, count, dtype):
    return cb.make_tensor(
        make_pointer(dtype, d.int64(addr), cb.MemLoc.GM),
        cb.make_layout((count,), stride=(1,)),
    )


@cb.jit
def local_barrier():
    vec_sync_all()
    vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
    # Epoch loads and flag polling run on PIPE.S. Waiting only on MTE2 lets
    # those scalar instructions race ahead of other cores' publication.
    vec_sync_block_wait(PIPE.S, 0, mode=0)


@cb.kernel
class RsMeshKernel:
    def __init__(self, plan, use_urma=False):
        self.p = plan.layout.ranks
        self.blocks = plan.blocks
        self.count = plan.layout.count
        self.repeats = plan.layout.repeats
        self.in_stride = plan.layout.input_slice_stride
        self.in_repeat = plan.layout.input_repeat_stride
        self.out_repeat = plan.layout.output_repeat_stride
        self.dtype = d.hifloat8
        self.size = plan.layout.itemsize
        self.op = plan.reduction
        self.tile = plan.tile_elements
        self.cap = plan.chunk_capacity
        self.row = plan.row_elements
        self.use_urma = use_urma
        self.copy_tile = HIF8_COPY_BYTES // self.size
        # Match AscendC's phase-local pool reuse. The send queue and
        # reduction tensors alias; the existing phase barrier orders reuse.
        self.copy_queue = cb.Channel(
            cb.MemLoc.UB, (self.copy_tile,), self.dtype, depth=1
        )
        # Depth one always selects the same slot. These aliases do not
        # allocate separate storage or advance the runtime copy pipeline.
        storage = self.copy_queue.consume()
        self.a = cb.reinterpret(storage, self.dtype, shape=(self.tile,))
        self.b = cb.reinterpret(
            storage, self.dtype, shape=(self.tile,), offset=self.tile
        )
        self.acc = cb.reinterpret(
            storage, d.float32, shape=(self.tile,), offset=2 * self.tile
        )
        self.a_bytes = cb.reinterpret(self.a, d.uint8, shape=(self.tile,))
        self.b_bytes = cb.reinterpret(self.b, d.uint8, shape=(self.tile,))
        if use_urma:
            self.comm_ub = Buffer(cb.MemLoc.UB, (512,), d.uint8)

    def __call__(self, x: cb.Tensor, y: cb.Tensor, ctx: cb.StructRef[RsMeshComm]):
        inp = address(x)
        out = address(y)
        handle = d.int64(0)
        if cb.const_expr(self.use_urma):
            handle = hcomm_init(self.comm_ub, 512, core=_CoreRole.AIV)
        # Each AIV participates even if its data partition is empty.
        for repeat in range(self.repeats):
            for offset in range(0, self.count, self.cap):
                n = min(self.cap, self.count - offset)
                epoch = self.next_epoch(ctx)
                # HIF8 matches the native first-use handshake: later rounds
                # reuse the window only after the preceding end barrier. Each
                # peer publishes end-of-round after all its local reads finish.
                # Epochs persist across launches/replays; ready and barrier rows
                # are separate, and monotonically increasing flags are not reset.
                if epoch == 1:
                    self.rank_barrier(ctx, epoch, 0)
                src = inp + repeat * self.in_repeat + offset * self.size
                dst = out + repeat * self.out_repeat + offset * self.size
                self.publish(ctx, handle, src, n, epoch)
                self.hif8(ctx, dst, n, epoch)
                # All ranks finish reading before reusing the single data bank.
                self.rank_barrier(ctx, epoch, 1)

    @cb.jit
    def copy(self, dst, src, count):
        for off in range(0, count, self.copy_tile):
            n = min(self.copy_tile, count - off)
            span = (slice(0, n),)
            cb.mem_copy(
                self.copy_queue.produce()[span],
                gm(src + off * self.size, n, self.dtype),
            )
            ready = self.copy_queue.consume()
            cb.mem_copy(gm(dst + off * self.size, n, self.dtype), ready[span])

    @cb.jit
    def put(self, ctx, handle, peer, inp, count):
        dst = ctx.comm_buffer_addrs[peer] + d.int64(ctx.rank_id) * self.row
        src = inp + peer * self.in_stride
        if cb.const_expr(self.use_urma):
            if ctx.remote_transport[peer] != 0:
                write_nbi(
                    handle,
                    ctx.channel_handles[peer],
                    dst,
                    src,
                    count * self.size,
                    core=_CoreRole.AIV,
                )
                hcomm_drain(handle, ctx.channel_handles[peer], core=_CoreRole.AIV)
            else:
                self.copy(dst, src, count)
        else:
            self.copy(dst, src, count)

    @cb.jit
    def signal(self, ctx, peer, row, epoch):
        off = 512 + row * 32
        vec_sync_all()
        # Data put() has completed its URMA Drain before ready is published.
        cb.scalar.vec_store_bypass(
            gm(ctx.flag_buffer_addrs[peer] + off, 1, d.int64).ptr((0,)), epoch
        )
        vec_sync_all()

    @cb.jit
    def wait(self, ctx, row, epoch):
        base = ctx.flag_buffer_addrs[d.int64(ctx.rank_id)]
        ptr = base + 512 + row * 32
        value = d.int64(0)
        while value < epoch:
            value = cb.scalar.load_bypass(gm(ptr, 1, d.int64).ptr((0,)))
            # 0.7.1 models load_bypass as a pure read. Without an observable
            # operation in the loop, lowering deletes this unused polling
            # loop entirely, allowing reads before the peer publishes data.
            vec_sync_all()
        vec_sync_all()

    @cb.jit
    def rank_barrier(self, ctx, epoch, phase):
        local_barrier()
        rank = d.int64(ctx.rank_id)
        # Use the same contiguous ownership as hif8(), also with blocks < ranks.
        block = cb.get_block_idx()
        begin = block * (self.p // self.blocks) + min(block, self.p % self.blocks)
        end = begin + self.p // self.blocks
        if block < self.p % self.blocks:
            end += 1
        for peer in range(begin, end):
            self.signal(ctx, peer, 2 * self.p + phase * self.p + rank, epoch)
        for peer in range(cb.get_block_idx(), self.p, self.blocks):
            self.wait(ctx, 2 * self.p + phase * self.p + peer, epoch)
        local_barrier()

    @cb.jit
    def next_epoch(self, ctx):
        clock = gm(ctx.flag_buffer_addrs[d.int64(ctx.rank_id)], 1, d.int64)
        local_barrier()
        if cb.get_block_idx() == 0:
            old = cb.scalar.load_bypass(clock.ptr((0,)))
            cb.scalar.vec_store_bypass(clock.ptr((0,)), old + 1)
            vec_sync_all()
        local_barrier()
        return cb.scalar.load_bypass(clock.ptr((0,)))

    @cb.jit
    def publish(self, ctx, handle, inp, count, epoch):
        rank = d.int64(ctx.rank_id)
        block = cb.get_block_idx()
        begin = block * (self.p // self.blocks) + min(block, self.p % self.blocks)
        end = begin + self.p // self.blocks
        if block < self.p % self.blocks:
            end += 1
        for target in range(begin, end):
            self.put(ctx, handle, target, inp, count)
            self.signal(ctx, target, rank, epoch)
        local_barrier()

    @cb.jit
    def hif8(self, ctx, out, count, epoch):
        block = cb.get_block_idx()
        per_core = count // self.blocks // 32 * 32
        off = block * per_core
        length = per_core
        if block == self.blocks - 1:
            length = count - off
        for start in range(0, length, self.tile):
            n = min(self.tile, length - start)
            self.reduce_tile(ctx, out, off + start, n, epoch)
        # The caller's completion barrier protects reuse of the single bank.

    @cb.jit
    def reduce_tile(self, ctx, out, offset, n, epoch):
        local = ctx.comm_buffer_addrs[d.int64(ctx.rank_id)]
        for source in range(self.p):
            self.wait(ctx, source, epoch)
            cb.mem_copy(self.a, gm(local + source * self.row + offset, n, self.dtype))
            vec_sync_all()
            self.accumulate(source, n)
            vec_sync_all()
        self.store_tile(out + offset, n)

    @cb.jit
    def accumulate(self, source, n):
        with cb.vf(mode="simd"):
            # Convert all lanes; arithmetic/stores below mask the tail.
            cast_mask = create_mask("all", elem_bits=8)
            for i in range(0, n, 64):
                mask, _ = update_mask(n - i, elem_bits=32)
                raw = vload_unpack(self.a_bytes, i, unpack_mode=UnpackMode.UNPACK4)
                value = vcast(
                    vreinterpret_lanes(raw, self.dtype),
                    d.float32,
                    mask=cast_mask,
                    reg_layout=cb.RegLayout.ZERO,
                )
                if source == 0:
                    vstore(self.acc, i, value, mask)
                else:
                    acc = vload(self.acc, i)
                    result = self.reduce_values(acc, value, mask)
                    vstore(self.acc, i, result, mask)

    @cb.jit
    def reduce_values(self, acc, value, mask):
        if cb.const_expr(self.op == "sum"):
            result = vadd(acc, value, mask=mask)
        elif cb.const_expr(self.op == "max"):
            result = vmax(acc, value, mask=mask)
        else:
            result = vmin(acc, value, mask=mask)
        return result

    @cb.jit
    def store_tile(self, out, n):
        with cb.vf(mode="simd"):
            for i in range(0, n, 64):
                mask, _ = update_mask(n - i, elem_bits=32)
                raw = vcast(vload(self.acc, i), self.dtype, mask=mask, rounding="rn")
                vstore_pack(
                    self.b_bytes,
                    i,
                    vreinterpret_lanes(raw, d.uint32),
                    mask,
                    pack_mode=PackMode.PACK_QUARTER,
                )
        vec_sync_all()
        span = (slice(0, n),)
        cb.mem_copy(gm(out, n, self.dtype), self.b[span])
        vec_sync_all()


class Program:
    def __init__(self, plan, use_urma=False):
        self.plan = plan
        self.use_urma = use_urma

    @cb.host
    def run(self, x: cb.Tensor, y: cb.Tensor, ctx: cb.StructRef[RsMeshComm]):
        op = RsMeshKernel(self.plan, self.use_urma)
        op[self.plan.blocks](x, y, ctx)


def plan_key(plan):
    return json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"))


def decode_plan(key):
    values = json.loads(key)
    values["layout"] = Layout(**values["layout"])
    return Plan(**values)


def compile_mesh(config: str, use_urma: bool = False):
    """Offline compilation needs no torch process group or live NPU."""
    plan = decode_plan(config)
    # HIF8's public torch storage carrier is int8, as in the repository's
    # HIF8 matmul sample. The device interprets the unchanged bytes as HIF8.
    dtype = cb.dtypes.int8
    return cb.compile(
        Program(plan, use_urma).run,
        cb.TensorSpec((plan.layout.input_elements,), dtype),
        cb.TensorSpec((plan.layout.output_elements,), dtype),
        cb.StructRef[RsMeshComm],
    )


# Some CI wheels do not expose the optional compile-factory cache.
# Direct compilation still returns the Provider used by save/load and execution.
if hasattr(opkit, "compile_cache"):
    compile_mesh = opkit.compile_cache(compile_mesh)


class AivReduceScatterSoleMesh:
    """Prepare private communication resources from a caller-owned torch group.

    All ranks construct and close operators in the same order. Calls enqueue on
    the preparation stream. Retire graphs before close(). The caller owns the
    HCCL group and must keep it alive until all operator users are closed.
    """

    def __init__(self, plan, *, group=None, transport="ubmem", artifact_path=None):
        self.plan = plan
        self._lock = threading.Lock()
        self._closed = False
        self._comm = _MeshCommunication(
            plan,
            RsMeshComm.__cannbotdsl_datastruct_schema__,
            plan_key(plan),
            group=group,
            transport=transport,
        )
        self.group = self._comm.group
        self.device = self._comm.device
        self.stream = self._comm.stream
        self.use_urma = self._comm.use_urma
        self.compiled = (
            compile_mesh(plan_key(plan), self.use_urma)
            if artifact_path is None
            else cb.load(str(artifact_path))
        )

    def __call__(self, x, out):
        import torch

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent host calls require separate operators")
        try:
            if self._closed:
                raise RuntimeError("ReduceScatter object is closed")
            if (
                torch.npu.current_stream(self.device) != self.stream
                or x.device != self.device
            ):
                raise RuntimeError(
                    "use the device and stream on which the operator was prepared"
                )
            return self._launch(x, out)
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Retire the private context; HCCL resources remain owned by the group."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close communication during a host submission")
        try:
            if self._closed:
                return
            self._comm.close()
            self._closed = True
            self.compiled = None
        finally:
            self._lock.release()

    def _launch(self, x, out):
        import torch

        context = self._comm.context
        layout = self.plan.layout
        if x.device.type != "npu" or x.dtype != torch.int8 or not x.is_contiguous():
            raise ValueError(
                "input must be contiguous HIF8 bytes (int8 storage) on NPU"
            )
        if out.device != x.device or out.dtype != x.dtype or not out.is_contiguous():
            raise ValueError("output must match input device/dtype and be contiguous")
        if x.numel() < layout.input_elements or out.numel() < layout.output_elements:
            raise ValueError("input/output storage is smaller than the strided layout")
        if (
            context.device != x.device
            or context.dtype != torch.uint8
            or not context.is_contiguous()
        ):
            raise ValueError(
                "context must be contiguous uint8 bytes on the input device"
            )
        if context.numel() != RsMeshComm.__cannbotdsl_datastruct_schema__.size:
            raise ValueError("context size must match one RsMeshComm")
        x0, y0 = x.data_ptr(), out.data_ptr()
        if max(x0, y0) < min(
            x0 + layout.input_elements * layout.itemsize,
            y0 + layout.output_elements * layout.itemsize,
        ):
            raise ValueError("overlapping input/output storage is unsupported")
        if layout.count:
            self.compiled(
                x.view(-1)[: layout.input_elements],
                out.view(-1)[: layout.output_elements],
                context,
            )
        return out
