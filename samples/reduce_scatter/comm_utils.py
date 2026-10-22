# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Private HIF8 communication resources backed by a caller-owned HCCL group."""

import ctypes
import hashlib

import cannbotdsl as cb

MAX_RANKS = 1024


class _HostComm(ctypes.Structure):
    _fields_ = [
        ("rank_id", ctypes.c_int32),
        ("rank_size", ctypes.c_int32),
        ("channel_handles", ctypes.c_int64 * MAX_RANKS),
        ("comm_buffer_addrs", ctypes.c_int64 * MAX_RANKS),
        ("remote_transport", ctypes.c_int32 * MAX_RANKS),
        ("flag_buffer_addrs", ctypes.c_int64 * MAX_RANKS),
    ]


def _collective_check(group, descriptor):
    import torch.distributed as dist

    descriptors = [None] * dist.get_world_size(group)
    dist.all_gather_object(descriptors, descriptor, group=group)
    if any(value != descriptor for value in descriptors):
        raise ValueError("ReduceScatter configuration differs between ranks")


class _MeshCommunication:
    """Build device context without creating or destroying process groups."""

    def __init__(self, plan, schema, key, *, group=None, transport="ubmem"):
        import torch_npu  # noqa: F401
        import torch
        import torch.distributed as dist

        self._schema = schema
        self._context = None
        if not dist.is_initialized():
            raise RuntimeError(
                "initialize torch.distributed before preparing communication"
            )
        if transport not in ("ubmem", "urma", "auto"):
            raise ValueError("transport must be ubmem, urma, or auto")
        self.plan = plan
        self.group = dist.group.WORLD if group is None else group
        if dist.get_backend(self.group) != "hccl":
            raise ValueError(
                "group must be an externally initialized HCCL process group"
            )
        self.device = torch.device("npu", torch.npu.current_device())
        self.stream = torch.npu.current_stream(self.device)
        if dist.get_world_size(self.group) != plan.layout.ranks:
            raise ValueError("plan rank count must match the communication group")
        limit = cb.get_platform_info().vector_core_num
        if limit <= 0 or plan.blocks > limit:
            raise ValueError(f"plan requires {plan.blocks} AIVs; available: {limit}")
        _collective_check(self.group, (key, transport))
        self.rank = dist.get_rank(self.group)
        # torch_npu's HCCL-name lookup requires this PyTorch backend accessor.
        backend = self.group._get_backend(self.device)
        self.group_name = backend.get_hccl_comm_name(self.rank)
        self._setup_context(transport, hashlib.sha256(key.encode()).hexdigest()[:24])
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.group)

    @property
    def context(self):
        """Device context for kernel submission, or None after close()."""
        return self._context

    def close(self):
        """Wait for users before dropping context; the caller owns HCCL resources."""
        import torch
        import torch.distributed as dist

        torch.npu.synchronize(self.device)
        dist.barrier(group=self.group)
        self._context = None

    def _setup_context(self, transport, tag):
        from cannbotdsl.distributed import hccl

        layer, peers = self._query_peers(transport)
        self.use_urma = any(
            protocol == hccl.COMM_PROTOCOL_UBC_CTP for _, protocol in peers.values()
        )
        host = self._allocate_buffers(transport, tag)
        for peer, link in peers.items():
            self._connect_peer(host, peer, layer, link)
        self._upload_context(host)

    def _query_peers(self, transport):
        from cannbotdsl.distributed import hccl

        self.comm_handle = hccl.get_comm_handle(self.group_name)
        if hccl.get_rank_size(self.comm_handle) != self.plan.layout.ranks:
            raise RuntimeError("HCCL and plan rank counts differ")
        rank = hccl.get_rank_id(self.comm_handle)
        if rank != self.rank:
            raise RuntimeError("HCCL and torch rank order differ")
        layer = hccl.get_rank_graph_layers(self.comm_handle)[0]
        peers = {}
        for peer in range(self.plan.layout.ranks):
            if peer == rank:
                continue
            links, _ = hccl.get_links(self.comm_handle, layer, rank, peer)
            # get_links returns ctypes views into HCCL-owned scratch storage.
            # The next topology query can overwrite those endpoints. Keep an
            # owned snapshot for each peer until channel acquisition finishes.
            links = [type(link).from_buffer_copy(link) for link in links]
            protocols = {link.linkAttr.linkProtocol for link in links}
            protocol = hccl.COMM_PROTOCOL_UB_MEM
            if transport == "urma" or (
                transport == "auto" and protocol not in protocols
            ):
                protocol = hccl.COMM_PROTOCOL_UBC_CTP
            if protocol not in protocols:
                raise RuntimeError(f"rank {rank} -> {peer} has no {transport} link")
            if hccl.COMM_PROTOCOL_UB_MEM not in protocols:
                raise RuntimeError(
                    f"HIF8 direct notifications require a UB_MEM mapping to peer {peer}"
                )
            peers[peer] = (links, protocol)
        return layer, peers

    def _allocate_buffers(self, transport, tag):
        from cannbotdsl.distributed import hccl

        # All ranks choose the same allocation source, including ranks whose
        # particular peers are exclusively UBMEM in an auto/mixed topology.
        self._registration = None
        if transport == "ubmem":
            ptr = hccl.engine_ctx_create(
                self.comm_handle, "rsdsl_" + tag, self.plan.buffer_bytes
            )
            self._registration = hccl.register_mem(
                self.comm_handle, "rsdsl_data", ptr, self.plan.buffer_bytes
            )
        else:
            ptr, size = hccl.get_hccl_buffer(self.comm_handle)
            if size < self.plan.buffer_bytes:
                raise ValueError(
                    f"HCCL window is {size} bytes; plan requires {self.plan.buffer_bytes}"
                )
        if not ptr or ptr % 32:
            raise RuntimeError(
                "communication buffer must have a nonzero 32-byte-aligned address"
            )
        hccl.mem_set(ptr, self.plan.buffer_bytes)
        host = _HostComm()
        host.rank_id, host.rank_size = self.rank, self.plan.layout.ranks
        host.comm_buffer_addrs[self.rank] = ptr
        self._channels = []
        # HIF8 keeps flags/epoch outside the data budget, like the native layout.
        # Exchange this separate allocation over UB_MEM for both transports.
        flag_ptr = hccl.engine_ctx_create(
            self.comm_handle, "rsdsl_flags_" + tag, self.plan.metadata_bytes
        )
        hccl.mem_set(flag_ptr, self.plan.metadata_bytes)
        self._flag_registration = hccl.register_mem(
            self.comm_handle, "rsdsl_flags", flag_ptr, self.plan.metadata_bytes
        )
        host.flag_buffer_addrs[self.rank] = flag_ptr
        return host

    def _connect_peer(self, host, peer, layer, link):
        from cannbotdsl.distributed import hccl

        links, protocol = link
        channel = hccl.acquire_channel(
            self.comm_handle,
            hccl.COMM_ENGINE_AIV,
            protocol,
            peer,
            layer,
            links,
            mem_handles=[self._registration]
            if self._registration is not None
            else None,
        )
        self._channels.append(channel)
        if self._registration is not None:
            remote = hccl.get_channel_remote_mems(
                self.comm_handle, channel, "rsdsl_data"
            )
        else:
            remote = hccl.get_channel_hccl_buffer(self.comm_handle, channel)
        if not remote:
            raise RuntimeError(f"peer {peer} returned a null communication address")
        host.channel_handles[peer] = channel
        host.comm_buffer_addrs[peer] = remote
        host.remote_transport[peer] = int(protocol == hccl.COMM_PROTOCOL_UBC_CTP)
        flag_channel = hccl.acquire_channel(
            self.comm_handle,
            hccl.COMM_ENGINE_AIV,
            hccl.COMM_PROTOCOL_UB_MEM,
            peer,
            layer,
            links,
            mem_handles=[self._flag_registration],
        )
        self._channels.append(flag_channel)
        flag_remote = hccl.get_channel_remote_mems(
            self.comm_handle, flag_channel, "rsdsl_flags"
        )
        if not flag_remote:
            raise RuntimeError(f"peer {peer} returned a null direct flag mapping")
        host.flag_buffer_addrs[peer] = flag_remote

    def _upload_context(self, host):
        import torch

        # Check ABI against the actual wheel rather than assume ctypes layout.
        schema = self._schema
        if ctypes.sizeof(host) != schema.size:
            raise RuntimeError("RsMeshComm host/device ABI size mismatch")
        for field in schema.fields:
            if getattr(_HostComm, field.name).offset != field.offset:
                raise RuntimeError(f"RsMeshComm ABI offset mismatch: {field.name}")
        payload = bytearray(
            ctypes.string_at(ctypes.addressof(host), ctypes.sizeof(host))
        )
        self._context = (
            torch.frombuffer(payload, dtype=torch.uint8).clone().to(self.device)
        )
