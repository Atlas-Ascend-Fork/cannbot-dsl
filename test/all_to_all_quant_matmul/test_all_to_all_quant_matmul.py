# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import inspect
import json
import logging
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import NamedTuple
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "samples" / "all_to_all_quant_matmul"))
sys.path.insert(0, str(_HERE))

MSG_ARTIFACT_READY = "ARTIFACT_READY"
MSG_START_DEVICE = "START_DEVICE"
MSG_RESULT = "RESULT"

MSG_DOMAIN_READY = "DOMAIN_READY"
MSG_CASE_GO = "CASE_GO"
MSG_CASE_RESULT = "CASE_RESULT"
MSG_SHUTDOWN = "SHUTDOWN"
MSG_WORKER_EXIT = "WORKER_EXIT"

_STAGE_COMPILE = "compile"
_STAGE_EXECUTE = "execute"

_PERF_INNER_ENV = "DISTRIBUTED_OP_RUNNER_MSPROF_INNER"


_LOGGER_OUT = logging.getLogger("a2a_test.stdout")
_LOGGER_ERR = logging.getLogger("a2a_test.stderr")
for _logger, _stream in ((_LOGGER_OUT, sys.stdout), (_LOGGER_ERR, sys.stderr)):
    _handler = logging.StreamHandler(_stream)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _log(tag, message, file=None):
    logger = _LOGGER_ERR if file is not None else _LOGGER_OUT
    logger.info(f"[{tag}] {message}")


def _distributed_group_name():
    return "hccl_world"


class WorkerCtx(NamedTuple):
    rank_id: int
    rank_size: int
    device_id: int
    group_name: str
    master_addr: str
    master_port: int


class RunCtx(NamedTuple):
    test: object
    args: argparse.Namespace
    extra_params: dict
    cases: list
    bundle_tmp_root: str = None


def _bootstrap_hccl(ctx):
    os.environ["RANK"] = str(ctx.rank_id)
    os.environ["WORLD_SIZE"] = str(ctx.rank_size)
    os.environ["MASTER_ADDR"] = ctx.master_addr
    os.environ["MASTER_PORT"] = str(ctx.master_port)
    import torch_npu  # noqa: F401 (注册 NPU 后端)

    torch.npu.set_device(ctx.device_id)
    torch.distributed.init_process_group(
        backend="hccl",
        init_method=f"tcp://{ctx.master_addr}:{ctx.master_port}",
        world_size=ctx.rank_size,
        rank=ctx.rank_id,
        group_name=ctx.group_name,
    )
    torch.distributed.barrier()
    lib = ctypes.CDLL("libhccl_fwk.so")
    lib.HcomCreateGroup.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.HcomCreateGroup.restype = ctypes.c_int32
    rank_ids = (ctypes.c_uint32 * ctx.rank_size)(*range(ctx.rank_size))
    ret = lib.HcomCreateGroup(ctx.group_name.encode(), ctx.rank_size, rank_ids)
    if ret != 0:
        raise RuntimeError(
            f"rank {ctx.rank_id}: HcomCreateGroup({ctx.group_name!r}) "
            f"failed with code {ret}"
        )


class DistributedOpTest:
    golden_is_rank_invariant = False

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Distributed operator test "
            "(compile-once / late-acquire, one worker per device)",
        )
        self.parser.add_argument(
            "--csv",
            required=True,
            help="cases CSV path (positional columns per case_column_map; "
            "header row is cosmetic)",
        )
        self.parser.add_argument(
            "--devices",
            default="0,1",
            help="comma-separated physical NPU device ids, one worker each "
            "(default: 0,1)",
        )
        self.parser.add_argument(
            "--cases",
            default=None,
            help="comma-separated 0-based CSV data-row selection, header "
            "not counted (default: all)",
        )
        self.kernel_name_pattern = None

    def case_column_map(self):
        raise NotImplementedError

    def compile_case(self, **params):
        raise NotImplementedError

    def determine_precision_tolerance(self):
        raise NotImplementedError

    def execute(self, prepare_res, **params):
        raise NotImplementedError

    def gen_inputs(self, rank_id, rank_size, **params):
        raise NotImplementedError

    def golden(self, rank_id, rank_size, all_inputs, **params):
        raise NotImplementedError

    def set_kernel_name_pattern(self, pattern):
        self.kernel_name_pattern = pattern

    def worker_prepare(self, ctx, inputs, callables=None, **params):
        raise NotImplementedError


def add_stage_arguments(parser):
    parser.add_argument(
        "--stage",
        choices=(_STAGE_COMPILE, _STAGE_EXECUTE),
        default=None,
        help="split the run into a device-free compile phase and an execute "
        "phase (default: combined compile + execute in one run)",
    )
    parser.add_argument(
        "--bundle-root",
        default="./bundles",
        help="artifact bundle directory used by --stage compile/execute "
        "(default: ./bundles)",
    )
    parser.add_argument(
        "--wait-devices",
        type=int,
        default=0,
        metavar="SECONDS",
        help="after ARTIFACT_READY, wait up to SECONDS for the requested "
        "devices to look idle before broadcasting START_DEVICE "
        "(default: 0 — no wait)",
    )
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=600,
        metavar="SECONDS",
        help="per-case wall-clock budget for worker messaging "
        "(ARTIFACT_READY collection and post-START result collection, "
        "each phase gets the full budget; 0 disables). A hung worker "
        "fails its case and is reaped with TERM/KILL so the batch can "
        "continue (default: 600)",
    )
    parser.add_argument(
        "--persistent-domain",
        action="store_true",
        help="EXPERIMENTAL: keep ONE HCCL domain per rank_size group alive "
        "across cases (workers load per-case bundles after domain "
        "bootstrap; case failure rebuilds the domain and continues). "
        "Requires the kernel's cross-launch sync protocol "
        "(launch-counter ack barrier + fixed ready flag)",
    )


def _load_cases(csv_path, column_map):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"cases CSV not found: {csv_path}")
    width = max(idx for idx, _ in column_map.values()) + 1
    cases = []
    saw_header = False
    with open(path, encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if not saw_header:
                saw_header = True
                continue
            if len(row) < width:
                raise ValueError(
                    f"{csv_path}: row {row!r} has {len(row)} cells, need "
                    f"{width} for columns {sorted(column_map)}"
                )
            case = {name: cast(row[idx]) for name, (idx, cast) in column_map.items()}
            cases.append(case)
    return cases


def _select_cases(cases, selection):
    if selection is None:
        return list(range(len(cases))), False
    wanted = []
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if not 0 <= idx < len(cases):
            raise ValueError(f"--cases index {idx} out of range (0..{len(cases) - 1})")
        wanted.append(idx)
    return wanted, True


def _parse_devices(devices_arg):
    devices = [int(tok) for tok in devices_arg.split(",") if tok.strip()]
    if not devices:
        raise ValueError(f"--devices produced no ids: {devices_arg!r}")
    return devices


def _devices_for_case(case, idx, devices):
    rank_size = case["rank_size"]
    if rank_size <= len(devices):
        chosen = list(devices[:rank_size])
        if rank_size < len(devices):
            _log(
                "parent",
                f"case {idx}: rank_size={rank_size} < pool of "
                f"{len(devices)}; using first {rank_size}: {chosen}",
            )
        return chosen
    _log(
        "parent",
        f"case {idx}: rank_size={rank_size} > "
        f"{len(devices)} device(s) in pool {devices}; skipping",
    )
    return None


def _bundle_dir(bundle_root, case_idx):
    return Path(bundle_root) / f"case_{case_idx:02d}"


_BUNDLE_FORMAT = "cannbotdsl-dist-bundle/1"


def _save_bundle(callables, params, case_dir, compile_seconds):
    artifacts = {}
    values = {}
    for name, item in callables.items():
        save = getattr(item, "save", None)
        if callable(save):
            so_name = f"{name}.so"
            save(str(case_dir / so_name))
            artifacts[name] = so_name
        else:
            values[name] = item
    meta = {
        "format": _BUNDLE_FORMAT,
        "params": params,
        "compile_seconds": round(compile_seconds, 3),
        "artifacts": artifacts,
        "values": values,
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _load_callables_from_bundle(case_dir, rank_size=None):
    meta_path = Path(case_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"bundle meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != _BUNDLE_FORMAT:
        raise ValueError(f"bundle {case_dir!r}: unknown format {meta.get('format')!r}")
    if rank_size is not None and meta.get("params", {}).get("rank_size") != rank_size:
        raise ValueError(
            f"bundle {case_dir!r} was compiled for rank_size "
            f"{meta.get('params', {}).get('rank_size')!r}, expected {rank_size}"
        )
    from cannbotdsl import load as _load_callable

    callables = dict(meta.get("values", {}))
    for name, so_name in meta.get("artifacts", {}).items():
        so_path = Path(case_dir) / so_name
        if not so_path.is_file():
            raise FileNotFoundError(f"bundle artifact missing: {so_path}")
        callables[name] = _load_callable(str(so_path))
    if not callables:
        raise ValueError(f"bundle {case_dir!r} carries no artifacts or values")
    return callables


def _bundle_load_supported():
    try:
        import cannbotdsl
    except ImportError:
        return False
    return callable(getattr(cannbotdsl, "load", None))


def _compile_to_bundle(test, params, bundle_root, case_idx):
    case_dir = _bundle_dir(bundle_root, case_idx)
    case_dir.mkdir(parents=True, exist_ok=True)
    _log("parent", f"case {case_idx}: compiling (rank-neutral, no device) ...")
    started = time.monotonic()
    callables = test.compile_case(**params)
    seconds = time.monotonic() - started
    meta = _save_bundle(callables, params, case_dir, seconds)
    _log(
        "parent",
        f"case {case_idx}: bundle saved to {case_dir} in "
        f"{seconds:.1f}s (artifacts={sorted(meta['artifacts'])})",
    )
    return callables


def _worker_main(conn, test, ctx, case_data, artifacts):
    params, inputs = case_data
    callables, bundle_dir = artifacts
    try:
        if callables is None and bundle_dir is not None:
            callables = _load_callables_from_bundle(bundle_dir, ctx.rank_size)
        conn.send({"kind": MSG_ARTIFACT_READY, "rank": ctx.rank_id})
        message = conn.recv()
        if message != MSG_START_DEVICE:
            raise RuntimeError(
                f"rank {ctx.rank_id}: expected {MSG_START_DEVICE}, got {message!r}"
            )
        _bootstrap_hccl(ctx)
        prepare_res = test.worker_prepare(ctx, inputs, callables=callables, **params)
        result = test.execute(prepare_res, **params)
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            _log(f"rank {ctx.rank_id}", f"worker exit error:\n{traceback.format_exc()}")
        conn.send(
            {"kind": MSG_RESULT, "rank": ctx.rank_id, "ok": True, "result": result}
        )
    except BaseException:
        try:
            conn.send(
                {
                    "kind": MSG_RESULT,
                    "rank": ctx.rank_id,
                    "ok": False,
                    "error": traceback.format_exc(),
                }
            )
        except OSError:
            pass
    finally:
        conn.close()


def _persistent_worker(conn, test, ctx):
    try:
        _bootstrap_hccl(ctx)
        conn.send({"kind": MSG_DOMAIN_READY, "rank": ctx.rank_id})
        message = conn.recv()
        if message != MSG_START_DEVICE:
            raise RuntimeError(
                f"rank {ctx.rank_id}: expected {MSG_START_DEVICE}, got {message!r}"
            )
        while True:
            message = conn.recv()
            kind = message.get("kind") if isinstance(message, dict) else message
            if kind == MSG_SHUTDOWN:
                break
            if kind != MSG_CASE_GO:
                raise RuntimeError(
                    f"rank {ctx.rank_id}: expected CASE_GO/SHUTDOWN, got {message!r}"
                )
            _persistent_case_once(conn, test, ctx, message)
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            _log(f"rank {ctx.rank_id}", f"worker exit error:\n{traceback.format_exc()}")
        conn.send({"kind": MSG_WORKER_EXIT, "rank": ctx.rank_id})
    except BaseException:
        try:
            conn.send(
                {
                    "kind": MSG_CASE_RESULT,
                    "rank": ctx.rank_id,
                    "ok": False,
                    "error": traceback.format_exc(),
                }
            )
        except OSError:
            pass
    finally:
        conn.close()


def _persistent_case_once(conn, test, ctx, message):
    callables = _load_callables_from_bundle(message["bundle_dir"], ctx.rank_size)
    prepare_res = test.worker_prepare(
        ctx,
        message["inputs"],
        callables=callables,
        **message["hook_params"],
    )
    result = test.execute(prepare_res, **message["hook_params"])
    conn.send(
        {
            "kind": MSG_CASE_RESULT,
            "rank": ctx.rank_id,
            "idx": message["idx"],
            "result": result,
        }
    )
    del prepare_res, callables, result


def _spawn_persistent_workers(test, rank_size, devices):
    master_addr, master_port = _pick_master_endpoint()
    group_name = _distributed_group_name()
    context = multiprocessing.get_context("fork")
    workers = []
    for rank_id, device_id in enumerate(devices):
        parent_conn, child_conn = context.Pipe()
        ctx = WorkerCtx(
            rank_id, rank_size, device_id, group_name, master_addr, master_port
        )
        proc = context.Process(
            target=_persistent_worker,
            args=(
                child_conn,
                test,
                ctx,
            ),
        )
        proc.start()
        workers.append((rank_id, parent_conn, proc))
    return workers


def _pick_master_endpoint():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", sock.getsockname()[1]


def _devices_look_idle(devices):
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        _log("parent", "npu-smi not found; skipping device-idleness probe")
        return True
    try:
        text = subprocess.run(
            [npu_smi, "info"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        _log("parent", f"npu-smi failed ({exc}); skipping idle probe")
        return True
    for line in text.splitlines():
        stripped = line.strip()
        cells = stripped.split()

        if len(cells) >= 2 and cells[0].isdigit():
            _log("parent", f"npu-smi reports a process row: {stripped!r}")
            return False
    _ = devices
    return True


def _wait_devices_free(devices, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    _log(
        "parent",
        f"waiting up to {timeout_seconds}s for devices {devices} to look idle ...",
    )
    while True:
        if _devices_look_idle(devices):
            _log("parent", "devices look idle")
            return
        if time.monotonic() >= deadline:
            _log("parent", "device wait timed out; proceeding anyway")
            return
        time.sleep(10.0)


def _recv_or_fail(conn, procs_by_rank, rank_id, deadline=None):
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not conn.poll(remaining):
            return {
                "kind": MSG_RESULT,
                "rank": rank_id,
                "ok": False,
                "error": (
                    "timeout waiting for worker message "
                    f"(case budget exhausted; pid={procs_by_rank[rank_id].pid})"
                ),
            }
    try:
        return conn.recv()
    except EOFError:
        exitcode = procs_by_rank[rank_id].exitcode
        return {
            "kind": MSG_RESULT,
            "rank": rank_id,
            "ok": False,
            "error": f"worker died before reporting (exitcode={exitcode})",
        }


def _generate_inputs_and_golden(test, rank_size, hook_params):
    all_inputs = [
        test.gen_inputs(rank_id, rank_size, **hook_params)
        for rank_id in range(rank_size)
    ]
    if getattr(test, "golden_is_rank_invariant", False):
        golden_by_rank = [
            test.golden(0, rank_size, all_inputs, **hook_params)
        ] * rank_size
    else:
        golden_by_rank = [
            test.golden(rank_id, rank_size, all_inputs, **hook_params)
            for rank_id in range(rank_size)
        ]
    return all_inputs, golden_by_rank, test.determine_precision_tolerance()


def _spawn_case_workers(test, rank_size, devices, case_data, artifacts):
    master_addr, master_port = _pick_master_endpoint()
    group_name = _distributed_group_name()
    context = multiprocessing.get_context("fork")
    workers = []
    for rank_id, device_id in enumerate(devices):
        ctx = WorkerCtx(
            rank_id, rank_size, device_id, group_name, master_addr, master_port
        )
        parent_conn, child_conn = context.Pipe()
        proc = context.Process(
            target=_worker_main,
            args=(
                child_conn,
                test,
                ctx,
                (case_data[0], case_data[1][rank_id]),
                artifacts,
            ),
        )
        proc.start()
        workers.append((rank_id, parent_conn, proc))
    return workers


def _reap_workers(workers, case_idx):
    for rank_id, conn, proc in workers:
        try:
            conn.close()
        except OSError:
            pass
        proc.join(5)
        if proc.is_alive():
            _log(
                "parent",
                f"case {case_idx}: rank {rank_id} worker pid {proc.pid} "
                "stuck; terminating",
            )
            proc.terminate()
            proc.join(10)
            if proc.is_alive():
                _log(
                    "parent",
                    f"case {case_idx}: rank {rank_id} worker pid {proc.pid} "
                    "ignored TERM; killing",
                )
                proc.kill()
                proc.join()


def _handshake_and_collect(ctx, case_idx, rank_size, devices, workers):
    case_timeout = getattr(ctx.args, "case_timeout", 0)
    procs_by_rank = {rank_id: proc for rank_id, _, proc in workers}
    results = {}
    try:
        if not _wait_artifacts_ready(ctx, case_idx, devices, workers, procs_by_rank):
            return _aborted_results(procs_by_rank)
        _broadcast_start(ctx, devices, workers)
        deadline = (time.monotonic() + case_timeout) if case_timeout > 0 else None
        for rank_id, conn, _ in workers:
            results[rank_id] = _recv_or_fail(conn, procs_by_rank, rank_id, deadline)
    finally:
        _reap_workers(workers, case_idx)
    return results


def _wait_artifacts_ready(ctx, case_idx, devices, workers, procs_by_rank):
    case_timeout = getattr(ctx.args, "case_timeout", 0)
    deadline = (time.monotonic() + case_timeout) if case_timeout > 0 else None
    for rank_id, conn, _ in workers:
        message = _recv_or_fail(conn, procs_by_rank, rank_id, deadline)
        if message.get("kind") == MSG_ARTIFACT_READY:
            continue
        _log(
            "parent",
            f"case {case_idx}: rank {rank_id} failed before "
            f"{MSG_START_DEVICE}; aborting case",
        )
        return False
    _log(
        "parent",
        f"case {case_idx}: all {len(workers)} workers {MSG_ARTIFACT_READY}",
    )
    if ctx.args.wait_devices > 0:
        _wait_devices_free(devices, ctx.args.wait_devices)
    return True


def _aborted_results(procs_by_rank):
    return {
        rank_id: {
            "kind": MSG_RESULT,
            "rank": rank_id,
            "ok": False,
            "error": "aborted before START_DEVICE (peer failed/died/timed out)",
        }
        for rank_id in procs_by_rank
    }


def _broadcast_start(ctx, devices, workers):
    for _, conn, _ in workers:
        try:
            conn.send(MSG_START_DEVICE)
        except OSError:
            pass


def _report_rank_results(case_idx, results, rank_size, golden_by_rank, tolerance):
    case_ok = True
    for rank_id in range(rank_size):
        message = results.get(rank_id)
        if message is None or not message.get("ok"):
            case_ok = False
            error = (message or {}).get("error", "no result")
            _log(f"rank {rank_id}", f"EXECUTION FAILED:\n{error}")
            continue
        rank_ok, detail = _compare_outputs(
            message["result"], golden_by_rank[rank_id], tolerance
        )
        case_ok = case_ok and rank_ok
        _log(f"rank {rank_id}", f"{'PASS' if rank_ok else 'FAIL'} {detail}")
    _log("parent", f"case {case_idx}: {'PASS' if case_ok else 'FAIL'}")
    return case_ok


def _run_case_on_devices(ctx, case_idx, case, devices, artifacts):
    callables, bundle_dir = artifacts
    rank_size = case["rank_size"]
    if rank_size != len(devices):
        raise ValueError(
            f"case {case_idx} rank_size={rank_size} but "
            f"{len(devices)} device(s) given ({devices})"
        )
    if callables is None and bundle_dir is None:
        raise ValueError(f"case {case_idx}: neither callables nor bundle_dir given")
    params = {**case, **ctx.extra_params}
    _log(
        "parent",
        f"case {case_idx}: {params} devices={devices} "
        f"artifacts={'bundle:' + str(bundle_dir) if bundle_dir else 'fork-inherited'}",
    )

    hook_params = {k: v for k, v in params.items() if k != "rank_size"}

    all_inputs, golden_by_rank, tolerance = _generate_inputs_and_golden(
        ctx.test, rank_size, hook_params
    )
    workers = _spawn_case_workers(
        ctx.test, rank_size, devices, (hook_params, all_inputs), (callables, bundle_dir)
    )
    results = _handshake_and_collect(ctx, case_idx, rank_size, devices, workers)
    return _report_rank_results(case_idx, results, rank_size, golden_by_rank, tolerance)


def _compare_outputs(result, golden, tolerance):
    details = []
    overall = True
    for name, (rtol, atol) in tolerance.items():
        if name not in result or name not in golden:
            overall = False
            details.append(
                f"{name}: missing from {'result' if name not in result else 'golden'}"
            )
            continue
        got = np.asarray(result[name])
        want = np.asarray(golden[name])
        if got.shape != want.shape:
            overall = False
            details.append(f"{name}: shape mismatch got={got.shape} want={want.shape}")
            continue
        close = np.allclose(got, want, rtol=rtol, atol=atol)
        max_abs = float(
            np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))
        )
        if not close:
            overall = False
        details.append(
            f"{name}: max_abs_err={max_abs:.3e} "
            f"(rtol={rtol:g}, atol={atol:g}, {'ok' if close else 'MISMATCH'})"
        )
    return overall, "; ".join(details)


def _validate_bundle(bundle_root, case_idx):
    case_dir = _bundle_dir(bundle_root, case_idx)
    meta_path = case_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"bundle meta not found for case {case_idx}: {meta_path} "
            f"(run --stage compile first)"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != _BUNDLE_FORMAT:
        raise ValueError(
            f"bundle {case_dir}: unknown format {meta.get('format')!r}; "
            f"re-run --stage compile with this runner version"
        )
    for name, so_name in meta.get("artifacts", {}).items():
        if not (case_dir / so_name).is_file():
            raise FileNotFoundError(
                f"bundle artifact for {name!r} missing: {case_dir / so_name}"
            )
    return meta.get("params", {}), case_dir


def _case_callables(test, args, extra_params, case, case_idx):
    params = {**case, **extra_params}
    if args.stage == _STAGE_EXECUTE:
        bundle_params, case_dir = _validate_bundle(args.bundle_root, case_idx)
        os.environ.pop("CANNBOTDSL_CACHE_DIR", None)
        if bundle_params != params:
            raise ValueError(
                f"case {case_idx}: bundle params {bundle_params} != current "
                f"{params}; re-run --stage compile for the new parameters"
            )
        if _bundle_load_supported():
            _log(
                "parent",
                f"case {case_idx}: workers will load bundle artifacts from {case_dir}",
            )
            return None, case_dir
        _log(
            "parent",
            f"case {case_idx}: cannbotdsl.load unavailable; falling back "
            "to fresh parent compile",
        )
    _log("parent", f"case {case_idx}: compiling (rank-neutral, no device) ...")
    return test.compile_case(**params), None


def _run_compile_stage(test, args, extra_params, cases, selected):
    bundle_root = Path(args.bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    for case_idx in selected:
        _compile_to_bundle(
            test, {**cases[case_idx], **extra_params}, bundle_root, case_idx
        )
    _log(
        "parent",
        f"compile stage done: {len(selected)} bundle(s) under {bundle_root}",
    )
    return True


def _prep_persistent_case(ctx, case_idx, bundle_tmp_root):
    case = ctx.cases[case_idx]
    params = {**case, **ctx.extra_params}
    hook_params = {k: v for k, v in params.items() if k != "rank_size"}
    if ctx.args.stage == _STAGE_EXECUTE:
        bundle_params, bundle_dir = _validate_bundle(ctx.args.bundle_root, case_idx)
        if bundle_params != params:
            raise ValueError(
                f"case {case_idx}: bundle params {bundle_params} != current "
                f"{params}; re-run --stage compile for the new parameters"
            )
    else:
        callables = ctx.test.compile_case(**params)
        bundle_dir = Path(bundle_tmp_root) / f"case_{case_idx:02d}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        _save_bundle(callables, params, bundle_dir, 0.0)
    rank_size = case["rank_size"]
    all_inputs, golden_by_rank, tolerance = _generate_inputs_and_golden(
        ctx.test, rank_size, hook_params
    )
    return {
        "idx": case_idx,
        "hook_params": hook_params,
        "all_inputs": all_inputs,
        "golden_by_rank": golden_by_rank,
        "tolerance": tolerance,
        "bundle_dir": str(bundle_dir),
    }


def _run_domain_group(ctx, case_idxs, devices):
    cases = ctx.cases
    rank_size = cases[case_idxs[0]]["rank_size"]
    group_devices = list(devices[:rank_size])
    remaining = list(case_idxs)
    judged = {}
    overall = True
    domain_epoch = 0
    bundle_tmp_root = None
    if ctx.args.stage != _STAGE_EXECUTE:
        bundle_tmp_root = tempfile.mkdtemp(prefix="a2a_persistent_bundles_")
    try:
        while remaining:
            domain_epoch += 1
            ctx = ctx._replace(bundle_tmp_root=bundle_tmp_root)
            judged, domain_failed = _run_one_domain(
                ctx, domain_epoch, rank_size, group_devices, remaining
            )
            overall = overall and all(judged.values())
            if domain_failed:
                remaining = [i for i in remaining if i not in judged]
            else:
                remaining = []
    finally:
        if bundle_tmp_root is not None:
            shutil.rmtree(bundle_tmp_root, ignore_errors=True)
    return overall, judged


def _run_one_domain(ctx, domain_epoch, rank_size, group_devices, remaining):
    case_timeout = getattr(ctx.args, "case_timeout", 0)
    bundle_tmp_root = ctx.bundle_tmp_root
    _log(
        "parent",
        f"persistent domain #{domain_epoch} (rank_size={rank_size}, "
        f"devices={group_devices}, {len(remaining)} case(s) left)",
    )
    workers = _spawn_persistent_workers(ctx.test, rank_size, group_devices)
    procs_by_rank = {r: p for r, _, p in workers}
    try:
        deadline = time.monotonic() + case_timeout if case_timeout > 0 else None
        if not _wait_domain_ready(workers, procs_by_rank, deadline):
            return {}, True
        _log("parent", f"persistent domain #{domain_epoch} READY")
        _broadcast_start(ctx, group_devices, workers)
        return _run_domain_cases(
            ctx, workers, procs_by_rank, remaining, bundle_tmp_root
        )
    finally:
        _reap_workers(workers, f"domain#{domain_epoch}")


def _run_domain_cases(ctx, workers, procs_by_rank, remaining, bundle_tmp_root):
    judged = {}
    for case_idx in remaining:
        prep = _prep_persistent_case(ctx, case_idx, bundle_tmp_root)
        for rank_id, conn, _ in workers:
            conn.send(
                {
                    "kind": MSG_CASE_GO,
                    "idx": case_idx,
                    "bundle_dir": prep["bundle_dir"],
                    "hook_params": prep["hook_params"],
                    "inputs": prep["all_inputs"][rank_id],
                }
            )
        case_ok = _judge_persistent_case(ctx, workers, procs_by_rank, case_idx, prep)
        judged[case_idx] = case_ok
        _log("parent", f"case {case_idx}: {'PASS' if case_ok else 'FAIL'}")
        if not case_ok:
            return judged, True
    return judged, False


def _wait_domain_ready(workers, procs_by_rank, deadline):
    for rank_id, conn, _ in workers:
        msg = _recv_or_fail(conn, procs_by_rank, rank_id, deadline)
        if msg.get("kind") != MSG_DOMAIN_READY:
            _log("parent", f"rank {rank_id} domain bootstrap failed: {msg}")
            return False
    return True


def _judge_persistent_case(ctx, workers, procs_by_rank, case_idx, prep):
    case_timeout = getattr(ctx.args, "case_timeout", 0)
    deadline = (time.monotonic() + case_timeout) if case_timeout > 0 else None
    case_ok = True
    for rank_id, conn, _ in workers:
        msg = _recv_or_fail(conn, procs_by_rank, rank_id, deadline)
        if msg.get("kind") != MSG_CASE_RESULT or msg.get("idx") != case_idx:
            error = msg.get("error") or msg.get("kind") or "no result"
            _log(f"rank {rank_id}", f"case {case_idx} FAILED:\n{error}")
            case_ok = False
            continue
        if "result" not in msg:
            _log(
                f"rank {rank_id}",
                f"case {case_idx} FAILED:\n{msg.get('error')}",
            )
            case_ok = False
            continue
        rank_ok, detail = _compare_outputs(
            msg["result"],
            prep["golden_by_rank"][rank_id],
            prep["tolerance"],
        )
        case_ok = case_ok and rank_ok
        _log(
            f"rank {rank_id}",
            f"case {case_idx}: {'PASS' if rank_ok else 'FAIL'} {detail}",
        )
    return case_ok


def _shutdown_domain(workers, procs_by_rank, deadline):
    for _, conn, _ in workers:
        try:
            conn.send({"kind": MSG_SHUTDOWN})
        except OSError:
            pass
    for rank_id, conn, _ in workers:
        msg = _recv_or_fail(conn, procs_by_rank, rank_id, deadline)
        if msg.get("kind") != MSG_WORKER_EXIT:
            _log(
                "parent",
                f"rank {rank_id} no clean exit: {msg.get('kind')}",
            )


def _run_persistent_batch(ctx, selected, devices):
    if not _bundle_load_supported():
        _log(
            "parent",
            "persistent mode requires cannbotdsl.load",
            file=sys.stderr,
        )
        return False
    os.environ.pop("CANNBOTDSL_CACHE_DIR", None)
    cases = ctx.cases
    groups = {}
    for idx in selected:
        rs = cases[idx]["rank_size"]
        if rs > len(devices):
            _log("parent", f"case {idx}: rank_size={rs} > pool {devices}; skipping")
            continue
        groups.setdefault(rs, []).append(idx)
    if not groups:
        _log("parent", "no runnable cases for the given devices")
        return False
    overall = True
    for rank_size, idxs in groups.items():
        _log(
            "parent",
            f"persistent group rank_size={rank_size}: {len(idxs)} case(s)",
        )
        group_ok, _judged = _run_domain_group(ctx, idxs, devices)
        overall = overall and group_ok
    _log("parent", f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


def _run_device_stage(ctx, selected, devices):
    cases = ctx.cases
    runnable = [
        idx
        for idx in selected
        if _devices_for_case(cases[idx], idx, devices) is not None
    ]
    if not runnable:
        _log("parent", "no runnable cases for the given devices")
        return False

    if getattr(ctx.args, "persistent_domain", False):
        return _run_persistent_batch(ctx, selected, devices)

    overall = True
    for case_idx in runnable:
        callables, bundle_dir = _case_callables(
            ctx.test, ctx.args, ctx.extra_params, cases[case_idx], case_idx
        )
        case_ok = _run_case_on_devices(
            ctx,
            case_idx,
            cases[case_idx],
            _devices_for_case(cases[case_idx], case_idx, devices),
            (callables, bundle_dir),
        )
        overall = overall and case_ok
    _log("parent", f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


def run_with_stage(test, args, extra_params=None):
    extra_params = dict(extra_params or {})
    if getattr(args, "persistent_domain", False):
        if args.stage == _STAGE_COMPILE:
            _log(
                "parent",
                "--persistent-domain is an execution-time mode; incompatible "
                "with --stage compile",
                file=sys.stderr,
            )
            return False
    try:
        cases = _load_cases(args.csv, test.case_column_map())
        selected, _ = _select_cases(cases, args.cases)
        devices = _parse_devices(args.devices)
        if args.stage == _STAGE_COMPILE:
            return _run_compile_stage(test, args, extra_params, cases, selected)
        ctx = RunCtx(test, args, extra_params, cases)
        return _run_device_stage(ctx, selected, devices)
    except Exception as exc:
        _log(
            "parent",
            f"run failed: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return False


BASE_M = 256
BASE_N = 256
BASE_K = 128
MX_GROUP_SIZE = 64
SCALE_C0 = 2
KERNEL_NAME_PATTERN = "AllToAllQuantMatmulKernel"

FP8_BYTE_LO = 0x38
FP8_BYTE_HI = 0x40
SCALE_EXP_LO = 123
SCALE_EXP_HI = 128
BIAS_AMP = 4096.0

E5M2_BYTE_LO = 0x3C
E5M2_BYTE_HI = 0x40

E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = (E2M1_MAGNITUDES[1:] + E2M1_MAGNITUDES[:-1]) / 2


def decode_fp8_e5m2(u8):
    u = u8.astype(np.uint32)
    sign = (u >> 7) & 1
    exp = (u >> 2) & 0x1F
    mant = u & 0x3
    val = np.where(
        exp == 0, mant * 2.0**-16, (1.0 + mant / 4.0) * 2.0 ** (exp - 15)
    ).astype(np.float32)
    return np.where(sign == 1, -val, val).astype(np.float32)


def decode_fp8_e4m3fn(u8):
    bits = u8.astype(np.uint32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    is_nan = (exp == 15) & (mant == 7)
    val = np.where(
        exp == 0,
        mant.astype(np.float32) * 2.0**-9,
        (1.0 + mant.astype(np.float32) / 8.0) * 2.0 ** (exp.astype(np.float32) - 7.0),
    )
    val = np.where(sign == 1, -val, val)
    return np.where(is_nan, np.nan, val).astype(np.float32)


def decode_e8m0(u8):
    return 2.0 ** (u8.astype(np.int32) - 127)


def quantize_mxfp4(x_f32):
    rows, cols = x_f32.shape
    groups = (cols + 31) // 32
    padded = np.zeros((rows, groups * 32), dtype=np.float32)
    padded[:, :cols] = x_f32
    seg = padded.reshape(rows, groups, 32)
    amax = np.abs(seg).max(axis=2, keepdims=True)
    with np.errstate(divide="ignore"):
        exponent = np.ceil(np.log2(np.where(amax > 0, amax, 1e-30) / 6.0))
    exponent = np.clip(exponent, -127, 127).astype(np.int32)
    scale = np.exp2(exponent.astype(np.float32))
    code = np.searchsorted(E2M1_MIDPOINTS, np.abs(seg) / scale, side="right").astype(
        np.uint8
    )
    sign = (seg < 0).astype(np.uint8)
    nibbles = (code + sign * 8).astype(np.uint8).reshape(rows, -1)[:, :cols]
    scale_bytes = (exponent + 127).astype(np.uint8)[..., 0]
    return nibbles, scale_bytes.reshape(rows, -1)


def pack_fp4_cols(nibbles):
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)


def test_comm_workspace_uses_planned_ub_address():
    import all_to_all_quant_matmul as _impl

    module_tree = ast.parse(inspect.getsource(_impl))
    init_node = None
    for node in ast.walk(module_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_init_comm_config":
            init_node = node
            break
    workspace_buffer = None
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "comm_workspace_ub":
                workspace_buffer = node.value
    assert workspace_buffer is not None

    assert isinstance(workspace_buffer, ast.Call)
    assert all(keyword.arg != "addr" for keyword in workspace_buffer.keywords)


_GEN_CODEGEN_CTOR = dict(
    m=256,
    k=256,
    n=256,
    rank_size=2,
    comm_mode="urma",
    base_m=BASE_M,
    base_n=BASE_N,
    base_k=BASE_K,
    used_core_num=2,
    head_m=256,
)


def _compile_dumped_asc(monkeypatch, tmp_path, **ctor_kwargs):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    import cannbotdsl

    from all_to_all_quant_matmul import AllToAllQuantMatmul

    op = AllToAllQuantMatmul(**ctor_kwargs)
    compiled = cannbotdsl.compile(op.run, *op.aot_compile_specs())
    close = getattr(compiled, "close", None)
    if callable(close):
        close()
    [asc_path] = list(tmp_path.glob("*.asc"))
    return asc_path.read_text(encoding="utf-8")


@pytest.mark.ascendc_toolchain
def test_codegen_uses_default_ratio_and_one_aiv_communication_owner(
    monkeypatch,
    tmp_path,
):
    code = _compile_dumped_asc(monkeypatch, tmp_path, **_GEN_CODEGEN_CTOR)
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in code

    aiv_start = code.index("if ASCEND_IS_AIV")
    aiv_code = code[aiv_start:]
    subblock = re.search(
        r"int64_t (\w+) = asc_get_sub_block_id\(\);",
        aiv_code,
    )
    assert subblock is not None
    owner_cmp = re.search(
        rf"bool (\w+) = {subblock.group(1)} == 0;",
        aiv_code,
    )
    assert owner_cmp is not None
    owner_guard = re.search(
        rf"if \({owner_cmp.group(1)}\) \{{\s*.*?\.Init\(",
        aiv_code,
        re.DOTALL,
    )
    assert owner_guard is not None

    assert aiv_code.count("cannbotdsl_get_ffts_config(14, 0)") == 2


@pytest.mark.parametrize(
    "shape_kwargs,match",
    [
        ({"k": 3520, "rank_size": 4}, "rank_size\\*base_k"),
        ({"k": 3585, "rank_size": 2}, "rank_size\\*base_k"),
        ({"base_k": 100}, "base_k"),
        ({"base_n": 100}, "base_n"),
    ],
)
def test_k_tail_constraint_rejected_by_host_tiling(shape_kwargs, match):
    from all_to_all_quant_matmul import AllToAllQuantMatmul

    kwargs = dict(
        m=2048,
        k=3584,
        n=4096,
        rank_size=2,
        comm_mode="urma",
        base_m=BASE_M,
        base_n=BASE_N,
        base_k=BASE_K,
        used_core_num=8,
        head_m=512,
    )
    kwargs.update(shape_kwargs)
    with pytest.raises(ValueError, match=match):
        AllToAllQuantMatmul(**kwargs)


@pytest.mark.ascendc_toolchain
def test_codegen_with_bias_compiles(monkeypatch, tmp_path):
    ctor = dict(_GEN_CODEGEN_CTOR, has_bias=True)
    code = _compile_dumped_asc(monkeypatch, tmp_path, **ctor)
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in code


@pytest.mark.ascendc_toolchain
def test_codegen_ub_mode_compiles(monkeypatch, tmp_path):
    ctor = dict(_GEN_CODEGEN_CTOR, comm_mode="ub")
    code = _compile_dumped_asc(monkeypatch, tmp_path, **ctor)
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in code
    aiv_start = code.index("if ASCEND_IS_AIV")
    aiv_code = code[aiv_start:]

    assert aiv_code.count("cannbotdsl_get_ffts_config(14, 0)") == 2


def test_comm_mode_default_and_validated():
    from all_to_all_quant_matmul import AllToAllQuantMatmul

    with pytest.raises(ValueError, match="comm_mode"):
        AllToAllQuantMatmul(
            2048,
            3584,
            4096,
            2,
            "rdma",
            base_m=BASE_M,
            base_n=BASE_N,
            base_k=BASE_K,
            used_core_num=8,
            head_m=512,
        )
    op = AllToAllQuantMatmul(
        2048,
        3584,
        4096,
        2,
        base_m=BASE_M,
        base_n=BASE_N,
        base_k=BASE_K,
        used_core_num=8,
        head_m=512,
    )
    assert op.comm_mode == "urma"
    assert op.required_comm_buffer_bytes() > 0


def prearrange_b_zn(b_fp8, k, n, n_padded):
    elem_bytes = b_fp8.dtype.itemsize
    m0 = 32 // elem_bytes
    n0 = 16
    m_padded = ((k + 63) // 64) * 64
    m1 = m_padded // m0
    n1 = n_padded // n0
    b_padded = np.zeros((m1 * m0, n1 * n0), dtype=b_fp8.dtype)
    b_padded[:k, :n] = b_fp8
    b_fractal = b_padded.reshape(m1, m0, n1, n0).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(b_fractal).ravel()


def pad_axis(arr, length, axis):
    if arr.shape[axis] >= length:
        return arr.copy()
    shape = list(arr.shape)
    shape[axis] = length
    out = np.zeros(shape, dtype=arr.dtype)
    index = [slice(None)] * arr.ndim
    index[axis] = slice(arr.shape[axis])
    out[tuple(index)] = arr
    return out


class TestAllToAllQuantMatmul(DistributedOpTest):
    __test__ = False

    golden_is_rank_invariant = True
    _shared_case_b = None
    _compile_memo = None
    _ctx_seq = 0

    def case_column_map(self):
        return {
            "case": (0, str),
            "M": (1, int),
            "K": (2, int),
            "N": (3, int),
            "rank_size": (4, int),
            "head_m": (5, int),
            "bias": (6, int),
            "tb": (7, int),
            "out": (8, str),
            "adt": (9, str),
            "bdt": (10, str),
        }

    def compile_case(self, **params):
        rank_size = params["rank_size"]
        m = params.get("M", 0)
        k = params.get("K", 0)
        n = params.get("N", 0)
        block_dim = params.get("block_dim", 28)
        head_m = params.get("head_m")
        signature = (
            rank_size,
            m,
            k,
            n,
            block_dim,
            head_m,
            params.get("comm_mode", "urma"),
            bool(params.get("bias", 0)),
            params.get("out", "bf16"),
            params.get("adt", "e4m3"),
            params.get("bdt", "e4m3"),
        )
        memo = getattr(self, "_compile_memo", None)
        if memo is None:
            memo = self._compile_memo = {}
        if signature in memo:
            _log("parent", f"compile memo hit: {signature}")
            return memo[signature]

        callables = self._compile_op((m, k, n, rank_size, block_dim, head_m), params)
        memo[signature] = callables
        return callables

    def determine_precision_tolerance(self):
        return {"y": (1e-2, 1e-2)}

    def execute(self, prepare_res, **params):
        launches = params.get("launches", 1)
        for _ in range(launches):
            prepare_res["main"](
                prepare_res["data_npu"],
                prepare_res["barrier_npu"],
                prepare_res["a_npu"],
                prepare_res["sa_npu"],
                prepare_res["b_npu"],
                prepare_res["sb_npu"],
                prepare_res["y_npu"],
                prepare_res["bias_npu"],
            )
        torch.npu.synchronize()
        result = prepare_res["y_npu"].cpu().float().numpy()
        return {"y": result}

    def gen_inputs(self, rank_id, rank_size, **params):
        m = params["M"]
        k = params["K"]
        n = params["N"]
        ka = k // rank_size
        scale_ka = (ka + MX_GROUP_SIZE - 1) // MX_GROUP_SIZE * SCALE_C0
        tb = bool(params.get("tb", 0))
        adt = params.get("adt", "e4m3")
        bdt = params.get("bdt", "e4m3")
        has_bias = bool(params.get("bias", 0))
        b_q, b_scale, bias = self._gen_shared_b(k, n, bdt, tb, has_bias)
        np.random.seed(42 + rank_id)
        if adt == "fp4":
            a_q, a_s32 = quantize_mxfp4(
                np.random.uniform(-1, 1, (m, ka)).astype(np.float32)
            )
            a_scale = a_s32.reshape(m, scale_ka)
        else:
            lo, hi = (
                (E5M2_BYTE_LO, E5M2_BYTE_HI)
                if adt == "e5m2"
                else (FP8_BYTE_LO, FP8_BYTE_HI)
            )
            a_q = np.random.randint(lo, hi, (m, ka)).astype(np.uint8)
            a_scale = np.random.randint(
                SCALE_EXP_LO, SCALE_EXP_HI, (m, scale_ka)
            ).astype(np.uint8)

        return {
            "a_q": a_q,
            "a_scale": a_scale,
            "b_q": b_q,
            "b_scale": b_scale,
            "bias": bias,
            "tb": tb,
            "adt": adt,
            "bdt": bdt,
            "ka": ka,
            "scale_ka": scale_ka,
            "scale_k": (k + MX_GROUP_SIZE - 1) // MX_GROUP_SIZE * SCALE_C0,
        }

    def golden(self, rank_id, rank_size, all_inputs, **params):
        m = params["M"]
        k = params["K"]
        n = params["N"]
        ins = all_inputs[rank_id]
        ka = ins["ka"]
        tb = ins["tb"]
        adt = ins["adt"]
        bdt = ins["bdt"]

        def decode_data(q, scale, dt, cols):
            if dt == "fp4":
                mag = E2M1_MAGNITUDES[(q & 0x7).astype(np.int64)]
                v = np.where((q & 0x8) != 0, -mag, mag).astype(np.float32)
                return v * np.repeat(decode_e8m0(scale), 32, axis=1)[:, :cols]
            dec = decode_fp8_e5m2 if dt == "e5m2" else decode_fp8_e4m3fn
            return dec(q) * np.repeat(decode_e8m0(scale), 32, axis=1)[:, :cols]

        def decode_b(q, scale):
            if tb:
                q = q.T
                scale = scale.transpose(1, 0, 2)
            col = decode_e8m0(scale).transpose(1, 0, 2).reshape(n, -1)
            sb = np.repeat(col, 32, axis=1)[:, :k]
            if bdt == "fp4":
                mag = E2M1_MAGNITUDES[(q & 0x7).astype(np.int64)]
                v = np.where((q & 0x8) != 0, -mag, mag).astype(np.float32)
                return v * sb.T
            dec = decode_fp8_e5m2 if bdt == "e5m2" else decode_fp8_e4m3fn
            return dec(q) * sb.T

        a_full = np.zeros((m, k), dtype=np.float32)
        for r in range(rank_size):
            r_in = all_inputs[r]
            col_lo, col_hi = r * ka, (r + 1) * ka
            a_full[:, col_lo:col_hi] = decode_data(
                r_in["a_q"], r_in["a_scale"], adt, ka
            )

        b_deq = decode_b(all_inputs[0]["b_q"], all_inputs[0]["b_scale"])
        bias = all_inputs[0]["bias"]

        golden_out = np.matmul(a_full, b_deq).astype(np.float32)
        if bias is not None:
            golden_out = golden_out + bias[None, :]
        return {"y": golden_out}

    def worker_prepare(self, ctx, inputs, callables=None, **params):
        rank_id = ctx.rank_id
        m = params.get("M", 0)
        k = params.get("K", 0)
        n = params.get("N", 0)

        data_npu, barrier_npu = self._make_comm_contexts(ctx, params, callables)

        _log(f"rank {rank_id}", "device acquired, creating rank-local state...")

        tb = inputs["tb"]
        adt = inputs["adt"]
        bdt = inputs["bdt"]
        out_t = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            params.get("out", "bf16")
        ]

        m_pad = (m + BASE_M - 1) // BASE_M * BASE_M
        n_pad = (n + BASE_N - 1) // BASE_N * BASE_N
        a_bytes = pack_fp4_cols(inputs["a_q"]) if adt == "fp4" else inputs["a_q"]
        a_npu = torch.from_numpy(pad_axis(a_bytes, m_pad, 0)).npu()
        sa_npu = torch.from_numpy(pad_axis(inputs["a_scale"], m_pad, 0)).npu()

        b_q = inputs["b_q"].T if tb else inputs["b_q"]
        b_scale = inputs["b_scale"].transpose(1, 0, 2) if tb else inputs["b_scale"]
        if bdt == "fp4":
            b_bytes = pack_fp4_cols(pad_axis(b_q, n_pad, 1))
            b_npu = torch.from_numpy(b_bytes.copy()).npu().view(k, n_pad // 2)
        else:
            b_fractal = prearrange_b_zn(b_q, k, n, n_pad)
            b_npu = torch.from_numpy(b_fractal.copy()).npu().view(k, n_pad)
        sb_npu = torch.from_numpy(pad_axis(b_scale, n_pad, 1)).npu()
        y_npu = torch.zeros(m, n, dtype=out_t).npu()

        bias_npu = (
            torch.from_numpy(pad_axis(inputs["bias"], n_pad, 0)).npu()
            if inputs["bias"] is not None
            else torch.empty(0, dtype=torch.float32).npu()
        )
        return {
            "main": callables["main"],
            "data_npu": data_npu,
            "barrier_npu": barrier_npu,
            "a_npu": a_npu,
            "sa_npu": sa_npu,
            "b_npu": b_npu,
            "sb_npu": sb_npu,
            "y_npu": y_npu,
            "bias_npu": bias_npu,
        }

    def _compile_op(self, shape, params):
        m, k, n, rank_size, block_dim, head_m = shape
        import cannbotdsl

        from all_to_all_quant_matmul import AllToAllQuantMatmul

        op = AllToAllQuantMatmul(
            m,
            k,
            n,
            rank_size,
            params.get("comm_mode", "urma"),
            base_m=BASE_M,
            base_n=BASE_N,
            base_k=BASE_K,
            used_core_num=block_dim,
            head_m=head_m,
            has_bias=bool(params.get("bias", 0)),
            out_dtype=params.get("out", "bf16"),
            a_dtype=params.get("adt", "e4m3"),
            b_dtype=params.get("bdt", "e4m3"),
        )
        _log(
            "parent",
            "compiling artifact bundle (rank-neutral, no device)...",
        )
        main = cannbotdsl.compile(op.run, *op.aot_compile_specs())
        return {
            "main": main,
            "data_buffer_bytes": op.required_comm_buffer_bytes(),
        }

    def _gen_shared_b(self, k, n, bdt, tb, has_bias):
        def gen_data(rows, cols, dt):
            if dt == "fp4":
                nib, _ = quantize_mxfp4(
                    np.random.uniform(-1, 1, (rows, cols)).astype(np.float32)
                )
                return nib
            lo, hi = (
                (E5M2_BYTE_LO, E5M2_BYTE_HI)
                if dt == "e5m2"
                else (FP8_BYTE_LO, FP8_BYTE_HI)
            )
            return np.random.randint(lo, hi, (rows, cols)).astype(np.uint8)

        n_groups = k // MX_GROUP_SIZE
        shared_key = (k, n, bdt, tb, has_bias)
        shared = getattr(self, "_shared_case_b", None)
        if shared is None or shared[0] != shared_key:
            np.random.seed(42)
            b_q = gen_data(n, k, bdt) if tb else gen_data(k, n, bdt)
            b_scale = np.random.randint(
                SCALE_EXP_LO,
                SCALE_EXP_HI,
                (n, n_groups, 2) if tb else (n_groups, n, 2),
            ).astype(np.uint8)
            bias = (
                np.random.uniform(-BIAS_AMP, BIAS_AMP, (n,)).astype(np.float32)
                if has_bias
                else None
            )
            shared = self._shared_case_b = (shared_key, (b_q, b_scale, bias))
        return shared[1]

    def _make_comm_contexts(self, ctx, params, callables):
        from cannbotdsl.ops.distributed import (
            CommContext,
            CommEngine,
            BufferSource,
            get_comm_context,
        )

        ctx_reuse = os.environ.get("A2A_CTX_REUSE") == "1"
        ctx_seq = getattr(self, "_ctx_seq", 0)
        self._ctx_seq = ctx_seq + 1
        if params["comm_mode"] == "urma":
            data_op_name = "a2a_data"
        else:
            data_op_name = "a2a_data" if ctx_reuse else f"a2a_data_s{ctx_seq}"
        barrier_op_name = "a2a_barrier"
        if params["comm_mode"] == "ub":
            data_engine = CommEngine(
                protocol=CommEngine.Protocol.UBMEM,
                buffer_source=BufferSource.SELF_REGISTERED,
                buffer_size=callables["data_buffer_bytes"],
            )
        else:
            data_engine = CommEngine(
                protocol=CommEngine.Protocol.URMA,
                buffer_source=BufferSource.HCCL,
            )
        data_npu = get_comm_context(
            CommContext, ctx.group_name, op_name=data_op_name, comm_engine=data_engine
        )
        barrier_engine = CommEngine(
            protocol=CommEngine.Protocol.UBMEM,
            buffer_source=BufferSource.SELF_REGISTERED,
            buffer_size=2 * 1024 * 1024,
        )
        barrier_npu = get_comm_context(
            CommContext,
            ctx.group_name,
            op_name=barrier_op_name,
            comm_engine=barrier_engine,
        )
        return data_npu, barrier_npu


if __name__ == "__main__":
    test = TestAllToAllQuantMatmul()
    test.set_kernel_name_pattern(KERNEL_NAME_PATTERN)
    test.parser.add_argument(
        "--comm-mode",
        default="urma",
        choices=["urma", "ub"],
        help="data-plane comm mode: urma (put/flush) / ub (peer UBMEM "
        "write) (default: urma)",
    )
    test.parser.add_argument(
        "--head_m",
        type=int,
        default=None,
        help="head_m / comm tile M-span (default: formula)",
    )
    test.parser.add_argument(
        "--launches",
        type=int,
        default=1,
        help="consecutive launches per correctness execution (default: 1)",
    )
    add_stage_arguments(test.parser)
    args = test.parser.parse_args()
    if args.launches <= 0:
        test.parser.error("--launches must be a positive integer")
    extra_params = {"comm_mode": args.comm_mode}
    if args.head_m is not None:
        extra_params["head_m"] = args.head_m
    if args.launches != 1:
        extra_params["launches"] = args.launches
    ok = run_with_stage(test, args, extra_params=extra_params)
    sys.exit(0 if ok else 1)
