# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""MegaReindex contracts, device accuracy, Native publication and profiling.

Run this file with test, smoke, verify-pa, verify-pa-batch, benchmark,
capture, compare or analyze-profile. Importing the host checks needs no NPU.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import runpy
import statistics
import sys
import time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "samples/attention/mega_reindex"
SOURCE = SAMPLE / "mega_reindex.py"


@lru_cache(maxsize=None)
def _load_sample(filename):
    spec = importlib.util.spec_from_file_location(
        "reindex_samples_path", ROOT / "test/_samples_path.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    return loader.load_sample("attention/mega_reindex/" + filename)


@pytest.fixture
def native_namespace(monkeypatch):
    """Exercise real export functions while replacing only the compile backend."""
    source = SAMPLE / "mega_reindex.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_native_signature", "export_mega_reindex"}
    ]
    assert len(functions) == 2
    for node in functions:
        node.decorator_list = []
    # Reuse the operator's fixed numeric dimensions, without importing runtime
    # adapters or duplicating their values in the fixture.
    runtime = ast.parse((SAMPLE / "reindex_runtime.py").read_text(encoding="utf-8"))
    constants = []
    for node in runtime.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == "FRACTAL_NZ" for t in node.targets
            ):
                break
            constants.append(node)
    calls, closed, clears = [], [], []

    def compile_profile(*args):
        calls.append(args)
        return SimpleNamespace(close=lambda: closed.append(args))

    compile_profile.cache_clear = lambda: clears.append(True)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "publication must not allocate tensors or query device quotas"
        )

    namespace = dict(
        math=math,
        torch=SimpleNamespace(
            bfloat16="bf16",
            float32="f32",
            uint8="u8",
            int8="i8",
            int32="i32",
            empty=forbidden,
            npu=SimpleNamespace(current_device=forbidden),
        ),
        SCALE_DTYPE="e8m0",
        check_platform=forbidden,
        compile_mega_reindex=compile_profile,
    )
    exec(
        compile(
            ast.Module(body=constants + functions, type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    monkeypatch.delenv("CANNBOTDSL_DS41_PROFILES", raising=False)
    return SimpleNamespace(
        namespace=namespace, calls=calls, closed=closed, clears=clears
    )


def _publish(state):
    state.namespace["export_mega_reindex"]()


def _override(monkeypatch, tmp_path, profiles):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profiles), encoding="utf-8")
    monkeypatch.setenv("CANNBOTDSL_DS41_PROFILES", str(path))


def test_mega_reindex_default_profiles(native_namespace):
    state = native_namespace
    _publish(state)
    assert len(state.calls) == len(state.closed) == 6
    assert state.closed == state.calls
    assert {(c[0], c[1], c[2], c[3], c[-2], c[-1]) for c in state.calls} == {
        (seq, 1, 8192, 512, page, 32) for seq in (6, 8, 16) for page in (0, 128)
    }
    assert all(c[4:7] == (1e-20, 1 / 64, 0) for c in state.calls)


def test_mega_reindex_profile_override(native_namespace, monkeypatch, tmp_path):
    state = native_namespace
    _override(
        monkeypatch,
        tmp_path,
        {
            "mega_reindex": [
                dict(
                    batch=2,
                    seq_len=6,
                    length=4096,
                    topk=256,
                    page_size=64,
                    cube_cores=28,
                    norm_eps=1e-6,
                    softmax_scale=0.25,
                )
            ],
        },
    )
    _publish(state)
    assert len(state.calls) == len(state.closed) == 1
    call = state.calls[0]
    assert call[:7] == (12, 2, 4096, 256, 1e-6, 0.25, 0)
    assert call[-2:] == (64, 28)


def test_other_operator_override_preserves_defaults(
    native_namespace, monkeypatch, tmp_path
):
    _override(monkeypatch, tmp_path, {"mega_mla": []})
    _publish(native_namespace)
    assert len(native_namespace.calls) == 6


@pytest.mark.parametrize(
    "profile",
    [
        {"batch": 0},
        {"seq_len": 3},
        {"seq_len": True},
        {"cube_cores": 1},
        {"topk": 65},
        {"topk": 16384},
        {"length": 16385},
        {"page_size": 17},
        {"page_size": 2048},
        {"page_size": 48},
        {"norm_eps": 0},
        {"norm_eps": float("nan")},
        {"softmax_scale": float("inf")},
    ],
)
def test_mega_reindex_rejects_invalid_profiles(
    native_namespace, monkeypatch, tmp_path, profile
):
    state = native_namespace
    _override(monkeypatch, tmp_path, {"mega_reindex": [profile]})
    with pytest.raises(ValueError, match="Invalid mega_reindex export profile"):
        _publish(state)
    assert state.calls == []
    assert len(state.clears) == 2  # Entry and finally, including validation errors.


def test_mega_reindex_rejects_unknown_fields(native_namespace, monkeypatch, tmp_path):
    _override(monkeypatch, tmp_path, {"mega_reindex": [{"seq_length": 8}]})
    with pytest.raises(ValueError, match="Unknown mega_reindex profile fields"):
        _publish(native_namespace)
    assert native_namespace.calls == []


@pytest.mark.parametrize("page", [0, 128])
def test_mega_reindex_export_tensor_contract(native_namespace, page):
    specs = native_namespace.namespace["_native_signature"](12, 2, 8192, 512, page)
    assert len(specs) == (42 if page else 39)
    assert specs[0] == ("tensor", (12, 5120), "bf16", (5120, 1))
    assert specs[7][:3] == ("tensor", (5120, 1280), "i8")
    assert specs[18][:3] == ("tensor", (4096, 1280), "i8")
    assert specs[24] == ("tensor", (16384, 1, 64), "u8", (64, 64, 1))
    assert specs[25][:3] == ("tensor", (16384, 1, 2, 2), "e8m0")
    assert specs[30] == ("tensor", (6144,), "i32", (1,))
    assert specs[31:33] == (("int64",), ("int64",))
    assert specs[33][:3] == ("tensor", (12, 1, 2048), "i32")
    assert specs[35][:3] == ("tensor", (3,), "i32")
    if page:
        assert specs[-3] == ("tensor", (128, 16, 544), "u8", (8704, 544, 1))
        assert specs[-2] == ("tensor", (2, 64), "i32", (64, 1))
        assert specs[-1] == specs[28] == ("tensor", (12,), "i32", (1,))


def test_mega_reindex_dependencies_and_ds41_registration():
    spec = importlib.util.spec_from_file_location(
        "reindex_packager", ROOT / "packaging/native/build.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    operators, groups = builder.load_catalog()
    registered = {Path(path).name for path in operators["mega_reindex"]}
    assert registered == {path.name for path in SAMPLE.glob("*.py")}
    assert groups["ds41"].count("mega_reindex") == 1
    assert builder.choose_operators(
        builder.parse_args(["--operator", "mega_reindex"]), operators, groups
    ) == (["mega_reindex"], Path("operators/mega_reindex"))


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_public_symbols_and_argument_contract():
    tree = _tree()
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "MegaReindex" in classes
    assert "MegaReindexTailWorkspace" in classes

    expected = (
        "x",
        "wqa",
        "descale_wqa",
        "ww",
        "norm_weight",
        "wqb",
        "descale_wqb",
        "rope",
        "key",
        "descale_key",
        "front_workspace",
        "tail_workspace",
        "norm_eps",
        "softmax_scale",
        "topk",
        "candidate_block_indices",
        "candidate_block_length",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "candidate_block_size",
        "mask_mode",
        "cmp_ratio",
        "layout_k",
        "block_table",
        "k",
        "descale_k",
    )
    for name in (
        "mega_reindex",
        "prepare_mega_reindex",
    ):
        fn = functions[name]
        actual = tuple(arg.arg for arg in fn.args.args + fn.args.kwonlyargs)
        assert actual == expected
        assert fn.args.vararg is None
        assert fn.args.kwarg is None
        # Preserve the source PR's optional arguments and their exact defaults.
        expected_defaults = {
            "key": "None",
            "descale_key": "None",
            "front_workspace": "None",
            "tail_workspace": "None",
            "norm_eps": "1e-20",
            "softmax_scale": "1 / 64",
            "topk": "512",
            "cu_seqlens_k": "None",
            "candidate_block_size": "8",
            "mask_mode": "3",
            "cmp_ratio": "1",
            "layout_k": '"TND"',
            "block_table": "None",
            "k": "None",
            "descale_k": "None",
        }
        defaults = dict(
            zip(
                (arg.arg for arg in fn.args.args[-len(fn.args.defaults) :]),
                fn.args.defaults,
            )
        )
        defaults.update(
            (arg.arg, value)
            for arg, value in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
            if value is not None
        )
        assert {key: ast.dump(value) for key, value in defaults.items()} == {
            key: ast.dump(ast.parse(value, mode="eval").body)
            for key, value in expected_defaults.items()
        }


def test_documented_supported_layers_and_contract():
    readme = (SAMPLE / "README.md").read_text(encoding="utf-8")
    for layer in ("L24", "L28", "L32", "L36"):
        assert layer in readme
    for name in (
        "candidate_block_indices",
        "candidate_block_length",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "norm_weight",
    ):
        assert name in readme


def test_standalone_import_dependencies():
    forbidden = {"indexer_front", "indexer_tail", "_indexer_common"}
    for source in SAMPLE.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] in forbidden for a in node.names), (
                    source
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in forbidden, source


def test_resource_checks_follow_device_and_stream_on_every_launch():
    import contextlib
    from types import SimpleNamespace

    state = {"device": 0, "cores": 32, "launches": []}
    observations = []

    @contextlib.contextmanager
    def device_context(device):
        previous = state["device"]
        state["device"] = device
        try:
            yield
        finally:
            state["device"] = previous

    def platform(**kwargs):
        observations.append((state["device"], kwargs["stream"]))
        cores = state["cores"] if state["device"] == 0 else 24
        return SimpleNamespace(cube_core_num=cores, vector_core_num=2 * cores)

    tree = ast.parse((SAMPLE / "reindex_runtime.py").read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "VECTOR_SUBBLOCKS"
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"check_platform", "ResourceCheckedProgram"}
        )
    ]
    namespace = {
        "torch": SimpleNamespace(
            npu=SimpleNamespace(
                device=device_context,
                current_stream=lambda: (state["device"], state["cores"]),
            )
        ),
        "cb": SimpleNamespace(get_platform_info=platform),
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "guard", "exec"),
        namespace,
    )
    compiled = []

    def build(cores):
        compiled.append(cores)
        return lambda: state["launches"].append(cores)

    checked = namespace["ResourceCheckedProgram"](build(32), 0, 32, build)
    checked()
    state["cores"] = 28
    checked()
    checked()
    assert compiled == [32, 28] and state["launches"] == [32, 28, 28]
    other = namespace["ResourceCheckedProgram"](build(32), 1, 32, build)
    other()
    assert state["launches"][-1] == 24
    assert state["device"] == 0 and observations[-1][0] == 1
    state["cores"] = 1
    with pytest.raises(RuntimeError, match="at least 2 Cube"):
        checked()
    assert state["launches"] == [32, 28, 28, 24]


def test_prepare_cache_hits_recheck_input_device_and_current_stream():
    import contextlib
    import functools
    from types import SimpleNamespace

    state = {"device": 0, "cores": 32, "checks": [], "compiles": []}

    @contextlib.contextmanager
    def device_context(device):
        previous = state["device"]
        state["device"] = device.index
        try:
            yield
        finally:
            state["device"] = previous

    def check():
        cores = state["cores"] if state["device"] == 0 else 24
        state["checks"].append((state["device"], cores))
        return cores

    @functools.lru_cache()
    def compile_program(*args):
        state["compiles"].append(args[-1])
        return object()

    workspace = SimpleNamespace(rq=None, rs=None, w=None)
    prepare = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_mega_reindex"
    )
    namespace = {
        "torch": SimpleNamespace(
            npu=SimpleNamespace(device=device_context), uint8="uint8"
        ),
        "functools": functools,
        "prepare_front": lambda *args, **kwargs: ((), workspace),
        "prepare_tail": lambda *args, **kwargs: ((), workspace),
        "check_platform": check,
        "signature": lambda args: (),
        "compile_mega_reindex": compile_program,
        "ResourceCheckedProgram": lambda program, device, cores, compiler: program,
    }
    exec(
        compile(ast.Module(body=[prepare], type_ignores=[]), "prepare", "exec"),
        namespace,
    )
    call = namespace[prepare.name]
    x = SimpleNamespace(shape=(8, 5120), device=SimpleNamespace(index=0))
    key = SimpleNamespace(shape=(8192, 1, 64))
    options = dict(
        candidate_block_indices=None,
        candidate_block_length=None,
        cu_seqlens_q=None,
        cu_seqlens_k=object(),
        seqused_q=SimpleNamespace(numel=lambda: 1),
        seqused_k=None,
    )
    args = (
        x,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        key,
        SimpleNamespace(dtype="uint8"),
    )
    call(*args, **options)
    call(*args, **options)
    assert state["compiles"] == [32] and len(state["checks"]) == 2
    state["cores"] = 28
    call(*args, **options)
    x.device = SimpleNamespace(index=1)
    call(*args, **options)
    assert state["compiles"] == [32, 28, 24] and state["checks"][-1] == (1, 24)
    assert state["device"] == 0


@pytest.mark.npu
def test_internal_imports_accept_unused_invalid_l2_override(monkeypatch):
    pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    import sys

    sys.path.insert(0, str(ROOT / "test"))
    from _samples_path import load_sample

    monkeypatch.setenv("QBMM_MX_L2_SIZE_BYTES", "invalid")
    for filename in ("reindex_matmul.py", "reindex_mx.py"):
        module = load_sample(f"attention/mega_reindex/{filename}")
        for name in getattr(module, "__all__", ()):
            assert hasattr(module, name)


@pytest.fixture
def npu_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    if not hasattr(torch.npu, "set_stream_limit"):
        pytest.skip("torch_npu stream resource limits unavailable")
    return torch


@pytest.mark.parametrize("capacity", [8192, 11999, 16384, 1_048_576, 1_048_704])
def test_pa_workspace_allocation(capacity):
    pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    workspace_type = _load_sample("mega_reindex.py").MegaReindexTailWorkspace
    workspace = workspace_type(8, capacity, "cpu")
    storage = min(capacity, 16384)
    padded = (storage + 255) // 256 * 256
    assert workspace.kv_s == capacity
    assert workspace.storage_kv_s == storage
    assert workspace.score.shape == (8, padded)
    assert workspace.block_mask.shape == (8, padded // 8)
    assert workspace.indices.shape == (8, 512)


@pytest.mark.npu
@pytest.mark.parametrize("capacity", [8192, 1_048_576, 1_048_704])
def test_pa_workspace_logical_and_compact_capacity(capacity, npu_runtime):
    torch = npu_runtime
    op = _load_sample("mega_reindex.py")
    device = f"npu:{torch.npu.current_device()}"
    x = torch.empty((8, 1), dtype=torch.bfloat16, device=device)
    key = torch.empty((1, 16, 544), dtype=torch.uint8, device=device)
    table = torch.zeros((1, capacity // 128), dtype=torch.int32, device=device)
    usedq = torch.tensor([8], dtype=torch.int32, device=device)
    usedk = torch.tensor([min(capacity, 1_048_578)], dtype=torch.int32, device=device)
    length = min(capacity, 16384)

    def prepare(workspace):
        return op.prepare_pa(x, key, None, table, usedq, usedk, None, workspace, 512)

    for declared in dict.fromkeys((None, length, capacity)):
        workspace = (
            None
            if declared is None
            else op.MegaReindexTailWorkspace(8, declared, device)
        )
        packed, scale, cuk, actual, pa_args, page_size = prepare(workspace)
        assert workspace is None or actual is workspace
        assert actual.kv_s == (length if declared is None else declared)
        assert actual.storage_kv_s == length
        assert packed.shape == (length, 1, 64)
        assert scale.shape == (length, 1, 2, 2)
        assert cuk.cpu().tolist() == [0, length]
        assert page_size == 128
        assert pa_args[-1] is actual.pa_map
        assert actual.pa_map.shape == ((8, 2049) if capacity > length else (8,))

    if capacity > length:
        mismatched = length + 128
        for rows, declared, placement, topk in (
            (8, mismatched, device, 512),
            (6, capacity, device, 512),
            (8, capacity, device, 256),
            (8, capacity, "cpu", 512),
        ):
            workspace = op.MegaReindexTailWorkspace(
                rows, declared, placement, topk=topk
            )
            with pytest.raises(ValueError, match="PA workspace T/capacity"):
                prepare(workspace)

        workspace = op.MegaReindexTailWorkspace(8, capacity, device)
        del workspace.storage_kv_s
        with pytest.raises(ValueError, match="storage capacity mismatch; recreate"):
            prepare(workspace)


@pytest.mark.npu
def test_real_stream_quota_is_checked_without_launch(npu_runtime):
    import sys

    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, str(ROOT / "test"))
    from _samples_path import load_sample

    runtime = load_sample("attention/mega_reindex/reindex_runtime.py")
    stream = torch.npu.Stream()
    compiled, called = [], []

    def build(cores):
        compiled.append(cores)
        return lambda: called.append(cores)

    initial = runtime.check_platform()
    checked = runtime.ResourceCheckedProgram(
        build(initial), torch.npu.current_device(), initial, build
    )
    with torch.npu.stream(stream):
        previous = torch.npu.get_stream_limit(stream)
        try:
            for cores in (28, 16, 28):
                torch.npu.set_stream_limit(stream, cube_num=cores, vector_num=2 * cores)
                checked()
                assert called[-1] == cores
        finally:
            torch.npu.set_stream_limit(
                stream,
                cube_num=previous["cube_core_num"],
                vector_num=previous["vector_core_num"],
            )
    assert called == [28, 16, 28]


@pytest.mark.npu
@pytest.mark.parametrize("command", ["smoke", "verify-pa", "verify-pa-batch"])
def test_device_regression(command, npu_runtime):
    """Keep device cases isolated and report each failure independently."""
    import subprocess

    try:
        completed = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        pytest.fail(f"{command} timed out after 900 seconds:\n{output}", pytrace=False)
    assert completed.returncode == 0, f"{command}:\n{completed.stdout}"
    print(completed.stdout)


@lru_cache(maxsize=1)
def _query_modules():
    pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    return _load_sample("reindex_query.py"), _load_sample("reindex_mx.py")


@pytest.mark.parametrize("cores", [2, 8, 16, 28, 32, 64])
def test_fixed_query_tile_accepts_core_profiles(monkeypatch, cores):
    query, mx = _query_modules()
    # Reproduce the old 28-core selector mismatch without a physical NPU.
    monkeypatch.setattr(mx, "AIC_NUM", cores)
    tile = query.get_query_tile_config(cores)
    phase = query.TailQueryPhase(24, 4096, 1280, tile, cores)
    assert (phase.base_m, phase.base_n, phase.base_k) == (16, 128, 256)
    assert phase.core_count == cores
    assert (phase.block_scheduler.m_tiles, phase.block_scheduler.n_tiles) == (2, 32)
    assert phase.k_l0_tiles == 5
    assert tile.full_load and tile.k_l1 == tile.scale_k_l1 == 1280
    assert tile.used_core_num == min(cores, 32)
    assert (tile.m_base_tail_split_count, tile.n_base_tail_split_count) == (1, 1)


@pytest.mark.parametrize("cores", [0, -1])
def test_invalid_query_quota(cores):
    query, _ = _query_modules()
    with pytest.raises(ValueError):
        query.get_query_tile_config(cores)


@pytest.mark.npu
@pytest.mark.parametrize("dim", [31, 64, 65, 1279, 1280])
def test_rms_norm_tail(dim, npu_runtime):
    import cannbotdsl as cb
    from cannbotdsl import dtypes as dt, simt

    torch = npu_runtime
    _front = _load_sample("reindex_front.py")
    _runtime = _load_sample("reindex_runtime.py")

    class NormProbe:
        def __init__(self, dim):
            self.dim = dim
            self.norm = _front.NormPhase(1, 1e-20, 1 / 64, 1)
            # Exercise only the normalization helper; MX quantization has its own ABI.
            self.norm.norm_dim = dim
            self.norm.norm_chunks = dim // _front.NORM_VECTOR_LANES
            self.norm.norm_tail = dim % _front.NORM_VECTOR_LANES
            self.padded = (
                (dim + _front.NORM_VECTOR_LANES - 1)
                // _front.NORM_VECTOR_LANES
                * _front.NORM_VECTOR_LANES
            )

        @cb.kernel
        def kernel(self, x: cb.Tensor, gamma: cb.Tensor, out: cb.Tensor):
            xx = _runtime.Buffer(cb.MemLoc.UB, (1, self.padded), dt.bfloat16)
            gg = _runtime.Buffer(cb.MemLoc.UB, (1, self.padded), dt.bfloat16)
            yy = _runtime.Buffer(cb.MemLoc.UB, (1, self.padded), dt.bfloat16)
            tmp = _runtime.Buffer(cb.MemLoc.UB, (_front.NORM_VECTOR_LANES,), dt.float32)
            # Poison unused lanes so accidental participation changes the answer.
            with cb.vf(mode="simt", thread=64):
                tid = dt.int32(simt.thread_idx()[0])
                for i in range(tid, self.padded, 64):
                    xx[0, i] = dt.bfloat16(10000.0)
                    gg[0, i] = dt.bfloat16(10000.0)
            cb.mem_copy(xx.view(self.padded)[: self.dim,], x)
            cb.mem_copy(gg.view(self.padded)[: self.dim,], gamma)
            self.norm._normalize(xx, gg, yy, tmp)
            cb.mem_copy(out[: self.dim,], yy.view(self.padded)[: self.dim,])

        @cb.host
        def run(self, x, gamma, out):
            self.kernel[1](x, gamma, out)

    op = NormProbe(dim)
    program = cb.compile(
        op.run,
        cb.TensorSpec((dim,), dt.bfloat16),
        cb.TensorSpec((dim,), dt.bfloat16),
        cb.TensorSpec((dim + 16,), dt.bfloat16),
    )
    torch.manual_seed(dim)
    gamma = (torch.rand(dim) + 0.5).bfloat16()
    scrub = torch.ones(200_000_000, device="npu", dtype=torch.float32)
    for x in (torch.ones(dim, dtype=torch.bfloat16), torch.randn(dim).bfloat16()):
        out = torch.full((dim + 16,), -1234.0, device="npu", dtype=torch.bfloat16)
        torch.sum(scrub)
        torch.npu.synchronize()
        program(x.npu(), gamma.npu(), out)
        torch.npu.synchronize()
        expected = (
            x.float()
            * torch.sqrt(1 / (x.float().square().mean() + 1e-20))
            * gamma.float()
        ).bfloat16()
        torch.testing.assert_close(out[:dim].cpu(), expected, rtol=0.008, atol=1e-4)
        assert torch.equal(
            out[dim:].cpu(), torch.full((16,), -1234.0, dtype=torch.bfloat16)
        )


def _make_case(compile_program=True):
    """Prepare shared seeded inputs, optionally compiling the TND program."""
    import json
    import time
    from importlib.metadata import version
    from pathlib import Path

    import cannbotdsl
    import torch

    def load_sample(path):
        return _load_sample(Path(path).name)

    _sample = load_sample("attention/mega_reindex/mega_reindex.py")
    FrontWorkspace = _sample.FrontWorkspace
    MegaReindexTailWorkspace = _sample.MegaReindexTailWorkspace
    prepare_mega_reindex = _sample.prepare_mega_reindex
    _front = load_sample("attention/mega_reindex/reindex_front.py")
    _formats = load_sample("attention/mega_reindex/reindex_front.py")
    prepare_weight_nz = _front.prepare_weight_nz
    prepare_wqa_nz = _formats.prepare_wqa_nz
    prepare_wqb_nz = _formats.prepare_wqb_nz

    B, S, KV_S, TOPK, SEED = 1, 8, 8192, 512, 2026
    T = B * S
    torch.npu.set_device(0)
    torch.manual_seed(SEED)

    cpu = {
        "x": torch.randn(T, 5120).to(torch.bfloat16),
        "wa": (torch.randn(1280, 5120) * 0.02).to(torch.float8_e4m3fn),
        "sa": torch.full((1280, 80, 2), 127, dtype=torch.uint8),
        "wb": (torch.randn(4096, 1280) * 0.02).to(torch.float8_e4m3fn),
        "sb": torch.full((4096, 20, 2), 127, dtype=torch.uint8),
        "ww": (torch.randn(32, 5120) * 0.02).to(torch.bfloat16),
        "g": torch.ones(1280, dtype=torch.bfloat16),
        "rope": torch.stack((torch.ones(T, 32), torch.zeros(T, 32)), dim=-1),
    }
    packed_fixture = (
        torch.randint(0, 256, (KV_S, 1, 64), dtype=torch.uint8),
        torch.full((KV_S, 1, 2, 2), 127, dtype=torch.uint8),
    )
    p = {name: value.npu() for name, value in cpu.items()}
    wqa, descale_wqa = prepare_wqa_nz(p["wa"], p["sa"])
    wqb, descale_wqb = prepare_wqb_nz(p["wb"], p["sb"])
    ww = prepare_weight_nz(p["ww"])
    key, descale_key = packed_fixture
    key, descale_key = key.npu(), descale_key.npu()

    num_blocks = (KV_S + 7) // 8
    candidate_block_indices = torch.full((T, 1, 2048), -1, dtype=torch.int32)
    candidate_block_indices[:, 0, :num_blocks] = torch.arange(
        num_blocks, dtype=torch.int32
    )
    candidate_block_indices = candidate_block_indices.npu()
    candidate_block_length = torch.full(
        (T, 1), num_blocks, dtype=torch.int32, device="npu"
    )
    cu_seqlens_q = torch.tensor([0, S], dtype=torch.int32, device="npu")
    cu_seqlens_k = torch.tensor([0, KV_S], dtype=torch.int32, device="npu")
    seqused_q = torch.tensor([S], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor([KV_S], dtype=torch.int32, device="npu")

    front_workspace = FrontWorkspace(T, p["x"].device)
    # Detect a Cube store that writes rows beyond the logical weights projection.
    w_raw = front_workspace.w_raw
    guard_elements = 16 * w_raw.shape[1]
    # 113.0 is an arbitrary sentinel that FP32 represents exactly.
    w_raw_backing = w_raw.new_full((w_raw.numel() + guard_elements,), 113.0)
    front_workspace.w_raw = w_raw_backing[: w_raw.numel()].view_as(w_raw)
    w_raw_guard = w_raw_backing[w_raw.numel() :]
    tail_workspace = MegaReindexTailWorkspace(T, KV_S, p["x"].device, topk=TOPK)

    if compile_program:
        compile_start = time.perf_counter()
        program, args, _, _ = prepare_mega_reindex(
            p["x"],
            wqa,
            descale_wqa,
            ww,
            p["g"],
            wqb,
            descale_wqb,
            p["rope"],
            key,
            descale_key,
            front_workspace=front_workspace,
            tail_workspace=tail_workspace,
            norm_eps=1e-20,
            softmax_scale=1 / 64,
            topk=TOPK,
            candidate_block_indices=candidate_block_indices,
            candidate_block_length=candidate_block_length,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            candidate_block_size=8,
            mask_mode=3,
            cmp_ratio=1,
        )
        compile_seconds = time.perf_counter() - compile_start

    # run_smoke() reads these entries from the returned mapping; naming them
    # here keeps the locals() contract visible to linters.
    return dict(locals(), json=json, version=version, cannbotdsl=cannbotdsl)


def run_smoke():
    case = _make_case()
    B = case["B"]
    KV_S = case["KV_S"]
    S = case["S"]
    T = case["T"]
    TOPK = case["TOPK"]
    _sample = case["_sample"]
    args = case["args"]
    candidate_block_indices = case["candidate_block_indices"]
    candidate_block_length = case["candidate_block_length"]
    cannbotdsl = case["cannbotdsl"]
    compile_seconds = case["compile_seconds"]
    cu_seqlens_k = case["cu_seqlens_k"]
    cu_seqlens_q = case["cu_seqlens_q"]
    descale_key = case["descale_key"]
    descale_wqa = case["descale_wqa"]
    descale_wqb = case["descale_wqb"]
    front_workspace = case["front_workspace"]
    json = case["json"]
    key = case["key"]
    num_blocks = case["num_blocks"]
    p = case["p"]
    prepare_mega_reindex = case["prepare_mega_reindex"]
    prepare_wqb_nz = case["prepare_wqb_nz"]
    program = case["program"]
    seqused_k = case["seqused_k"]
    seqused_q = case["seqused_q"]
    tail_workspace = case["tail_workspace"]
    torch = case["torch"]
    version = case["version"]
    w_raw_guard = case["w_raw_guard"]
    wqa = case["wqa"]
    wqb = case["wqb"]
    ww = case["ww"]
    scrub = torch.ones(200 * 1024 * 1024 // 4, dtype=torch.float32, device="npu")
    torch.sum(scrub)
    assert _sample._native_signature(T, B, KV_S, TOPK) == _sample.signature(args)
    program(*args)
    torch.npu.synchronize()
    assert bool((w_raw_guard.cpu() == 113.0).all()), "weights_proj wrote beyond w_raw"
    indices = tail_workspace.indices.cpu()
    torch.sum(scrub)
    program(*args)
    torch.npu.synchronize()
    assert torch.equal(indices, tail_workspace.indices.cpu())
    candidate_block_length.zero_()
    torch.sum(scrub)
    program(*args)
    torch.npu.synchronize()
    assert bool((tail_workspace.indices.cpu() == -1).all())
    candidate_block_length.fill_(num_blocks)
    torch.sum(scrub)
    program(*args)
    torch.npu.synchronize()
    assert torch.equal(indices, tail_workspace.indices.cpu())
    valid = indices.ge(0)
    row_sorted = bool(
        torch.all((indices[:, 1:] >= indices[:, :-1]) | (indices[:, 1:] < 0))
    )
    range_ok = bool(torch.all((indices >= -1) & (indices < KV_S)))
    result = {
        "cannbotdsl_version": version("cannbot-dsl"),
        "module_version": cannbotdsl.__version__,
        "case": {"B": B, "S": S, "T": T, "kv_s": KV_S, "topk": TOPK},
        "compile_seconds": compile_seconds,
        "output_shape": list(indices.shape),
        "output_dtype": str(indices.dtype),
        "valid_counts": valid.sum(dim=1).tolist(),
        "range_ok": range_ok,
        "ascending_or_padded": row_sorted,
        "first_row_head": indices[0, :16].tolist(),
        "pass": (
            tuple(indices.shape) == (T, TOPK)
            and indices.dtype == torch.int32
            and range_ok
            and row_sorted
        ),
    }
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)

    # Non-identity, pair-dependent RoPE with one-hot WQB gives an exact CPU oracle.
    # Each head projects the same first 128 normalized query elements.
    identity_wqb = torch.zeros((4096, 1280), dtype=torch.float32)
    identity_wqb[torch.arange(4096), torch.arange(4096) % 128] = 1
    probe_wqb, probe_scale = prepare_wqb_nz(
        identity_wqb.to(torch.float8_e4m3fn).npu(), p["sb"]
    )
    rotation = (torch.arange(T)[:, None] + torch.arange(32)[None, :]) % 2
    probe_rope_cpu = torch.stack((1 - rotation, rotation), dim=-1).float()
    probe_rope = probe_rope_cpu.npu()
    probe_program, probe_args, probe_front, probe_tail = prepare_mega_reindex(
        p["x"],
        wqa,
        descale_wqa,
        ww,
        p["g"],
        probe_wqb,
        probe_scale,
        probe_rope,
        key,
        descale_key,
        front_workspace=front_workspace,
        tail_workspace=tail_workspace,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
    )
    probe_program(*probe_args)
    torch.npu.synchronize()
    qr_scale = torch.pow(2.0, probe_front.rs.cpu().reshape(T, 40).float() - 127)
    qr = probe_front.rq.cpu().view(torch.float8_e4m3fn).float()
    qr = (qr * qr_scale.repeat_interleave(32, dim=1)).to(torch.bfloat16).float()
    expected = qr[:, :128].clone()
    even, odd = expected[:, 64::2].clone(), expected[:, 65::2].clone()
    cosine, sine = probe_rope_cpu[:, :, 0], probe_rope_cpu[:, :, 1]
    expected[:, 64::2] = even * cosine - odd * sine
    expected[:, 65::2] = even * sine + odd * cosine
    expected = expected.to(torch.bfloat16).float().reshape(T, 4, 32)
    maxima = expected.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
    exponents = torch.ceil(torch.log2(maxima / 6.0))
    normalized = expected / torch.pow(2.0, exponents)[..., None]
    absolute = normalized.abs()
    codes = torch.zeros_like(absolute, dtype=torch.uint8)
    for code, threshold, inclusive in (
        (1, 0.25, False),
        (2, 0.75, True),
        (3, 1.25, False),
        (4, 1.75, True),
        (5, 2.5, False),
        (6, 3.5, True),
        (7, 5.0, False),
    ):
        take = absolute >= threshold if inclusive else absolute > threshold
        codes = torch.where(take, code, codes)
    codes += (normalized < 0).to(torch.uint8) * 8
    codes = codes.reshape(T, 128)
    packed_expected = codes[:, 0::2] + codes[:, 1::2] * 16
    assert torch.equal(
        probe_tail.q.cpu(), packed_expected[:, None, :].expand(T, 32, 64)
    )
    assert torch.equal(
        probe_tail.q_scale.cpu().reshape(T, 32, 4),
        (exponents + 127).to(torch.uint8)[:, None, :].expand(T, 32, 4),
    )
    print(
        "PASS non-identity RoPE: token/pair cos-sin order and shared head frequencies"
    )

    # Cross the metadata UB budget without changing the per-batch numeric fixture.
    # One batch is the reference; duplicated batches must produce identical rows.
    batch_results = []
    metadata_stream = torch.npu.Stream()
    for batch_count in (1, 256):
        batch_rows = batch_count * 2
        batch_x = p["x"][:2].repeat(batch_count, 1)
        batch_rope = p["rope"][:2].repeat(batch_count, 1, 1)
        batch_key = key[:512].repeat(batch_count, 1, 1)
        batch_scale = descale_key[:512].repeat(batch_count, 1, 1, 1)
        batch_candidates = torch.full((batch_rows, 1, 2048), -1, dtype=torch.int32)
        batch_candidates[:, 0, :64] = torch.arange(64, dtype=torch.int32)
        torch.npu.synchronize()
        with torch.npu.stream(metadata_stream):
            quota = 8 if batch_count == 256 else 32
            torch.npu.set_stream_limit(
                metadata_stream, cube_num=quota, vector_num=2 * quota
            )
            batch_program, batch_args, _, batch_tail = prepare_mega_reindex(
                batch_x,
                wqa,
                descale_wqa,
                ww,
                p["g"],
                wqb,
                descale_wqb,
                batch_rope,
                batch_key,
                batch_scale,
                topk=64,
                candidate_block_indices=batch_candidates.npu(),
                candidate_block_length=torch.full(
                    (batch_rows, 1), 64, dtype=torch.int32, device="npu"
                ),
                cu_seqlens_q=torch.arange(
                    batch_count + 1, dtype=torch.int32, device="npu"
                )
                * 2,
                cu_seqlens_k=torch.arange(
                    batch_count + 1, dtype=torch.int32, device="npu"
                )
                * 512,
                seqused_q=torch.full(
                    (batch_count,), 2, dtype=torch.int32, device="npu"
                ),
                seqused_k=torch.full(
                    (batch_count,), 512, dtype=torch.int32, device="npu"
                ),
            )
            batch_program(*batch_args)
            torch.npu.synchronize()
        batch_results.append(
            (
                batch_tail.indices.cpu(),
                batch_tail.score.cpu(),
                batch_tail.visible_lengths.cpu(),
                batch_tail.block_mask.cpu(),
            )
        )
    for reference, actual in zip(batch_results[0], batch_results[1]):
        repeats = (256,) + (1,) * (reference.ndim - 1)
        assert torch.equal(reference.repeat(*repeats), actual)
    print(
        "PASS B=256 metadata fallback: same indices, score, visibility and mask as duplicated B=1"
    )

    # Native E8M0 parameters and the source patch's byte views encode the same scales.
    assert descale_wqa.dtype == torch.float8_e8m0fnu
    assert descale_wqb.dtype == torch.float8_e8m0fnu
    scale_results = []
    candidate_block_length.fill_(num_blocks)
    for scale_dtype in (torch.uint8, torch.float8_e8m0fnu):
        scale_program, scale_args, scale_front, scale_tail = prepare_mega_reindex(
            p["x"],
            wqa,
            descale_wqa.view(scale_dtype),
            ww,
            p["g"],
            wqb,
            descale_wqb.view(scale_dtype),
            p["rope"],
            key,
            descale_key.view(scale_dtype),
            front_workspace=front_workspace,
            tail_workspace=tail_workspace,
            candidate_block_indices=candidate_block_indices,
            candidate_block_length=candidate_block_length,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )
        for index in (6, 8, 19, 20, 25):
            assert scale_args[index].dtype == torch.float8_e8m0fnu
        scale_program(*scale_args)
        torch.npu.synchronize()
        scale_results.append(
            {
                f"{stage}/{name}": tensor.cpu().contiguous().view(torch.uint8).clone()
                for stage, workspace in (("front", scale_front), ("tail", scale_tail))
                for name, tensor in vars(workspace).items()
                if isinstance(tensor, torch.Tensor) and tensor.numel()
            }
        )
    assert scale_results[0].keys() == scale_results[1].keys()
    for name in scale_results[0]:
        assert torch.equal(scale_results[0][name], scale_results[1][name]), name
    print(
        "PASS native E8M0 and legacy UINT8 scale arguments: all workspace bytes identical"
    )

    # Reuse a prepared program after quota changes, including Graph capture.
    program(*args)
    torch.npu.synchronize()

    def workspace_bytes():
        return {
            f"{stage}/{name}": tensor.cpu().contiguous().view(torch.uint8).clone()
            for stage, workspace in (
                ("front", front_workspace),
                ("tail", tail_workspace),
            )
            for name, tensor in vars(workspace).items()
            if isinstance(tensor, torch.Tensor) and tensor.numel()
        }

    quota_reference = workspace_bytes()
    quota_stream = torch.npu.Stream()
    with torch.npu.stream(quota_stream):
        previous = torch.npu.get_stream_limit(quota_stream)
        try:
            for quota in (28, 8, 2, 32):
                torch.npu.set_stream_limit(
                    quota_stream, cube_num=quota, vector_num=2 * quota
                )
                for workspace in (front_workspace, tail_workspace):
                    for tensor in vars(workspace).values():
                        if isinstance(tensor, torch.Tensor) and tensor.numel():
                            tensor.fill_(37)
                program(*args)
                torch.npu.synchronize()
                for name, expected in quota_reference.items():
                    assert torch.equal(workspace_bytes()[name], expected), (quota, name)
                assert program._core_count == quota
                if quota == 28:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=quota_stream):
                        program(*args)
                    for workspace in (front_workspace, tail_workspace):
                        for tensor in vars(workspace).values():
                            if isinstance(tensor, torch.Tensor) and tensor.numel():
                                tensor.fill_(53)
                    graph.replay()
                    torch.npu.synchronize()
                    for name, expected in quota_reference.items():
                        assert torch.equal(workspace_bytes()[name], expected), (
                            "graph",
                            quota,
                            name,
                        )
                print(
                    f"PASS TND core quota {quota}: workspace bytes and Graph consistent"
                )
        finally:
            torch.npu.set_stream_limit(
                quota_stream,
                cube_num=previous["cube_core_num"],
                vector_num=previous["vector_core_num"],
            )

    # Exercise more M tiles than physical cores; no stale query/score data may survive.
    large_reference = {
        name: getattr(batch_tail, name).cpu().contiguous().view(torch.uint8).clone()
        for name in (
            "q",
            "q_scale",
            "indices",
            "score",
            "visible_lengths",
            "block_mask",
        )
    }
    with torch.npu.stream(quota_stream):
        previous = torch.npu.get_stream_limit(quota_stream)
        try:
            torch.npu.set_stream_limit(quota_stream, cube_num=8, vector_num=16)
            for name in large_reference:
                getattr(batch_tail, name).fill_(37)
            batch_program(*batch_args)
            torch.npu.synchronize()
            for name, expected in large_reference.items():
                actual = getattr(batch_tail, name).cpu().contiguous().view(torch.uint8)
                assert torch.equal(actual, expected), name
            print("PASS B=256 with 8 cores: all M tiles recomputed bitwise identically")
        finally:
            torch.npu.set_stream_limit(
                quota_stream,
                cube_num=previous["cube_core_num"],
                vector_num=previous["vector_core_num"],
            )


def run_pa():
    import torch

    ns = _make_case(compile_program=True)
    prepare = ns["prepare_mega_reindex"]
    scrub = torch.ones(200 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)

    def run(prog, args):
        torch.sum(scrub)
        prog(*args)
        torch.npu.synchronize()

    run(ns["program"], ns["args"])
    gold = ns["tail_workspace"].indices.cpu()
    score = ns["tail_workspace"].score.cpu()
    results = []
    for page_size in (128, 64, 16, 1024):
        pages = 8192 // page_size
        torch.manual_seed(900 + page_size)
        perm = torch.randperm(pages)
        # Pad physical page axis and reverse logical mapping.
        host = torch.zeros((pages, page_size // 8 + 1, 544), dtype=torch.uint8)
        data = torch.cat(
            (
                ns["key"].cpu().reshape(pages, page_size // 8, 512),
                ns["descale_key"].cpu().reshape(pages, page_size // 8, 32),
            ),
            dim=-1,
        )
        host[perm, : page_size // 8] = data
        packed = host.npu()[:, : page_size // 8, :]
        table = perm.to(torch.int32).view(1, pages).npu()
        common = dict(
            front_workspace=ns["front_workspace"],
            topk=512,
            candidate_block_indices=ns["candidate_block_indices"],
            candidate_block_length=ns["candidate_block_length"],
            cu_seqlens_q=ns["cu_seqlens_q"],
            seqused_q=ns["seqused_q"],
            seqused_k=ns["seqused_k"],
            layout_k="PA_BBND",
            block_table=table,
            k=packed,
            descale_k=None,
        )
        p = ns["p"]
        pos = (
            p["x"],
            ns["wqa"],
            ns["descale_wqa"],
            ns["ww"],
            p["g"],
            ns["wqb"],
            ns["descale_wqb"],
            p["rope"],
        )
        print("COMPILE_PA", page_size, flush=True)
        start = time.perf_counter()
        prog, args, fw, tw = prepare(*pos, **common)
        print("COMPILED", time.perf_counter() - start, flush=True)
        run(prog, args)
        check = {
            "key_exact": torch.equal(tw.pa_key.cpu(), ns["key"].cpu()),
            "scale_exact": torch.equal(tw.pa_scale.cpu(), ns["descale_key"].cpu()),
            "score_exact": torch.equal(tw.score.cpu(), score),
            "indices_exact": torch.equal(tw.indices.cpu(), gold),
        }
        print(page_size, check, flush=True)
        assert all(check.values())
        mega_reindex = ns["_sample"].mega_reindex
        torch.sum(scrub)
        public = mega_reindex(*pos, tail_workspace=tw, **common)
        torch.npu.synchronize()
        assert public.data_ptr() == tw.indices.data_ptr()
        assert torch.equal(public.cpu(), gold)
        for bad in (
            dict(descale_k=ns["descale_key"]),
            dict(block_table=None),
            dict(cu_seqlens_k=ns["cu_seqlens_k"]),
            dict(layout_k="invalid"),
        ):
            options = dict(common)
            options.update(bad)
            try:
                prepare(*pos, **options)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(("accepted invalid PA arguments", bad.keys()))

        for case in ("short", "empty", "inactive_query", "remap"):
            ns["seqused_k"].fill_(8192 - page_size - 3 if case == "short" else 8192)
            ns["candidate_block_length"].fill_(0 if case == "empty" else 1024)
            ns["seqused_q"].fill_(6 if case == "inactive_query" else 8)
            table.copy_(perm.to(torch.int32).view(1, pages).npu())
            if case == "short":
                table[0, -1] = -1
            if case == "remap":
                # Logical page zero changes in-place without recompilation.
                table[0, 0] = perm[1].item()
                ns["key"][:page_size].copy_(
                    ns["key"][page_size : 2 * page_size].clone()
                )
                ns["descale_key"][:page_size].copy_(
                    ns["descale_key"][page_size : 2 * page_size].clone()
                )
            run(ns["program"], ns["args"])
            run(prog, args)
            assert torch.equal(tw.indices.cpu(), ns["tail_workspace"].indices.cpu()), (
                case
            )
            assert torch.equal(tw.score.cpu(), ns["tail_workspace"].score.cpu()), case
            check[case] = True
        # Restore both layouts and metadata, then verify workspace reuse.
        ns["key"].copy_(ns["packed_fixture"][0].npu())
        ns["descale_key"].copy_(ns["packed_fixture"][1].npu())
        ns["seqused_k"].fill_(8192)
        ns["seqused_q"].fill_(8)
        ns["candidate_block_length"].fill_(1024)
        table.copy_(perm.to(torch.int32).view(1, pages).npu())
        run(ns["program"], ns["args"])
        run(prog, args)
        assert torch.equal(tw.indices.cpu(), gold)
        results.append({"page_size": page_size, "accuracy": check})
        print("PASS", page_size, check, flush=True)

    print(json.dumps(results, indent=2))


def run_pa_batch():
    import torch

    ns = _make_case(compile_program=False)
    prepare = ns["prepare_mega_reindex"]
    p = ns["p"]
    x = p["x"][:6].repeat(2, 1)
    rope = p["rope"][:6].repeat(2, 1, 1)
    key = torch.cat((ns["key"].cpu(), ns["key"].cpu().flip(0))).npu()
    scale = torch.cat((ns["descale_key"].cpu(), ns["descale_key"].cpu().flip(0))).npu()
    candidates = ns["candidate_block_indices"][:6].repeat(2, 1, 1)
    counts = ns["candidate_block_length"][:6].repeat(2, 1)
    args = (
        x,
        ns["wqa"],
        ns["descale_wqa"],
        ns["ww"],
        p["g"],
        ns["wqb"],
        ns["descale_wqb"],
        rope,
    )
    common = dict(
        candidate_block_indices=candidates,
        candidate_block_length=counts,
        cu_seqlens_q=torch.tensor([0, 6, 12], dtype=torch.int32, device="npu"),
        seqused_q=torch.tensor([6, 4], dtype=torch.int32, device="npu"),
        seqused_k=torch.tensor([8192, 8001], dtype=torch.int32, device="npu"),
    )
    tnd, ta, _, tw = prepare(
        *args,
        k=key,
        descale_k=scale,
        cu_seqlens_k=torch.tensor([0, 8192, 16384], dtype=torch.int32, device="npu"),
        **common,
    )
    pages = 128
    perm = torch.randperm(pages)
    packed = torch.cat(
        (key.cpu().reshape(pages, 16, 512), scale.cpu().reshape(pages, 16, 32)), dim=-1
    )
    physical = torch.empty_like(packed)
    physical[perm] = packed
    pa, paargs, _, pw = prepare(
        *args,
        k=physical.npu(),
        layout_k="PA_BBND",
        block_table=perm.to(torch.int32).reshape(2, 64).npu(),
        **common,
    )
    scrub = torch.ones(200 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
    for prog, inputs in ((tnd, ta), (pa, paargs)):
        torch.sum(scrub)
        prog(*inputs)
        torch.npu.synchronize()
    assert torch.equal(tw.score.cpu(), pw.score.cpu())
    assert torch.equal(tw.indices.cpu(), pw.indices.cpu())
    assert (pw.indices.cpu()[10:] == -1).all()
    print(
        "PASS B=2 S=6 PA128 random distinct batch pages, shortened K and inactive Q",
        flush=True,
    )

    # The same logical PA pages must be independent of the available core quota.
    reference = {
        name: getattr(pw, name).cpu().clone()
        for name in ("indices", "score", "pa_key", "pa_scale")
    }
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        previous = torch.npu.get_stream_limit(stream)
        try:
            for quota in (28, 8, 2, 32):
                torch.npu.set_stream_limit(stream, cube_num=quota, vector_num=2 * quota)
                for workspace in (tw, pw):
                    for name, tensor in vars(workspace).items():
                        if (
                            name != "pa_cuk"
                            and isinstance(tensor, torch.Tensor)
                            and tensor.numel()
                        ):
                            tensor.fill_(37)
                for prog, inputs in ((tnd, ta), (pa, paargs)):
                    prog(*inputs)
                torch.npu.synchronize()
                assert torch.equal(tw.score.cpu(), pw.score.cpu())
                assert torch.equal(tw.indices.cpu(), pw.indices.cpu())
                for name, expected in reference.items():
                    assert torch.equal(getattr(pw, name).cpu(), expected), (quota, name)
                if quota == 28:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        pa(*paargs)
                    for name, tensor in vars(pw).items():
                        if (
                            name != "pa_cuk"
                            and isinstance(tensor, torch.Tensor)
                            and tensor.numel()
                        ):
                            tensor.fill_(53)
                    graph.replay()
                    torch.npu.synchronize()
                    for name, expected in reference.items():
                        assert torch.equal(getattr(pw, name).cpu(), expected), (
                            "graph",
                            quota,
                            name,
                        )
                print(
                    f"PASS PA core quota {quota}: TND parity and Graph consistent",
                    flush=True,
                )
        finally:
            torch.npu.set_stream_limit(
                stream,
                cube_num=previous["cube_core_num"],
                vector_num=previous["vector_core_num"],
            )


def run_example(
    tensors, *, topk=512, norm_eps=1e-20, softmax_scale=1 / 64, layout_k="TND"
):
    """Run prepared NPU tensors from a dict and return INT32 [T, topk] indices."""
    sample = _load_sample("mega_reindex.py")
    FrontWorkspace = sample.FrontWorkspace
    MegaReindexTailWorkspace = sample.MegaReindexTailWorkspace
    mega_reindex = sample.mega_reindex
    x = tensors["x"]
    batch = tensors["seqused_q"].numel()
    key = tensors.get("k", tensors.get("key"))
    kv_s = (
        tensors["block_table"].shape[1] * key.shape[1] * 8
        if layout_k == "PA_BBND"
        else key.shape[0] // batch
    )
    front_workspace = FrontWorkspace(x.shape[0], x.device)
    tail_workspace = MegaReindexTailWorkspace(x.shape[0], kv_s, x.device, topk=topk)
    return mega_reindex(
        x,
        tensors["wqa"],
        tensors["descale_wqa"],
        tensors["ww"],
        tensors["norm_weight"],
        tensors["wqb"],
        tensors["descale_wqb"],
        tensors["rope"],
        key,
        tensors.get("descale_k", tensors.get("descale_key")),
        front_workspace=front_workspace,
        tail_workspace=tail_workspace,
        norm_eps=norm_eps,
        softmax_scale=softmax_scale,
        topk=topk,
        candidate_block_indices=tensors["candidate_block_indices"],
        candidate_block_length=tensors["candidate_block_length"],
        cu_seqlens_q=tensors["cu_seqlens_q"],
        cu_seqlens_k=tensors.get("cu_seqlens_k"),
        seqused_q=tensors["seqused_q"],
        seqused_k=tensors["seqused_k"],
        layout_k=layout_k,
        block_table=tensors.get("block_table"),
    )


def _load_setup(source):
    source = Path(source).expanduser().resolve()
    if source == Path(__file__).resolve():
        return _make_case()
    test_root = next(
        (
            parent
            for parent in source.parents
            if (parent / "_samples_path.py").is_file()
        ),
        source.parent.parent,
    )
    sys.path.insert(0, str(test_root))
    return runpy.run_path(str(source), run_name="mega_reindex_setup")


def capture(source, output):
    import torch

    namespace = _load_setup(source)
    tensors = {}
    for case in ("full", "empty", "restore"):
        namespace["candidate_block_length"].fill_(
            0 if case == "empty" else namespace["num_blocks"]
        )
        namespace["program"](*namespace["args"])
        torch.npu.synchronize()
        for prefix, key in (("front", "front_workspace"), ("tail", "tail_workspace")):
            for name, tensor in vars(namespace[key]).items():
                if isinstance(tensor, torch.Tensor) and tensor.numel():
                    tensors[f"{case}/{prefix}/{name}"] = tensor.cpu().clone()
    torch.save(tensors, output)
    print(json.dumps({"captured": len(tensors), "output": str(output)}))


def compare(reference, candidate):
    import torch

    before = torch.load(reference, map_location="cpu", weights_only=True)
    after = torch.load(candidate, map_location="cpu", weights_only=True)
    assert before.keys() == after.keys(), "capture tensor sets differ"
    results = {}
    for name, expected in before.items():
        actual = after[name]
        results[name] = (
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )
    print(json.dumps({"comparisons": results, "pass": all(results.values())}, indent=2))
    assert all(results.values()), "migration differs from the reference"


def benchmark_attention(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="fixture from test_mega_reindex.py capture on the reference implementation",
    )
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args(argv)
    import cannbotdsl
    import torch
    import torch_npu

    reference_path = args.reference.expanduser().resolve()
    if not reference_path.is_file():
        parser.error(
            f"reference fixture not found: {reference_path}; "
            "generate it with test_mega_reindex.py capture before profiling"
        )
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    required = ("full/tail/indices", "full/tail/score")
    if not isinstance(reference, dict) or any(
        name not in reference or not isinstance(reference[name], torch.Tensor)
        for name in required
    ):
        parser.error(f"reference fixture must contain tensor keys: {required}")
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ns = _load_setup(source)
    program, inputs = ns["program"], ns["args"]
    scrub = torch.ones(200_000_000, dtype=torch.float32, device="npu")

    def clear_cache():
        torch.sum(scrub)
        torch.npu.synchronize()

    for _ in range(5):
        clear_cache()
        program(*inputs)
        torch.npu.synchronize()
    assert torch.equal(
        ns["tail_workspace"].indices.cpu(), reference["full/tail/indices"]
    )
    assert torch.equal(ns["tail_workspace"].score.cpu(), reference["full/tail/score"])
    clear_cache()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        program(*inputs)
    torch.npu.synchronize()
    for _ in range(3):
        clear_cache()
        graph.replay()
        torch.npu.synchronize()
    metadata = {
        "dsl_version": cannbotdsl.__version__,
        "dsl_path": cannbotdsl.__file__,
        "sample_path": str(source),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "case": {"B": 1, "S": 8, "KV": 8192, "topk": 512, "seed": 2026},
        "device": torch.npu.get_device_name(0),
        "platform": str(cannbotdsl.get_platform_info()),
        "samples": args.samples,
        "reference_path": str(reference_path),
        "clear_cache_elements": scrub.numel(),
        "clear_cache_bytes": scrub.numel() * scrub.element_size(),
        "cache_protocol": "FP32 ReduceSum then device synchronize before every invocation; eviction workload, not a cache-invalidate instruction",
        "accuracy": "score and indices bit-exact against the supplied reference fixture before and after profiling",
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(
                sys.modules[ns["prepare_mega_reindex"].__module__].__file__
            ).parent.glob("*.py")
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PROFILE", metadata["dsl_version"], args.samples, flush=True)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=args.samples, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for iteration in range(args.samples + 1):
            clear_cache()
            with torch.profiler.record_function(f"target_replay_{iteration:03d}"):
                graph.replay()
                torch.npu.synchronize()
            prof.step()
    assert torch.equal(
        ns["tail_workspace"].indices.cpu(), reference["full/tail/indices"]
    )
    assert torch.equal(ns["tail_workspace"].score.cpu(), reference["full/tail/score"])
    print("DONE", output, flush=True)


MAIN = "mega_reindex_kernel"
FILL = "aclnnInplaceFillScalar_FillAiCore_Fill"
SCRUB = "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum"


def stats(values):
    return dict(
        n=len(values),
        p50_us=statistics.median(values),
        p95_us=statistics.quantiles(values, n=100, method="inclusive")[94],
        min_us=min(values),
        max_us=max(values),
    )


def analyze_profile(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = {
        "protocol": "A/B/B/A independent processes; 30 profiled replays per batch; 800 MB FP32 ReduceSum + synchronize before each replay",
        "batches": {},
    }
    combined = {"baseline": [], "candidate": []}
    for batch in ("a1", "b1", "b2", "a2"):
        directory = args.root / batch
        files = list(directory.rglob("kernel_details.csv"))
        assert len(files) == 1, (batch, files)
        csv_path = files[0]
        trace_path = next(directory.rglob("trace_view.json"))
        trace = json.loads(trace_path.read_text())
        events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        device = {
            (x["name"], Decimal(str(x["ts"]))): x
            for x in events
            if x.get("ph") == "X"
            and x.get("name") in (MAIN, FILL, SCRUB)
            and "Task Type" in x.get("args", {})
        }
        rows = list(csv.DictReader(csv_path.open()))
        assert all(row["Name"] in (MAIN, FILL, SCRUB) for row in rows)
        rows.sort(key=lambda x: Decimal(x["Start Time(us)"].strip()))
        assert len(rows) == 120, (batch, len(rows))
        # Host profiler Step IDs may assign the next asynchronous scrub to the
        # previous step. The synchronized device stream has an unambiguous
        # ReduceSum -> Fill -> Fill -> main sequence for each replay.
        groups = {str(i + 1): rows[i * 4 : (i + 1) * 4] for i in range(30)}
        meta = json.loads((directory / "metadata.json").read_text())
        assert len(groups) == meta["samples"] == 30, (batch, len(groups))
        samples = []
        for step, rows in sorted(groups.items(), key=lambda x: int(x[0])):
            rows.sort(key=lambda x: Decimal(x["Start Time(us)"].strip()))
            assert [x["Name"] for x in rows] == [SCRUB, FILL, FILL, MAIN], (
                batch,
                step,
                rows,
            )
            tasks = []
            for row in rows:
                start = Decimal(row["Start Time(us)"].strip())
                event = device[(row["Name"], start)]
                duration = float(row["Duration(us)"])
                assert abs(float(event["dur"]) - duration) < 1e-6
                tasks.append(
                    dict(
                        name=row["Name"],
                        start_us=str(start),
                        end_us=str(start + Decimal(row["Duration(us)"])),
                        duration_us=duration,
                        device=row["Device_id"],
                        stream=event["tid"],
                        task=event["args"],
                    )
                )
            assert Decimal(tasks[0]["end_us"]) <= Decimal(tasks[1]["start_us"])
            samples.append(
                dict(
                    iteration=int(step),
                    main_us=tasks[3]["duration_us"],
                    fill_sum_us=sum(t["duration_us"] for t in tasks[1:3]),
                    total_kernel_us=sum(t["duration_us"] for t in tasks[1:]),
                    tasks=tasks,
                )
            )
        variant = "baseline" if batch.startswith("a") else "candidate"
        assert meta.get("dsl_version")
        combined[variant].extend(samples)
        result["batches"][batch] = dict(
            metadata=meta,
            csv=str(csv_path),
            trace=str(trace_path),
            main=stats([s["main_us"] for s in samples]),
            total_kernel=stats([s["total_kernel_us"] for s in samples]),
            samples=samples,
        )
    result["summary"] = {
        v: {
            metric: stats([s[key] for s in samples])
            for metric, key in [
                ("main", "main_us"),
                ("fill_sum", "fill_sum_us"),
                ("total_kernel", "total_kernel_us"),
            ]
        }
        for v, samples in combined.items()
    }
    result["change_percent"] = {
        metric: {
            p: 100
            * (
                result["summary"]["candidate"][metric][p]
                / result["summary"]["baseline"][metric][p]
                - 1
            )
            for p in ("p50_us", "p95_us")
        }
        for metric in ("main", "total_kernel")
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "change_percent": result["change_percent"],
                "batches": {
                    k: {"main": v["main"], "total_kernel": v["total_kernel"]}
                    for k, v in result["batches"].items()
                },
            },
            indent=2,
        )
    )


def migration_command(command, argv):
    parser = argparse.ArgumentParser(prog=f"{Path(__file__).name} {command}")
    if command == "capture":
        parser.add_argument(
            "source", type=Path, help="this test file or a setup-only reference script"
        )
        parser.add_argument("output", type=Path)
        args = parser.parse_args(argv)
        capture(args.source, args.output)
    else:
        parser.add_argument("reference", type=Path)
        parser.add_argument("candidate", type=Path)
        args = parser.parse_args(argv)
        compare(args.reference, args.candidate)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "smoke": run_smoke,
        "verify-pa": run_pa,
        "verify-pa-batch": run_pa_batch,
        "benchmark": benchmark_attention,
        "analyze-profile": analyze_profile,
    }
    if argv[:1] in (["-h"], ["--help"]):
        print(
            "Usage: test_mega_reindex.py [test|smoke|verify-pa|verify-pa-batch|benchmark|capture|compare|analyze-profile] [arguments]"
        )
        print(
            "test: pytest options; benchmark/capture/compare/analyze-profile: use --help"
        )
        return 0
    if argv and argv[0] in ("capture", "compare"):
        return migration_command(argv[0], argv[1:])
    if argv and argv[0] in commands:
        command, *arguments = argv
        if command in ("benchmark", "analyze-profile"):
            return commands[command](arguments)
        if arguments:
            raise SystemExit(f"{command} does not accept additional arguments")
        commands[command]()
        return 0
    if argv[:1] == ["test"]:
        argv.pop(0)
    return pytest.main([str(Path(__file__).resolve()), *argv])


if __name__ == "__main__":
    raise SystemExit(main())
