# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "samples" / "attention" / "mega_reindex"
SOURCE = SAMPLE / "mega_reindex.py"


def _tree():
    return ast.parse(SOURCE.read_text())


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
    readme = (SAMPLE / "README.md").read_text()
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
        tree = ast.parse(source.read_text())
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

    tree = ast.parse((SAMPLE / "reindex_runtime.py").read_text())
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
    import sys

    sys.path.insert(0, str(ROOT / "test"))
    from _samples_path import load_sample

    monkeypatch.setenv("QBMM_MX_L2_SIZE_BYTES", "invalid")
    for filename in ("reindex_matmul.py", "reindex_mx.py"):
        module = load_sample(f"attention/mega_reindex/{filename}")
        for name in getattr(module, "__all__", ()):
            assert hasattr(module, name)


@pytest.mark.npu
def test_existing_device_regressions():
    """Wire the existing serial smoke/PA scripts into normal NPU CI."""
    import subprocess
    import sys

    directory = Path(__file__).resolve().parent
    for name in ("smoke.py", "verify_pa.py", "verify_pa_batch.py"):
        completed = subprocess.run(
            [sys.executable, "-u", str(directory / name)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
            check=False,
        )
        assert completed.returncode == 0, f"{name}:\n{completed.stdout}"
        print(completed.stdout)


@pytest.mark.npu
def test_real_stream_quota_is_checked_without_launch():
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
