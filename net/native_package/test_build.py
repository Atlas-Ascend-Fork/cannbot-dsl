# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only regression checks for the standalone CANNBotDSL package project."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ds41_native_build", PROJECT / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_catalog_exports_and_staged_resources(tmp_path, monkeypatch):
    operators, _ = builder.load_catalog()
    sources = tmp_path / "sources"
    builder.materialize_flat_sources(sources, list(operators), operators)
    # Exercise the DSL 0.7 static discovery used by the package builder.
    from cannbotdsl.aot.publication import _declares_registration

    exports = set()
    for source in sources.glob("*.py"):
        if _declares_registration(source, {"export"}):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if not isinstance(decorator, ast.Call):
                            continue
                        function = decorator.func
                        if isinstance(function, ast.Name):
                            name = function.id
                        elif isinstance(function, ast.Attribute):
                            name = function.attr
                        else:
                            continue
                        if name == "export":
                            exports.add(ast.literal_eval(decorator.args[0]))
    assert exports == set(operators)
    package = tmp_path / "package"
    builder.stage_python_package(package, list(operators), operators, sources)
    for source in sources.glob("*.py"):
        assert (
            package / "src" / "ops" / source.name
        ).read_bytes() == source.read_bytes()
    import cannbotdsl.aot as aot

    calls = []
    monkeypatch.setattr(aot, "register_directory", lambda *args: calls.append(args))
    init = package / "src" / "ops" / "__init__.py"
    exec(
        compile(init.read_text(), str(init), "exec"),
        {"__package__": "ops", "__file__": str(init)},
    )
    assert calls == [(package / "src" / "ops" / "_native",)]


@pytest.fixture
def runner(tmp_path):
    project = tmp_path / "net" / "native_package"
    project.mkdir(parents=True)
    shutil.copy2(PROJECT / "run-build.sh", project / "run-build.sh")
    cann_env = tmp_path / "set_env.sh"
    cann_env.write_text("# test CANN environment\n")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BUILD_LOG'], 'a') as f:\n"
        "    f.write(json.dumps([sys.argv[1:], os.environ.get('PYTHONPATH')]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    log = tmp_path / "log"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in ("CANNBOTDSL_ROOT", "CANNBOTDSL_WHEEL", "OPKIT_ROOT", "OPKIT_WHEEL")
    }
    env.update(CANN_ENV=str(cann_env), PYTHON=str(fake_python), BUILD_LOG=str(log))

    def run():
        return subprocess.run(
            ["bash", str(project / "run-build.sh"), "--group", "ds41"],
            env=env,
            capture_output=True,
            text=True,
        )

    return project, env, log, run


def test_dsl_discovery_and_cache_invalidation(runner):
    project, env, log, run = runner
    dsl = project.parent / 'cannbotdsl-0.7-py3-none-any.whl'
    dsl.write_text('dsl')
    # Stale OpKit settings must not require or install a separate distribution.
    env['OPKIT_WHEEL'] = '/missing/opkit.whl'
    for _ in range(2):
        result = run()
        assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    installs = [args for args, _ in records if args[:3] == ['-m', 'pip', 'install']]
    assert len(installs) == 1
    assert Path(installs[0][-1]).resolve() == dsl
    assert not any('opkit' in arg for arg in installs[0])
    first_site = records[-1][1]
    dsl.write_text('dsl-updated')
    result = run()
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(args[:3] == ['-m', 'pip', 'install'] for args, _ in records) == 2
    assert records[-1][1] != first_site


@pytest.mark.parametrize('selection', ['explicit', 'dsl_root'])
def test_external_dsl_wheel(runner, tmp_path, selection):
    project, env, log, run = runner
    location = tmp_path / 'external' / 'build' / 'run' / 'payload'
    location.mkdir(parents=True)
    wheel = location / 'cannbotdsl-0.7-py3-none-any.whl'
    wheel.write_text('dsl')
    if selection == 'explicit':
        env['CANNBOTDSL_WHEEL'] = str(wheel)
    else:
        env['CANNBOTDSL_ROOT'] = str(tmp_path / 'external')
    result = run()
    assert result.returncode == 0, result.stderr
    assert json.loads(log.read_text().splitlines()[0])[0][-1] == str(wheel)


def test_missing_dsl_fails_before_install(runner):
    project, env, log, run = runner
    result = run()
    assert result.returncode != 0
    assert 'CANNBOTDSL_WHEEL' in result.stderr
    assert not log.exists()


def test_build_wheel_contains_sources_and_native_resources(tmp_path, monkeypatch):
    """Build a real wheel while substituting only the device compiler."""
    import zipfile
    import cannbotdsl.aot as aot

    source = tmp_path / "example.py"
    source.write_text(
        'from cannbotdsl.aot import export\n@export("example")\ndef collect():\n    pass\n'
    )
    monkeypatch.setattr(builder, "_SOURCE_INDEX", {"example.py": source})
    monkeypatch.setattr(
        builder, "load_catalog", lambda: ({"example": ["example.py"]}, {})
    )
    # Avoid leaking main()'s process-wide environment into other tests.
    monkeypatch.setattr(
        builder,
        "configure_intermediate_environment",
        lambda work: work.mkdir(parents=True),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    def compile_native(**kwargs):
        assert kwargs["operators"] == ["example"]
        assert kwargs["target"] == "dav-3510"
        assert (kwargs["collectors"] / "example.py").read_bytes() == source.read_bytes()
        output = kwargs["output"]
        variant = output / "operators" / "example" / "variants" / "key"
        variant.mkdir(parents=True)
        (output / "manifest.json").write_text("{}")
        (variant / "kernel.so").write_bytes(b"test binary")
        return output

    monkeypatch.setattr(aot, "build", compile_native)
    assert (
        builder.main(
            [
                "--operator",
                "example",
                "--output-root",
                str(tmp_path / "out"),
                "--work-root",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    (wheel,) = (tmp_path / "out" / "operators" / "example" / "wheels").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("ops/example.py") == source.read_bytes()
        assert (
            archive.read("ops/_native/operators/example/variants/key/kernel.so")
            == b"test binary"
        )
        assert archive.read("ops/_native/manifest.json") == b"{}"
        assert b"from cannbotdsl.aot import register_directory" in archive.read(
            "ops/__init__.py"
        )
        (metadata,) = (
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        assert "Requires-Dist: opkit" not in archive.read(metadata).decode()
        assert "Requires-Dist: cannbotdsl<0.8,>=0.7.0" in archive.read(metadata).decode()


def test_ds41_selection():
    operators, groups = builder.load_catalog()
    assert {
        "mqsmla",
        "qli",
        "qsli",
        "indexer_prologue_qw",
        "indexer_prologue_k",
        "engram_gate",
    } <= set(operators)
    assert groups["ds41"] == [
        "mqsmla",
        "qli",
        "qsli",
        "indexer_prologue_qw",
        "indexer_prologue_k",
        "engram_gate",
    ]
    assert builder.choose_operators(
        builder.parse_args(["--group", "ds41"]), operators, groups
    ) == (groups["ds41"], Path("groups/ds41"))
    with pytest.raises(ValueError, match="unknown group"):
        builder.choose_operators(
            builder.parse_args(["--group", "unknown"]), operators, groups
        )


@pytest.mark.parametrize("operator", ["qli", "qsli"])
def test_indexer_export_profiles(operator, monkeypatch):
    """Check collector routing without compiling or executing device kernels."""
    operators, _ = builder.load_catalog()
    source = builder.source_index()[operators[operator][0]]
    tree = ast.parse(source.read_text())
    collector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == f"export_{operator}"
    )
    collector.decorator_list = []
    calls = []
    namespace = {
        "os": os,
        "TILE_N": 256,
        "clear_caches": lambda: None,
        "clear_tnd_caches": lambda: None,
        "_get_compiled_fused_runner": lambda *args: calls.append(("PA_BBND", args)),
        "_get_tnd_compiled_fused_runner": lambda *args: calls.append(("TND", args)),
    }
    monkeypatch.delenv("CANNBOTDSL_DS41_PROFILES", raising=False)
    exec(
        compile(ast.Module(body=[collector], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    namespace[f"export_{operator}"]()
    assert len(calls) == 18
    assert {layout for layout, _ in calls} == {"PA_BBND", "TND"}
    for layout, config in calls:
        if operator == "qli":
            assert config[-2:] == (32, 6)
            assert config[9] in ((64, 128) if layout == "PA_BBND" else (256,))
        else:
            assert config[10] in ((64, 128) if layout == "PA_BBND" else (0,))


def _extract_export_collector(operator):
    """Return (source, collector) for one operator, decorator removed."""
    operators, _ = builder.load_catalog()
    source = builder.source_index()[operators[operator][0]]
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == f"export_{operator}":
            node.decorator_list = []
            return source, node
    raise AssertionError(f"export_{operator} collector not found in {source}")


def _exec_collector(source, collector, namespace):
    """Define the collector in the namespace, then invoke it once."""
    exec(
        compile(ast.Module(body=[collector], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    run_export = namespace.get(collector.name)
    assert callable(run_export)
    run_export()


def _recording_run_callable(calls, cache):
    """Record positional configs and park closable handles in the cache."""
    from types import SimpleNamespace

    def fake_get_run_callable(*args):
        calls.append(args)
        handle = SimpleNamespace(close=lambda: None)
        cache[args] = handle
        return handle

    return fake_get_run_callable


def test_engram_gate_default_profiles(monkeypatch):
    """Check the default token/quota/mask catalogue without device kernels."""
    calls = []
    cache = {}
    source, collector = _extract_export_collector("engram_gate")
    namespace = {
        "os": os,
        "_get_run_callable": _recording_run_callable(calls, cache),
        "_RUN_CALLABLE_CACHE": cache,
    }
    monkeypatch.delenv("CANNBOTDSL_DS41_PROFILES", raising=False)
    _exec_collector(source, collector, namespace)
    assert len(calls) == 4
    assert sorted({call[4] for call in calls}) == [56, 64]
    assert {call[5] for call in calls} == {72}
    assert {call[6] for call in calls} == {False, True}
    assert all(call[:4] == (1.0e-6, 1.0e-6, 5120, 4) for call in calls)
    assert cache == {}


def test_engram_gate_profile_override(monkeypatch, tmp_path):
    """CANNBOTDSL_DS41_PROFILES replaces the default engram_gate catalogue."""
    calls = []
    cache = {}
    source, collector = _extract_export_collector("engram_gate")
    namespace = {
        "os": os,
        "_get_run_callable": _recording_run_callable(calls, cache),
        "_RUN_CALLABLE_CACHE": cache,
    }
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        '{"engram_gate": [{"max_blocks": 56, "tokens": [72, 100],'
        ' "has_mask": [false]}]}'
    )
    monkeypatch.setenv("CANNBOTDSL_DS41_PROFILES", str(profiles))
    _exec_collector(source, collector, namespace)
    assert [call[4:] for call in calls] == [(56, 72, False), (56, 100, False)]


def test_aicpu_metadata_resources_do_not_collide(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cannbotdsl.aicpu import toolchain

    compiled_workdirs = []

    def compile_metadata(kernel, *, workdir, launch_mode, npu_arch):
        assert launch_mode == "interface"
        assert npu_arch == "dav-3510"
        work = Path(workdir)
        work.mkdir(parents=True)
        binary = work / "metadata_kernel.so"
        binary.write_text(work.name)
        compiled_workdirs.append(work)
        return SimpleNamespace(so_path=binary)

    module = SimpleNamespace(metadata_kernel=object(), _mqsmla_metadata_kernel=object())
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(toolchain, "compile_aicpu_kernel", compile_metadata)
    output = tmp_path / "aicpu"
    builder.build_aicpu_metadata(
        ["mqsmla", "qli", "qsli"], output, tmp_path / "work", "dav-3510"
    )
    assert len(set(compiled_workdirs)) == 3
    assert {path.name: path.read_text() for path in output.iterdir()} == {
        "mqsmla_metadata_kernel.so": "mqsmla",
        "metadata_kernel.so": "qli",
        "qsli_metadata_kernel.so": "qsli",
    }


def test_list_without_frameworks(runner):
    project, env, log, _ = runner
    result = subprocess.run(
        ["bash", str(project / "run-build.sh"), "--list"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0][0] == [str(project / "build.py"), "--list"]
