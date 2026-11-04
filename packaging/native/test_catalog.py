# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Metadata/build regression checks; not part of the operator performance loop."""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    return load_module("catalog_native_builder", PROJECT / "build.py")


@pytest.fixture
def generator():
    return load_module("catalog_generator", ROOT / "scripts/ci/sync_catalog.py")


@pytest.fixture
def metadata_tree(tmp_path):
    catalog = {
        "schema_version": 1,
        "operators": {
            "op_a": {
                "source": "samples/attention/op_a",
                "test": "test/attention/op_a",
                "native": {"export": "export_a", "sources": ["@op_a/source/kernel.py"]},
            },
            "op_b": {
                "source": "samples/vector/op_b",
                "test": "test/vector/op_b",
                "native": {"export": "export_b", "sources": ["@op_b/source/kernel.py"]},
            },
        },
        "native_exports": ["op_b", "op_a"],
        "ci": {
            "skip_all_tests": False,
            "ignore": [],
            "inventory": ["op_a", "op_b"],
            "rules": [
                {
                    "name": "op_a",
                    "enabled": True,
                    "watch": ["@op_a/source", "@op_a/test"],
                    "tests": ["@op_a/test", "@op_b/test"],
                },
                {
                    "name": "op_b",
                    "enabled": False,
                    "watch": ["@op_b/source", "@op_b/test"],
                    "tests": ["@op_b/test"],
                },
            ],
        },
    }
    (tmp_path / "catalog").mkdir()
    for name, operator in catalog["operators"].items():
        source = tmp_path / operator["source"]
        source.mkdir(parents=True)
        (source / "kernel.py").write_text(f"VALUE = {name!r}\n")
        (tmp_path / operator["test"]).mkdir(parents=True)
    network = tmp_path / "net/example"
    network.mkdir(parents=True)
    (network / "network.json").write_text(
        json.dumps(
            {"schema_version": 1, "name": "example", "operators": ["op_a", "op_b"]}
        )
    )

    def save():
        (tmp_path / "catalog/operators.json").write_text(json.dumps(catalog))

    save()
    return tmp_path, catalog, save


def test_generated_views_are_current(generator):
    for path, content in generator.render(ROOT).items():
        assert (ROOT / path).read_text() == content


def test_catalog_generates_routing_and_export_order(generator, metadata_tree):
    root, _, _ = metadata_tree
    outputs = generator.render(root)
    routes = yaml.safe_load(outputs[Path("test/test_config.yaml")])
    assert routes["op_a"]["src"] == ["samples/attention/op_a", "test/attention/op_a"]
    assert routes["op_a"]["options"] == ["attention/op_a", "vector/op_b"]
    assert routes["op_b"]["module"] is False
    native = outputs[Path("packaging/native/operator_groups.toml")]
    assert native.index("export_b =") < native.index("export_a =")
    assert 'example = ["export_a", "export_b"]' in native


def test_catalog_path_change_updates_all_consumers(generator, metadata_tree):
    root, catalog, save = metadata_tree
    old = root / catalog["operators"]["op_a"]["source"]
    new = root / "samples/attention/moved_a"
    old.rename(new)
    catalog["operators"]["op_a"]["source"] = str(new.relative_to(root))
    save()
    outputs = generator.render(root)
    assert "samples/attention/moved_a" in outputs[Path("test/test_config.yaml")]
    assert (
        "attention/moved_a/kernel.py"
        in outputs[Path("packaging/native/operator_groups.toml")]
    )
    assert "attention/moved_a" in outputs[Path("scripts/ci/operator_list.yaml")]


def test_unknown_network_operator_is_rejected(generator, metadata_tree):
    root, _, _ = metadata_tree
    manifest = root / "net/example/network.json"
    manifest.write_text(json.dumps({"name": "example", "operators": ["missing"]}))
    with pytest.raises(ValueError, match="unknown Native operator"):
        generator.render(root)


@pytest.mark.parametrize(
    "reference", ["@missing/source", "@op_a/missing", "@op_a/source/../../outside.py"]
)
def test_invalid_catalog_references_are_rejected(generator, metadata_tree, reference):
    root, catalog, save = metadata_tree
    catalog["operators"]["op_a"]["native"]["sources"] = [reference]
    save()
    with pytest.raises(ValueError):
        generator.render(root)


def test_stale_generated_views_fail_check(generator, metadata_tree, monkeypatch):
    root, _, _ = metadata_tree
    monkeypatch.setattr(generator, "ROOT", root)
    # Default function arguments are intentionally bound at definition time.
    render = generator.render
    monkeypatch.setattr(generator, "render", lambda: render(root))
    assert generator.main(["--check"]) == 1
    assert generator.main(["--write"]) == 0
    assert generator.main(["--check"]) == 0
    (root / "test/test_config.yaml").write_text("skip_all_tests: true\n")
    assert generator.main(["--check"]) == 1


@pytest.fixture
def explicit_sources(builder, metadata_tree, generator, monkeypatch):
    root, _, _ = metadata_tree
    outputs = generator.render(root)
    catalog = root / "native.toml"
    catalog.write_text(outputs[Path("packaging/native/operator_groups.toml")])
    monkeypatch.setattr(builder, "CATALOG_FILE", catalog)
    monkeypatch.setattr(builder, "OPS_ROOT", root / "samples")
    monkeypatch.setattr(builder, "_SOURCE_INDEX", None)
    return builder, root


def test_independent_same_named_sources_can_build_separately(
    explicit_sources, tmp_path
):
    builder, _ = explicit_sources
    operators, _ = builder.load_catalog()
    for name, expected in (("export_a", "op_a"), ("export_b", "op_b")):
        staged = tmp_path / name
        builder.materialize_flat_sources(staged, [name], operators)
        assert (staged / "kernel.py").read_text() == f"VALUE = {expected!r}\n"


def test_combined_filename_collision_fails_before_staging(explicit_sources, tmp_path):
    builder, _ = explicit_sources
    operators, _ = builder.load_catalog()
    staged = tmp_path / "collision"
    with pytest.raises(ValueError, match="collide"):
        builder.materialize_flat_sources(staged, list(operators), operators)
    assert not staged.exists()


def test_explicit_cli_selection_and_ambiguous_basename(explicit_sources):
    builder, _ = explicit_sources
    operators, groups = builder.load_catalog()
    args = builder.parse_args(["--operator", "attention/op_a/kernel.py"])
    assert builder.choose_operators(args, operators, groups)[0] == ["export_a"]
    with pytest.raises(ValueError, match="ambiguous"):
        builder.choose_operators(
            builder.parse_args(["--operator", "kernel.py"]), operators, groups
        )


def test_unregistered_duplicate_filenames_do_not_affect_build(
    builder, tmp_path, monkeypatch
):
    root = tmp_path / "samples"
    (root / "registered").mkdir(parents=True)
    (root / "unregistered").mkdir()
    (root / "registered/kernel.py").write_text("VALUE = 1\n")
    (root / "unregistered/kernel.py").write_text("VALUE = 2\n")
    catalog = tmp_path / "catalog.toml"
    catalog.write_text('[operators]\nexample = ["registered/kernel.py"]\n[groups]\n')
    monkeypatch.setattr(builder, "CATALOG_FILE", catalog)
    monkeypatch.setattr(builder, "OPS_ROOT", root)
    operators, _ = builder.load_catalog()
    builder.materialize_flat_sources(tmp_path / "stage", ["example"], operators)
    assert (tmp_path / "stage/kernel.py").read_text() == "VALUE = 1\n"


def test_python310_toml_fallback_matches_standard_parser(builder, monkeypatch):
    expected = builder.load_catalog()
    monkeypatch.setattr(builder, "tomllib", None)
    monkeypatch.setattr(builder, "_SOURCE_INDEX", None)
    assert builder.load_catalog() == expected


@pytest.mark.parametrize(
    "path", ["../outside.py", "/tmp/outside.py", "attention/op/../file.py"]
)
def test_native_paths_cannot_escape_samples(builder, path):
    with pytest.raises(ValueError, match="unsafe"):
        builder._resolve_source(path)


def test_native_exports_include_relative_sample_dependencies(builder):
    operators, _ = builder.load_catalog()
    for name, files in operators.items():
        selected = {(builder.OPS_ROOT / file).resolve() for file in files}
        for source in selected:
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.ImportFrom) or node.level != 1:
                    continue
                modules = (
                    [node.module]
                    if node.module
                    else [alias.name for alias in node.names]
                )
                for module in modules:
                    dependency = source.parent / (module.split(".")[0] + ".py")
                    if dependency.is_file():
                        assert dependency.resolve() in selected, (
                            f"{name}: {source.name} requires {dependency.name} in the Native package"
                        )


def test_shell_caches_explicit_wheel(tmp_path):
    project = tmp_path / "packaging/native"
    project.mkdir(parents=True)
    shutil.copy2(PROJECT / "run-build.sh", project / "run-build.sh")
    wheel = tmp_path / "cannbot_dsl-0.8.0-py3-none-any.whl"
    wheel.write_text("framework")
    cann_env = tmp_path / "set_env.sh"
    cann_env.write_text("# test CANN environment\n")
    log = tmp_path / "build-log.jsonl"
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BUILD_LOG'], 'a') as f:\n"
        "    f.write(json.dumps([sys.argv[1:], os.environ.get('PYTHONPATH')]) + '\\n')\n"
    )
    python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PYTHON=str(python),
        CANN_ENV=str(cann_env),
        CANNBOTDSL_WHEEL=str(wheel),
        BUILD_LOG=str(log),
    )
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(project / "run-build.sh"), "--group", "ds41"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    installs = [
        record for record in records if record[0][:3] == ["-m", "pip", "install"]
    ]
    assert len(installs) == 1
    assert installs[0][0][-1] == str(wheel)
    assert records[-1][0] == [str(project / "build.py"), "--group", "ds41"]
    assert records[-1][1].startswith(str(project / ".build/frameworks"))
