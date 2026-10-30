# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build selected sample operators into isolated native and wheel artifacts.

Sources are the repository's ``samples/`` tree: the catalog references them by
basename and the build flattens the selection into one directory, because the
native collector imports every module by filename from a single location.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    tomllib = None


PROJECT = Path(__file__).resolve().parent
NET_ROOT = PROJECT.parent
# Operator sources are the repository's samples, one directory per operator.
# The native collector imports every module by filename from a single
# directory, so the build flattens the selected sources into a staging tree;
# the index below maps each catalogued filename back to its real location.
OPS_ROOT = NET_ROOT.parent / "samples"
CATALOG_FILE = PROJECT / "operator_groups.toml"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
sys.dont_write_bytecode = True

_SOURCE_INDEX: dict[str, Path] | None = None


def source_index() -> dict[str, Path]:
    """Map each sample module's basename to its path.

    Filenames double as module names once flattened, so two samples sharing a
    basename would silently shadow each other.
    """
    global _SOURCE_INDEX
    if _SOURCE_INDEX is None:
        index: dict[str, Path] = {}
        for path in sorted(OPS_ROOT.rglob("*.py")):
            if path.name in index:
                raise ValueError(
                    f"ambiguous sample module {path.name!r}: "
                    f"{index[path.name]} and {path} both flatten to that name"
                )
            index[path.name] = path
        _SOURCE_INDEX = index
    return _SOURCE_INDEX


def load_catalog() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if tomllib is not None:
        with CATALOG_FILE.open("rb") as stream:
            data = tomllib.load(stream)
    else:
        data = _parse_catalog_without_tomllib()

    operators = data.get("operators", {})
    groups = data.get("groups", {})
    if not isinstance(operators, dict):
        raise ValueError(f"{CATALOG_FILE} [operators] must be a table")
    if not isinstance(groups, dict):
        raise ValueError(f"{CATALOG_FILE} [groups] must be a table")

    normalized_operators: dict[str, list[str]] = {}
    for name, files in operators.items():
        if not SAFE_NAME.fullmatch(name) or not isinstance(files, list) or not files:
            raise ValueError(f"invalid operator entry: {name!r} = {files!r}")
        normalized_files: list[str] = []
        for filename in files:
            if not isinstance(filename, str):
                raise ValueError(f"operator {name!r} has a non-string source filename")
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
                raise ValueError(
                    f"operator {name!r} has an unsafe source path: {filename!r}"
                )
            if path.name not in source_index():
                raise ValueError(
                    f"operator {name!r} source does not exist under {OPS_ROOT}: "
                    f"{filename!r}"
                )
            normalized_files.append(filename)
        normalized_operators[name] = normalized_files

    normalized_groups: dict[str, list[str]] = {}
    for group, names in groups.items():
        if not SAFE_NAME.fullmatch(group) or not isinstance(names, list) or not names:
            raise ValueError(f"invalid group entry: {group!r} = {names!r}")
        unknown = [name for name in names if name not in normalized_operators]
        if unknown:
            raise ValueError(
                f"group {group!r} contains unknown operators: {', '.join(unknown)}"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"group {group!r} contains duplicate operators")
        normalized_groups[group] = list(names)

    return normalized_operators, normalized_groups


def _parse_catalog_without_tomllib() -> dict[str, dict[str, list[str]]]:
    """Parse the small, array-only catalog when running on Python 3.10."""
    data: dict[str, dict[str, list[str]]] = {"operators": {}, "groups": {}}
    section: str | None = None
    pending_key: str | None = None
    pending_value = ""
    for raw_line in CATALOG_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section not in data:
                raise ValueError(f"unsupported catalog section: {section!r}")
            continue
        if section is None:
            raise ValueError("catalog entry appears before a section header")
        if pending_key is None:
            if "=" not in line:
                raise ValueError(f"invalid catalog line: {raw_line!r}")
            pending_key, value = (part.strip() for part in line.split("=", 1))
            pending_value = value
        else:
            pending_value += " " + line
        if pending_value.endswith("]"):
            try:
                parsed = ast.literal_eval(pending_value)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"invalid catalog array for {pending_key!r}"
                ) from error
            data[section][pending_key] = parsed
            pending_key = None
            pending_value = ""
    if pending_key is not None:
        raise ValueError(f"unterminated catalog array for {pending_key!r}")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group", help="build one group from operator_groups.toml")
    selection.add_argument(
        "--operators",
        "--operator",
        nargs="+",
        dest="operators",
        metavar="NAME",
        help="build one or more registered operator names",
    )
    parser.add_argument(
        "--list", action="store_true", help="list operators and groups, then exit"
    )
    parser.add_argument(
        "--target", choices=("dav-2201", "dav-3510"), default="dav-3510"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT / "output",
        help="root for final artifacts, separated by selection",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=PROJECT / ".build",
        help="root for intermediate artifacts, separated by selection",
    )
    return parser.parse_args(argv)


def choose_operators(
    args: argparse.Namespace,
    operators: dict[str, list[str]],
    groups: dict[str, list[str]],
) -> tuple[list[str], Path]:
    if not operators:
        raise ValueError(
            "no operators are registered; add entries to "
            "net/native_package/operator_groups.toml before building"
        )
    if args.group:
        if args.group not in groups:
            raise ValueError(
                f"unknown group {args.group!r}; available groups: {', '.join(groups) or '(none)'}"
            )
        return groups[args.group], Path("groups") / args.group

    if args.operators:
        file_to_operator = {
            filename: operator
            for operator, filenames in operators.items()
            for filename in filenames
        }
        requested = [
            name if name in operators else file_to_operator.get(Path(name).name, name)
            for name in args.operators
        ]
        unknown = [name for name in requested if name not in operators]
        if unknown:
            raise ValueError(
                f"unknown operators: {', '.join(unknown)}; available operators: "
                f"{', '.join(operators)}"
            )
        selected = list(dict.fromkeys(requested))
        label = selected[0] if len(selected) == 1 else "__".join(selected)
        return selected, Path("operators") / label

    return list(operators), Path("all")


def print_catalog(
    operators: dict[str, list[str]], groups: dict[str, list[str]]
) -> None:
    print("Operators:")
    for name, files in operators.items():
        print(f"  {name}: {', '.join(files)}")
    print("Groups:")
    if not groups:
        print("  (none)")
    for group, names in groups.items():
        print(f"  {group}: {', '.join(names)}")


def configure_intermediate_environment(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    temp_root = work / "tmp"
    temp_root.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["CANNBOTDSL_CACHE_DIR"] = str(work / "cache")
    os.environ["PYTHONPYCACHEPREFIX"] = str(work / "pycache")
    os.environ["PIP_CACHE_DIR"] = str(work / "pip-cache")
    tempfile.tempdir = str(temp_root)


def selected_filenames(
    selected: list[str],
    operators: dict[str, list[str]],
) -> list[str]:
    """Source files to ship, de-duplicated and in a stable order."""
    return list(
        dict.fromkeys(
            filename for operator in selected for filename in operators[operator]
        )
    )


def materialize_flat_sources(
    destination: Path,
    selected: list[str],
    operators: dict[str, list[str]],
) -> None:
    """Copy the selected sources into one flat directory.

    Two consumers need this layout: the native collector imports each module by
    filename from a single directory, and the samples' sibling imports resolve
    either through the packaged ``ops`` package or as flat modules.
    """
    destination.mkdir(parents=True, exist_ok=True)
    index = source_index()
    for filename in selected_filenames(selected, operators):
        shutil.copy2(index[filename], destination / filename)


def stage_python_package(
    package: Path,
    selected: list[str],
    operators: dict[str, list[str]],
    sources: Path,
) -> None:
    package.mkdir()
    for name in ("pyproject.toml", "setup.py"):
        shutil.copy2(PROJECT / name, package / name)

    staged_ops = package / "src" / "ops"
    staged_ops.mkdir(parents=True)
    (staged_ops / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from cannbotdsl.aot import register_directory\n"
        "\n"
        "register_directory(Path(__file__).resolve().parent / '_native')\n",
        encoding="utf-8",
    )
    for filename in selected_filenames(selected, operators):
        shutil.copy2(sources / filename, staged_ops / filename)


def build_aicpu_metadata(
    selected: list[str], output: Path, work: Path, target: str
) -> None:
    """Compile each metadata kernel separately to avoid same-name collisions."""
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

    metadata = {
        "mqsmla": (
            "mixed_quant_sparse_flash_mla_metadata",
            "_mqsmla_metadata_kernel",
            "mqsmla_metadata_kernel.so",
        ),
        "qli": (
            "quant_lightning_indexer_metadata",
            "metadata_kernel",
            "metadata_kernel.so",
        ),
        "qsli": (
            "quant_sparse_lightning_indexer_metadata",
            "metadata_kernel",
            "qsli_metadata_kernel.so",
        ),
    }
    for operator in selected:
        if operator not in metadata:
            continue
        module_name, kernel_name, filename = metadata[operator]
        module = importlib.import_module(module_name)
        compiled = compile_aicpu_kernel(
            getattr(module, kernel_name),
            workdir=str(work / operator),
            launch_mode="interface",
            npu_arch=target,
        )
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled.so_path, output / filename)


def publish(final: Path, destination: Path, work: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = Path(tempfile.mkdtemp(prefix="previous-output-", dir=work))
        shutil.move(destination, previous / "artifacts")
        print(f"Previous successful artifacts: {previous / 'artifacts'}")
    shutil.move(final, destination)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        operators, groups = load_catalog()
        if args.list:
            print_catalog(operators, groups)
            return 0
        selected, selection_path = choose_operators(args, operators, groups)
    except ValueError as error:
        raise SystemExit(f"configuration error: {error}") from error

    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    output = output_root / selection_path
    work = work_root / selection_path
    for generated in (output_root, work_root):
        if generated == OPS_ROOT or OPS_ROOT in generated.parents:
            raise SystemExit(
                "output and work roots must be outside the operator sources"
            )
    if (
        output_root == work_root
        or output_root in work_root.parents
        or work_root in output_root.parents
    ):
        raise SystemExit("output and work roots must not overlap")

    configure_intermediate_environment(work)

    from cannbotdsl.aot import build

    with tempfile.TemporaryDirectory(prefix="native-build-", dir=work) as temp:
        stage = Path(temp)
        # Flatten the per-operator sample directories. The collector imports
        # modules by filename, and the samples' sibling imports resolve against
        # this directory; the staged wheel ships the same set.
        sources = stage / "sources"
        materialize_flat_sources(sources, selected, operators)
        if str(sources) not in sys.path:
            sys.path.insert(0, str(sources))

        package = stage / "package"
        stage_python_package(package, selected, operators, sources)

        final = stage / "final"
        final.mkdir()
        exported = build(
            collectors=sources,
            operators=selected,
            target=args.target,
            output=final / "native",
        )
        shutil.copytree(exported, package / "src" / "ops" / "_native")
        build_aicpu_metadata(
            selected, final / "aicpu", stage / "aicpu-work", args.target
        )
        if (final / "aicpu").exists():
            shutil.copytree(final / "aicpu", package / "src" / "ops" / "_aicpu")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                "--wheel-dir",
                str(final / "wheels"),
                str(package),
            ],
            check=True,
            cwd=stage,
        )
        publish(final, output, work)

    print(f"Selection: {', '.join(selected)}")
    print(f"Final binaries: {output / 'native'}")
    print(f"Final wheels: {output / 'wheels'}")
    print(f"Intermediate files: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
