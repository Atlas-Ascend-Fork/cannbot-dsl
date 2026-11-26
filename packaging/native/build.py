# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build selected sample operators into isolated native and wheel artifacts.

Sources are explicitly registered relative to the repository's ``samples/``
tree. Selected sources use their filenames as wheel module names;
colliding filenames are rejected before compilation or artifact publication.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from threading import Event, Thread
from time import monotonic

PROJECT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT.parents[1]
OPS_ROOT = REPO_ROOT / "samples"
sys.dont_write_bytecode = True
SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")
AICORE_PROGRESS_INTERVAL = 30
SOC_TARGETS = {
    "ascend950": "dav-3510",
}


def load_catalog() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    catalog = json.loads((PROJECT / "operators.json").read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported Native catalog schema_version")
    definitions = catalog["operators"]
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("Native operators must be a nonempty object")
    operators = {}
    exports = {}
    samples = OPS_ROOT.resolve()
    for name, definition in definitions.items():
        if not SAFE_ID.fullmatch(name):
            raise ValueError(f"invalid operator ID: {name!r}")
        export = definition["export"]
        if (
            not isinstance(export, str)
            or not SAFE_ID.fullmatch(export)
            or export in operators
        ):
            raise ValueError(f"invalid or duplicate Native export: {export!r}")
        files = definition["sources"]
        if not isinstance(files, list) or not files:
            raise ValueError(f"Native sources must be a nonempty list: {name}")
        normalized = []
        for filename in files:
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"invalid Native source: {filename!r}")
            path = Path(filename)
            source = (samples / path).resolve()
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in filename
                or path.suffix != ".py"
                or not source.is_relative_to(samples)
                or not source.is_file()
            ):
                raise ValueError(f"invalid Native source: {filename!r}")
            normalized.append(path.as_posix())
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"duplicate Native sources: {name}")
        operators[export] = normalized
        exports[name] = export

    manifest = PROJECT.parent / "net/network.json"
    network = json.loads(manifest.read_text(encoding="utf-8"))
    if network.get("schema_version") != 1:
        raise ValueError("unsupported network schema_version")
    definitions = network["groups"]
    if not isinstance(definitions, dict):
        raise ValueError("network groups must be an object")
    groups = {}
    for group, names in definitions.items():
        if not SAFE_ID.fullmatch(group):
            raise ValueError(f"invalid network group: {group!r}")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
        ):
            raise ValueError(
                f"network operators must be a nonempty list of IDs: {group}"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate network operators: {group}")
        if any(name not in exports for name in names):
            raise ValueError(f"network {group} references an unknown Native operator")
        groups[group] = [exports[name] for name in names]
    return operators, groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--group", help="build one group from packaging/net/network.json"
    )
    selection.add_argument(
        "--operators",
        "--operator",
        nargs="+",
        action="extend",
        dest="operators",
        metavar="NAME",
        help="build one or more registered operator names",
    )
    parser.add_argument(
        "--list", action="store_true", help="list operators and groups, then exit"
    )
    parser.add_argument(
        "--soc",
        choices=SOC_TARGETS,
        default="ascend950",
        help="target SoC (default: %(default)s)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=32,
        help="cap parallel operator builds (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show CANNBotDSL compiler logs and kernel progress",
    )
    parser.add_argument(
        "--no-partial",
        dest="allow_partial",
        action="store_false",
        help="publish a wheel only if all selected operators build successfully",
    )
    parser.add_argument(
        "--allow-partial",
        dest="allow_partial",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "output",
        help="directory for the built wheel",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT / ".build",
        help="directory for the compiler cache and temporary build files",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def choose_operators(
    args: argparse.Namespace,
    operators: dict[str, list[str]],
    groups: dict[str, list[str]],
) -> list[str]:
    if not operators:
        raise ValueError(
            "no operators are registered; add entries to "
            "packaging/native/operators.json before building"
        )
    if args.group:
        if args.group not in groups:
            raise ValueError(
                f"unknown group {args.group!r}; available groups: {', '.join(groups) or '(none)'}"
            )
        return groups[args.group]

    if args.operators:
        file_to_operator = {
            filename: operator
            for operator, filenames in operators.items()
            for filename in filenames
        }
        basename_exports: dict[str, set[str]] = {}
        for operator, filenames in operators.items():
            for filename in filenames:
                basename_exports.setdefault(Path(filename).name, set()).add(operator)
        requested = []
        for name in args.operators:
            if name in operators:
                requested.append(name)
            elif name in file_to_operator:
                requested.append(file_to_operator[name])
            else:
                matches = basename_exports.get(Path(name).name, set())
                if len(matches) > 1:
                    raise ValueError(
                        f"ambiguous source filename {name!r}; select an export or full source path"
                    )
                requested.append(next(iter(matches)) if matches else name)
        unknown = [name for name in requested if name not in operators]
        if unknown:
            raise ValueError(
                f"unknown operators: {', '.join(unknown)}; available operators: "
                f"{', '.join(operators)}"
            )
        return list(dict.fromkeys(requested))

    return list(operators)


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


@contextmanager
def configure_intermediate_environment(work: Path, stage: Path):
    temp_root = stage / "tool-tmp"
    temp_root.mkdir()
    overrides = {
        "TMPDIR": str(temp_root),
        "CANNBOTDSL_CACHE_DIR": str(work / "cache"),
        "PYTHONPYCACHEPREFIX": str(temp_root / "pycache"),
        "PIP_CACHE_DIR": str(temp_root / "pip-cache"),
    }
    previous = {name: os.environ.get(name) for name in overrides}
    previous_tempdir = tempfile.tempdir
    os.environ.update(overrides)
    tempfile.tempdir = str(temp_root)
    try:
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def observe_aicore_artifacts(aot, label, started, output):
    def report_artifact(artifact, source):
        print(
            f"  {label}Kernel ready: {source.__name__} "
            f"({monotonic() - started:.0f}s elapsed)",
            file=output,
            flush=True,
        )

    # The public build API has no progress callback. Older DSL versions can
    # still report elapsed time if their collection module lacks the observer.
    observe = getattr(getattr(aot, "collection", None), "observe_compilations", None)
    return observe(report_artifact) if observe is not None else nullcontext()


@contextmanager
def report_aicore_progress(aot, output):
    started = monotonic()
    done = Event()

    def heartbeat():
        while not done.wait(AICORE_PROGRESS_INTERVAL):
            print(
                f"  Still compiling ({monotonic() - started:.0f}s)",
                file=output,
                flush=True,
            )

    thread = Thread(target=heartbeat)
    thread.start()
    try:
        with observe_aicore_artifacts(aot, "", started, output):
            yield
    finally:
        done.set()
        thread.join()
    print(
        f"  Compilation finished in {monotonic() - started:.1f}s",
        file=output,
        flush=True,
    )


@contextmanager
def compiler_output(verbose: bool):
    """Keep explicit build events and DSL logs apart from dependency output."""
    with open(os.devnull, "wb") as dependency_output, ExitStack() as stack:
        output = stack.enter_context(os.fdopen(os.dup(sys.stdout.fileno()), "w"))
        streams = (sys.stdout, sys.stderr)
        for stream in streams:
            stream.flush()
            fd = stream.fileno()
            saved = os.dup(fd)
            stack.callback(os.close, saved)
            stack.callback(os.dup2, saved, fd)
            os.dup2(dependency_output.fileno(), fd)
        if verbose:
            # The package configures its logger on first import.
            importlib.import_module("cannbotdsl")
            logger = logging.getLogger("cannbotdsl")
            handler = logging.StreamHandler(output)
            handler.setFormatter(
                logging.Formatter("%(levelname)s %(name)s: %(message)s")
            )
            stack.callback(logger.setLevel, logger.level)
            stack.callback(setattr, logger, "propagate", logger.propagate)
            stack.callback(logger.removeHandler, handler)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        try:
            yield output
        finally:
            for stream in streams:
                stream.flush()


def compile_operator(
    operator: str,
    sources: Path,
    target: str,
    work: Path,
    shard: Path,
    verbose: bool,
    interactive: bool,
) -> tuple[Path, float]:
    started = monotonic()
    shard.mkdir()
    if not interactive:
        print(f"  {operator} started", flush=True)
    output = shard / "artifacts"
    with (
        compiler_output(verbose) as progress,
        configure_intermediate_environment(work, shard),
    ):
        from cannbotdsl import aot

        observer = (
            observe_aicore_artifacts(aot, f"{operator}: ", started, progress)
            if verbose
            else nullcontext()
        )
        with observer:
            aot.build(
                collectors=sources, operators=[operator], target=target, output=output
            )
    return output, monotonic() - started


class AicoreTerminalProgress:
    def __init__(self):
        self.results = []
        self.rows = 0
        self.color = not os.getenv("NO_COLOR")

    def colorize(self, message, color):
        return f"\x1b[{color}m{message}\x1b[0m" if self.color else message

    def show(self, status):
        columns, lines = shutil.get_terminal_size()
        visible_results = max(1, lines - 3)
        contents = [(status, "1;36"), *self.results[-visible_results:]]
        width = max(1, columns - 1)
        rendered = [
            self.colorize(
                line
                if len(line) <= width
                else line[: max(0, width - 3)] + "." * min(3, width),
                color,
            )
            for line, color in contents
        ]
        # Truncate before adding color so the cursor moves by physical lines.
        rewind = f"\x1b[{self.rows}F" if self.rows else ""
        sys.stdout.write(rewind + "\x1b[J" + "\n".join(rendered) + "\n")
        sys.stdout.flush()
        self.rows = len(rendered)


@contextmanager
def connect_jobserver(auth: str):
    """Open a nonblocking reader without changing Make's inherited pipe flags."""
    with ExitStack() as stack:
        try:
            if auth.startswith("fifo:"):
                reader = writer = os.open(
                    auth.removeprefix("fifo:"),
                    os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW,
                )
                stack.callback(os.close, reader)
                if not stat.S_ISFIFO(os.fstat(reader).st_mode):
                    raise OSError("jobserver path is not a FIFO")
            else:
                read_fd, write_fd = map(int, auth.split(","))
                read_stat = os.fstat(read_fd)
                write_stat = os.fstat(write_fd)
                if not stat.S_ISFIFO(read_stat.st_mode) or not os.path.samestat(
                    read_stat, write_stat
                ):
                    raise OSError(
                        "jobserver descriptors do not reference the same pipe"
                    )
                # Reopen the inherited pipe on Linux to get independent file
                # status flags. dup()/set_blocking() would change Make's reader.
                reader = os.open(
                    f"/proc/self/fd/{read_fd}", os.O_RDONLY | os.O_NONBLOCK
                )
                stack.callback(os.close, reader)
                writer = write_fd
        except (OSError, ValueError) as error:
            print(
                f"  Make jobserver unavailable ({error}); using one job",
                file=sys.stderr,
            )
            yield None
            return
        yield reader, writer


@contextmanager
def defer_jobserver_interrupts(enabled):
    interrupted = None

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def check():
        if interrupted is not None:
            raise SystemExit(128 + interrupted)

    with ExitStack() as stack:
        if enabled:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous = signal.signal(signum, interrupt)
                stack.callback(signal.signal, signum, previous)
        yield check


def compile_parallel(
    selected,
    sources,
    target,
    work,
    stage,
    jobs,
    jobserver,
    verbose,
    group,
    allow_partial,
):
    pending = iter(enumerate(selected))
    active = {}
    shards = [None] * len(selected)
    next_operator = next(pending, None)
    completed = 0
    started = monotonic()
    next_report = started + AICORE_PROGRESS_INTERVAL
    interactive = (
        sys.stdout.isatty()
        and os.getenv("TERM") not in (None, "", "dumb")
        and not verbose
    )
    screen = AicoreTerminalProgress() if interactive else None
    scope = f"[{group}] " if group else ""

    def show_running():
        names = ", ".join(selected[index] for index, _ in active.values()) or "-"
        status = f"  {scope}[{completed}/{len(selected)}] running: {names}"
        screen.show(status)

    with defer_jobserver_interrupts(jobserver is not None) as check_interrupt:
        try:
            with ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                while next_operator is not None or active:
                    check_interrupt()
                    submitted = False
                    while next_operator is not None and len(active) < jobs:
                        token = None
                        if jobserver is not None and any(
                            slot is None for _, slot in active.values()
                        ):
                            try:
                                token = os.read(jobserver[0], 1)
                            except BlockingIOError:
                                break
                            if not token:
                                break
                        index, operator = next_operator
                        try:
                            future = executor.submit(
                                compile_operator,
                                operator,
                                sources,
                                target,
                                work,
                                stage / f"aicore-{operator}",
                                verbose,
                                interactive,
                            )
                        except BaseException:
                            if token is not None:
                                os.write(jobserver[1], token)
                            raise
                        active[future] = index, token
                        next_operator = next(pending, None)
                        submitted = True
                    if screen and submitted:
                        show_running()
                    timeout = None if screen else max(0, next_report - monotonic())
                    if jobserver:
                        timeout = 0.1 if timeout is None else min(timeout, 0.1)
                    done, _ = wait(active, timeout=timeout, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, token = active.pop(future)
                        if token is not None:
                            os.write(jobserver[1], token)
                        operator = selected[index]
                        try:
                            shards[index], duration = future.result()
                        except Exception as error:
                            if allow_partial:
                                message = f"  {operator} FAILED"
                                if screen:
                                    screen.results.append((message, "1;31"))
                                    show_running()
                                else:
                                    print(
                                        f"{message}: {error}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                continue
                            if screen:
                                show_running()
                            failure = f"  {operator} failed; waiting for running workers to exit"
                            if screen and sys.stderr.isatty():
                                failure = screen.colorize(failure, "1;31")
                            print(failure, file=sys.stderr, flush=True)
                            raise RuntimeError(f"AICore {operator} failed") from error
                        completed += 1
                        message = (
                            f"  [{completed}/{len(selected)}] "
                            f"{operator} finished in {duration:.1f}s"
                        )
                        if screen:
                            screen.results.append((message, "32"))
                            show_running()
                        else:
                            print(message, flush=True)
                    now = monotonic()
                    if active and now >= next_report and not screen:
                        names = ", ".join(
                            selected[index] for index, _ in active.values()
                        )
                        print(
                            f"  {scope}[{completed}/{len(selected)}] {now - started:.0f}s elapsed"
                            f" | running: {names}",
                            flush=True,
                        )
                        next_report = now + AICORE_PROGRESS_INTERVAL
                    check_interrupt()
        finally:
            # The executor has joined outstanding workers before slots are freed.
            for _, token in active.values():
                if token is not None:
                    os.write(jobserver[1], token)
        check_interrupt()
    return shards


def merge_native_shards(
    selected: list[str], shards: list[Path], output: Path, target: str
) -> None:
    """Merge the Native layout emitted by DSL, retaining first-owner paths."""
    metadata = None
    files = {}
    index = {}
    output.mkdir()

    def write_json(relative, value):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        path.write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()

    for operator, shard in zip(selected, shards, strict=True):
        manifest = json.loads((shard / "manifest.json").read_bytes())
        shard_files = manifest.pop("files")
        if manifest["target"] != target or manifest["index"] != "index.json":
            raise ValueError(f"unexpected Native manifest for {operator}")
        if metadata is None:
            metadata = manifest
        elif manifest != metadata:
            raise ValueError(f"incompatible Native manifest for {operator}")

        operator_path = f"operators/{operator}/operator.json"
        info = json.loads((shard / operator_path).read_bytes())
        variants = json.loads((shard / "index.json").read_bytes())
        if info["name"] != operator or info["variants"] != variants:
            raise ValueError(f"inconsistent Native index for {operator}")
        for key, entry in variants.items():
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError(f"invalid Native binary key for {operator}: {key}")
            base = f"operators/{operator}/variants/{key}"
            if entry != {"descriptor": f"{base}/variant.json"}:
                raise ValueError(f"unexpected Native variant layout for {operator}")
            artifacts = ("provider.so", "provider.so.meta.json", "variant.json")
            if key in index:
                owner = index[key]["descriptor"].rsplit("/", 1)[0]
                if any(
                    shard_files[f"{base}/{name}"] != files[f"{owner}/{name}"]
                    for name in artifacts[:2]
                ):
                    raise ValueError(
                        f"different Native artifacts share binary key {key}"
                    )
                continue
            destination = output / base
            destination.mkdir(parents=True)
            for name in artifacts:
                relative = f"{base}/{name}"
                shutil.copyfile(shard / relative, output / relative)
                files[relative] = shard_files[relative]
            index[key] = entry
        info["variants"] = {key: index[key] for key in variants}
        write_json(operator_path, info)

    write_json("index.json", index)
    write_json("manifest.json", dict(metadata, files=files))


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
    selected_sources: dict[str, Path] = {}
    filenames = dict.fromkeys(
        filename for operator in selected for filename in operators[operator]
    )
    for filename in filenames:
        source = (OPS_ROOT / filename).resolve()
        previous = selected_sources.get(source.name)
        if previous is not None and previous != source:
            raise ValueError(
                f"selected sources collide as {source.name!r}: {previous} and {source}"
            )
        selected_sources[source.name] = source
    destination.mkdir(parents=True, exist_ok=True)
    for filename, source in selected_sources.items():
        shutil.copy2(source, destination / filename)


def stage_python_package(
    package: Path,
    sources: Path,
    group: str | None,
) -> None:
    package.mkdir()
    shutil.copy2(PROJECT / "setup.py", package / "setup.py")
    metadata = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    if group is not None:
        metadata, count = re.subn(
            r'^(name = ")([^"\n]+)"$',
            lambda match: f'{match[1]}{match[2]}-{group}"',
            metadata,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError("expected one project name in pyproject.toml")
    (package / "pyproject.toml").write_text(metadata, encoding="utf-8")

    staged_ops = package / "src" / "ops"
    staged_ops.mkdir(parents=True)
    (staged_ops / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from cannbotdsl.aot import register_directory\n"
        "\n"
        "register_directory(Path(__file__).resolve().parent / '_native')\n",
        encoding="utf-8",
    )
    for source in sources.glob("*.py"):
        shutil.copy2(source, staged_ops / source.name)


def build_aicpu_metadata(
    selected: list[str],
    output: Path,
    work: Path,
    target: str,
    *,
    allow_partial=False,
    verbose=False,
) -> list[str]:
    """Compile each metadata kernel separately to avoid same-name collisions."""
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
    required = [operator for operator in selected if operator in metadata]
    if not required:
        print("[3/5] Skipping AICPU metadata (none required)", flush=True)
        return []

    print(f"[3/5] Building AICPU metadata: {', '.join(required)}", flush=True)
    with compiler_output(verbose):
        from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

    failed = []
    for operator in required:
        module_name, kernel_name, filename = metadata[operator]
        try:
            with compiler_output(verbose):
                module = importlib.import_module(module_name)
                compiled = compile_aicpu_kernel(
                    getattr(module, kernel_name),
                    workdir=str(work / operator),
                    launch_mode="interface",
                    npu_arch=target,
                )
        except Exception as error:
            if not allow_partial:
                raise
            print(
                f"  {operator} FAILED (AICPU metadata): {error}",
                file=sys.stderr,
                flush=True,
            )
            failed.append(operator)
            continue
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled.so_path, output / filename)
    return failed


def publish_wheel(wheel: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / wheel.name
    # Stage on the destination filesystem so the completed wheel replaces atomically.
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=output) as temp:
        incoming = Path(temp) / wheel.name
        shutil.move(wheel, incoming)
        incoming.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    makeflags = shlex.split(os.getenv("MAKEFLAGS", ""))
    if not args.list and makeflags and re.fullmatch(r"-?[A-Za-z]+", makeflags[0]):
        if any(flag in makeflags[0] for flag in "nqt"):
            return 0
    auth = next(
        (
            flag.split("=", 1)[1]
            for flag in reversed(makeflags)
            if flag.startswith(("--jobserver-auth=", "--jobserver-fds="))
        ),
        None,
    )
    try:
        operators, groups = load_catalog()
        if args.list:
            print_catalog(operators, groups)
            return 0
        selected = choose_operators(args, operators, groups)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"configuration error: {error}") from error

    target = SOC_TARGETS[args.soc]
    output_root = args.output_dir.resolve()
    work_root = args.work_dir.resolve()
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

    staging_root = work_root / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    print(f"Selection: {', '.join(selected)}", flush=True)
    print(f"SoC: {args.soc}", flush=True)
    print(f"Build workspace: {work_root}", flush=True)
    print("[1/5] Preparing sources and wheel project", flush=True)
    with (
        tempfile.TemporaryDirectory(prefix="build-", dir=staging_root) as temp,
        configure_intermediate_environment(work_root, Path(temp)),
    ):
        stage = Path(temp)
        # Flatten the per-operator sample directories. The collector imports
        # modules by filename, and the samples' sibling imports resolve against
        # this directory; the staged wheel ships the same set.
        sources = stage / "compile-sources"
        materialize_flat_sources(sources, selected, operators)
        if str(sources) not in sys.path:
            sys.path.insert(0, str(sources))

        package = stage / "wheel-project"
        allow_partial = args.allow_partial and len(selected) > 1
        if not allow_partial:
            stage_python_package(package, sources, args.group)

        staged_ops = package / "src" / "ops"
        print(f"[2/5] Compiling AICore kernels ({len(selected)} selected)", flush=True)
        native_output = staged_ops / "_native"
        jobs = min(args.jobs, len(selected))
        connection = connect_jobserver(auth) if auth and jobs > 1 else nullcontext()
        with connection as jobserver:
            if auth and jobserver is None:
                jobs = 1
            if jobs == 1 and not allow_partial:
                with compiler_output(args.verbose) as progress:
                    from cannbotdsl import aot

                    with report_aicore_progress(aot, progress):
                        aot.build(
                            collectors=sources,
                            operators=selected,
                            target=target,
                            output=native_output,
                        )
            else:
                if jobserver:
                    print(
                        f"  Using Make jobserver (up to {jobs} operators)", flush=True
                    )
                shards = compile_parallel(
                    selected,
                    sources,
                    target,
                    work_root,
                    stage,
                    jobs,
                    jobserver,
                    args.verbose,
                    args.group,
                    allow_partial,
                )
                if not allow_partial:
                    merge_native_shards(selected, shards, native_output, target)
        if allow_partial:
            built = {
                operator: shard
                for operator, shard in zip(selected, shards, strict=True)
                if shard is not None
            }
            if not built:
                raise SystemExit(
                    "No operators compiled successfully; no wheel published"
                )
            aicpu_output = stage / "aicpu-output"
            for operator in build_aicpu_metadata(
                list(built),
                aicpu_output,
                stage / "compile-aicpu",
                target,
                allow_partial=True,
                verbose=args.verbose,
            ):
                del built[operator]
            if not built:
                raise SystemExit(
                    "No operators compiled successfully; no wheel published"
                )
            wheel_sources = sources
            distribution_suffix = args.group
            if len(built) != len(selected):
                failed = ", ".join(
                    operator for operator in selected if operator not in built
                )
                print(
                    f"Partial build: {len(built)}/{len(selected)} succeeded; failed: {failed}",
                    flush=True,
                )
                distribution_suffix = (
                    f"{args.group}-partial" if args.group else "partial"
                )
                wheel_sources = stage / "wheel-sources"
                materialize_flat_sources(wheel_sources, list(built), operators)
            stage_python_package(package, wheel_sources, distribution_suffix)
            merge_native_shards(
                list(built), list(built.values()), native_output, target
            )
            if aicpu_output.is_dir():
                aicpu_output.rename(staged_ops / "_aicpu")
        else:
            build_aicpu_metadata(
                selected,
                staged_ops / "_aicpu",
                stage / "compile-aicpu",
                target,
                verbose=args.verbose,
            )
        wheels = stage / "wheel-output"
        print("[4/5] Building wheel", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--quiet",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(package),
            ],
            check=True,
            cwd=stage,
        )
        (wheel,) = wheels.glob("*.whl")
        print(f"[5/5] Publishing wheel to {output_root}", flush=True)
        published = publish_wheel(wheel, output_root)

    print(f"Wheel: {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
