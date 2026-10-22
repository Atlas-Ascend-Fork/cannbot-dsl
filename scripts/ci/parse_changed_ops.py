#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Map PR changed files to the operators that need verification.

The CI writes the changed-file list (from ``git diff``) into a file, then calls
this script to decide which operators must be compiled/tested.

Usage:
    python3 scripts/ci/parse_changed_ops.py <changed_list> \
        [--config test/test_config.yaml] [--op-list scripts/ci/operator_list.yaml]

Input format of <changed_list> (one file per line):
    samples/matmul/matmul.py            (git diff --name-only)
    M\tsamples/matmul/matmul.py         (git diff --name-status)

Output (stdout):
    op1;op2;op3   — operators to verify
    all           — full-suite verification (a shared file changed)
    ""            — no operator affected (or the skip_all_tests switch is on)

``skip_all_tests`` in the rule file is an escape hatch for landing windows: when
it is ``true`` configuration is still validated, but no operator test runs,
while the remaining pipeline stages keep working.

``module: false`` keeps a routing node as an explicit opt-out: changes matching
its sources do not select tests or require inventory entries. Other routing
nodes and shared-file full-suite triggers remain independent.
"""

import argparse
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 结果输出专用 logger：写 stdout 且无前缀，供 CI 读取 ops 结果
result_logger = logging.getLogger("result")
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter("%(message)s"))
result_logger.addHandler(_stdout_handler)
result_logger.setLevel(logging.INFO)
result_logger.propagate = False

_SKIP_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".svg", ".gif")
_STATUS_PREFIXES = ("M\t", "A\t", "D\t", "R\t", "C\t")


class Module:
    """A rule node in test_config.yaml: guarded sources -> related options."""

    def __init__(self, name, desc):
        self.name = name
        self.enabled = desc.get("module", True)
        src = desc.get("src", [])
        self.src_files = [Path(p) for p in (src if isinstance(src, list) else [src])]
        options = desc.get("options", [])
        self.options = list(options) if isinstance(options, list) else [options]

    def matches(self, file_path):
        changed = Path(file_path)
        for src in self.src_files:
            try:
                changed.relative_to(src)
                return True
            except ValueError:
                continue
        return False


def load_modules(config_path):
    """Recursively walk the yaml tree and collect all module nodes."""
    with open(config_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f)

    modules = []

    def walk(obj, prefix=""):
        if not isinstance(obj, dict):
            return
        if "module" in obj:
            modules.append(Module(prefix, obj))
            return
        for key, value in obj.items():
            walk(value, f"{prefix}/{key}" if prefix else key)

    walk(desc)
    return modules


def load_operator_list(op_list_path):
    with open(op_list_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f)
    return desc.get("operators", [])


def load_ignored_paths(config_path):
    """Load exact paths that do not affect the operator test scope."""
    with open(config_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f)

    ignored = desc.get("ignore", [])
    if not isinstance(ignored, list):
        ignored = [ignored]
    return {Path(path) for path in ignored}


def is_test_skipped(config_path):
    """Return True when the global 'skip_all_tests' switch is turned on.

    The switch is meant for landing windows with a large batch of pending
    changes: the pipeline still runs, but no operator is verified.
    """
    with open(config_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f) or {}

    return bool(desc.get("skip_all_tests", False))


def is_skippable(file_path):
    return file_path.endswith(_SKIP_SUFFIXES)


def normalize_line(line):
    """Strip git diff --name-status prefixes such as 'M\\tfile'."""
    line = line.strip()
    for prefix in _STATUS_PREFIXES:
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return line


def matching_module(modules, line):
    """Return the first matching rule, including explicit disabled rules."""
    for mod in modules:
        if mod.matches(line):
            return mod
    return None


class MissingOperatorConfig(ValueError):
    """An operator change has no valid routing or inventory entry."""


def target_is_registered(target, operators):
    """Allow inventory directories to cover nested sample paths."""
    target = Path(target)
    return any(target == Path(op) or Path(op) in target.parents for op in operators)


def warn_missing_config(missing_samples, missing_tests, config_path, op_list_path):
    """Show actionable YAML examples for only the configuration that is missing."""
    required = []
    if missing_samples:
        names = list(
            dict.fromkeys(
                Path(path).parent.relative_to("samples").as_posix()
                for path in missing_samples
            )
        )
        logger.warning(
            "New operators must be registered in %s.\n"
            "Changed files missing configuration:\n  %s\n"
            "Example configuration (merge into the existing operators list "
            "and adjust the operator paths as needed):\n%s",
            op_list_path,
            "\n  ".join(dict.fromkeys(missing_samples)),
            yaml.safe_dump({"operators": names}, sort_keys=False),
        )
        required.append(str(op_list_path))
    if missing_tests:
        examples = {}
        for path in missing_tests:
            name = Path(path).parent.relative_to("test").as_posix()
            examples[name.replace("/", "_")] = {
                "module": True,
                "src": [f"samples/{name}", f"test/{name}"],
                "options": [name],
            }
        logger.warning(
            "New operator test files require a valid targeted rule in %s.\n"
            "Changed files missing configuration:\n  %s\n"
            "Example configuration (adjust the operator paths as needed):\n%s",
            config_path,
            "\n  ".join(dict.fromkeys(missing_tests)),
            yaml.safe_dump(examples, sort_keys=False),
        )
        required.append(str(config_path))
    raise MissingOperatorConfig(
        "Please update " + " and ".join(required) + " and trigger CI again."
    )


@dataclass
class RoutingConfig:
    """Related rules, inventory and paths used to route a changed file."""

    modules: list[Module]
    ignored_paths: set[Path]
    operators: list[str]
    config_path: Path
    op_list_path: Path


def route_changed_file(line, routing):
    """Return selected tests and the missing configuration category, if any."""
    module = matching_module(routing.modules, line)
    if module is not None and not module.enabled:
        logger.info("module disabled, skip tests: %s (%s)", module.name, line)
        return [], None
    options = module.options if module is not None else []
    path = Path(line)
    if len(path.parts) < 3:
        return options, None
    if path.parts[0] == "samples":
        if not target_is_registered(path.relative_to("samples"), routing.operators):
            return [], "samples"
        if not options:
            logger.info("sample registered without test routing: %s", line)
    elif path.parts[0] == "test" and (not options or "all" in options):
        return [], "test"
    return options, None


def collect_related_ops(routing, changed_list_path):
    """Collect test targets, rejecting operator changes without configuration."""
    related = []
    missing_files = {"samples": [], "test": []}
    with open(changed_list_path, encoding="utf-8") as f:
        for line in f:
            line = normalize_line(line)
            if not line or line.startswith("#"):
                continue
            if Path(line) in routing.ignored_paths:
                logger.info("ignored metadata file: %s", line)
                continue
            path = Path(line)
            if path.parent == Path("test"):
                related.append("all")
                continue
            if is_skippable(line):
                continue
            options, missing = route_changed_file(line, routing)
            if missing is not None:
                missing_files.setdefault(missing, []).append(line)
                continue
            related.extend(options)
    if any(missing_files.values()):
        warn_missing_config(
            missing_files["samples"],
            missing_files["test"],
            routing.config_path,
            routing.op_list_path,
        )
    return list(dict.fromkeys(related))


def resolve_ops_output(related, op_list_path):
    """Convert collected options to the final ops string for stdout."""
    if not related:
        logger.info("no operators affected by this change")
        return ""
    if "all" in related:
        operators = load_operator_list(op_list_path)
        logger.info("shared file changed, triggering full suite: %s", operators)
        return "all"
    logger.info("related operators: %s", related)
    return ";".join(related)


def main():
    parser = argparse.ArgumentParser(
        description="Parse changed files and print related operators."
    )
    parser.add_argument("changed_list", type=Path, help="changed-file list from CI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("test/test_config.yaml"),
        help="rule file (test/test_config.yaml)",
    )
    parser.add_argument(
        "--op-list",
        type=Path,
        default=Path("scripts/ci/operator_list.yaml"),
        help="full operator list file",
    )
    args = parser.parse_args()

    if not args.changed_list.exists():
        logger.error("changed-list file not found: %s", args.changed_list)
        return 1, ""
    if not args.config.exists():
        logger.error("config file not found: %s", args.config)
        return 1, ""

    if not args.op_list.exists():
        logger.warning("operator-list file not found: %s", args.op_list)
        return 2, ""

    routing = RoutingConfig(
        modules=load_modules(args.config),
        ignored_paths=load_ignored_paths(args.config),
        operators=load_operator_list(args.op_list),
        config_path=args.config,
        op_list_path=args.op_list,
    )
    try:
        related = collect_related_ops(routing, args.changed_list)
    except MissingOperatorConfig as exc:
        logger.warning("%s", exc)
        return 2, ""

    if is_test_skipped(args.config):
        logger.warning(
            "skip_all_tests is enabled in %s: no operator test will run",
            args.config,
        )
        return 0, ""

    return 0, resolve_ops_output(related, args.op_list)


if __name__ == "__main__":
    ret_code, ops_output = main()
    result_logger.info("%s", ops_output)
    sys.exit(ret_code)
