# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Monotonic CI phase timings; measurement must preserve command exit status."""

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time

_LOCK = threading.Lock()
_FIELDS = ["timestamp_utc", "phase", "scope", "elapsed_seconds", "exit_code"]
_LOGGER = logging.getLogger(__name__)


def _append_record(path, row):
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


def record(path, phase, scope, elapsed, exit_code):
    row = dict(
        zip(
            _FIELDS,
            [
                datetime.now(timezone.utc).isoformat(),
                phase,
                scope,
                f"{elapsed:.6f}",
                exit_code,
            ],
        )
    )
    _LOGGER.info(
        "[TIMING] phase=%s scope=%s elapsed_seconds=%.3f exit_code=%s",
        phase,
        scope,
        elapsed,
        exit_code,
    )
    try:
        _append_record(Path(path), row)
    except OSError as error:
        _LOGGER.warning("[WARN] cannot save timing report: %s", error)


class Timer:
    def __init__(self, path, phase, scope):
        self.path, self.phase, self.scope = path, phase, scope
        self.exit_code = 0
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, kind, value, traceback):
        self.elapsed = time.perf_counter() - self.started
        code = self.exit_code if kind is None else 1
        record(self.path, self.phase, self.scope, self.elapsed, code)
        return False


def run_command(command, path, phase, scope):
    with Timer(path, phase, scope) as timer:
        try:
            code = subprocess.run(command).returncode
            timer.exit_code = code if code >= 0 else 128 - code
        except OSError as error:
            _LOGGER.error("[ERROR] cannot start timed command: %s", error)
            timer.exit_code = 127
        return timer.exit_code


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    return run_command(command, args.output, args.phase, args.scope)


if __name__ == "__main__":
    raise SystemExit(main())
