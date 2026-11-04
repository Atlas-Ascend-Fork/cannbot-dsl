# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Distributed-test session cleanup.

Flushes test output and forwards the pytest exit status during process shutdown."""

import atexit
import sys
from os import _exit as _os_hard_exit

_EXIT_CODE = [1]  # 缺省 1：未跑到 sessionfinish 的异常退出不报假绿


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "full: 全量矩阵用例（默认门禁不跑，--full 启用）"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", help="跑全量矩阵（含 full 标记用例）"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return
    items[:] = [i for i in items if "full" not in i.keywords]


def pytest_sessionfinish(session, exitstatus):
    _EXIT_CODE[0] = int(exitstatus)


def _hard_exit():
    sys.stdout.flush()
    sys.stderr.flush()
    _os_hard_exit(_EXIT_CODE[0])


atexit.register(_hard_exit)
