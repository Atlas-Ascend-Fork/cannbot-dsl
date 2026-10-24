# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep 测试目录级收尾：报告打印完成后硬退出，跳过 ElasticBuffer 逐个销毁。

本机 ElasticBuffer Destroy 的 HcclBarrier 建 AICPU_TS 通道必等 ~120s/个，
池化 7 个 buffer 串行销毁会把作业拖过调度窗口（测试结果有效但退出码 124）。
池引用保活到进程结束，硬退出跳过 GC 析构，设备资源由驱动随进程回收。
pytest 退出码经 pytest_sessionfinish 捕获并透传，失败不被掩盖。
"""

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
