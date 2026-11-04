# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Call tracing for indexer_prologue_qw. Not used by the kernel launch path.

Enable with IPQW_LOG=1 (every call) or IPQW_LOG=once (one line per shape).
``recent_calls`` reads the same ring buffer after an asynchronous device fault.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
from collections import deque

from indexer_prologue_qw.indexer_prologue_qw import ceil_div

_CALL_LOG: deque = deque(maxlen=int(os.environ.get("IPQW_CALL_LOG", 32)))
_LOGGED_ONCE: set = set()


def _describe(record) -> str:
    t, n_tiles, partial_rows, workspace_rows, hblk, grid, plan_t = record
    template = "split-T" if hblk == 1 else "split-K"
    return (
        f"T={t} {template} tiles={n_tiles} hblk={hblk} grid={grid} "
        f"workspace={workspace_rows}({partial_rows}+{grid} marker) "
        f"plan_built_at_T={plan_t}"
    )


def recent_calls() -> str:
    """The last ``IPQW_CALL_LOG`` calls, oldest first, as text."""
    if not _CALL_LOG:
        return "[ipqw] no calls recorded"
    lines = [f"[ipqw] last {len(_CALL_LOG)} call(s), oldest first:"]
    lines += [f"[ipqw]   {_describe(record)}" for record in _CALL_LOG]
    return "\n".join(lines)


def trace_call(op, t, n_tiles, partial_rows, workspace_rows) -> None:
    """Record one host launch and optionally print it. Does not launch the kernel."""
    plan = op.p
    record = (
        t, n_tiles, partial_rows, workspace_rows,
        plan.n_head_blocks, plan.used_cores, plan.t,
    )
    _CALL_LOG.append(record)

    if plan.w_workspace:
        need = plan.n_head_blocks * ceil_div(t, plan.base_m) * plan.base_m
        if partial_rows < need:
            raise RuntimeError(
                f"Workspace too small for T={t}: {partial_rows} partial rows "
                f"allocated, {need} needed.\n" + recent_calls()
            )

    mode = os.environ.get("IPQW_LOG", "")
    if mode == "once":
        key = (plan.shape_key(), t)
        if key in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(key)
    elif mode != "1":
        return
    logger.info("[ipqw] %s", _describe(record))
