# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AOT export coverage for indexer_prologue_qw."""

from __future__ import annotations

from importlib import import_module

op = import_module("indexer_prologue_qw.indexer_prologue_qw")


class _Artifact:
    def close(self) -> None:
        pass


def _record_export(monkeypatch, t_values):
    compiled: list[str] = []
    monkeypatch.setattr(op, "EXPORT_T", t_values)
    monkeypatch.setattr(op, "resolve_cube_cores", lambda cube_cores=None: 36)

    def fake_compile(fn, *specs):
        compiled.append(specs[0].shape[0].name)
        return _Artifact()

    monkeypatch.setattr(op, "dsl_compile", fake_compile)
    op.export_indexer_prologue_qw()
    return compiled


def test_default_export_covers_every_plan_bucket(monkeypatch) -> None:
    """Unset ``IPQW_EXPORT_T`` packages every template, not only DEPLOYMENT_T.

    T is dynamic, but the head split is a compile-time shape key.  Restricting
    the AOT build to the deployment matrix leaves the T ranges served by the
    other keys without a shipped binary and forces a JIT there.
    """
    compiled = _record_export(monkeypatch, None)
    geo = op.Geometry(
        op.FROZEN_DIM,
        op.FROZEN_Q_LORA,
        op.FROZEN_N_HEADS,
        op.FROZEN_D,
        op.FROZEN_DR,
        36,
    )
    expected = [f"T{i}" for i in range(len(op.plan_buckets(geo)))]
    assert compiled == expected


def test_export_t_subset_builds_the_5620_template(monkeypatch) -> None:
    """An explicit ``IPQW_EXPORT_T`` still narrows the build to its templates."""
    compiled = _record_export(monkeypatch, (5620,))
    assert compiled == ["T11"]
