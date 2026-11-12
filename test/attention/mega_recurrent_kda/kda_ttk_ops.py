# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime compatibility checks and explicit stream quotas for PA/TH TTK."""

import logging
import os
import sys
from pathlib import Path

import torch

__all__ = []

VALID_CORE_QUOTAS = (8, 16, 24, 28, 32)


def _assert_environment(require_npu: bool) -> None:
    """Check supported runtime versions before executing the operator."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"mega_recurrent_kda TTK assets require CPython 3.12, got {sys.version.split()[0]}"
        )
    if not str(torch.__version__).startswith("2.12."):
        raise RuntimeError(f"expected torch 2.12.x, got {torch.__version__}")
    if require_npu:
        import torch_npu  # noqa: PLC0415

        if not str(torch_npu.__version__).startswith("2.12."):
            raise RuntimeError(
                f"expected torch_npu 2.12.x, got {torch_npu.__version__}"
            )
        import cannbotdsl  # noqa: PLC0415

        if str(cannbotdsl.__version__) != "0.8.2":
            raise RuntimeError(
                f"expected cannbotdsl 0.8.2, got {cannbotdsl.__version__} "
                f"at {getattr(cannbotdsl, '__file__', '?')}"
            )
        if not torch.npu.is_available():
            raise RuntimeError(
                "torch.npu.is_available() is False; NPU backend required"
            )
        logging.info(
            "[mkda-ttk] env python=%s torch=%s torch_npu=%s cannbotdsl=%s npu_count=%d",
            sys.version.split()[0],
            torch.__version__,
            torch_npu.__version__,
            cannbotdsl.__version__,
            torch.npu.device_count(),
        )
    else:
        logging.info(
            "[mkda-ttk] CPU chain path: python=%s torch=%s",
            sys.version.split()[0],
            torch.__version__,
        )


def _available_core_quotas():
    """Quotas supported by both the physical device and the current stream."""
    import cannbotdsl  # noqa: PLC0415

    device_id = torch.npu.current_device()
    hardware = torch.npu.get_device_properties(device_id)
    stream = torch.npu.current_stream(device_id)
    platform = cannbotdsl.get_platform_info(stream=stream)
    cube = min(int(hardware.cube_core_num), int(platform.cube_core_num))
    vector = min(int(hardware.vector_core_num), int(platform.vector_core_num))
    return tuple(
        core for core in VALID_CORE_QUOTAS if core <= cube and 2 * core <= vector
    )


def _call_with_quota(sample, tensors, scalars, cores: int, *, guard_case=False):
    """Invoke the sample inside a stream whose AIC/AIV quota matches the pytest semantics.

    ``cores == 0`` means "max available" (pytest ``max``/``auto``); every invocation
    sets the stream quota explicitly, exactly like
    ``test_precision._check_precision`` does.
    """
    available = _available_core_quotas()
    if not available:
        raise RuntimeError(
            "device/stream exposes no supported AIC quota (needs >= 8 AIC and 16 AIV)"
        )
    quota = available[-1] if not cores else int(cores)
    if guard_case:
        hardware = torch.npu.get_device_properties(torch.npu.current_device())
        if quota > hardware.cube_core_num or 2 * quota > hardware.vector_core_num:
            raise RuntimeError(
                "Requested negative stream quota unavailable on physical device"
            )
    elif quota not in available:
        raise ValueError(
            f"requested cores={quota} is not in the device/stream supported set {available}; "
            "regenerate cases with cores=0 for max/auto or use a supported quota"
        )

    stream = torch.npu.Stream()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=quota, vector_num=2 * quota)
    try:
        with torch.npu.stream(stream):
            out = sample(*tensors, *scalars)
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        torch.npu.reset_stream_limit(stream)
    return out


# Real-input adapters and strict TTK specification.
"""TTK adapter builds real PA tensors deterministically and records their hashes.

The scalar marker only selects TTK's device. Actual inputs are generated from
the same frozen case on NPU/golden/peer; references never consume captured NPU
values. Direct inplace/alias contracts are additionally covered by pytest.
"""
import json
import re
import traceback
from assets import inputs as cases
from assets import inputs as fixtures
from assets import golden


def prepare(case_id):
    c = cases.get(case_id)
    a, _, _ = fixtures.make(c)
    return a, c


def prepare_commit(case_id):
    c = cases.get(case_id)
    return fixtures.make_commit(c), c


def save(kind, case_id, value):
    if not os.environ.get("MKDA_RUN_DIR"):
        return
    root = Path(os.environ["MKDA_RUN_DIR"]).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Keep validation evidence outside repository")
    dest = root / kind
    dest.mkdir(parents=True, exist_ok=True)
    (dest / (case_id + ".json")).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8"
    )


def _span(t):
    if t.ndim and t.shape[0] and t.numel():
        length = 1 + sum(
            (extent - 1) * stride for extent, stride in zip(t.shape, t.stride())
        )
        return t.as_strided((length,), (1,))
    return t.flatten()


def _pool_storage(t):
    import torch_npu

    return _span(t) if torch_npu.get_npu_format(t) in (0, 2) else t


def _snapshot(a, mode):
    pools = tuple(k for k in fixtures.PA_POOLS if k in a)
    return {
        k: (_pool_storage(v) if k in pools else v).detach().cpu().clone()
        for k, v in a.items()
        if isinstance(v, torch.Tensor) and (k in pools)
    }


def _unchanged(actual, before):
    # Compare storage bits: unchanged NaN payloads and signed zero must also survive.
    assert fixtures.digest(actual) == fixtures.digest(before), (
        "Unexpected write outside the designed output region"
    )


def _pool_untouched(t, before, changed):
    after = _span(t).detach().cpu()
    stride = t.stride(0)
    payload = t[0].numel()
    for slot in range(t.shape[0]):
        start = slot * stride
        rows = changed.get(slot, 0)
        offset = rows * t[0, 0].numel() if rows else 0
        _unchanged(
            after[start + offset : start + payload],
            before[start + offset : start + payload],
        )
        if slot < t.shape[0] - 1:
            _unchanged(
                after[start + payload : start + stride],
                before[start + payload : start + stride],
            )


def _check_mega(a, c, before, result, module):
    assert (
        len(result) == 5
        and result[0].shape == a["hidden_states"].shape
        and result[0].dtype == torch.bfloat16
    )
    for actual, key in zip(result[1:], ("conv_state", *fixtures.REPLAY_POOLS)):
        assert actual is a[key], "InplaceOutput must return the original input Tensor"
    _unchanged(_span(a["checkpoint_state"]), before["checkpoint_state"])
    lengths = c["lengths"]
    slots = c["state_indices"]
    selected = dict(zip(slots, lengths))
    _pool_untouched(
        a["conv_state"],
        before["conv_state"],
        {slot: length + 2 for slot, length in selected.items()},
    )
    for key in fixtures.REPLAY_POOLS:
        tensor = a[key]
        after = _span(tensor).detach().cpu()
        current = after.as_strided(tensor.shape, tensor.stride())
        original = before[key].as_strided(tensor.shape, tensor.stride())
        stride = tensor.stride(0)
        payload = tensor[0].numel()
        for slot in range(tensor.shape[0]):
            length = selected.get(slot, 0)
            if length:
                assert torch.isfinite(current[slot, :, :length]).all(), (
                    key + " produced NaN/Inf"
                )
            _unchanged(current[slot, :, length:], original[slot, :, length:])
            if slot < tensor.shape[0] - 1:
                _unchanged(
                    after[slot * stride + payload : (slot + 1) * stride],
                    before[key][slot * stride + payload : (slot + 1) * stride],
                )
    assert torch.isfinite(result[0]).all(), "out produced NaN/Inf"
    tp = 96 // c["N"]
    sequence = lengths[0]
    optimized = (
        sequence in (4, 8)
        and all(n == sequence for n in lengths)
        and c["B"] % tp == 0
        and 1 <= c["B"] // tp <= 32
    )
    assert module._last_backend == (
        f"optimized_s{sequence}" if optimized else "variable_length"
    ), "Unexpected execution branch"


def _device_mega(a, c):
    _assert_environment(True)
    module = fixtures.load_operator()
    device = fixtures.to_npu(a)
    save(
        "input_manifests",
        c["case_id"],
        dict(
            case_id=c["case_id"],
            configuration=c,
            inputs=fixtures.manifest(a),
            actual_inputs=fixtures.manifest(device),
        ),
    )
    before = _snapshot(device, "Verify")
    result = _call_with_quota(
        lambda: module.mega_recurrent_kda(**device), (), (), c["cores"]
    )
    _check_mega(device, c, before, result, module)
    save(
        "invariant_rows",
        c["case_id"],
        dict(
            status="PASS",
            checks=[
                "shape_dtype",
                "input_aliases",
                "checkpoint_readonly",
                "pool_gaps_unselected_slots",
                "conv_tail",
                "record_tails",
                "finite_outputs",
                "execution_branch",
            ],
        ),
    )
    return result, device


def verify(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type in ("npu", "privateuseone"), (
        "Verify requires an NPU marker"
    )
    arguments, config = prepare(case_id)
    return _device_mega(arguments, config)[0]


def verify_hp(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", "FP64 golden requires a CPU marker"
    arguments, _ = prepare(case_id)
    return golden.run_fp64(**arguments)


def verify_peer(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", (
        "Original third-party golden must execute on CPU"
    )
    a, c = prepare(case_id)
    before = fixtures.manifest(a)
    result = golden.run_original(**a)
    assert all(t.device.type == "cpu" for t in result)
    after = fixtures.manifest(a)
    assert after == before, "Original golden adapter modified caller inputs"
    save(
        "peer_manifests",
        case_id,
        dict(
            case_id=case_id,
            configuration=c,
            reference="specified_original_cpu_golden",
            inputs=before,
            actual_inputs=after,
            output_devices=[str(t.device) for t in result],
        ),
    )
    return result


def _device_commit(device, c):
    module = fixtures.load_operator("Commit")
    before = _snapshot(device, "Commit")
    readonly = {
        k: fixtures.digest(_span(v) if k in fixtures.REPLAY_POOLS else v)
        for k, v in device.items()
        if isinstance(v, torch.Tensor) and k != "checkpoint_state"
    }
    result = _call_with_quota(
        lambda: module.commit_recurrent_kda(**device), (), (), c["cores"]
    )
    assert result is None, "Commit returns None and updates checkpoint_state in place"
    assert all(
        fixtures.digest(_span(device[k]) if k in fixtures.REPLAY_POOLS else device[k])
        == v
        for k, v in readonly.items()
    ), "Commit modified readonly inputs"
    slots = device["state_indices"].cpu().tolist()
    counts = device["num_accepted_tokens"].cpu().tolist()
    # A whole block can change only for a positive accepted prefix.
    _pool_untouched(
        device["checkpoint_state"],
        before["checkpoint_state"],
        {
            slot: device["checkpoint_state"].shape[1]
            for slot, n in zip(slots, counts)
            if n > 0
        },
    )
    assert torch.isfinite(device["checkpoint_state"]).all(), "Commit produced NaN/Inf"
    save(
        "invariant_rows",
        c["case_id"],
        dict(
            status="PASS",
            checks=[
                "return_none",
                "records_controls_readonly",
                "zero_accepted_unselected_slots",
                "pool_gaps",
                "finite_checkpoint",
            ],
        ),
    )
    return device["checkpoint_state"]


def commit(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type in ("npu", "privateuseone"), (
        "Commit requires an NPU marker"
    )
    arguments, config = prepare_commit(case_id)
    _assert_environment(True)
    device = fixtures.to_npu(arguments)
    save(
        "input_manifests",
        case_id,
        dict(
            case_id=case_id,
            configuration=config,
            inputs=fixtures.manifest(arguments),
            actual_inputs=fixtures.manifest(device),
        ),
    )
    return _device_commit(device, config)


def commit_hp(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", "FP64 golden requires a CPU marker"
    arguments, _ = prepare_commit(case_id)
    return golden.commit_fp64(
        arguments["checkpoint_state"],
        arguments["replay_u"],
        arguments["replay_k"],
        arguments["replay_decay"],
        arguments["num_accepted_tokens"],
        arguments["state_indices"],
    )


def commit_peer(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", (
        "Original third-party golden must execute on CPU"
    )
    a, c = prepare_commit(case_id)
    before = fixtures.manifest(a)
    result = golden.commit_original(
        a["checkpoint_state"],
        a["replay_u"],
        a["replay_k"],
        a["replay_decay"],
        a["num_accepted_tokens"],
        a["state_indices"],
    )
    after = fixtures.manifest(a)
    assert after == before, "Original Commit adapter modified caller inputs"
    save(
        "peer_manifests",
        case_id,
        dict(
            case_id=case_id,
            configuration=c,
            reference="specified_original_cpu_golden",
            inputs=before,
            actual_inputs=after,
            output_devices=[str(result.device)],
        ),
    )
    return result


def expected(marker: torch.Tensor, *, case_id: str):
    return torch.ones(3, dtype=torch.float32, device="cpu")


def reject(marker: torch.Tensor, *, case_id: str):
    if marker.device.type not in ("npu", "privateuseone"):
        return expected(marker, case_id=case_id)
    _assert_environment(True)
    c = cases.get(case_id)
    assert c["expected"] == "REJECT_ATOMIC"
    hardware = torch.npu.get_device_properties(torch.npu.current_device())
    if c["cores"] > hardware.cube_core_num or 2 * c["cores"] > hardware.vector_core_num:
        raise RuntimeError(
            "Required designed stream quota is unavailable; environment blocked, not an operator contract result"
        )
    a, _ = prepare_commit(case_id) if c["mode"] == "Commit" else prepare(case_id)
    device = fixtures.to_npu(a)
    fixtures.mutate(device, c["mutation"], c.get("mutation_args"))
    save(
        "input_manifests",
        case_id,
        dict(
            case_id=case_id,
            configuration=c,
            actual_inputs=fixtures.manifest(device),
            check_type="CONTRACT_NOT_NUMERICAL",
        ),
    )
    watched = _snapshot(device, c["mode"])
    caught = None
    trace = None
    module = fixtures.load_operator(c["mode"])
    target = (
        module.commit_recurrent_kda
        if c["mode"] == "Commit"
        else module.mega_recurrent_kda
    )

    def invoke():
        return target(**device)

    try:
        _call_with_quota(
            invoke,
            (),
            (),
            c["cores"],
            guard_case=c.get("validation_layer") == "STREAM_CORE_GUARD",
        )
    except Exception as error:
        caught = error
        trace = traceback.format_exc()
    atomic = True
    for key, value in watched.items():
        try:
            _unchanged(_pool_storage(device[key]), value)
        except AssertionError:
            atomic = False
    flags = [
        caught is not None and type(caught).__name__ in c["exception_types"],
        caught is not None
        and all(re.search(p, str(caught)) is not None for p in c["match_all"]),
        atomic,
    ]
    record = dict(
        case_id=case_id,
        configuration=c,
        check_type="CONTRACT_NOT_NUMERICAL",
        actual_exception=type(caught).__name__ if caught else None,
        actual_message=str(caught),
        flags=flags,
        status="PASS" if all(flags) else "FAIL",
        traceback=trace,
    )
    save("contract_rows", case_id, record)
    assert all(flags), json.dumps(record, ensure_ascii=False)
    return torch.tensor(flags, dtype=torch.float32, device=marker.device)


def capture(marker: torch.Tensor, *, case_id: str):
    if marker.device.type not in ("npu", "privateuseone"):
        return expected(marker, case_id=case_id)
    _assert_environment(True)
    c = cases.get(case_id)
    is_commit = c["mode"] == "Commit"
    cpu, _ = prepare_commit(case_id) if is_commit else prepare(case_id)
    device = fixtures.to_npu(cpu)
    module = fixtures.load_operator(c["mode"])
    target = module.commit_recurrent_kda if is_commit else module.mega_recurrent_kda
    writable = (
        ("checkpoint_state",) if is_commit else ("conv_state", *fixtures.REPLAY_POOLS)
    )

    def check():
        spans = {name: _span(device[name]) for name in writable}
        baseline = {name: span.clone() for name, span in spans.items()}
        eager = target(**device)
        eager_values = (
            [device["checkpoint_state"].clone()]
            if is_commit
            else [t.clone() for t in eager]
        )
        device["state_indices"].zero_()
        try:
            target(**device)
        except ValueError as error:
            assert "distinct" in str(error)
        else:
            raise AssertionError("Duplicate indices must be rejected in eager mode")
        stream = torch.npu.current_stream()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            assert torch.npu.is_current_stream_capturing()
            captured = target(**device)
        device["state_indices"].copy_(cpu["state_indices"])
        for name, span in spans.items():
            span.copy_(baseline[name])
        graph.replay()
        stream.synchronize()
        outputs = [device["checkpoint_state"]] if is_commit else captured
        for actual, reference in zip(outputs, eager_values, strict=True):
            torch.testing.assert_close(
                actual, reference, rtol=0, atol=0, equal_nan=True
            )
        graph.reset()

    _call_with_quota(check, (), (), c["cores"])
    save(
        "capture_rows",
        case_id,
        dict(
            status="PASS",
            checks=[
                "duplicate_eager_rejected",
                "duplicate_placeholder_captured",
                "legal_indices_replayed",
            ],
        ),
    )
    return expected(marker, case_id=case_id).to(marker.device)


def replay_commit(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type in ("npu", "privateuseone"), (
        "Replay and Commit require an NPU marker"
    )
    arguments, config = prepare(case_id)
    result, device = _device_mega(arguments, config)
    commit_args = {
        key: device[key]
        for key in (
            "checkpoint_state",
            "replay_u",
            "replay_k",
            "replay_decay",
            "state_indices",
        )
    }
    commit_args["num_accepted_tokens"] = torch.tensor(
        config["accepted"], dtype=torch.int32, device=marker.device
    )
    return (*result, _device_commit(commit_args, config))


def _reference_chain(case_id, run_reference, commit_reference):
    arguments, config = prepare(case_id)
    result = run_reference(**arguments)
    accepted = torch.tensor(config["accepted"], dtype=torch.int32, device="cpu")
    checkpoint = commit_reference(
        arguments["checkpoint_state"], *result[2:], accepted, arguments["state_indices"]
    )
    return (*result, checkpoint)


def replay_commit_hp(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", "FP64 golden requires a CPU marker"
    return _reference_chain(case_id, golden.run_fp64, golden.commit_fp64)


def replay_commit_peer(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", (
        "Original third-party golden must execute on CPU"
    )
    return _reference_chain(case_id, golden.run_original, golden.commit_original)


def run_pytest_case(case_id, reference_verify=None, reference_commit=None):
    c = cases.get(case_id)
    marker = torch.empty(1, device="npu")
    if c["expected"] == "REJECT_ATOMIC":
        return reject(marker, case_id=case_id)
    if c["expected"] == "PASS_CAPTURE":
        return capture(marker, case_id=case_id)
    if c["mode"] == "Commit":
        a, _ = prepare_commit(case_id)
        wanted = reference_commit(
            a["checkpoint_state"],
            a["replay_u"],
            a["replay_k"],
            a["replay_decay"],
            a["num_accepted_tokens"],
            a["state_indices"],
            fp64=False,
        )
        actual = commit(marker, case_id=case_id)
        torch.testing.assert_close(
            actual.cpu(), wanted, atol=1e-6, rtol=1e-6, equal_nan=True
        )
    else:
        a, _ = prepare(case_id)
        wanted = reference_verify(**a)
        actual = (
            replay_commit(marker, case_id=case_id)
            if c["mode"] == "ReplayCommit"
            else verify(marker, case_id=case_id)
        )
        for i, (value, reference) in enumerate(zip(actual, wanted)):
            atol = (
                5e-5
                if i == 0
                and all(n == 8 for n in c["lengths"])
                and c["B"] == 96 // c["N"]
                else 0.005
            )
            torch.testing.assert_close(
                value.cpu(), reference, atol=atol, rtol=0.005, equal_nan=True
            )
        if c["mode"] == "ReplayCommit":
            expected_checkpoint = reference_commit(
                a["checkpoint_state"],
                *wanted[2:],
                torch.tensor(c["accepted"], dtype=torch.int32),
                a["state_indices"],
                fp64=False,
            )
            torch.testing.assert_close(
                actual[-1].cpu(),
                expected_checkpoint,
                atol=1e-6,
                rtol=1e-6,
                equal_nan=True,
            )
    save(
        "pytest_rows", case_id, dict(status="PASS", framework="pytest", configuration=c)
    )


"""Unchanged strict cross_check/L2, including FP32 U under the new ABI."""
__spec__ = {
    "kda_ttk_ops.verify": "VerifySpec",
    "kda_ttk_ops.commit": "CommitSpec",
    "kda_ttk_ops.reject": "RejectSpec",
    "kda_ttk_ops.capture": "RejectSpec",
    "kda_ttk_ops.replay_commit": "ReplayCommitSpec",
}


class Base:
    tolerance = {
        dtype: {"standard": "cross_check", "level": "L2"}
        for dtype in ("bfloat16", "float16", "float32")
    }

    @staticmethod
    def describe(params):
        cases.get(params["case_id"])
        return dict(shapes=((1,),), dtypes=("float32",))


class VerifySpec(Base):
    golden = staticmethod(verify_hp)
    third_party = {"torch": "kda_ttk_ops.verify_peer"}


class CommitSpec(Base):
    golden = staticmethod(commit_hp)
    third_party = {"torch": "kda_ttk_ops.commit_peer"}


class RejectSpec(Base):
    golden = staticmethod(expected)
    third_party = {"torch": "kda_ttk_ops.expected"}


class ReplayCommitSpec(Base):
    golden = staticmethod(replay_commit_hp)
    third_party = {"torch": "kda_ttk_ops.replay_commit_peer"}


# Normalize lru-cache calls: f() and f("Verify") must load one Python module.
_mode_cached_load_operator = fixtures.load_operator


def _single_load_operator(mode="Verify"):
    return _mode_cached_load_operator(mode)


fixtures.load_operator = _single_load_operator


# Observe an invalid internal-format tensor without changing the operator input.
def _snapshot(a, mode):
    import torch_npu

    pools = tuple(k for k in fixtures.PA_POOLS if k in a)
    watched = {}
    for key, value in a.items():
        if not isinstance(value, torch.Tensor) or key not in pools:
            continue
        observed = _pool_storage(value) if key in pools else value
        if observed.device.type in ("npu", "privateuseone"):
            fmt = torch_npu.get_npu_format(observed)
            if fmt not in (0, 2):
                observed = torch_npu.npu_format_cast(observed, 2)
                assert torch_npu.get_npu_format(value) == fmt, (
                    "Observer changed the input storage format"
                )
        watched[key] = observed.detach().cpu().clone()
    return watched


# Preserve the historical transpose view and its PA stride; do not materialize it.
_original_mutate_for_views = fixtures.mutate


def _historical_view_mutation(args, kind, details=None):
    if kind == "generic" and details and details.get("op") == "transpose":
        key = details["target"]
        args[key] = args[key].transpose(*details["dims"])
        return args
    return _original_mutate_for_views(args, kind, details)


fixtures.mutate = _historical_view_mutation

# Compatibility only: build the same negative inputs through supported ND operations.
_portable_mutation_before = fixtures.mutate


def _portable_negative_mutation(args, kind, details=None):
    import torch_npu

    d = details or {}
    key = d.get("target")
    t = args.get(key)
    op = d.get("op")
    if (
        kind == "generic"
        and isinstance(t, torch.Tensor)
        and t.device.type in ("npu", "privateuseone")
    ):
        fmt = int(torch_npu.get_npu_format(t))
        if op in ("dtype", "noncontiguous") and fmt == 29:
            nd = torch_npu.npu_format_cast(t, 2)
            if op == "dtype":
                result = torch_npu.npu_format_cast(
                    nd.to(getattr(torch, d["dtype"])), fmt
                )
                assert (
                    result.dtype == getattr(torch, d["dtype"])
                    and result.shape == t.shape
                    and int(torch_npu.get_npu_format(result)) == fmt
                )
                assert fixtures.digest(result) == fixtures.digest(
                    nd.cpu().to(getattr(torch, d["dtype"]))
                )
            else:
                shape = list(t.shape)
                shape[-1] *= 2
                base = torch.empty(shape, dtype=t.dtype, device=t.device)
                result = base[..., ::2]
                result.copy_(nd)
                assert (
                    result.shape == t.shape
                    and result.dtype == t.dtype
                    and not result.is_contiguous()
                    and fixtures.digest(result) == fixtures.digest(t)
                )
            args[key] = result
            save(
                "mutation_compat_rows",
                os.environ.get("MKDA_ACTIVE_CASE", "unknown"),
                dict(
                    op=op,
                    target=key,
                    source_format=fmt,
                    result_format=int(torch_npu.get_npu_format(result)),
                    shape=list(result.shape),
                    dtype=str(result.dtype),
                    stride=list(result.stride()),
                ),
            )
            return args
        if op == "format_nz":
            nd = torch_npu.npu_format_cast(t, 2)
            result = torch_npu.npu_format_cast(nd, 29)
            assert (
                result.dtype == t.dtype
                and result.shape == t.shape
                and int(torch_npu.get_npu_format(result)) == 29
                and fixtures.digest(result) == fixtures.digest(t)
            )
            args[key] = result
            save(
                "mutation_compat_rows",
                os.environ.get("MKDA_ACTIVE_CASE", "unknown"),
                dict(
                    op=op,
                    target=key,
                    source_format=fmt,
                    result_format=29,
                    shape=list(result.shape),
                    dtype=str(result.dtype),
                ),
            )
            return args
    return _portable_mutation_before(args, kind, details)


fixtures.mutate = _portable_negative_mutation

_portable_transfer_before = fixtures.to_npu


def _portable_empty_transfer(args):
    empty = {
        k: v
        for k, v in args.items()
        if k in fixtures.PA_POOLS and isinstance(v, torch.Tensor) and v.numel() == 0
    }
    if not empty:
        return _portable_transfer_before(args)
    result = _portable_transfer_before(
        {k: v for k, v in args.items() if k not in empty}
    )
    for k, v in empty.items():
        result[k] = torch.empty_strided(
            v.shape, v.stride(), dtype=v.dtype, device="npu"
        )
        assert (
            result[k].shape == v.shape
            and result[k].stride() == v.stride()
            and result[k].numel() == 0
        )
    return result


fixtures.to_npu = _portable_empty_transfer

_negative_before_observation = reject


def reject(marker, *, case_id):
    os.environ["MKDA_ACTIVE_CASE"] = case_id
    return _negative_before_observation(marker, case_id=case_id)


# Rank-four negative-format views use a real two-dimensional NZ allocation.
_rank4_digest_before = fixtures.digest


def _read_rank4_nz(t):
    import torch_npu

    assert t.ndim == 4 and int(torch_npu.get_npu_format(t)) == 29
    return torch_npu.npu_format_cast(t.view(-1, t.shape[-1]), 2).cpu().reshape(t.shape)


def _rank4_safe_digest(t):
    import torch_npu

    if (
        t.device.type in ("npu", "privateuseone")
        and t.ndim == 4
        and int(torch_npu.get_npu_format(t)) == 29
    ):
        return _rank4_digest_before(_read_rank4_nz(t))
    return _rank4_digest_before(t)


fixtures.digest = _rank4_safe_digest

_rank4_snapshot_before = _snapshot


def _snapshot(args, mode):
    import torch_npu

    result = {}
    wanted = ("checkpoint_state", "replay_u", "replay_k", "replay_decay") + (
        () if mode == "Commit" else ("conv_state",)
    )
    for key, t in args.items():
        if key not in wanted or not isinstance(t, torch.Tensor):
            continue
        if (
            t.ndim == 4
            and t.device.type in ("npu", "privateuseone")
            and int(torch_npu.get_npu_format(t)) == 29
        ):
            result[key] = _read_rank4_nz(t).clone()
        else:
            result.update(_rank4_snapshot_before({key: t}, mode))
    return result


_rank4_mutation_before = fixtures.mutate


def _rank4_negative_format(args, kind, details=None):
    import torch_npu

    d = details or {}
    key = d.get("target")
    t = args.get(key)
    if (
        kind == "generic"
        and d.get("op") == "format_nz"
        and isinstance(t, torch.Tensor)
        and t.ndim == 4
    ):
        assert t.device.type in ("npu", "privateuseone")
        expected = fixtures.digest(t)
        matrix = t.reshape(-1, t.shape[-1])
        result = torch_npu.npu_format_cast(matrix, 29).view(t.shape)
        assert (
            result.shape == t.shape
            and result.dtype == t.dtype
            and int(torch_npu.get_npu_format(result)) == 29
        )
        assert fixtures.digest(result) == expected
        args[key] = result
        save(
            "mutation_compat_rows",
            os.environ["MKDA_ACTIVE_CASE"],
            dict(
                op="format_nz",
                target=key,
                result_format=29,
                shape=list(result.shape),
                dtype=str(result.dtype),
                allocation_shape=list(matrix.shape),
                logical_values_preserved=True,
            ),
        )
        return args
    return _rank4_mutation_before(args, kind, details)


fixtures.mutate = _rank4_negative_format
