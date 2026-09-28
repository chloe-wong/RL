# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import json
import os
from typing import Any, Protocol

import rich
import torch

NRL_NSYS_WORKER_PATTERNS = os.environ.get("NRL_NSYS_WORKER_PATTERNS", "")
NRL_NSYS_PROFILE_STEP_RANGE = os.environ.get("NRL_NSYS_PROFILE_STEP_RANGE", "")


def _parse_extra_options(raw: str) -> dict[str, Any]:
    """Parse NRL_NSYS_EXTRA_OPTIONS as a JSON object of nsys flag -> value.

    Empty string parses to an empty dict. Invalid JSON or non-object payloads raise
    so misconfiguration surfaces immediately rather than silently being ignored.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse NRL_NSYS_EXTRA_OPTIONS as JSON: {e}. "
            "Expected a JSON object mapping nsys flag names (without leading --) to "
            'values, e.g. \'{"gpu-metrics-device": "all", "cuda-memory-usage": "true"}\'. '
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"NRL_NSYS_EXTRA_OPTIONS must be a JSON object, got {type(parsed).__name__}. "
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        )
    return parsed


NRL_NSYS_EXTRA_OPTIONS: dict[str, Any] = _parse_extra_options(
    os.environ.get("NRL_NSYS_EXTRA_OPTIONS", "")
)


class ProfilablePolicy(Protocol):
    def start_gpu_profiling(self) -> None: ...

    def stop_gpu_profiling(self) -> None: ...


def maybe_gpu_profile_step(policy: ProfilablePolicy, step: int):
    # NRL_NSYS_PROFILE_STEP_RANGE selects the steps for every profiler, not just nsys:
    # start_gpu_profiling/stop_gpu_profiling also drive the torch profiler and the CUDA
    # memory snapshot. Gating on NRL_NSYS_WORKER_PATTERNS alone made those two
    # reachable only by also launching nsys, which shares CUPTI with them.
    any_profiler = bool(
        NRL_NSYS_WORKER_PATTERNS or NRL_TORCH_PROFILE_DIR or NRL_MEMORY_SNAPSHOT_DIR
    )
    assert not (any_profiler ^ bool(NRL_NSYS_PROFILE_STEP_RANGE)), (
        "NRL_NSYS_PROFILE_STEP_RANGE must be set together with one of NRL_NSYS_WORKER_PATTERNS, NRL_TORCH_PROFILE_DIR or NRL_MEMORY_SNAPSHOT_DIR, and left unset otherwise. See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
    )

    if not any_profiler:
        return

    NSYS_START_STEP, NSYS_STOP_STEP = NRL_NSYS_PROFILE_STEP_RANGE.split(":")
    try:
        NSYS_START_STEP = int(NSYS_START_STEP)
        NSYS_STOP_STEP = int(NSYS_STOP_STEP)
    except ValueError as e:
        raise ValueError(
            f"Error parsing NRL_NSYS_PROFILE_STEP_RANGE: {str(e)}. "
            "Please ensure the format is 'start:stop' where both values are integers. "
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        ) from e

    assert NSYS_START_STEP < NSYS_STOP_STEP, (
        f"{NRL_NSYS_PROFILE_STEP_RANGE=} must be a non-empty range"
    )
    assert NSYS_START_STEP >= 1, (
        f"The start step in {NRL_NSYS_PROFILE_STEP_RANGE=} must be >= 1"
    )

    # Use slice syntax of left inclusive and right exclusive
    if NSYS_START_STEP <= step < NSYS_STOP_STEP:
        if not getattr(policy, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Starting GPU profiling for {policy} for step {step}[/bold red]"
            )
            policy.start_gpu_profiling()
            policy.__NRL_PROFILE_STARTED = True

            def stop_profiler_on_exit():
                rich.print(
                    f"[bold red]Stopping GPU profiling on exit for {policy} for step {step}[/bold red]"
                )
                policy.stop_gpu_profiling()

            atexit.register(stop_profiler_on_exit)
    else:
        if getattr(policy, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Stopping GPU profiling for {policy} for step {step}[/bold red]"
            )
            policy.stop_gpu_profiling()
            policy.__NRL_PROFILE_STARTED = False


def wrap_with_nvtx_name(name: str):
    """A decorator to wrap a function with an NVTX range with the given name."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            ret = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return ret

        return wrapper

    return decorator


# Optional PyTorch profiler and CUDA memory snapshot, driven by the same
# NRL_NSYS_PROFILE_STEP_RANGE start/stop calls. Each writes one file per rank.
# Do not combine NRL_TORCH_PROFILE_DIR with nsys or ncu: they share CUPTI.
NRL_TORCH_PROFILE_DIR = os.environ.get("NRL_TORCH_PROFILE_DIR", "")
NRL_MEMORY_SNAPSHOT_DIR = os.environ.get("NRL_MEMORY_SNAPSHOT_DIR", "")
_memory_recording = False


def start_torch_profiling() -> Any:
    """Start the PyTorch profiler and/or CUDA memory history if enabled."""
    global _memory_recording
    if NRL_MEMORY_SNAPSHOT_DIR:
        torch.cuda.memory._record_memory_history(max_entries=500_000)
        _memory_recording = True
    if not NRL_TORCH_PROFILE_DIR:
        return None
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )
    prof.start()
    return prof


def stop_torch_profiling(prof: Any) -> None:
    """Stop profilers started by start_torch_profiling and write per-rank files."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if prof is not None:
        prof.stop()
        os.makedirs(NRL_TORCH_PROFILE_DIR, exist_ok=True)
        prof.export_chrome_trace(
            os.path.join(NRL_TORCH_PROFILE_DIR, f"rank{rank}.pt.trace.json.gz")
        )
    global _memory_recording
    # stop can run twice (step range end, then atexit); only dump while recording.
    if _memory_recording:
        _memory_recording = False
        os.makedirs(NRL_MEMORY_SNAPSHOT_DIR, exist_ok=True)
        torch.cuda.memory._dump_snapshot(
            os.path.join(NRL_MEMORY_SNAPSHOT_DIR, f"rank{rank}.pickle")
        )
        torch.cuda.memory._record_memory_history(enabled=None)
