from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.distributed as dist

_GIB = 1024**3


class CudaPhaseMemoryTracker:
    """Measure coarse CUDA phase peaks without changing allocator state."""

    def __init__(self, rank: dict[str, int]):
        self.rank = rank
        self.phases: list[dict[str, float | str]] = []

    @contextmanager
    def phase(self, name: str):
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)
        start_free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            end_free, _ = torch.cuda.mem_get_info(device)
            self.phases.append(
                {
                    "name": name,
                    "seconds": time.monotonic() - started,
                    "start_allocated_gib": start_allocated / _GIB,
                    "end_allocated_gib": torch.cuda.memory_allocated(device) / _GIB,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / _GIB,
                    "start_reserved_gib": start_reserved / _GIB,
                    "end_reserved_gib": torch.cuda.memory_reserved(device) / _GIB,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / _GIB,
                    "start_driver_free_gib": start_free / _GIB,
                    "end_driver_free_gib": end_free / _GIB,
                    "driver_total_gib": total / _GIB,
                }
            )

    def gather(self, group: dist.ProcessGroup) -> list[dict[str, Any]]:
        local = {"rank": self.rank, "phases": self.phases}
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, local, group=group)
        return [record for record in gathered if record is not None]


def cuda_phase(tracker: CudaPhaseMemoryTracker | None, name: str):
    return tracker.phase(name) if tracker is not None else nullcontext()


def global_phase_peaks(records: list[dict[str, Any]]) -> tuple[float, float]:
    phases = [phase for record in records for phase in record["phases"]]
    return (
        max(phase["peak_allocated_gib"] for phase in phases),
        max(phase["peak_reserved_gib"] for phase in phases),
    )
