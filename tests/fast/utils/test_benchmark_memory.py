from __future__ import annotations

from miles.utils import benchmark_memory
from miles.utils.benchmark_memory import CudaPhaseMemoryTracker, global_phase_peaks

GIB = 1024**3


def test_phase_records_isolated_peak_and_driver_memory(monkeypatch):
    values = {
        "allocated": iter([10 * GIB, 13 * GIB]),
        "reserved": iter([12 * GIB, 16 * GIB]),
        "free": iter([(70 * GIB, 80 * GIB), (64 * GIB, 80 * GIB)]),
    }
    resets = []
    synchronizations = []
    monkeypatch.setattr(benchmark_memory.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(benchmark_memory.torch.cuda, "synchronize", lambda: synchronizations.append(True))
    monkeypatch.setattr(benchmark_memory.torch.cuda, "memory_allocated", lambda _device: next(values["allocated"]))
    monkeypatch.setattr(benchmark_memory.torch.cuda, "memory_reserved", lambda _device: next(values["reserved"]))
    monkeypatch.setattr(benchmark_memory.torch.cuda, "mem_get_info", lambda _device: next(values["free"]))
    monkeypatch.setattr(benchmark_memory.torch.cuda, "reset_peak_memory_stats", lambda device: resets.append(device))
    monkeypatch.setattr(benchmark_memory.torch.cuda, "max_memory_allocated", lambda _device: 15 * GIB)
    monkeypatch.setattr(benchmark_memory.torch.cuda, "max_memory_reserved", lambda _device: 17 * GIB)

    tracker = CudaPhaseMemoryTracker({"global": 5, "pp": 1})
    with tracker.phase("forward_backward"):
        pass

    assert synchronizations == [True, True]
    assert resets == [3]
    assert tracker.phases == [
        {
            "name": "forward_backward",
            "seconds": tracker.phases[0]["seconds"],
            "start_allocated_gib": 10.0,
            "end_allocated_gib": 13.0,
            "peak_allocated_gib": 15.0,
            "start_reserved_gib": 12.0,
            "end_reserved_gib": 16.0,
            "peak_reserved_gib": 17.0,
            "start_driver_free_gib": 70.0,
            "end_driver_free_gib": 64.0,
            "driver_total_gib": 80.0,
        }
    ]


def test_global_phase_peaks_uses_all_phases_and_ranks():
    records = [
        {"phases": [{"peak_allocated_gib": 10.0, "peak_reserved_gib": 12.0}]},
        {
            "phases": [
                {"peak_allocated_gib": 11.0, "peak_reserved_gib": 11.5},
                {"peak_allocated_gib": 9.0, "peak_reserved_gib": 14.0},
            ]
        },
    ]

    assert global_phase_peaks(records) == (11.0, 14.0)
