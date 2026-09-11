from types import SimpleNamespace

import pytest
import torch

from miles.utils import profile_utils


def test_profiler_waits_without_collecting_then_records(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    captured_keys = set()
    monkeypatch.setattr(
        profile_utils,
        "_trace_handler",
        lambda _args, _name: lambda profiler: captured_keys.update(event.key for event in profiler.key_averages()),
    )
    args = SimpleNamespace(
        profile_step_start=1,
        profile_step_end=2,
        tensorboard_dir=str(tmp_path),
        pytorch_profiler_collect_shapes=False,
        pytorch_profiler_collect_callstack=False,
    )

    profiler = profile_utils._create_torch_profiler(args, "smoke")
    profiler.start()
    with torch.profiler.record_function("warmup.must_not_be_recorded"):
        torch.ones(8) + 1
    profiler.step()
    with torch.profiler.record_function("active.must_be_recorded"):
        torch.ones(8) + 1
    profiler.step()

    assert "active.must_be_recorded" in captured_keys
    assert "warmup.must_not_be_recorded" not in captured_keys


def test_empty_device_profile_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.profiler, "tensorboard_trace_handler", lambda *_args, **_kwargs: lambda _profiler: None)

    class EmptyProfiler:
        @staticmethod
        def key_averages():
            return []

    args = SimpleNamespace(tensorboard_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="captured no device activity"):
        profile_utils._trace_handler(args, "smoke")(EmptyProfiler())
