import json
from types import SimpleNamespace

from miles.utils import train_metric_utils


def test_benchmark_output_excludes_warmup(tmp_path):
    path = tmp_path / "benchmark.json"
    args = SimpleNamespace(benchmark_output=str(path), start_rollout_id=0)

    train_metric_utils._write_benchmark_step(
        args,
        rollout_id=0,
        step_seconds=99.0,
        peak_memory_gib=180.0,
    )
    train_metric_utils._write_benchmark_step(
        args,
        rollout_id=1,
        step_seconds=10.0,
        peak_memory_gib=175.0,
    )

    assert json.loads(path.read_text()) == {
        "warmup_steps": 1,
        "step_seconds": [10.0],
        "peak_memory_gib": [175.0],
    }
