import gzip
import json

from miles.ray.rollout.benchmark_data import load_benchmark_samples


def test_load_benchmark_samples(tmp_path):
    records = [
        {
            "group_index": 0,
            "index": 0,
            "tokens": [1, 2, 3],
            "response_length": 2,
            "loss_mask": [1, 1],
            "reward": -1.0,
            "rollout_log_probs": [-0.1, -0.2],
            "status": "completed",
            "metadata": {
                "sample_id": "sample-1",
                "rollout_id": "rollout-1",
                "task_id": "task-1",
            },
        },
        {
            "group_index": 0,
            "index": 1,
            "tokens": [4, 5],
            "response_length": 1,
            "loss_mask": [1],
            "reward": 1.0,
            "rollout_log_probs": [-0.3],
            "status": "completed",
            "metadata": {
                "sample_id": "sample-2",
                "rollout_id": "rollout-2",
                "task_id": "task-1",
            },
        },
    ]
    manifest = {
        "version": 1,
        "rollout_count": 2,
        "task_count": 1,
        "sample_count": 2,
        "training_tokens": 5,
        "loss_tokens": 3,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with gzip.open(tmp_path / "miles.jsonl.gz", "wt") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    samples = load_benchmark_samples(tmp_path)

    assert [sample.tokens for sample in samples] == [[1, 2, 3], [4, 5]]
    assert samples[0].status.value == "completed"
    assert samples[1].reward == 1.0
