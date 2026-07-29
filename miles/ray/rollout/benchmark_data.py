import gzip
import json
from pathlib import Path

from miles.utils.types import Sample


def load_benchmark_samples(path: Path) -> list[Sample]:
    manifest_path = path / "manifest.json"
    records_path = path / "miles.jsonl.gz"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"Prepared RL benchmark artifact is incomplete: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = []
    with gzip.open(records_path, "rt", encoding="utf-8") as file:
        for line in file:
            sample = Sample.from_dict(json.loads(line))
            sample.validate()
            samples.append(sample)

    actual = (
        len({sample.metadata["rollout_id"] for sample in samples}),
        len({sample.metadata["task_id"] for sample in samples}),
        len(samples),
        sum(len(sample.tokens) for sample in samples),
        sum(sum(sample.loss_mask) for sample in samples),
    )
    expected = (
        manifest["rollout_count"],
        manifest["task_count"],
        manifest["sample_count"],
        manifest["training_tokens"],
        manifest["loss_tokens"],
    )
    if manifest["version"] != 1:
        raise ValueError(f"Unsupported Miles benchmark artifact version: {manifest['version']}")
    if actual != expected:
        raise ValueError(f"Miles benchmark records do not match manifest totals: expected {expected}, found {actual}")

    return samples
