import json
import logging
import os
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from miles.utils.device_flops import local_peak_bf16_tflops
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.timer import Timer
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    compute_total_fwd_flops: Callable,
    extra_metrics: dict | None = None,
    benchmark_memory_by_rank: list[dict] | None = None,
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    if extra_metrics:
        log_dict.update(extra_metrics)

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

            peak_tflops = getattr(args, "mfu_peak_tflops", None) or local_peak_bf16_tflops()
            if peak_tflops:
                log_dict["perf/mfu_peak_tflops"] = peak_tflops
                log_dict["perf/actor_train_mfu"] = log_dict["perf/actor_train_tflops"] / peak_tflops

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    if (
        "perf/train_time" in log_dict
        and "perf/peak_memory_gib" in log_dict
        and "perf/peak_allocated_memory_gib" in log_dict
    ):
        _write_benchmark_step(
            args,
            rollout_id,
            step_seconds=log_dict["perf/train_time"],
            peak_memory_gib=log_dict["perf/peak_memory_gib"],
            peak_allocated_memory_gib=log_dict["perf/peak_allocated_memory_gib"],
            benchmark_memory_by_rank=benchmark_memory_by_rank,
        )

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking.log(args, log_dict, step_key="rollout/step")


def _write_benchmark_step(
    args: Namespace,
    rollout_id: int,
    *,
    step_seconds: float,
    peak_memory_gib: float,
    peak_allocated_memory_gib: float,
    benchmark_memory_by_rank: list[dict] | None = None,
) -> None:
    if args.benchmark_output is None:
        return

    path = Path(args.benchmark_output)
    result = {
        "warmup_steps": 1,
        "step_seconds": [],
        "peak_memory_gib": [],
        "peak_allocated_memory_gib": [],
        "phase_memory_by_step": [],
    }
    is_warmup = rollout_id == args.start_rollout_id
    if not is_warmup and path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
    result["phase_memory_by_step"].append(
        {
            "rollout_id": rollout_id,
            "is_warmup": is_warmup,
            "ranks": benchmark_memory_by_rank or [],
        }
    )
    if not is_warmup:
        result["step_seconds"].append(float(step_seconds))
        result["peak_memory_gib"].append(float(peak_memory_gib))
        result["peak_allocated_memory_gib"].append(float(peak_allocated_memory_gib))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
