"""Runtime helpers for hardware-aware Step1 training configuration."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist


def _read_host_mem_total_gb() -> float:
    """Best-effort host RAM detection from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_kb = float(parts[1])
                        return mem_kb / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def _round_down_to_multiple(value: float, multiple: int) -> int:
    """Round down to a positive multiple, keeping at least `multiple`."""
    multiple = max(1, int(multiple))
    return max(multiple, (int(value) // multiple) * multiple)


def apply_cuda_allocator_settings(opt_cfg: Dict, is_main_process: bool) -> None:
    """Apply allocator tuning knobs before CUDA context creation."""
    if not bool(opt_cfg.get("cuda_allocator_guard", True)):
        return

    conf_parts = []
    max_split = int(opt_cfg.get("cuda_allocator_max_split_size_mb", 256))
    if max_split > 0:
        conf_parts.append(f"max_split_size_mb:{max_split}")

    gc_threshold = float(opt_cfg.get("cuda_allocator_gc_threshold", 0.8))
    if gc_threshold > 0.0:
        conf_parts.append(f"garbage_collection_threshold:{gc_threshold}")

    if bool(opt_cfg.get("cuda_allocator_expandable_segments", True)):
        conf_parts.append("expandable_segments:True")

    if not conf_parts:
        return

    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if current:
        existing_keys = {part.split(":", 1)[0] for part in current.split(",") if ":" in part}
        missing_parts = [part for part in conf_parts if part.split(":", 1)[0] not in existing_keys]
        if not missing_parts:
            return
        merged = current + "," + ",".join(missing_parts)
    else:
        merged = ",".join(conf_parts)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = merged
    if is_main_process:
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={merged}")


def detect_hardware_profile(
    world_size: int,
    local_world_size: int,
    device: str,
    distributed: bool,
) -> Dict:
    """Detect runtime GPU/CPU resources for current training job."""
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    slurm_mem_per_node_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0") or 0)
    slurm_mem_per_cpu_mb = int(os.environ.get("SLURM_MEM_PER_CPU", "0") or 0)

    profile = {
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
        "visible_gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "host_cpu_count": int(os.cpu_count() or 1),
        "host_mem_total_gb": round(_read_host_mem_total_gb(), 2),
        "slurm_cpus_per_task": slurm_cpus_per_task,
        "slurm_mem_per_node_mb": slurm_mem_per_node_mb,
        "slurm_mem_per_cpu_mb": slurm_mem_per_cpu_mb,
    }

    per_rank_mem_gb = 0.0
    if slurm_mem_per_node_mb > 0:
        per_rank_mem_gb = (slurm_mem_per_node_mb / 1024.0) / max(1, local_world_size)
    elif slurm_mem_per_cpu_mb > 0 and slurm_cpus_per_task > 0:
        per_rank_mem_gb = (slurm_mem_per_cpu_mb * slurm_cpus_per_task) / 1024.0
    elif profile["host_mem_total_gb"] > 0:
        per_rank_mem_gb = profile["host_mem_total_gb"] / max(1, local_world_size)
    profile["cpu_mem_gb_per_rank_est"] = round(per_rank_mem_gb, 2) if per_rank_mem_gb > 0 else 0.0

    if not torch.cuda.is_available():
        return profile

    local_gpu_mem_gb = 0.0
    gpu_name = "unknown"
    try:
        dev = torch.device(device)
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        local_gpu_mem_gb = float(props.total_memory) / float(1024 ** 3)
        gpu_name = props.name
    except Exception:
        pass

    profile["gpu_name_local_rank"] = gpu_name
    profile["gpu_mem_gb_local_rank"] = round(local_gpu_mem_gb, 2) if local_gpu_mem_gb > 0 else 0.0

    if distributed and dist.is_available() and dist.is_initialized() and local_gpu_mem_gb > 0:
        backend = dist.get_backend()
        collect_device = (
            torch.device(device)
            if backend == "nccl" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        mem_tensor = torch.tensor([local_gpu_mem_gb], device=collect_device, dtype=torch.float32)
        min_tensor = mem_tensor.clone()
        max_tensor = mem_tensor.clone()
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        profile["gpu_mem_gb_min"] = round(float(min_tensor.item()), 2)
        profile["gpu_mem_gb_max"] = round(float(max_tensor.item()), 2)
    else:
        profile["gpu_mem_gb_min"] = profile["gpu_mem_gb_local_rank"]
        profile["gpu_mem_gb_max"] = profile["gpu_mem_gb_local_rank"]

    return profile


def save_hardware_profile(path: Path, profile: Dict) -> None:
    """Persist hardware profile JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)


def maybe_apply_hardware_aware_batching(
    config: Dict,
    backbone_config: Dict,
    hardware_profile: Dict,
    world_size: int,
    is_main_process: bool,
) -> None:
    """Choose per-device micro-batch from runtime hardware and model depth."""
    opt_cfg = config.get("optimization", {})
    if not bool(opt_cfg.get("auto_batch_hardware_aware", True)):
        return

    gpu_mem_gb = float(
        hardware_profile.get(
            "gpu_mem_gb_min",
            hardware_profile.get("gpu_mem_gb_local_rank", 0.0),
        )
    )
    if gpu_mem_gb <= 0.0:
        return

    ref_depth = max(1.0, float(opt_cfg.get("auto_batch_ref_depth", 6.0)))
    ref_mem_gb = max(1.0, float(opt_cfg.get("auto_batch_ref_gpu_mem_gb", 80.0)))
    ref_micro = max(
        1,
        int(opt_cfg.get("auto_batch_ref_per_device_micro_batch", config["training_backbone"]["batch_size"])),
    )
    ref_seq_len = max(1.0, float(opt_cfg.get("auto_batch_ref_seq_len_tokens", 256.0)))
    seq_len = max(
        1.0,
        float(
            opt_cfg.get(
                "compute_opt_seq_len_tokens",
                backbone_config.get(
                    "max_position_embeddings",
                    config.get("backbone", {}).get("max_position_embeddings", 256),
                ),
            )
        ),
    )
    depth = max(
        1.0,
        float(backbone_config.get("num_layers", config.get("backbone", {}).get("num_layers", ref_depth))),
    )

    depth_exponent = float(opt_cfg.get("auto_batch_mem_depth_exponent", -0.35))
    mem_safety = float(opt_cfg.get("auto_batch_gpu_mem_safety_factor", 1.0))
    mem_safety = min(1.0, max(0.5, mem_safety))
    fp8_gain = float(opt_cfg.get("auto_batch_fp8_gain", 1.15))
    if not bool(opt_cfg.get("fp8_training", False)):
        fp8_gain = 1.0

    min_micro = max(1, int(opt_cfg.get("auto_batch_min_per_device_micro_batch", 8)))
    max_micro = max(min_micro, int(opt_cfg.get("auto_batch_max_per_device_micro_batch", 256)))
    round_multiple = max(1, int(opt_cfg.get("auto_batch_micro_batch_multiple", 8)))

    mem_scale = max(0.25, (gpu_mem_gb * mem_safety) / ref_mem_gb)
    depth_scale = max(0.25, (depth / ref_depth) ** depth_exponent)
    seq_scale = max(0.25, math.sqrt(ref_seq_len / seq_len))
    raw_micro = float(ref_micro) * mem_scale * depth_scale * seq_scale * fp8_gain

    clipped_micro = min(max_micro, max(min_micro, int(math.floor(raw_micro))))
    chosen_micro = _round_down_to_multiple(clipped_micro, round_multiple)
    chosen_micro = min(max_micro, max(min_micro, chosen_micro))

    prev_micro = int(config["training_backbone"]["batch_size"])
    config["training_backbone"]["batch_size"] = chosen_micro

    low_mem_threshold = float(opt_cfg.get("gpu_oom_guard_low_mem_threshold_gb", 64.0))
    if bool(opt_cfg.get("gpu_oom_guard_force_dynamic_padding", True)) and gpu_mem_gb < low_mem_threshold:
        config["optimization"]["dynamic_padding"] = True

    if is_main_process:
        per_micro_global = chosen_micro * max(1, int(world_size))
        print(
            "Hardware-aware micro-batch: "
            f"gpu_mem={gpu_mem_gb:.1f}GB, depth={int(depth)}, "
            f"micro_batch={prev_micro}->{chosen_micro}, per_micro_global={per_micro_global}"
        )


def maybe_apply_cpu_oom_guards(
    opt_cfg: Dict,
    hardware_profile: Dict,
    local_world_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    is_main_process: bool,
) -> Tuple[int, int, bool]:
    """Apply conservative DataLoader settings when per-rank host RAM is tight."""
    if not bool(opt_cfg.get("cpu_oom_guard", True)):
        return num_workers, prefetch_factor, pin_memory

    per_rank_mem_gb = float(hardware_profile.get("cpu_mem_gb_per_rank_est", 0.0))
    if per_rank_mem_gb <= 0:
        host_mem_gb = float(hardware_profile.get("host_mem_total_gb", 0.0))
        if host_mem_gb > 0:
            per_rank_mem_gb = host_mem_gb / max(1, local_world_size)

    min_mem_gb = float(opt_cfg.get("cpu_oom_guard_min_mem_gb_per_rank", 24.0))
    if per_rank_mem_gb <= 0 or per_rank_mem_gb >= min_mem_gb:
        return num_workers, prefetch_factor, pin_memory

    updated_workers = int(num_workers)
    updated_prefetch = int(prefetch_factor)
    updated_pin_memory = bool(pin_memory)

    max_workers = max(1, int(opt_cfg.get("cpu_oom_guard_max_workers", 8)))
    if updated_workers > max_workers:
        updated_workers = max_workers

    target_prefetch = max(1, int(opt_cfg.get("cpu_oom_guard_prefetch_factor", 1)))
    if updated_workers > 0 and updated_prefetch > target_prefetch:
        updated_prefetch = target_prefetch

    if bool(opt_cfg.get("cpu_oom_guard_disable_pin_memory", True)):
        updated_pin_memory = False

    if is_main_process and (
        updated_workers != num_workers
        or updated_prefetch != prefetch_factor
        or updated_pin_memory != pin_memory
    ):
        print(
            "CPU OOM guard applied: "
            f"mem_per_rank={per_rank_mem_gb:.1f}GB, "
            f"workers={num_workers}->{updated_workers}, "
            f"prefetch={prefetch_factor}->{updated_prefetch}, "
            f"pin_memory={pin_memory}->{updated_pin_memory}"
        )

    return updated_workers, updated_prefetch, updated_pin_memory
