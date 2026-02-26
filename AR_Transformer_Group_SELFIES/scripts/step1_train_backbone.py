#!/usr/bin/env python
"""Step 1: Train autoregressive backbone model."""

import os
import sys
import argparse
import math
from pathlib import Path
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.model_scales import (
    get_model_config, get_training_config, estimate_params,
    get_results_dir, print_model_info
)
from src.data.hf_tokenizer import load_polymer_tokenizer
from src.data.dataset import PolymerDataset, collate_fn, dynamic_collate_fn
from src.data.samplers import LengthBucketBatchSampler
from src.model.hf_ar import build_polymer_ar_model, load_polymer_ar_checkpoint
from src.training.trainer_backbone import BackboneTrainer
from src.utils.reproducibility import seed_everything, save_run_metadata
from shared.unlabeled_data import require_preprocessed_unlabeled_splits
from shared.training_runtime import (
    apply_cuda_allocator_settings,
    detect_hardware_profile,
    save_hardware_profile,
    maybe_apply_hardware_aware_batching,
    maybe_apply_cpu_oom_guards,
)


def init_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if not dist.is_available():
        return False, 0, 1, 0, None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0, None
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return True, rank, world_size, local_rank, device


def _nearest_power_of_two(value: float) -> int:
    """Round positive value to nearest power of two."""
    return int(2 ** round(math.log2(max(1.0, float(value)))))


def _estimate_scaling_params(backbone_config: dict, vocab_size: int) -> int:
    """Estimate nanochat-style scaling params: transformer matrices + lm_head."""
    hidden_size = int(backbone_config['hidden_size'])
    num_layers = int(backbone_config['num_layers'])
    ffn_hidden_size = int(backbone_config['ffn_hidden_size'])

    attn_params = 4 * hidden_size * hidden_size
    ffn_params = 2 * hidden_size * ffn_hidden_size
    ln_params = 4 * hidden_size
    transformer_params = num_layers * (attn_params + ffn_params + ln_params)
    lm_head_params = vocab_size * hidden_size
    return int(transformer_params + lm_head_params)


def _maybe_apply_compute_optimal_scaling(
    config: dict,
    backbone_config: dict,
    vocab_size: int,
    world_size: int,
    is_main_process: bool,
) -> None:
    """Apply nanochat-style compute-optimal scaling for max_steps and LR."""
    opt_cfg = config.get('optimization', {})
    if not bool(opt_cfg.get('compute_optimal_scaling', False)):
        return

    target_ratio = float(opt_cfg.get('compute_opt_target_param_data_ratio', 10.5))
    if target_ratio <= 0:
        raise ValueError("optimization.compute_opt_target_param_data_ratio must be > 0.")

    seq_len_tokens = int(
        opt_cfg.get(
            'compute_opt_seq_len_tokens',
            backbone_config.get(
                'max_position_embeddings',
                config.get('backbone', {}).get('max_position_embeddings', 256),
            ),
        )
    )
    if seq_len_tokens <= 0:
        raise ValueError("optimization.compute_opt_seq_len_tokens must be > 0.")

    global_batch_samples = int(
        config['training_backbone']['batch_size'] *
        config['optimization']['gradient_accumulation_steps'] *
        world_size
    )
    if global_batch_samples <= 0:
        raise ValueError("Global batch samples must be > 0 for compute-optimal scaling.")
    global_batch_tokens = global_batch_samples * seq_len_tokens

    scaling_params = _estimate_scaling_params(backbone_config, vocab_size)
    target_tokens = int(target_ratio * scaling_params)
    target_max_steps = max(1, target_tokens // max(1, global_batch_tokens))
    prev_max_steps = int(config['training_backbone']['max_steps'])
    config['training_backbone']['max_steps'] = target_max_steps

    warmup_ratio = float(opt_cfg.get('compute_opt_warmup_ratio', -1.0))
    if warmup_ratio >= 0.0:
        config['training_backbone']['warmup_steps'] = max(1, int(round(target_max_steps * warmup_ratio)))

    lr_scale = 1.0
    if bool(opt_cfg.get('compute_opt_lr_batch_scaling', True)):
        ref_global_batch_tokens = int(opt_cfg.get('compute_opt_ref_global_batch_tokens', 0))
        if ref_global_batch_tokens <= 0:
            ref_global_batch_samples = int(opt_cfg.get('auto_batch_ref_global_batch', 1024))
            ref_global_batch_tokens = max(1, ref_global_batch_samples * seq_len_tokens)

        lr_scale = math.sqrt(float(global_batch_tokens) / float(ref_global_batch_tokens))
        config['training_backbone']['learning_rate'] = float(config['training_backbone']['learning_rate']) * lr_scale
        for lr_key in ['adam_embedding_lr', 'adam_unembedding_lr', 'adam_scalar_lr', 'muon_matrix_lr']:
            if lr_key in config['optimization']:
                config['optimization'][lr_key] = float(config['optimization'][lr_key]) * lr_scale

    if is_main_process:
        print(
            "Compute-optimal scaling enabled: "
            f"scaling_params={scaling_params:,}, target_tokens={target_tokens:,}, "
            f"global_batch_tokens={global_batch_tokens:,}, max_steps={target_max_steps:,} "
            f"(was {prev_max_steps:,}), lr_scale={lr_scale:.4f}"
        )


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)
    preflight_is_main = int(os.environ.get("RANK", "0") or 0) == 0
    apply_cuda_allocator_settings(config.get('optimization', {}), preflight_is_main)

    distributed, rank, world_size, local_rank, dist_device = init_distributed()
    is_main_process = (not distributed) or rank == 0

    # Set device
    device = dist_device if distributed else ('cuda' if torch.cuda.is_available() else 'cpu')
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
    if is_main_process:
        print(f"Using device: {device}")

    # Override results_dir if model_size specified
    base_results_dir = config['paths']['results_dir']
    results_dir = Path(get_results_dir(args.model_size, base_results_dir))
    step_dir = results_dir / 'step1_backbone'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    hardware_profile = detect_hardware_profile(
        world_size=world_size,
        local_world_size=local_world_size,
        device=device,
        distributed=distributed,
    )
    if is_main_process:
        save_hardware_profile(metrics_dir / 'hardware_profile.json', hardware_profile)
        gpu_mem_min = float(hardware_profile.get('gpu_mem_gb_min', 0.0))
        gpu_mem_max = float(hardware_profile.get('gpu_mem_gb_max', 0.0))
        gpu_desc = "cpu-only"
        if gpu_mem_min > 0.0:
            gpu_desc = (
                f"{world_size} training GPU(s), visible_local={hardware_profile.get('visible_gpu_count', 0)}, "
                f"memory_range={gpu_mem_min:.1f}-{gpu_mem_max:.1f}GB"
            )
        print(
            "Detected hardware: "
            f"{gpu_desc}, local_world_size={local_world_size}, "
            f"cpu_mem_per_rank~{float(hardware_profile.get('cpu_mem_gb_per_rank_est', 0.0)):.1f}GB"
        )

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    if is_main_process:
        save_config(config, step_dir / 'config_used.yaml')
        save_run_metadata(step_dir, args.config, seed_info)

    if is_main_process:
        print("=" * 50)
        print("Step 1: Training Autoregressive Backbone")
        print("=" * 50)

    # Get model and training config based on model_size
    backbone_config = get_model_config(args.model_size, config, model_type='sequence')
    if args.model_size:
        training_config = get_training_config(args.model_size, config, model_type='sequence')
        # Override training_backbone config
        config['training_backbone']['batch_size'] = training_config['batch_size']
        config['training_backbone']['learning_rate'] = training_config['learning_rate']
        config['training_backbone']['max_steps'] = training_config['max_steps']
        config['training_backbone']['warmup_steps'] = training_config['warmup_steps']
        config['optimization']['gradient_accumulation_steps'] = training_config['gradient_accumulation_steps']

    maybe_apply_hardware_aware_batching(
        config=config,
        backbone_config=backbone_config,
        hardware_profile=hardware_profile,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    # Optional nanochat-style auto global batch scaling from model depth.
    opt_cfg = config.get('optimization', {})
    if bool(opt_cfg.get('auto_batch_scaling', False)):
        ref_depth = max(1.0, float(opt_cfg.get('auto_batch_ref_depth', 6)))
        ref_global_batch = max(1.0, float(opt_cfg.get('auto_batch_ref_global_batch', 1024)))
        exponent = float(opt_cfg.get('auto_batch_exponent', 0.766))
        round_pow2 = bool(opt_cfg.get('auto_batch_power_of_two', True))

        depth = float(backbone_config.get('num_layers', config['backbone']['num_layers']))
        target_global_batch = ref_global_batch * ((depth / ref_depth) ** exponent)
        if round_pow2:
            target_global_batch = float(_nearest_power_of_two(target_global_batch))

        per_micro_global = float(config['training_backbone']['batch_size'] * world_size)
        auto_grad_accum = max(1, int(round(target_global_batch / max(1.0, per_micro_global))))
        config['optimization']['gradient_accumulation_steps'] = auto_grad_accum
        achieved_global_batch = int(config['training_backbone']['batch_size'] * auto_grad_accum * world_size)

        if is_main_process:
            print(
                "Auto batch scaling enabled: "
                f"depth={int(depth)}, target_global_batch={int(target_global_batch)}, "
                f"grad_accum={auto_grad_accum}, achieved_global_batch={achieved_global_batch}"
            )

    # Load tokenizer (from base results dir which has the tokenizer)
    if is_main_process:
        print("\n1. Loading tokenizer...")
    tokenizer = load_polymer_tokenizer(results_dir, Path(base_results_dir))
    if is_main_process:
        print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Optional nanochat-style compute-optimal scaling for max_steps and learning rates.
    _maybe_apply_compute_optimal_scaling(
        config=config,
        backbone_config=backbone_config,
        vocab_size=tokenizer.vocab_size,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    # Keep a single effective training_config view after all runtime overrides.
    training_config = {
        'batch_size': int(config['training_backbone']['batch_size']),
        'learning_rate': float(config['training_backbone']['learning_rate']),
        'max_steps': int(config['training_backbone']['max_steps']),
        'warmup_steps': int(config['training_backbone']['warmup_steps']),
        'gradient_accumulation_steps': int(config['optimization']['gradient_accumulation_steps']),
    }
    if is_main_process:
        effective_global_batch = (
            training_config['batch_size'] *
            training_config['gradient_accumulation_steps'] *
            world_size
        )
        print(f"Effective global batch (samples/update): {effective_global_batch}")

    # Print model info if model_size specified
    if args.model_size and is_main_process:
        print_model_info(args.model_size, backbone_config, training_config,
                        tokenizer.vocab_size, model_type='sequence')

    # Load data (from base results dir which has the data)
    if is_main_process:
        print("\n2. Loading data...")
    repo_root = Path(__file__).resolve().parents[2]
    train_path, val_path = require_preprocessed_unlabeled_splits(repo_root)
    gs_cfg = config.get('group_selfies', {})
    cache_train_file = str(gs_cfg.get('step1_cache_train_file', 'train_group_selfies_cache.csv.gz'))
    cache_val_file = str(gs_cfg.get('step1_cache_val_file', 'val_group_selfies_cache.csv.gz'))
    cache_train_candidates = [
        results_dir / cache_train_file,
        Path(base_results_dir) / cache_train_file,
    ]
    cache_val_candidates = [
        results_dir / cache_val_file,
        Path(base_results_dir) / cache_val_file,
    ]
    train_cache_path = next((p for p in cache_train_candidates if p.exists()), None)
    val_cache_path = next((p for p in cache_val_candidates if p.exists()), None)
    using_group_selfies_cache = train_cache_path is not None and val_cache_path is not None

    if using_group_selfies_cache:
        train_df = pd.read_csv(train_cache_path, usecols=['p_smiles', 'group_selfies'])
        val_df = pd.read_csv(val_cache_path, usecols=['p_smiles', 'group_selfies'])
        if is_main_process:
            print(f"Using precomputed train Group SELFIES cache: {train_cache_path}")
            print(f"Using precomputed val Group SELFIES cache: {val_cache_path}")
    else:
        if is_main_process and ((train_cache_path is None) != (val_cache_path is None)):
            print("Partial Group SELFIES cache found; falling back to shared p_smiles splits.")
        train_df = pd.read_csv(train_path, usecols=['p_smiles'])
        val_df = pd.read_csv(val_path, usecols=['p_smiles'])
        if is_main_process:
            print("Group SELFIES Step1 cache not found; using on-the-fly tokenization.")

    # Optionally subsample training data.
    train_fraction = config.get('data', {}).get('train_fraction', 1.0)
    if train_fraction <= 0 or train_fraction > 1:
        raise ValueError("data.train_fraction must be within (0, 1].")
    if train_fraction < 1.0:
        full_train_count = len(train_df)
        n_train = max(1, int(round(full_train_count * train_fraction)))
        train_df = train_df.sample(
            n=n_train, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Using {n_train}/{full_train_count} train samples ({train_fraction:.2%})")

    # Optionally subsample validation data for faster periodic evaluation.
    train_cfg = config.get('training_backbone', {})
    val_fraction = float(train_cfg.get('val_fraction', 1.0))
    if val_fraction <= 0 or val_fraction > 1:
        raise ValueError("training_backbone.val_fraction must be within (0, 1].")
    if val_fraction < 1.0:
        full_val_count = len(val_df)
        n_val = max(1, int(round(full_val_count * val_fraction)))
        val_df = val_df.sample(
            n=n_val, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Using {n_val}/{full_val_count} val samples ({val_fraction:.2%})")
    val_max_samples = int(train_cfg.get('val_max_samples', 0))
    if val_max_samples > 0 and len(val_df) > val_max_samples:
        full_val_count = len(val_df)
        val_df = val_df.sample(
            n=val_max_samples, random_state=config['data']['random_seed']
        ).reset_index(drop=True)
        if is_main_process:
            print(f"Capping val samples to {val_max_samples}/{full_val_count} for faster eval")

    # Get optimization settings
    opt_config = config.get('optimization', {})
    cache_tokenization = opt_config.get('cache_tokenization', False)
    cache_max_samples = int(opt_config.get('cache_tokenization_max_samples', 500000))
    num_workers = int(opt_config.get('num_workers', 4))
    tokenize_canonicalize = bool(opt_config.get('tokenize_canonicalize', False))
    persistent_workers = bool(opt_config.get('persistent_workers', False)) and num_workers > 0
    pin_memory = opt_config.get('pin_memory', True)
    prefetch_factor = opt_config.get('prefetch_factor', 2)
    dynamic_padding = bool(opt_config.get('dynamic_padding', False))
    length_bucket_sampler = bool(opt_config.get('length_bucket_sampler', False))
    bucket_size_multiplier = int(opt_config.get('bucket_size_multiplier', 50))
    if bucket_size_multiplier <= 0:
        raise ValueError("optimization.bucket_size_multiplier must be > 0.")

    # Bound DataLoader workers to per-rank CPU budget to avoid oversubscription.
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    host_cpus = os.cpu_count() or 1
    if slurm_cpus_per_task > 0:
        per_rank_cpu_budget = max(1, slurm_cpus_per_task // max(1, local_world_size))
    else:
        per_rank_cpu_budget = max(1, host_cpus // max(1, local_world_size))
    per_rank_worker_cap = max(1, per_rank_cpu_budget - 2)
    if num_workers <= 0:
        num_workers = per_rank_worker_cap
        if is_main_process:
            print(
                "Auto-selected DataLoader workers per rank: "
                f"{num_workers} (cpu_budget={per_rank_cpu_budget}, local_world_size={local_world_size})"
            )
    elif num_workers > per_rank_worker_cap:
        if is_main_process:
            print(
                f"Capping num_workers from {num_workers} to {per_rank_worker_cap} "
                f"(cpu_budget={per_rank_cpu_budget}, local_world_size={local_world_size})"
            )
        num_workers = per_rank_worker_cap
    num_workers, prefetch_factor, pin_memory = maybe_apply_cpu_oom_guards(
        opt_cfg=opt_config,
        hardware_profile=hardware_profile,
        local_world_size=local_world_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        is_main_process=is_main_process,
    )
    persistent_workers = bool(opt_config.get('persistent_workers', False)) and num_workers > 0

    # Guard against memory blow-up: full-cache can be too large for multi-million datasets.
    total_samples = len(train_df) + len(val_df)
    if cache_tokenization and distributed:
        if is_main_process:
            print("Disabling cache_tokenization under DDP to avoid per-rank RAM duplication.")
        cache_tokenization = False
    elif cache_tokenization and total_samples > cache_max_samples:
        if is_main_process:
            print(
                f"Disabling cache_tokenization for {total_samples:,} samples "
                f"(limit={cache_max_samples:,})."
            )
        cache_tokenization = False

    # Create datasets
    dataset_smiles_col = 'group_selfies' if using_group_selfies_cache else 'p_smiles'
    train_dataset = PolymerDataset(
        train_df,
        tokenizer,
        smiles_col=dataset_smiles_col,
        cache_tokenization=cache_tokenization,
        pad_to_max_length=not dynamic_padding,
        canonicalize=tokenize_canonicalize and not using_group_selfies_cache,
        pretokenized_group_selfies=using_group_selfies_cache
    )
    val_dataset = PolymerDataset(
        val_df,
        tokenizer,
        smiles_col=dataset_smiles_col,
        cache_tokenization=cache_tokenization,
        pad_to_max_length=not dynamic_padding,
        canonicalize=tokenize_canonicalize and not using_group_selfies_cache,
        pretokenized_group_selfies=using_group_selfies_cache
    )

    active_collate_fn = collate_fn
    if dynamic_padding:
        active_collate_fn = partial(dynamic_collate_fn, pad_token_id=tokenizer.pad_token_id)

    # Create dataloaders
    batch_size = config['training_backbone']['batch_size']
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    if length_bucket_sampler:
        train_lengths = train_df['p_smiles'].astype(str).str.len().tolist()
        train_batch_sampler = LengthBucketBatchSampler(
            lengths=train_lengths,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            seed=config['data']['random_seed'],
            bucket_size_multiplier=bucket_size_multiplier,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=active_collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=active_collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=active_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    if is_main_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        if dynamic_padding:
            print("Using dynamic batch padding for Step1 dataloaders.")
        if length_bucket_sampler:
            print(
                f"Using length-bucket batching (bucket_size_multiplier={bucket_size_multiplier})."
            )
        print(f"DataLoader workers per rank: {num_workers}")
        if using_group_selfies_cache:
            print("Using precomputed Group SELFIES cache in Step1 (RDKit/grammar bypass in Dataset).")
        elif not tokenize_canonicalize:
            print("Using fast Group SELFIES tokenization in Step1 (canonicalize=False).")

    # Create model
    if is_main_process:
        print("\n3. Creating model...")
    model = build_polymer_ar_model(backbone_config, tokenizer, config['diffusion'])

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")

    # Resume from checkpoint if specified
    if args.resume:
        if is_main_process:
            print(f"\nResuming from checkpoint: {args.resume}")
        load_polymer_ar_checkpoint(model, args.resume, map_location=device)

    # Create trainer
    if is_main_process:
        print("\n4. Starting training...")
    trainer = BackboneTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
        output_dir=str(step_dir),
        step_dir=str(step_dir),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )

    # Train
    history = trainer.train()

    # Create loss plot
    if is_main_process:
        print("\n5. Creating loss plot...")
        plotter = PlotUtils(
            figure_size=tuple(config['plotting']['figure_size']),
            font_size=config['plotting']['font_size'],
            dpi=config['plotting']['dpi']
        )

        plotter.loss_curve(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            xlabel='Step',
            ylabel='Loss',
            title='Backbone Training Loss',
            save_path=figures_dir / 'backbone_loss_curve.png'
        )

        # Convert nats/token to bits/token (BpB) and plot against epoch progress.
        inv_ln2 = 1.0 / math.log(2.0)
        train_bpb = [loss * inv_ln2 for loss in history['train_losses']]
        val_bpb = [loss * inv_ln2 for loss in history['val_losses']]

        micro_batches_per_epoch = max(1, len(train_loader))
        train_epochs = [
            (idx + 1) / micro_batches_per_epoch
            for idx in range(len(train_bpb))
        ]
        grad_accum_steps = max(1, int(config['optimization']['gradient_accumulation_steps']))
        update_steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / grad_accum_steps))
        val_steps = history.get('val_steps', [])
        val_epochs = None
        if val_bpb and val_steps and len(val_bpb) == len(val_steps):
            val_epochs = [step / update_steps_per_epoch for step in val_steps]

        plotter.loss_curve(
            train_losses=train_bpb,
            val_losses=val_bpb,
            xlabel='Epoch',
            ylabel='BpB',
            title='Backbone Training BpB',
            save_path=figures_dir / 'backbone_bpb_curve.png',
            train_x=train_epochs,
            val_x=val_epochs
        )

        # Export best checkpoint in HF-pretrained format.
        best_ckpt_path = step_dir / 'checkpoints' / 'backbone_best.pt'
        hf_export_dir = step_dir / 'checkpoints' / 'backbone_best_hf'
        if best_ckpt_path.exists():
            export_model = build_polymer_ar_model(backbone_config, tokenizer, config['diffusion'])
            load_polymer_ar_checkpoint(export_model, best_ckpt_path, map_location='cpu')
            export_model.save_pretrained(str(hf_export_dir))
            tokenizer.save_pretrained(str(hf_export_dir))
            print(f"HF checkpoint exported to: {hf_export_dir}")
        else:
            print(f"Skipping HF export: missing {best_ckpt_path}")

        print("\n" + "=" * 50)
        print("Backbone training complete!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Checkpoints saved to: {step_dir / 'checkpoints'}")
        print("=" * 50)

    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train autoregressive backbone')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~12M, medium: ~50M, large: ~150M, xl: ~400M)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (.pt or HF model directory)')
    args = parser.parse_args()
    main(args)
