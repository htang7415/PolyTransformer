"""Trainer for autoregressive backbone model."""

import math
import warnings
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm


def _to_float(value, name: str) -> float:
    """Convert config value to float with a clear error on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r} ({type(value).__name__})")


def _to_int(value, name: str) -> int:
    """Convert config value to int with a clear error on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be integer-like, got {value!r} ({type(value).__name__})")


def _is_cuda_device(device) -> bool:
    """Return True if the provided device resolves to CUDA."""
    try:
        return torch.device(device).type == 'cuda'
    except (TypeError, ValueError):
        return str(device).startswith('cuda')


def _supports_torch_compile(device) -> bool:
    """Return True if torch.compile can safely run on the current GPU."""
    if not _is_cuda_device(device) or not torch.cuda.is_available():
        return False
    try:
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    return major >= 7


def _supports_fp8_training(device) -> bool:
    """Return True when hardware likely supports performant FP8 training."""
    if not _is_cuda_device(device) or not torch.cuda.is_available():
        return False
    try:
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    # Hopper (SM90) or newer.
    return major >= 9


def _extract_loss(outputs, context: str) -> torch.Tensor:
    """Extract loss tensor from model outputs with clear failure messages."""
    loss = getattr(outputs, "loss", None)
    if loss is None and isinstance(outputs, dict):
        loss = outputs.get("loss")
    if loss is None:
        raise ValueError(
            f"Model outputs from {context} did not include a loss tensor."
        )
    return loss


def _forward_for_loss(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Forward model and return outputs, preferring HF-style loss computation."""
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    except TypeError as exc:
        # Legacy wrappers may not accept labels; fall back to original call style.
        if "labels" not in str(exc):
            raise
        return model(input_ids=input_ids, attention_mask=attention_mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for matrix params, AdamW for remaining params."""

    def __init__(self, param_groups: List[Dict]):
        super().__init__(param_groups, defaults={})

    @staticmethod
    def _orthogonalize_newton_schulz(
        matrix: torch.Tensor,
        steps: int,
        eps: float = 1.0e-7,
    ) -> torch.Tensor:
        """Orthogonalize update matrix using Newton-Schulz iterations."""
        x = matrix.float()
        transposed = False
        if x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)
            transposed = True

        x = x / (x.norm() + eps)
        for _ in range(max(1, int(steps))):
            x = 1.5 * x - 0.5 * (x @ (x.transpose(0, 1) @ x))

        if transposed:
            x = x.transpose(0, 1)
        return x

    @torch.no_grad()
    def _step_adamw(self, group: Dict) -> None:
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Sparse gradients are not supported by MuonAdamW AdamW groups.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = state["step"]

            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def _step_muon(self, group: Dict) -> None:
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        ns_steps = int(group["ns_steps"])
        weight_decay = float(group["weight_decay"])
        nesterov = bool(group.get("nesterov", True))

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.ndim < 2:
                continue
            if grad.is_sparse:
                raise RuntimeError("Sparse gradients are not supported by Muon groups.")

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            momentum_buffer = state["momentum_buffer"]
            momentum_buffer.mul_(momentum).add_(grad, alpha=1.0 - momentum)
            update = grad.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer

            original_shape = update.shape
            update_matrix = update
            if update_matrix.ndim > 2:
                update_matrix = update_matrix.view(update_matrix.shape[0], -1)

            ortho_update = self._orthogonalize_newton_schulz(update_matrix, steps=ns_steps)
            if update.ndim > 2:
                ortho_update = ortho_update.view(original_shape)

            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)

            # Muon reference scales updates for tall matrices.
            scale = 1.0
            if update_matrix.ndim == 2 and update_matrix.shape[1] > 0:
                scale = math.sqrt(max(1.0, update_matrix.shape[0] / update_matrix.shape[1]))
            p.add_(ortho_update.to(dtype=p.dtype), alpha=-(lr * scale))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            kind = group.get("kind", "adamw")
            if kind == "adamw":
                self._step_adamw(group)
            elif kind == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")
        return loss


class BackboneTrainer:
    """Trainer for autoregressive causal LM backbone."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results',
        step_dir: str = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: Optional[int] = None
    ):
        """Initialize trainer.

        Args:
            model: Autoregressive backbone model.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            config: Training configuration.
            device: Device for training.
            output_dir: Output directory for shared artifacts (checkpoints).
            step_dir: Step-specific output directory for metrics/figures.
            distributed: Whether to use DistributedDataParallel.
            rank: Global rank.
            world_size: Total number of ranks.
            local_rank: Local rank (GPU index).
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = (not self.distributed) or self.rank == 0
        self.output_dir = Path(output_dir)
        self.step_dir = Path(step_dir) if step_dir else self.output_dir

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.step_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Optimization config
        opt_config = config.get('optimization', {})
        self.use_amp = opt_config.get('use_amp', False) and _is_cuda_device(device)
        self.compile_model = opt_config.get('compile_model', False)
        self.compile_mode = opt_config.get('compile_mode', 'default')
        self.fp8_training = bool(opt_config.get('fp8_training', False))
        self.fp8_backend = str(opt_config.get('fp8_backend', 'torchao')).lower()
        self.fp8_strict = bool(opt_config.get('fp8_strict', False))
        self.fp8_enabled = False
        self.optimizer_type = str(opt_config.get('optimizer_type', 'adamw')).lower()
        if bool(opt_config.get('use_muon', False)):
            self.optimizer_type = 'muon_adamw'
        self.grad_accum_steps = max(1, _to_int(opt_config.get('gradient_accumulation_steps', 1), 'gradient_accumulation_steps'))

        # Enable cuDNN benchmark for consistent input sizes
        if opt_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True

        # Suppress SequentialLR deprecation warning (PyTorch internal issue)
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Optional FP8 conversion (experimental, H100/Hopper-oriented).
        self.model = self._maybe_enable_fp8(self.model)

        # Compile model for faster execution (guard against older GPUs)
        if self.compile_model and self.fp8_training:
            warnings.warn("torch.compile disabled when fp8_training is enabled.")
            self.compile_model = False
        if self.compile_model and self.distributed:
            warnings.warn("torch.compile disabled for DDP to avoid compilation issues.")
            self.compile_model = False
        if self.compile_model and _is_cuda_device(device):
            if not _supports_torch_compile(device):
                warnings.warn("torch.compile disabled: GPU compute capability < 7.0")
                self.compile_model = False
            else:
                print(f"Compiling model with torch.compile(mode='{self.compile_mode}')...")
                self.model = torch.compile(self.model, mode=self.compile_mode)
        if self.distributed:
            self.model = self._wrap_ddp(self.model)

        # Training config
        train_config = config['training_backbone']
        self.learning_rate = _to_float(train_config['learning_rate'], 'learning_rate')
        self.weight_decay = _to_float(train_config['weight_decay'], 'weight_decay')
        self.warmup_steps = _to_int(train_config['warmup_steps'], 'warmup_steps')
        self.max_steps = _to_int(train_config['max_steps'], 'max_steps')
        self.gradient_clip_norm = _to_float(train_config['gradient_clip_norm'], 'gradient_clip_norm')
        self.eval_every = _to_int(train_config['eval_every'], 'eval_every')
        self.save_every = _to_int(train_config['save_every'], 'save_every')
        self.num_epochs = _to_int(train_config.get('num_epochs', 50), 'num_epochs')

        ckpt_cfg = config.get('checkpointing', {})
        self.save_best_only = ckpt_cfg.get('save_best_only', True)
        self.save_last = ckpt_cfg.get('save_last', False)
        self.save_periodic = ckpt_cfg.get('save_periodic', False)

        self.optimizer = self._build_optimizer(opt_config)

        # Initialize scheduler (warmup + cosine decay)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.max_steps - self.warmup_steps),
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )

        # Training state
        self.global_step = 0
        self.micro_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_steps = []
        self.learning_rates = []

        # GPU memory monitoring
        self.memory_log_interval = opt_config.get('memory_log_interval', 500)
        self.memory_stats = []

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel when enabled."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return model
        if _is_cuda_device(self.device):
            device_index = torch.device(self.device).index
            return DDP(model, device_ids=[device_index], output_device=device_index)
        return DDP(model)

    def _unwrap_model_for_optimizer(self) -> nn.Module:
        """Return the underlying module for parameter grouping."""
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _build_optimizer(self, opt_config: Dict) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        if self.optimizer_type in {"muon", "muon_adamw", "muon+adamw"}:
            return self._build_muon_adamw_optimizer(opt_config)

        adam_beta1 = _to_float(opt_config.get('adam_beta1', 0.9), 'optimization.adam_beta1')
        adam_beta2 = _to_float(opt_config.get('adam_beta2', 0.999), 'optimization.adam_beta2')
        adam_eps = _to_float(opt_config.get('adam_eps', 1.0e-8), 'optimization.adam_eps')
        use_fused_adamw = bool(opt_config.get('fused_adamw', True)) and _is_cuda_device(self.device)

        optimizer_kwargs = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': (adam_beta1, adam_beta2),
            'eps': adam_eps,
        }
        if use_fused_adamw:
            optimizer_kwargs['fused'] = True
        try:
            optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        except TypeError:
            if use_fused_adamw and self.is_main_process:
                warnings.warn("Fused AdamW unavailable in this torch build; falling back to standard AdamW.")
            optimizer_kwargs.pop('fused', None)
            optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)

        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _build_muon_adamw_optimizer(self, opt_config: Dict) -> torch.optim.Optimizer:
        """Build Muon/AdamW multi-group optimizer."""
        adam_beta1 = _to_float(opt_config.get('adam_beta1', 0.9), 'optimization.adam_beta1')
        adam_beta2 = _to_float(opt_config.get('adam_beta2', 0.999), 'optimization.adam_beta2')
        adam_eps = _to_float(opt_config.get('adam_eps', 1.0e-8), 'optimization.adam_eps')

        adam_group_weight_decay = _to_float(
            opt_config.get('adamw_group_weight_decay', 0.0),
            'optimization.adamw_group_weight_decay',
        )
        muon_matrix_lr = _to_float(
            opt_config.get('muon_matrix_lr', self.learning_rate),
            'optimization.muon_matrix_lr',
        )
        muon_weight_decay = _to_float(
            opt_config.get('muon_weight_decay', self.weight_decay),
            'optimization.muon_weight_decay',
        )
        muon_momentum = _to_float(
            opt_config.get('muon_momentum', 0.95),
            'optimization.muon_momentum',
        )
        muon_ns_steps = _to_int(
            opt_config.get('muon_ns_steps', 5),
            'optimization.muon_ns_steps',
        )
        muon_nesterov = bool(opt_config.get('muon_nesterov', True))

        adam_embedding_lr = _to_float(
            opt_config.get('adam_embedding_lr', self.learning_rate),
            'optimization.adam_embedding_lr',
        )
        adam_unembedding_lr = _to_float(
            opt_config.get('adam_unembedding_lr', self.learning_rate),
            'optimization.adam_unembedding_lr',
        )
        adam_scalar_lr = _to_float(
            opt_config.get('adam_scalar_lr', self.learning_rate),
            'optimization.adam_scalar_lr',
        )

        model = self._unwrap_model_for_optimizer()
        hidden_size = None
        model_cfg = getattr(model, "config", None)
        if model_cfg is not None:
            hidden_size = getattr(model_cfg, "hidden_size", None)
        if hidden_size is None:
            backbone = getattr(model, "backbone", None)
            hidden_size = getattr(backbone, "hidden_size", None)

        dmodel_lr_scale = 1.0
        if bool(opt_config.get('muon_dmodel_lr_scale', True)) and hidden_size is not None:
            dmodel_lr_scale = (float(hidden_size) / 768.0) ** -0.5

        embedding_keywords = opt_config.get(
            'muon_embedding_keywords',
            ['token_embedding', 'position_embedding', 'wte', 'wpe', 'embed_tokens'],
        )
        unembedding_keywords = opt_config.get(
            'muon_unembedding_keywords',
            ['output_proj', 'lm_head', 'unembed'],
        )

        if isinstance(embedding_keywords, str):
            embedding_keywords = [embedding_keywords]
        if isinstance(unembedding_keywords, str):
            unembedding_keywords = [unembedding_keywords]
        embedding_keywords = tuple(str(k).lower() for k in embedding_keywords)
        unembedding_keywords = tuple(str(k).lower() for k in unembedding_keywords)

        embedding_params = []
        unembedding_params = []
        scalar_params = []
        muon_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            is_embedding = any(keyword in lname for keyword in embedding_keywords)
            is_unembedding = any(keyword in lname for keyword in unembedding_keywords)
            is_norm_or_bias = (
                param.ndim <= 1 or
                lname.endswith('.bias') or
                'norm' in lname or
                'layernorm' in lname or
                '.ln_' in lname
            )
            is_matrix_candidate = (
                param.ndim == 2 and
                (not is_embedding) and
                (not is_unembedding) and
                (not is_norm_or_bias)
            )

            if is_matrix_candidate:
                muon_params.append(param)
            elif is_embedding:
                embedding_params.append(param)
            elif is_unembedding:
                unembedding_params.append(param)
            else:
                scalar_params.append(param)

        param_groups: List[Dict] = []
        if unembedding_params:
            param_groups.append(dict(
                kind='adamw',
                params=unembedding_params,
                lr=adam_unembedding_lr * dmodel_lr_scale,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=adam_group_weight_decay,
            ))
        if embedding_params:
            param_groups.append(dict(
                kind='adamw',
                params=embedding_params,
                lr=adam_embedding_lr * dmodel_lr_scale,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=adam_group_weight_decay,
            ))
        if scalar_params:
            param_groups.append(dict(
                kind='adamw',
                params=scalar_params,
                lr=adam_scalar_lr * dmodel_lr_scale,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=adam_group_weight_decay,
            ))

        # Group Muon params by shape (nanochat-style) for better kernel locality.
        by_shape = {}
        for param in muon_params:
            by_shape.setdefault(tuple(param.shape), []).append(param)
        for shape in sorted(by_shape.keys()):
            group_params = by_shape[shape]
            param_groups.append(dict(
                kind='muon',
                params=group_params,
                lr=muon_matrix_lr,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                nesterov=muon_nesterov,
                weight_decay=muon_weight_decay,
            ))

        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer construction.")

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        if self.is_main_process:
            adam_count = len(embedding_params) + len(unembedding_params) + len(scalar_params)
            print(
                "Using Muon/AdamW optimizer: "
                f"muon_tensors={len(muon_params)}, adam_tensors={adam_count}, "
                f"dmodel_lr_scale={dmodel_lr_scale:.4f}"
            )
        return optimizer

    def _maybe_enable_fp8(self, model: nn.Module) -> nn.Module:
        """Convert supported modules to experimental FP8 training path."""
        if not self.fp8_training:
            return model

        def _handle_failure(message: str, exc: Optional[Exception] = None) -> nn.Module:
            if self.fp8_strict:
                if exc is not None:
                    raise RuntimeError(f"{message} ({exc})") from exc
                raise RuntimeError(message)
            warning_msg = message if exc is None else f"{message} ({exc})"
            warnings.warn(f"{warning_msg}. Continuing without FP8.")
            return model

        if not _supports_fp8_training(self.device):
            return _handle_failure(
                "fp8_training requested but current device is not Hopper-class CUDA"
            )

        if self.fp8_backend != 'torchao':
            return _handle_failure(
                f"Unsupported fp8_backend={self.fp8_backend!r}; only 'torchao' is currently integrated"
            )

        try:
            from torchao.float8 import convert_to_float8_training
        except Exception as exc:
            return _handle_failure(
                "fp8_training requested but torchao.float8 is not available",
                exc,
            )

        try:
            converted_model = convert_to_float8_training(model)
        except TypeError:
            # Keep compatibility with older/newer call signatures.
            try:
                converted_model = convert_to_float8_training(model=model)
            except Exception as exc:
                return _handle_failure("Failed to convert model with torchao FP8 backend", exc)
        except Exception as exc:
            return _handle_failure("Failed to convert model with torchao FP8 backend", exc)

        self.fp8_enabled = True
        if self.is_main_process:
            print("Enabled experimental FP8 training via torchao.float8.")
        return converted_model

    def _get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get a clean state_dict for saving (strip DDP/compile wrappers)."""
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model.state_dict()

    def _reduce_mean(self, value: float) -> float:
        """Average a scalar across ranks when using DDP."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / self.world_size).item()

    def _maybe_mark_cudagraph_step_begin(self) -> None:
        """Mark the beginning of a cudagraph step if supported."""
        if not self.compile_model or not _is_cuda_device(self.device):
            return

        compiler_mod = getattr(torch, "compiler", None)
        if compiler_mod is None:
            return

        mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
        if mark_step is None:
            return

        mark_step()

    def _log_gpu_memory(self) -> Dict:
        """Log GPU memory usage.

        Returns:
            Dictionary with memory statistics in GB.
        """
        if not _is_cuda_device(self.device) or not torch.cuda.is_available():
            return {}

        device_obj = torch.device(self.device)
        device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        stats = {
            'step': self.global_step,
            'allocated_gb': torch.cuda.memory_allocated(device_index) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(device_index) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device_index) / 1e9,
        }

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(device_index).total_memory / 1e9
        stats['total_gb'] = total_memory
        stats['free_gb'] = total_memory - stats['reserved_gb']
        stats['utilization_pct'] = (stats['allocated_gb'] / total_memory) * 100

        return stats

    def train(self) -> Dict:
        """Run training loop.

        Returns:
            Training history.
        """
        if self.is_main_process:
            print(f"Starting training for {self.num_epochs} epochs...")
            print(f"Train batches: {len(self.train_dataloader)}")
            print(f"Val batches: {len(self.val_dataloader)}")

        for epoch in range(self.num_epochs):
            # Training epoch
            train_loss = self._train_epoch(epoch)

            # Validation
            val_loss = self._validate()

            # Save checkpoint
            self._save_checkpoint(val_loss, epoch)

            # Barrier after checkpoint to prevent rank drift
            if self.distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()

            if self.is_main_process:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if self.global_step >= self.max_steps:
                if self.is_main_process:
                    print(f"Reached max steps ({self.max_steps}), stopping training.")
                break

        # Save final checkpoint
        self._save_checkpoint(val_loss, epoch, final=True)

        # Save training history
        self._save_history()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_steps': self.val_steps,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if self.distributed:
            data_sampler = getattr(self.train_dataloader, "sampler", None)
            if hasattr(data_sampler, "set_epoch"):
                data_sampler.set_epoch(epoch)
            batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not self.is_main_process)
        for batch_idx, batch in enumerate(pbar):
            if self.global_step >= self.max_steps:
                break

            is_last_batch = (batch_idx + 1) == len(self.train_dataloader)
            hits_step_limit = (self.global_step + 1) >= self.max_steps
            should_step = (
                ((batch_idx + 1) % self.grad_accum_steps == 0) or
                is_last_batch or
                hits_step_limit
            )
            loss = self._train_step(batch, should_step=should_step)
            total_loss += loss
            num_batches += 1
            self.micro_step += 1

            if self.is_main_process:
                self.train_losses.append(loss)
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if self.is_main_process:
                pbar.set_postfix({'loss': f'{loss:.4f}'})

            if should_step:
                self.global_step += 1

                # Periodic validation
                if self.global_step > 0 and self.global_step % self.eval_every == 0:
                    val_loss = self._validate()
                    self.model.train()
                    if self.is_main_process:
                        self.val_losses.append(val_loss)
                        self.val_steps.append(self.global_step)
                        self._save_checkpoint(val_loss, epoch)
                    if self.distributed and dist.is_available() and dist.is_initialized():
                        dist.barrier()

                # Periodic save
                if (
                    not self.save_best_only and self.save_periodic and
                    self.global_step > 0 and self.global_step % self.save_every == 0
                ):
                    self._save_periodic_checkpoint(epoch)
                    if self.distributed and dist.is_available() and dist.is_initialized():
                        dist.barrier()

                # GPU memory monitoring
                if self.global_step > 0 and self.global_step % self.memory_log_interval == 0:
                    if self.is_main_process:
                        mem_stats = self._log_gpu_memory()
                        if mem_stats:
                            self.memory_stats.append(mem_stats)
                            pbar.set_postfix({
                                'loss': f'{loss:.4f}',
                                'mem': f'{mem_stats["allocated_gb"]:.1f}/{mem_stats["total_gb"]:.0f}GB'
                            })

        # Barrier before final reduce to ensure all ranks exit loop together
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

        avg_loss = total_loss / max(num_batches, 1)
        return self._reduce_mean(avg_loss)

    def _train_step(self, batch: Dict[str, torch.Tensor], should_step: bool) -> float:
        """Single training step.

        Args:
            batch: Batch of data.
            should_step: Whether to apply optimizer/scheduler updates this micro-batch.

        Returns:
            Loss value.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Skip DDP gradient sync on accumulation microsteps to reduce communication overhead.
        sync_context = nullcontext()
        if self.distributed and isinstance(self.model, DDP) and not should_step:
            sync_context = self.model.no_sync()
        with sync_context:
            # Forward pass with AMP
            self._maybe_mark_cudagraph_step_begin()
            with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = _forward_for_loss(self.model, input_ids, attention_mask)
                loss = _extract_loss(outputs, context="_train_step") / self.grad_accum_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

        if should_step:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
            )

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        return loss.item() * self.grad_accum_steps

    def _validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                self._maybe_mark_cudagraph_step_begin()
                with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    outputs = _forward_for_loss(self.model, input_ids, attention_mask)
                total_loss += _extract_loss(outputs, context="_validate").item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return self._reduce_mean(avg_loss)

    def _save_checkpoint(self, val_loss: float, epoch: int, final: bool = False):
        """Save model checkpoint.

        Args:
            val_loss: Validation loss.
            epoch: Current epoch.
            final: Whether this is the final checkpoint.
        """
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'micro_step': self.micro_step,
            'model_state_dict': self._get_model_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'backbone_best.pt')
            print(f"New best model saved with val_loss: {val_loss:.4f}")

        # Save final checkpoint
        if final and not self.save_best_only and self.save_last:
            torch.save(checkpoint, self.checkpoint_dir / 'backbone_last.pt')

    def _save_periodic_checkpoint(self, epoch: int):
        """Save periodic checkpoint.

        Args:
            epoch: Current epoch.
        """
        if self.save_best_only or not self.save_periodic:
            return
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'micro_step': self.micro_step,
            'model_state_dict': self._get_model_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f'backbone_step_{self.global_step}.pt')

    def _save_history(self):
        """Save training history to CSV."""
        if not self.is_main_process:
            return
        # Round floats to 4 decimal places
        rounded_train_losses = [round(loss, 4) for loss in self.train_losses]
        rounded_learning_rates = [round(lr, 8) for lr in self.learning_rates]  # LR needs more precision
        rounded_val_losses = [round(loss, 4) for loss in self.val_losses]

        # Save as DataFrame
        train_df = pd.DataFrame({
            'step': list(range(len(self.train_losses))),
            'train_loss': rounded_train_losses,
            'learning_rate': rounded_learning_rates
        })
        train_df.to_csv(self.metrics_dir / 'backbone_loss_curve.csv', index=False)

        if self.val_losses and self.val_steps:
            paired_count = min(len(self.val_losses), len(self.val_steps))
            val_df = pd.DataFrame({
                'step': self.val_steps[:paired_count],
                'val_loss': rounded_val_losses[:paired_count]
            })
            val_df.to_csv(self.metrics_dir / 'backbone_val_loss.csv', index=False)

        # Save memory stats
        if self.memory_stats:
            mem_df = pd.DataFrame(self.memory_stats)
            mem_df.to_csv(self.metrics_dir / 'gpu_memory_stats.csv', index=False)
            print(f"GPU Memory Stats saved. Peak usage: {mem_df['max_allocated_gb'].max():.2f} GB")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.micro_step = checkpoint.get('micro_step', self.global_step * self.grad_accum_steps)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from step {self.global_step}")
