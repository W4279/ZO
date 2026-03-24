# -*- coding: utf-8 -*-
"""
FO/MeZO/ZOAdamW/ZO_Ours training for LoRA fine-tuning on NarrativeQA.
"""

import argparse
import hashlib
import json
import math
import os
import signal
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

try:
    import wandb
except Exception:
    wandb = None

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_optional_float(value: str, field_name: str) -> Optional[float]:
    val = value.strip().lower()
    if val in {"none", "null", ""}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{field_name} must be a float or 'none', got: {value}"
        ) from exc


def parse_lora_dropout(value: str) -> float:
    parsed = parse_optional_float(value, "lora_dropout")
    if parsed is None:
        return 0.0
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("lora_dropout must be >= 0 or 'none'")
    return parsed


def parse_max_grad_norm(value: str) -> Optional[float]:
    parsed = parse_optional_float(value, "max_grad_norm")
    if parsed is None:
        return None
    if parsed <= 0.0:
        return None
    return parsed


def parse_int_schedule(schedule_text: str, default_value: int) -> List[Tuple[int, int]]:
    text = schedule_text.strip()
    if not text:
        return [(0, default_value)]

    schedule: List[Tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid schedule item '{item}', expected format step:value"
            )
        step_text, value_text = item.split(":", 1)
        step = int(step_text.strip())
        value = int(value_text.strip())
        if step < 0:
            raise argparse.ArgumentTypeError("Schedule step must be >= 0")
        if value <= 0:
            raise argparse.ArgumentTypeError("Schedule value must be > 0")
        schedule.append((step, value))

    if not schedule:
        return [(0, default_value)]

    schedule.sort(key=lambda x: x[0])
    if schedule[0][0] != 0:
        schedule.insert(0, (0, default_value))

    deduped: List[Tuple[int, int]] = []
    for step, value in schedule:
        if deduped and deduped[-1][0] == step:
            deduped[-1] = (step, value)
        else:
            deduped.append((step, value))
    return deduped


def get_schedule_value(schedule: Sequence[Tuple[int, int]], global_step: int) -> int:
    current = schedule[0][1]
    for step, value in schedule:
        if global_step >= step:
            current = value
        else:
            break
    return current


def parse_float_schedule(schedule_text: str) -> List[Tuple[int, float]]:
    text = schedule_text.strip()
    if not text:
        return []
    schedule: List[Tuple[int, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid schedule item '{item}', expected format step:value"
            )
        step_text, value_text = item.split(":", 1)
        step = int(step_text.strip())
        value = float(value_text.strip())
        if step < 0:
            raise argparse.ArgumentTypeError("Schedule step must be >= 0")
        if not (0.0 <= value <= 1.0):
            raise argparse.ArgumentTypeError("subspace_alpha schedule value must be in [0, 1]")
        schedule.append((step, value))
    schedule.sort(key=lambda x: x[0])
    return schedule


def parse_lr_schedule(schedule_text: str) -> List[Tuple[int, float]]:
    text = schedule_text.strip()
    if not text:
        return []

    schedule: List[Tuple[int, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid lr schedule item '{item}', expected format step:value"
            )
        step_text, value_text = item.split(":", 1)
        step = int(step_text.strip())
        value = float(value_text.strip())
        if step < 0:
            raise argparse.ArgumentTypeError("LR schedule step must be >= 0")
        if value < 0.0:
            raise argparse.ArgumentTypeError("LR schedule value must be >= 0")
        schedule.append((step, value))

    schedule.sort(key=lambda x: x[0])
    deduped: List[Tuple[int, float]] = []
    for step, value in schedule:
        if deduped and deduped[-1][0] == step:
            deduped[-1] = (step, value)
        else:
            deduped.append((step, value))
    return deduped


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def dist_barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_rank_info() -> Tuple[int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0
    return local_rank, world_size, is_main


def compute_grad_norm_from_model(model, norm_type: float = 2.0) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    device = grads[0].device
    total_norm = torch.zeros(1, device=device, dtype=torch.float32)
    for g in grads:
        param_norm = g.detach().float().norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm.item()


def compute_grad_mean_from_model(model) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    total_abs_sum = 0.0
    total_numel = 0
    for g in grads:
        total_abs_sum += g.detach().float().abs().sum().item()
        total_numel += g.numel()

    if total_numel == 0:
        return 0.0
    return total_abs_sum / total_numel


def compute_grad_norm_from_list(grads: List[torch.Tensor], norm_type: float = 2.0) -> float:
    if not grads:
        return 0.0
    device = grads[0].device
    total_norm = torch.zeros(1, device=device, dtype=torch.float32)
    for g in grads:
        param_norm = g.detach().float().norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm.item()


def compute_grad_mean_from_list(grads: List[torch.Tensor]) -> float:
    if not grads:
        return 0.0

    total_abs_sum = 0.0
    total_numel = 0
    for g in grads:
        total_abs_sum += g.detach().float().abs().sum().item()
        total_numel += g.numel()

    if total_numel == 0:
        return 0.0
    return total_abs_sum / total_numel


def clip_grad_norm_inplace(grads: List[torch.Tensor], max_norm: float) -> float:
    grad_norm = compute_grad_norm_from_list(grads)
    if max_norm > 0 and grad_norm > max_norm:
        clip_coef = max_norm / (grad_norm + 1e-6)
        for g in grads:
            g.mul_(clip_coef)
        return max_norm
    return grad_norm

def init_loss_smoother(owner, args) -> None:
    owner._loss_ema = None
    owner._loss_ema_alpha = float(getattr(args, "loss_smooth_alpha", 0.98))

def update_loss_smoother(owner, loss_val: float) -> float:
    a = owner._loss_ema_alpha
    if owner._loss_ema is None:
        owner._loss_ema = float(loss_val)
    else:
        owner._loss_ema = a * owner._loss_ema + (1.0 - a) * float(loss_val)
    return owner._loss_ema

def maybe_log_loss_smooth(
    owner,
    logs: Dict[str, float],
    digits: int = 4,
    source_key: str = "loss",
) -> None:
    """Update EMA from the logged loss value and emit loss_smooth."""
    if source_key not in logs:
        return
    val = update_loss_smoother(owner, float(logs[source_key]))
    logs["loss_smooth"] = round(val, digits)


class FOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_schedule: List[Tuple[int, float]] = list(getattr(self.args, "lr_schedule_parsed", []))
        self._window_grad_norms: List[float] = []
        self._window_grad_means: List[float] = []
        self._window_grad_norms_clipped: List[float] = []
        self._window_ppl: List[float] = []
        self._window_entropy: List[float] = []
        init_loss_smoother(self, self.args)

    def _get_custom_lr(self) -> Optional[float]:
        if not self.lr_schedule:
            return None
        current = self.lr_schedule[0][1]
        for step, lr_val in self.lr_schedule:
            if self.state.global_step >= step:
                current = lr_val
            else:
                break
        return current

    def _set_optimizer_lr(self, lr: float) -> None:
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # For custom lr_schedule, disable HF scheduler to avoid double scheduling.
        if self.lr_schedule:
            self.lr_scheduler = None
            return None
        return super().create_scheduler(num_training_steps, optimizer)

    def optimizer_step(self, *args, **kwargs):
        custom_lr = self._get_custom_lr()
        if custom_lr is not None:
            self._set_optimizer_lr(custom_lr)
        return super().optimizer_step(*args, **kwargs)

    def _compute_entropy(self, model, inputs) -> float:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs.get("labels", None)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != -100).float()
        else:
            shift_logits = logits
            mask = torch.ones(shift_logits.shape[:2], dtype=torch.float, device=logits.device)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = log_probs.exp()
        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy_per_token * mask).sum() / mask.sum().clamp(min=1.0)
        return entropy.item()

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)

        loss_val = loss.item()
        ppl = math.exp(min(loss_val, 20.0))
        inputs = self._prepare_inputs(inputs)
        entropy = self._compute_entropy(model, inputs)
        self._window_ppl.append(ppl)
        self._window_entropy.append(entropy)
        return loss

    def _inner_training_loop(self, *args, **kwargs):
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_
        trainer_self = self

        def hooked_clip_grad_norm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
            grad_norm = compute_grad_norm_from_model(trainer_self.model, norm_type)
            grad_mean = compute_grad_mean_from_model(trainer_self.model)
            trainer_self._window_grad_norms.append(grad_norm)
            trainer_self._window_grad_means.append(grad_mean)

            clipped = original_clip_grad_norm(parameters, max_norm, norm_type, error_if_nonfinite)
            grad_norm_clipped = min(grad_norm, max_norm) if max_norm and max_norm > 0 else grad_norm
            trainer_self._window_grad_norms_clipped.append(grad_norm_clipped)
            return clipped

        torch.nn.utils.clip_grad_norm_ = hooked_clip_grad_norm
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm

    def log(self, logs: Dict[str, float]) -> None:
        if self._window_grad_norms:
            logs["grad_norm"] = round(self._window_grad_norms[-1], 4)
            logs["grad_mean"] = round(self._window_grad_means[-1], 8)
            self._window_grad_norms = []
            self._window_grad_means = []

        if self._window_grad_norms_clipped:
            logs["grad_norm_clipped"] = round(self._window_grad_norms_clipped[-1], 4)
            self._window_grad_norms_clipped = []

        if self._window_ppl:
            logs["perplexity"] = round(sum(self._window_ppl) / len(self._window_ppl), 4)
            self._window_ppl = []

        if self._window_entropy:
            logs["entropy"] = round(sum(self._window_entropy) / len(self._window_entropy), 6)
            self._window_entropy = []

        maybe_log_loss_smooth(self, logs)

        super().log(logs)


class ZOAdamW:
    """MeZO-style zeroth-order AdamW optimizer."""
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        zo_eps=1e-3,
        zo_samples=1,
        max_grad_norm: Optional[float] = None,
        lr_schedule: Optional[List[Tuple[int, float]]] = None,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.zo_eps = zo_eps
        self.zo_samples = zo_samples
        self.max_grad_norm = max_grad_norm
        self.lr_schedule: List[Tuple[int, float]] = list(lr_schedule) if lr_schedule else []

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0

        self.grad_stats = {
            "grad_norm": 0.0,
            "grad_mean": 0.0,
            "grad_norm_clipped": 0.0,
            "zo_samples_used": zo_samples,
        }

    def set_zo_samples(self, zo_samples: int) -> None:
        self.zo_samples = zo_samples

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def _get_current_lr(self) -> float:
        """Return the current learning rate based on step_count and schedule.
        If no custom lr_schedule is set, returns the fixed self.lr.
        """
        if not self.lr_schedule:
            return self.lr
        current = self.lr_schedule[0][1]
        for step, lr_val in self.lr_schedule:
            if self.step_count >= step:
                current = lr_val
            else:
                break
        return current

    def get_grad_stats(self) -> Dict[str, float]:
        return self.grad_stats.copy()

    def _sample_perturbation(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        return[torch.randn_like(p) for p in self.params]
        # return [torch.empty_like(p).bernoulli_(0.5).mul_(2.0).add_(-1.0) for p in self.params]

    def _perturb_parameters(self, perturbations, scale=1.0):
        for p, z in zip(self.params, perturbations):
            p.data.add_(z, alpha=scale * self.zo_eps)

    def _sync_seed(self) -> int:
        device = self.params[0].device
        base_seed = torch.randint(0, 2**31, (1,), device=device)
        if is_distributed():
            torch.distributed.broadcast(base_seed, src=0)
        return base_seed.item()

    def _all_reduce_loss(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        if is_distributed():
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
        return loss_tensor

    def zo_grad_estimate(self, model, inputs, loss_fn, zo_samples_override: Optional[int] = None):
        current_samples = zo_samples_override if zo_samples_override is not None else self.zo_samples
        base_seed = self._sync_seed()

        was_training = model.training
        model.eval()

        with torch.no_grad():
            loss_original = loss_fn(model, inputs)

        accumulated_grads = None

        for sample_idx in range(current_samples):
            seed = base_seed + sample_idx
            perturbations = self._sample_perturbation(seed)

            self._perturb_parameters(perturbations, scale=1.0)
            with torch.no_grad():
                loss_pos = loss_fn(model, inputs)

            self._perturb_parameters(perturbations, scale=-2.0)
            with torch.no_grad():
                loss_neg = loss_fn(model, inputs)

            self._perturb_parameters(perturbations, scale=1.0)

            loss_pos = self._all_reduce_loss(loss_pos)
            loss_neg = self._all_reduce_loss(loss_neg)

            grad_coef = (loss_pos - loss_neg) / (2.0 * self.zo_eps)
            sample_grads = [grad_coef * z for z in perturbations]

            if accumulated_grads is None:
                accumulated_grads = sample_grads
            else:
                for i, g in enumerate(sample_grads):
                    accumulated_grads[i].add_(g)

        grads = [g / current_samples for g in accumulated_grads]
        self.grad_stats["zo_samples_used"] = current_samples

        if was_training:
            model.train()

        return grads, loss_original

    def step(self, grads: List[torch.Tensor]) -> None:
        self.step_count += 1

        self.grad_stats["grad_norm"] = compute_grad_norm_from_list(grads)
        self.grad_stats["grad_mean"] = compute_grad_mean_from_list(grads)

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            self.grad_stats["grad_norm_clipped"] = clip_grad_norm_inplace(grads, self.max_grad_norm)
        else:
            self.grad_stats["grad_norm_clipped"] = self.grad_stats["grad_norm"]

        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)

            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            denom = v_hat.sqrt().add_(self.eps)
            p.data.addcdiv_(m_hat, denom, value=-self.lr)

    def state_dict(self) -> Dict:
        return {
            "m": [m.cpu().clone() for m in self.m],
            "v": [v.cpu().clone() for v in self.v],
            "step_count": self.step_count,
            "lr": self.lr,
            "zo_samples": self.zo_samples,
            "lr_schedule": self.lr_schedule,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        device = self.params[0].device
        for i, (m_saved, v_saved) in enumerate(zip(state_dict["m"], state_dict["v"])):
            self.m[i].copy_(m_saved.to(device))
            self.v[i].copy_(v_saved.to(device))
        self.step_count = state_dict["step_count"]
        self.lr = state_dict["lr"]
        self.zo_samples = state_dict.get("zo_samples", self.zo_samples)
        self.lr_schedule = state_dict.get("lr_schedule", [])


class MeZOOptimizer(ZOAdamW):
    """Pure MeZO optimizer with direct zeroth-order updates."""
    def __init__(
        self,
        params,
        lr=1e-4,
        weight_decay=0.01,
        zo_eps=1e-3,
        zo_samples=1,
        max_grad_norm: Optional[float] = None,
        lr_schedule: Optional[List[Tuple[int, float]]] = None,
    ):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.zo_eps = zo_eps
        self.zo_samples = zo_samples
        self.max_grad_norm = max_grad_norm
        self.lr_schedule: List[Tuple[int, float]] = list(lr_schedule) if lr_schedule else []

        self.step_count = 0
        self.grad_stats = {
            "grad_norm": 0.0,
            "grad_mean": 0.0,
            "grad_norm_clipped": 0.0,
            "zo_samples_used": zo_samples,
        }

    def step(self, grads: List[torch.Tensor]) -> None:
        self.step_count += 1

        self.grad_stats["grad_norm"] = compute_grad_norm_from_list(grads)
        self.grad_stats["grad_mean"] = compute_grad_mean_from_list(grads)

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            self.grad_stats["grad_norm_clipped"] = clip_grad_norm_inplace(grads, self.max_grad_norm)
        else:
            self.grad_stats["grad_norm_clipped"] = self.grad_stats["grad_norm"]

        for p, g in zip(self.params, grads):
            if self.weight_decay > 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)
            p.data.add_(g, alpha=-self.lr)

    def state_dict(self) -> Dict:
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "zo_samples": self.zo_samples,
            "lr_schedule": self.lr_schedule,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.step_count = state_dict["step_count"]
        self.lr = state_dict["lr"]
        self.zo_samples = state_dict.get("zo_samples", self.zo_samples)
        self.lr_schedule = state_dict.get("lr_schedule", [])


class ZO_Ours(ZOAdamW):
    """Improved zeroth-order optimizer with adaptive eps + momentum-guided subspace."""
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.95, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        zo_eps=1e-3,
        zo_samples=1,
        max_grad_norm: Optional[float] = None,
        u_buffer_k: int = 1,
        subspace_alpha: float = 0.5,
        subspace_alpha_schedule: Optional[List[Tuple[int, float]]] = None,
        normalize_by_sqrt_d: bool = False,
        use_adaptive_eps: bool = True,
        adaptive_eps_min_scale: float = 0.01,
        adaptive_eps_max_scale: float = 10.0,
        adaptive_eps_delta: float = 1e-8,
        lr_schedule: Optional[List[Tuple[int, float]]] = None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            zo_eps=zo_eps,
            zo_samples=zo_samples,
            max_grad_norm=max_grad_norm,
            lr_schedule=lr_schedule,
        )

        self.u_buffer_k = max(1, int(u_buffer_k))
        self.subspace_alpha = float(subspace_alpha)
        self.subspace_alpha_schedule: List[Tuple[int, float]] = list(subspace_alpha_schedule) if subspace_alpha_schedule else []
        self.normalize_by_sqrt_d = bool(normalize_by_sqrt_d)
        self.use_adaptive_eps = bool(use_adaptive_eps)
        self.adaptive_eps_min_scale = float(adaptive_eps_min_scale)
        self.adaptive_eps_max_scale = float(adaptive_eps_max_scale)
        self.adaptive_eps_delta = float(adaptive_eps_delta)

        self.total_d = sum(p.numel() for p in self.params)
        self.sqrt_d = math.sqrt(max(1, self.total_d))

        self.u_buffer: List[List[torch.Tensor]] = []

    def _get_subspace_alpha(self) -> float:
        """Return the current subspace_alpha based on step_count and schedule.
        If no schedule is set, returns the fixed self.subspace_alpha.
        """
        if not self.subspace_alpha_schedule:
            return self.subspace_alpha
        current = self.subspace_alpha_schedule[0][1]
        for step, alpha in self.subspace_alpha_schedule:
            if self.step_count >= step:
                current = alpha
            else:
                break
        return current

    @staticmethod
    def _inner_product(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
        return sum((ai * bi).sum().item() for ai, bi in zip(a, b))

    def _update_u_buffer(self, guided_v: List[torch.Tensor]) -> None:
        norm_sq = sum(v.detach().float().pow(2).sum().item() for v in guided_v)
        norm = math.sqrt(norm_sq) + 1e-8
        normed_v = [v.detach().clone() / norm for v in guided_v]

        self.u_buffer.append(normed_v)
        if len(self.u_buffer) > self.u_buffer_k:
            self.u_buffer.pop(0)

        ortho: List[List[torch.Tensor]] = []
        for vec in self.u_buffer:
            q = [v.clone() for v in vec]
            for base in ortho:
                proj = self._inner_product(q, base)
                for q_i, base_i in zip(q, base):
                    q_i.add_(base_i, alpha=-proj)

            q_norm = math.sqrt(self._inner_product(q, q))
            if q_norm < 1e-6:
                continue
            for q_i in q:
                q_i.div_(q_norm)
            ortho.append(q)

        self.u_buffer = ortho

    def _sample_perturbation_elliptical(self, seed: Optional[int] = None):
        # z_i = self._sample_perturbation(seed)
        z_i = [z / self.sqrt_d for z in self._sample_perturbation(seed)] if self.normalize_by_sqrt_d else self._sample_perturbation(seed)
        if len(self.u_buffer) == 0:
            return z_i

        k = len(self.u_buffer) # 1
        device = self.params[0].device

        coeff = torch.randn(k, device=device) # torch.Size([1])
        # coeff = torch.empty(k, device=device).bernoulli_(0.5).mul_(2.0).add_(-1.0)

        z_u = [torch.zeros_like(p) for p in self.params]

        for c, u_j in zip(coeff, self.u_buffer):
            c_val = c.item()
            for z_u_i, u_ji in zip(z_u, u_j):
                z_u_i.add_(u_ji, alpha=c_val)

        alpha = self._get_subspace_alpha()
        sa = math.sqrt(alpha)
        sb = math.sqrt(max(0.0, 1.0 - alpha))

        return [sa * zi + sb * zu for zi, zu in zip(z_i, z_u)]

    def get_adaptive_eps_scales(self) -> List[torch.Tensor]:
        scales: List[torch.Tensor] = []
        for m, v in zip(self.m, self.v):
            var = v - m**2
            var = torch.clamp(var, min=self.adaptive_eps_delta)
            scale = self.zo_eps / (var.sqrt() + self.adaptive_eps_delta)
            scale = torch.clamp(
                scale,
                min=self.adaptive_eps_min_scale * self.zo_eps,
                max=self.adaptive_eps_max_scale * self.zo_eps,
            )
            scales.append(scale)
        return scales

    def _perturb_parameters_scaled(self, perturbations, scales):
        if isinstance(scales, (float, int)):
            for p, z in zip(self.params, perturbations):
                p.data.add_(z, alpha=float(scales) * self.zo_eps)
        else:
            for p, z, s in zip(self.params, perturbations, scales):
                p.data.add_(z * s)

    def zo_grad_estimate(self, model, inputs, loss_fn, zo_samples_override: Optional[int] = None):
        current_samples = zo_samples_override if zo_samples_override is not None else self.zo_samples
        base_seed = self._sync_seed()

        was_training = model.training
        model.eval()

        with torch.no_grad():
            loss_original = loss_fn(model, inputs)

        accumulated_grads = None
        use_adaptive = self.use_adaptive_eps and self.step_count > 0
        adaptive_scales = self.get_adaptive_eps_scales() if use_adaptive else None

        for sample_idx in range(current_samples):
            seed = base_seed + sample_idx
            perturbation = self._sample_perturbation_elliptical(seed)

            if use_adaptive:
                self._perturb_parameters_scaled(perturbation, adaptive_scales)
                with torch.no_grad():
                    loss_pos = loss_fn(model, inputs)

                neg_scales = [-2.0 * s for s in adaptive_scales]
                self._perturb_parameters_scaled(perturbation, neg_scales)
                with torch.no_grad():
                    loss_neg = loss_fn(model, inputs)

                self._perturb_parameters_scaled(perturbation, adaptive_scales)

                loss_pos = self._all_reduce_loss(loss_pos)
                loss_neg = self._all_reduce_loss(loss_neg)

                grad_coef = (loss_pos - loss_neg) / 2.0
                sample_grads = [grad_coef * z / s for z, s in zip(perturbation, adaptive_scales)]
            else:
                self._perturb_parameters_scaled(perturbation, 1.0)
                with torch.no_grad():
                    loss_pos = loss_fn(model, inputs)

                self._perturb_parameters_scaled(perturbation, -2.0)
                with torch.no_grad():
                    loss_neg = loss_fn(model, inputs)

                self._perturb_parameters_scaled(perturbation, 1.0)

                loss_pos = self._all_reduce_loss(loss_pos)
                loss_neg = self._all_reduce_loss(loss_neg)

                grad_coef = (loss_pos - loss_neg) / (2.0 * self.zo_eps)
                sample_grads = [grad_coef * z for z in perturbation]

            if accumulated_grads is None:
                accumulated_grads = sample_grads
            else:
                for i, g in enumerate(sample_grads):
                    accumulated_grads[i].add_(g)

        grads = [g / current_samples for g in accumulated_grads]

        if self.normalize_by_sqrt_d:
            grads = [g * self.sqrt_d for g in grads]

        self.grad_stats["zo_samples_used"] = current_samples

        if was_training:
            model.train()

        return grads, loss_original

    def step(self, grads: List[torch.Tensor]) -> None:
        super().step(grads)
        self._update_u_buffer(self.m)

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state.update({
            "u_buffer": [[v.cpu().clone() for v in u_j] for u_j in self.u_buffer],
            "normalize_by_sqrt_d": self.normalize_by_sqrt_d,
            "use_adaptive_eps": self.use_adaptive_eps,
            "adaptive_eps_min_scale": self.adaptive_eps_min_scale,
            "adaptive_eps_max_scale": self.adaptive_eps_max_scale,
            "adaptive_eps_delta": self.adaptive_eps_delta,
        })
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        super().load_state_dict(state_dict)
        device = self.params[0].device
        saved_u = state_dict.get("u_buffer", [])
        self.u_buffer = [[v.to(device) for v in u_j] for u_j in saved_u]


class ZOTrainer(Trainer):
    """Trainer for MeZO, ZOAdamW, and ZO_Ours modes."""
    def __init__(
        self,
        optimizer_mode: str,
        sample_schedule: Sequence[Tuple[int, int]],
        zo_config: Dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_mode = optimizer_mode
        self.sample_schedule = list(sample_schedule)
        self.zo_config = dict(zo_config)

        self.zo_optimizer: Optional[ZOAdamW] = None
        self._globalstep_last_logged = 0

        self._accumulated_grads: Optional[List[torch.Tensor]] = None
        self._accumulation_count = 0

        self._recent_losses: List[float] = []
        self._window_grad_norms: List[float] = []
        self._window_grad_means: List[float] = []
        self._window_grad_norms_clipped: List[float] = []
        self._window_zo_samples: List[float] = []
        self._window_ppl: List[float] = []
        self._window_entropy: List[float] = []
        init_loss_smoother(self, self.args)

    def create_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        common_kwargs = dict(
            params=trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            zo_eps=self.zo_config["zo_eps"],
            zo_samples=self.zo_config["zo_samples_init"],
            max_grad_norm=self.zo_config["max_grad_norm"],
        )

        if self.optimizer_mode == "MeZO":
            self.zo_optimizer = MeZOOptimizer(
                **common_kwargs,
                lr_schedule=self.zo_config.get("lr_schedule", []),
            )
        elif self.optimizer_mode == "ZOAdamW":
            self.zo_optimizer = ZOAdamW(
                **common_kwargs,
                betas=(self.zo_config["adam_beta1"], self.zo_config["adam_beta2"]),
                eps=self.zo_config["adam_eps"],
                lr_schedule=self.zo_config.get("lr_schedule", []),
            )
        elif self.optimizer_mode == "ZO_Ours":
            self.zo_optimizer = ZO_Ours(
                **common_kwargs,
                betas=(self.zo_config["adam_beta1"], self.zo_config["adam_beta2"]),
                eps=self.zo_config["adam_eps"],
                u_buffer_k=self.zo_config["u_buffer_k"],
                subspace_alpha=self.zo_config["subspace_alpha"],
                subspace_alpha_schedule=self.zo_config.get("subspace_alpha_schedule", []),
                normalize_by_sqrt_d=self.zo_config["normalize_by_sqrt_d"],
                use_adaptive_eps=self.zo_config["use_adaptive_eps"],
                adaptive_eps_min_scale=self.zo_config["adaptive_eps_min_scale"],
                adaptive_eps_max_scale=self.zo_config["adaptive_eps_max_scale"],
                adaptive_eps_delta=self.zo_config["adaptive_eps_delta"],
                lr_schedule=self.zo_config.get("lr_schedule", []),
            )
        else:
            raise ValueError(f"Unsupported optimizer_mode for ZOTrainer: {self.optimizer_mode}")

        # Dummy optimizer only for HF scheduler plumbing.
        self.optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=self.args.learning_rate)
        return self.optimizer

    def optimizer_step(self, *args, **kwargs):
        # Only call HF scheduler.step() if NOT using custom lr_schedule.
        # When using custom lr_schedule, we manage LR in training_step().
        if self.lr_scheduler is not None and not (hasattr(self.zo_optimizer, 'lr_schedule') and self.zo_optimizer.lr_schedule):
            self.lr_scheduler.step()

    def _loss_fn(self, model, inputs):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    def _compute_entropy(self, model, inputs) -> float:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs.get("labels", None)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != -100).float()
        else:
            shift_logits = logits
            mask = torch.ones(shift_logits.shape[:2], dtype=torch.float, device=logits.device)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = log_probs.exp()
        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy_per_token * mask).sum() / mask.sum().clamp(min=1.0)
        return entropy.item()

    def _get_current_sample_size(self) -> int:
        return get_schedule_value(self.sample_schedule, self.state.global_step)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        current_samples = self._get_current_sample_size()
        grads, loss = self.zo_optimizer.zo_grad_estimate(
            model,
            inputs,
            self._loss_fn,
            zo_samples_override=current_samples,
        )

        loss_val = loss.item()
        ppl = math.exp(min(loss_val, 20.0))
        entropy = self._compute_entropy(model, inputs)
        self._window_ppl.append(ppl)
        self._window_entropy.append(entropy)

        if self._accumulated_grads is None:
            self._accumulated_grads = [g / self.args.gradient_accumulation_steps for g in grads]
        else:
            for i, g in enumerate(grads):
                self._accumulated_grads[i].add_(g, alpha=1.0 / self.args.gradient_accumulation_steps)

        self._accumulation_count += 1

        if self._accumulation_count >= self.args.gradient_accumulation_steps:
            if hasattr(self.zo_optimizer, 'lr_schedule') and self.zo_optimizer.lr_schedule:
                current_lr = self.zo_optimizer._get_current_lr()
                self.zo_optimizer.set_lr(current_lr)
            elif self.lr_scheduler is not None:
                current_lr = self.lr_scheduler.get_last_lr()[0]
                self.zo_optimizer.set_lr(current_lr)
            # else: keep initial lr set during __init__

            self.zo_optimizer.step(self._accumulated_grads)

            stats = self.zo_optimizer.get_grad_stats()
            self._window_grad_norms.append(stats["grad_norm"])
            self._window_grad_means.append(stats["grad_mean"])
            self._window_grad_norms_clipped.append(stats["grad_norm_clipped"])
            self._window_zo_samples.append(stats["zo_samples_used"])

            self._accumulated_grads = None
            self._accumulation_count = 0

        self._recent_losses.append(loss_val)
        return loss.detach()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs = {}

            if self._recent_losses:
                logs["loss"] = round(sum(self._recent_losses) / len(self._recent_losses), 4)
                self._recent_losses = []
            else:
                tr_loss_scalar = tr_loss.item() if isinstance(tr_loss, torch.Tensor) else tr_loss
                denom = max(1, self.state.global_step - self._globalstep_last_logged)
                logs["loss"] = round(tr_loss_scalar / denom, 4)

            maybe_log_loss_smooth(self, logs)

            logs["learning_rate"] = (
                self.zo_optimizer.lr if self.zo_optimizer is not None else self.args.learning_rate
            )
            logs["global_step"] = (
                self.zo_optimizer.step_count if self.zo_optimizer is not None else 0
            )

            if self._window_grad_norms:
                logs["grad_norm"] = round(sum(self._window_grad_norms) / len(self._window_grad_norms), 4)
                logs["grad_mean"] = round(sum(self._window_grad_means) / len(self._window_grad_means), 8)
                logs["grad_norm_clipped"] = round(
                    sum(self._window_grad_norms_clipped) / len(self._window_grad_norms_clipped), 4
                )
                logs["zo_samples"] = round(sum(self._window_zo_samples) / len(self._window_zo_samples), 2)

                self._window_grad_norms = []
                self._window_grad_means = []
                self._window_grad_norms_clipped = []
                self._window_zo_samples = []

            if self._window_ppl:
                logs["perplexity"] = round(sum(self._window_ppl) / len(self._window_ppl), 4)
                self._window_ppl = []

            if self._window_entropy:
                logs["entropy"] = round(sum(self._window_entropy) / len(self._window_entropy), 6)
                self._window_entropy = []

            self._globalstep_last_logged = self.state.global_step
            self.log(logs)


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    if len(prediction.strip()) == 0:
        return 0.0
    scores = [
        scorer.score(ref, prediction)["rougeL"].fmeasure
        for ref in references
        if len(ref.strip()) > 0
    ]
    return max(scores) if scores else 0.0


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def extract_final_loss_metrics(trainer: Trainer, smooth_alpha: float = 0.98) -> Tuple[Optional[float], Optional[float]]:
    loss_values: List[float] = []
    for item in trainer.state.log_history:
        if isinstance(item, dict) and "loss" in item:
            try:
                loss_values.append(float(item["loss"]))
            except (TypeError, ValueError):
                continue

    if not loss_values:
        return None, None

    final_loss = loss_values[-1]
    ema = None
    for loss_val in loss_values:
        ema = loss_val if ema is None else smooth_alpha * ema + (1.0 - smooth_alpha) * loss_val
    return final_loss, ema


def append_run_summary(summary_file: str, summary_record: Dict) -> None:
    summary_dir = os.path.dirname(summary_file)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(summary_record), ensure_ascii=False) + "\n")


def run_generation_evaluation(
    model,
    tokenizer,
    eval_data,
    max_input_length: int,
    max_target_length: int,
    max_samples: int,
):
    model.eval()
    device = next(model.parameters()).device
    predictions, references_list = [], []

    sample_count = min(len(eval_data), max_samples) if max_samples > 0 else len(eval_data)

    for idx in tqdm(range(sample_count), desc="Evaluating", leave=False):
        example = eval_data[idx]
        summary = example["document"]["summary"]["text"]
        question = example["question"]["text"]
        refs = [ans["text"] for ans in example["answers"]]

        prompt = f"Summary: {summary}\n\nQuestion: {question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_target_length,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in generated:
            answer = generated.split("Answer:")[-1].strip()
        else:
            answer = generated[len(prompt) :].strip()

        predictions.append(answer)
        references_list.append(refs)

    rouge_scores = [compute_rouge_l(p, r) for p, r in zip(predictions, references_list)]
    model.train()
    return {"ROUGE-L": 100.0 * np.mean(rouge_scores) if rouge_scores else 0.0}


def call_save_checkpoint(trainer: Trainer, model) -> None:
    """Best-effort adapter for Trainer private _save_checkpoint signature differences."""
    try:
        trainer._save_checkpoint(model, trial=None, metrics=None)
        return
    except TypeError:
        pass

    try:
        trainer._save_checkpoint(model, trial=None)
        return
    except TypeError:
        pass

    trainer._save_checkpoint(model=model, trial=None)


def save_zo_optimizer_state(trainer: Trainer, global_step: int) -> None:
    if not hasattr(trainer, "zo_optimizer"):
        return
    zo_optimizer = getattr(trainer, "zo_optimizer", None)
    if zo_optimizer is None:
        return

    ckpt_dir = os.path.join(trainer.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(zo_optimizer.state_dict(), os.path.join(ckpt_dir, "zo_optimizer.pt"))


def load_zo_optimizer_state(trainer: Trainer, checkpoint_dir: str, is_main: bool) -> None:
    if not hasattr(trainer, "zo_optimizer"):
        return

    zo_optimizer = getattr(trainer, "zo_optimizer", None)
    if zo_optimizer is None:
        trainer.create_optimizer()
        zo_optimizer = getattr(trainer, "zo_optimizer", None)

    if zo_optimizer is None:
        return

    zo_path = os.path.join(checkpoint_dir, "zo_optimizer.pt")
    if not os.path.exists(zo_path):
        if is_main:
            print(f"[Resume] No zo_optimizer.pt under {checkpoint_dir}, skip zeroth-order optimizer restore")
        return

    state = torch.load(zo_path, map_location="cpu")
    zo_optimizer.load_state_dict(state)
    if is_main:
        print(f"[Resume] Loaded zeroth-order optimizer state from: {zo_path}")


class PeriodicCheckpointCallback(TrainerCallback):
    """Save checkpoints by custom step/epoch interval and keep Trainer-compatible state."""
    def __init__(self, train_mode: str, save_steps: int, save_epochs: int, optimizer_mode: str):
        self.train_mode = train_mode
        self.save_steps = save_steps
        self.save_epochs = save_epochs
        self.optimizer_mode = optimizer_mode
        self.trainer: Optional[Trainer] = None

    def attach_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def _save(self, state, model) -> None:
        if self.trainer is None:
            return
        call_save_checkpoint(self.trainer, model)
        if self.optimizer_mode in {"MeZO", "ZOAdamW", "ZO_Ours"}:
            save_zo_optimizer_state(self.trainer, state.global_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.train_mode != "step":
            return
        if not state.is_world_process_zero:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.save_steps != 0:
            return
        self._save(state, model)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.train_mode != "epoch":
            return
        if not state.is_world_process_zero:
            return
        if state.epoch is None:
            return

        current_epoch = int(round(state.epoch))
        if current_epoch <= 0:
            return
        if current_epoch % self.save_epochs != 0:
            return
        self._save(state, model)


class EvalDuringTrainingCallback(TrainerCallback):
    """ROUGE-L generation evaluation every N global steps."""
    def __init__(
        self,
        tokenizer,
        eval_data,
        eval_steps: int,
        max_eval_samples: int,
        max_input_length: int,
        max_target_length: int,
        use_wandb: bool,
    ):
        self.tokenizer = tokenizer
        self.eval_data = eval_data
        self.eval_steps = eval_steps
        self.max_eval_samples = max_eval_samples
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_wandb = use_wandb

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps <= 0:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.eval_steps != 0:
            return

        if state.is_world_process_zero:
            print(f"\n[Eval] Running ROUGE-L at step {state.global_step}...")
            metrics = run_generation_evaluation(
                model=model,
                tokenizer=self.tokenizer,
                eval_data=self.eval_data,
                max_input_length=self.max_input_length,
                max_target_length=self.max_target_length,
                max_samples=self.max_eval_samples,
            )
            print(f"[Eval] Step {state.global_step} ROUGE-L: {metrics['ROUGE-L']:.2f}")
            if self.use_wandb and wandb is not None and wandb.run is not None:
                wandb.log({"eval/ROUGE-L": metrics["ROUGE-L"]}, step=state.global_step)

        dist_barrier()


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    hash_keys = [
        "optimizer_mode",
        "model_path",
        "dataset_name",
        "dataset_config_name",
        "train_split",
        "validation_split",
        "max_input_length",
        "max_target_length",
        "min_answer_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "scheduler_type",
        "warmup_ratio",
        "lr_schedule",
        "train_mode",
        "max_steps",
        "num_train_epochs",
        "save_steps",
        "save_epochs",
        "save_total_limit",
        "max_grad_norm",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "weight_decay",
        "seed",
        "data_seed",
        "zo_eps",
        "zo_samples_init",
        "zo_sample_schedule",
        "u_buffer_k",
        "subspace_alpha",
        "subspace_alpha_schedule",
        "normalize_by_sqrt_d",
        "use_adaptive_eps",
        "adaptive_eps_min_scale",
        "adaptive_eps_max_scale",
        "adaptive_eps_delta",
    ]
    payload = {k: getattr(args, k, None) for k in hash_keys}
    digest = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    return (
        f"{args.optimizer_mode}"
        f"-bs{args.batch_size}x{args.accumulation_steps}"
        f"-lr{args.learning_rate}"
        f"-{args.train_mode}"
        f"-{digest}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FO/MeZO/ZOAdamW/ZO_Ours training for OPT-1.3B on NarrativeQA with LoRA.")

    parser.add_argument(
        "--optimizer_mode",
        type=str,
        default="FO",
        choices=["FO", "MeZO", "ZOAdamW", "ZO_Ours"],
        help="FO=first-order AdamW, MeZO=pure zeroth-order baseline, ZOAdamW=MeZO-style AdamW, ZO_Ours=our MeZO-Adam variant.",
    )

    parser.add_argument("--model_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)

    parser.add_argument("--dataset_name", type=str, default="narrativeqa")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--max_train_samples", type=int, default=-1)

    parser.add_argument("--max_input_length", type=int, default=1280, help="Max input length after tokenization (truncated if exceeded).")
    parser.add_argument("--max_target_length", type=int, default=64, help="Max target length after tokenization (truncated if exceeded).")
    parser.add_argument("--min_answer_length", type=int, default=4, help="Min answer length after tokenization (filtered if shorter).")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=parse_lora_dropout, default=0.0, help="LoRA dropout. Use 'none' to disable (mapped to 0.0).")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,fc1,fc2")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="",
        help=("Custom learning rate schedule (step:lr pairs), e.g. '0:1e-4,300:7e-5,800:4.5e-5'. "
              "If set, custom schedule is used for optimizer LR steps; if empty, use scheduler_type. ")
    )

    parser.add_argument("--train_mode", type=str, default="step", choices=["step", "epoch"])
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)

    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--enable_checkpoint", type=str2bool, default=True)

    parser.add_argument(
        "--max_grad_norm",
        type=parse_max_grad_norm,
        default=1.0,
        help="Set float value to enable clipping; use 'none' or <=0 to disable.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every N steps during training. Set <=0 to disable.")
    parser.add_argument("--eval_max_samples", type=int, default=200, help="Max samples for intermediate evaluation during training. Set <=0 for full evaluation.")
    parser.add_argument("--final_eval_max_samples", type=int, default=-1, help="Max samples for final evaluation after training. Set <=0 for full evaluation.")

    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--project_name", type=str, default="OPT1.3B-LoRA-NarrativeQA-FOZO")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--output_root", type=str, default="./outputs/train_fozo")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--final_model_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=-1)

    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False)

    parser.add_argument("--auto_resume", type=str2bool, default=True, help="Whether to automatically find and resume from the latest checkpoint in output_dir.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume from (overrides auto_resume).")
    parser.add_argument("--save_on_interrupt", type=str2bool, default=True, help="Whether to save a checkpoint if training is interrupted.")
    parser.add_argument(
        "--enable_interrupt_checkpoint_only",
        type=str2bool,
        default=False,
        help="If true, only save checkpoint on interrupt; delete all checkpoints after successful completion.",
    )

    parser.add_argument("--do_final_eval", type=str2bool, default=True)
    parser.add_argument("--save_final_model", type=str2bool, default=True)
    parser.add_argument("--save_run_summary", type=str2bool, default=True)
    parser.add_argument("--summary_file", type=str, default=None)
    parser.add_argument("--loss_smooth_alpha", type=float, default=0.98)

    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--zo_samples_init", type=int, default=4)
    parser.add_argument(
        "--zo_sample_schedule",
        type=str,
        default="0:4",
        help="Sample schedule for MeZO / ZOAdamW / ZO_Ours, e.g. '0:4,2000:8,5000:16'. Empty => constant zo_samples_init.",
    )

    parser.add_argument("--u_buffer_k", type=int, default=1)
    parser.add_argument("--subspace_alpha", type=float, default=0.5)
    parser.add_argument(
        "--subspace_alpha_schedule",
        type=str,
        default="",
        help=("Step schedule for subspace_alpha, e.g. '0:1.0,200:0.98,800:0.95,2000:0.92,4000:0.88,6500:0.85,8500:0.81,9500:0.79'. "
              "Empty string means use fixed --subspace_alpha. "),
    )
    parser.add_argument("--normalize_by_sqrt_d", type=str2bool, default=False)

    parser.add_argument("--use_adaptive_eps", type=str2bool, default=True)
    parser.add_argument("--adaptive_eps_min_scale", type=float, default=0.01)
    parser.add_argument("--adaptive_eps_max_scale", type=float, default=10.0)
    parser.add_argument("--adaptive_eps_delta", type=float, default=1e-8)

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("batch_size must be > 0")
    if args.accumulation_steps <= 0:
        parser.error("accumulation_steps must be > 0")

    if args.train_mode == "step" and args.max_steps <= 0:
        parser.error("max_steps must be > 0 in step mode")
    if args.train_mode == "epoch" and args.num_train_epochs <= 0:
        parser.error("num_train_epochs must be > 0 in epoch mode")

    if args.save_steps <= 0:
        parser.error("save_steps must be > 0")
    if args.save_epochs <= 0:
        parser.error("save_epochs must be > 0")
    if not (0.0 <= args.loss_smooth_alpha < 1.0):
        parser.error("loss_smooth_alpha must be in [0, 1)")

    if args.data_seed < 0:
        args.data_seed = args.seed

    if not (0.0 <= args.subspace_alpha <= 1.0):
        parser.error("subspace_alpha must be in [0, 1]")

    args.subspace_alpha_schedule_parsed = parse_float_schedule(args.subspace_alpha_schedule)
    args.lr_schedule_parsed = parse_lr_schedule(args.lr_schedule)

    args.zo_sample_schedule_parsed = parse_int_schedule(args.zo_sample_schedule, args.zo_samples_init)

    args.run_name = build_run_name(args)
    if args.output_dir is None:
        args.output_dir = os.path.join(args.output_root, args.run_name)

    if args.final_model_dir is None:
        args.final_model_dir = os.path.join(args.output_dir, "lora_model")
    if args.summary_file is None:
        args.summary_file = os.path.join(args.output_root, "all_runs_summary.jsonl")

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_raw_datasets(args: argparse.Namespace):
    if args.dataset_config_name:
        return load_dataset(args.dataset_name, args.dataset_config_name)
    return load_dataset(args.dataset_name)


def prepare_train_features_builder(args: argparse.Namespace, tokenizer):
    max_length = args.max_input_length + args.max_target_length

    def prepare_train_features(examples):
        input_ids_list, attn_list, labels_list = [], [], []

        for doc, question, answers in zip(examples["document"], examples["question"], examples["answers"]):
            summary = doc["summary"]["text"]
            q_text = question["text"]
            answer = answers[0]["text"] if answers else ""

            prompt = f"Summary: {summary}\n\nQuestion: {q_text.strip()}\n\nAnswer:"
            prompt_tok = tokenizer(prompt, max_length=args.max_input_length, truncation=True)
            prompt_len = len(prompt_tok["input_ids"])

            max_answer_len = max_length - prompt_len - 1
            if max_answer_len < args.min_answer_length:
                continue

            answer_text = " " + answer + tokenizer.eos_token
            answer_tok = tokenizer(
                answer_text,
                max_length=max_answer_len,
                truncation=True,
                add_special_tokens=False,
            )

            input_ids = prompt_tok["input_ids"] + answer_tok["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * prompt_len + answer_tok["input_ids"]

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }

    return prepare_train_features


def maybe_init_wandb(args: argparse.Namespace, world_size: int, is_main: bool) -> bool:
    if not args.use_wandb:
        return False

    if wandb is None:
        if is_main:
            print("[Warn] wandb is not installed. Disable wandb logging.")
        return False

    if not is_main:
        return True

    config_payload = dict(vars(args))
    if "zo_sample_schedule_parsed" in config_payload:
        config_payload["zo_sample_schedule_parsed"] = list(config_payload["zo_sample_schedule_parsed"])
    config_payload["world_size"] = world_size

    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=config_payload,
        settings=wandb.Settings(console="wrap"),
    )
    return True


def build_training_args(args: argparse.Namespace, use_wandb: bool, is_main: bool) -> TrainingArguments:
    report_to = ["wandb"] if use_wandb else []

    trainer_max_grad_norm = 0.0
    if args.optimizer_mode == "FO" and args.max_grad_norm is not None:
        trainer_max_grad_norm = args.max_grad_norm

    max_steps = args.max_steps if args.train_mode == "step" else -1

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_eps,
        max_grad_norm=trainer_max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="no",
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.data_seed,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        disable_tqdm=not is_main,
        ignore_data_skip=False,
    )


def _is_valid_checkpoint(ckpt_dir: str) -> bool:
    """Check that a checkpoint directory contains files the Trainer can actually load."""
    if not os.path.isdir(ckpt_dir):
        return False
    valid_markers = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    return any(os.path.isfile(os.path.join(ckpt_dir, m)) for m in valid_markers)


def find_resume_checkpoint(args: argparse.Namespace, is_main: bool) -> Optional[str]:
    if args.resume_from_checkpoint:
        if is_main:
            print(f"[Resume] Using explicit checkpoint: {args.resume_from_checkpoint}")
        return args.resume_from_checkpoint

    if not args.auto_resume:
        return None

    last_ckpt = get_last_checkpoint(args.output_dir)
    if last_ckpt:
        if _is_valid_checkpoint(last_ckpt):
            if is_main:
                print(f"[Resume] Found latest checkpoint: {last_ckpt}")
            return last_ckpt
        else:
            if is_main:
                print(f"[Resume] Skipping incomplete checkpoint: {last_ckpt}")
            return None
    return last_ckpt


def main() -> None:
    # Convert SIGTERM (sent by torchrun on Ctrl+C) into KeyboardInterrupt
    # so that the save_on_interrupt handler works in DDP mode.
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt("SIGTERM received")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    args = parse_args()
    local_rank, world_size, is_main = get_rank_info()
    run_start_time = datetime.now().isoformat(timespec="seconds")

    setup_seed(args.seed)

    if is_main:
        print("=" * 80)
        print("train_fozo.py")
        print(f"Optimizer mode: {args.optimizer_mode}")
        print(f"Train mode: {args.train_mode}")
        print(f"World size: {world_size}")
        print(f"Output dir: {args.output_dir}")
        print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    # config_dump_path = os.path.join(args.output_dir, "run_config.json")
    # if is_main:
    #     with open(config_dump_path, "w", encoding="utf-8") as f:
    #         json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)

    use_wandb = maybe_init_wandb(args, world_size, is_main)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.use_cache = False

    lora_targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    if is_main:
        print("[Info] Trainable params after LoRA:")
        model.print_trainable_parameters()

    raw_datasets = load_raw_datasets(args)
    train_raw = raw_datasets[args.train_split]
    val_raw = raw_datasets[args.validation_split]

    if args.max_train_samples > 0:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))

    prepare_train_features = prepare_train_features_builder(args, tokenizer)
    train_dataset = train_raw.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Preparing train dataset" if is_main else None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )

    training_args = build_training_args(args, use_wandb=use_wandb, is_main=is_main)

    eval_callback = EvalDuringTrainingCallback(
        tokenizer=tokenizer,
        eval_data=val_raw,
        eval_steps=args.eval_steps,
        max_eval_samples=args.eval_max_samples,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        use_wandb=use_wandb,
    )

    callbacks: List[TrainerCallback] = [eval_callback]
    checkpoint_callback: Optional[PeriodicCheckpointCallback] = None
    if args.enable_checkpoint:
        checkpoint_callback = PeriodicCheckpointCallback(
            train_mode=args.train_mode,
            save_steps=args.save_steps,
            save_epochs=args.save_epochs,
            optimizer_mode=args.optimizer_mode,
        )
        callbacks.append(checkpoint_callback)
    elif is_main:
        print("[Info] Checkpoint saving is disabled (--enable_checkpoint false).")

    if args.optimizer_mode == "FO":
        trainer = FOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    else:
        zo_config = {
            "zo_eps": args.zo_eps,
            "zo_samples_init": args.zo_samples_init,
            "max_grad_norm": args.max_grad_norm,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            "u_buffer_k": args.u_buffer_k,
            "subspace_alpha": args.subspace_alpha,
            "subspace_alpha_schedule": args.subspace_alpha_schedule_parsed,
            "normalize_by_sqrt_d": args.normalize_by_sqrt_d,
            "use_adaptive_eps": args.use_adaptive_eps,
            "adaptive_eps_min_scale": args.adaptive_eps_min_scale,
            "adaptive_eps_max_scale": args.adaptive_eps_max_scale,
            "adaptive_eps_delta": args.adaptive_eps_delta,
            "lr_schedule": args.lr_schedule_parsed,
        }

        trainer = ZOTrainer(
            optimizer_mode=args.optimizer_mode,
            sample_schedule=args.zo_sample_schedule_parsed,
            zo_config=zo_config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

    if checkpoint_callback is not None:
        checkpoint_callback.attach_trainer(trainer)

    resume_from_checkpoint = find_resume_checkpoint(args, is_main)

    if args.optimizer_mode in {"MeZO", "ZOAdamW", "ZO_Ours"} and resume_from_checkpoint is not None:
        trainer.create_optimizer()
        load_zo_optimizer_state(trainer, resume_from_checkpoint, is_main=is_main)

    interrupted = False
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        if is_main:
            print("\n[Interrupt] Caught KeyboardInterrupt / SIGTERM.")
        if args.save_on_interrupt and args.enable_checkpoint:
            try:
                if is_main:
                    print("[Interrupt] Saving emergency checkpoint...")
                call_save_checkpoint(trainer, trainer.model)
                if args.optimizer_mode in {"MeZO", "ZOAdamW", "ZO_Ours"}:
                    save_zo_optimizer_state(trainer, trainer.state.global_step)
                if is_main:
                    print(f"[Interrupt] Checkpoint saved at step {trainer.state.global_step}.")
            except Exception as save_err:
                if is_main:
                    print(f"[Interrupt] Failed to save checkpoint: {save_err}")

    final_loss, final_loss_smooth = extract_final_loss_metrics(
        trainer, smooth_alpha=args.loss_smooth_alpha
    )
    final_val_rouge_l = None

    dist_barrier()

    if is_main and args.do_final_eval and not interrupted:
        max_samples = args.final_eval_max_samples
        if max_samples <= 0:
            max_samples = len(val_raw)

        print("\n[Final Eval] Running validation ROUGE-L...")
        val_metrics = run_generation_evaluation(
            model=model,
            tokenizer=tokenizer,
            eval_data=val_raw,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            max_samples=max_samples,
        )
        print(f"[Final Eval] Validation ROUGE-L: {val_metrics['ROUGE-L']:.2f}")
        final_val_rouge_l = float(val_metrics["ROUGE-L"])

        if use_wandb and wandb is not None and wandb.run is not None:
            wandb.log({"final/val_ROUGE-L": val_metrics["ROUGE-L"]})
            wandb.run.summary["final/val_ROUGE-L"] = val_metrics["ROUGE-L"]

    if is_main and args.save_final_model:
        os.makedirs(args.final_model_dir, exist_ok=True)
        model.save_pretrained(args.final_model_dir)
        tokenizer.save_pretrained(args.final_model_dir)
        print(f"[Save] LoRA adapter + tokenizer saved to: {args.final_model_dir}")

    if is_main and args.save_run_summary:
        summary_record = {
            "run_name": args.run_name,
            "optimizer_mode": args.optimizer_mode,
            "train_mode": args.train_mode,
            "output_dir": args.output_dir,
            # "config_path": config_dump_path,
            "resume_from_checkpoint": resume_from_checkpoint,
            "enable_checkpoint": args.enable_checkpoint,
            "interrupted": interrupted,
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
            "metrics": {
                "final/val_ROUGE-L": final_val_rouge_l,
                "final_loss": final_loss,
                "final_loss_smooth": final_loss_smooth,
            },
            "run_start_time": run_start_time,
            "run_end_time": datetime.now().isoformat(timespec="seconds"),
            "params": vars(args),
        }
        append_run_summary(args.summary_file, summary_record)
        print(f"[Summary] Appended run summary to: {args.summary_file}")

    if use_wandb and is_main and wandb is not None and wandb.run is not None:
        wandb.finish()

    # clean up checkpoints if enabled and training completed without interruption
    if is_main and not interrupted and args.enable_interrupt_checkpoint_only:
        import shutil
        checkpoint_dirs = []
        if os.path.exists(args.output_dir):
            for item in os.listdir(args.output_dir):
                if item.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint_path = os.path.join(args.output_dir, item)
                    if os.path.isdir(checkpoint_path):
                        checkpoint_dirs.append(checkpoint_path)
        
        if checkpoint_dirs:
            print(f"[Cleanup] Deleting {len(checkpoint_dirs)} temporary checkpoint(s)...")
            for ckpt_dir in checkpoint_dirs:
                try:
                    shutil.rmtree(ckpt_dir)
                    print(f"  Deleted: {ckpt_dir}")
                except Exception as e:
                    print(f"  Failed to delete {ckpt_dir}: {e}")

    if is_main:
        print("\nTraining run finished.")


if __name__ == "__main__":
    main()
