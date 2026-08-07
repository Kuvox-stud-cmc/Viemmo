import time
from typing import Any, Dict, List, Optional
import torch
from transformers import TrainerCallback, EarlyStoppingCallback


class VRAMAndGradientCallback(TrainerCallback):
    """Callback to monitor peak VRAM usage, step duration, and non-finite gradients."""

    def __init__(self):
        super().__init__()
        self.step_start_time = None
        self.metrics_history: List[Dict[str, Any]] = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step_duration = time.perf_counter() - self.step_start_time if self.step_start_time else 0.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )

        has_non_finite_grad = False
        if model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_non_finite_grad = True
                        break

        log_entry = {
            "step": state.global_step,
            "step_duration_sec": round(step_duration, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "non_finite_grad_detected": has_non_finite_grad,
        }

        if state.log_history and "loss" in state.log_history[-1]:
            log_entry["loss"] = state.log_history[-1]["loss"]
        if state.log_history and "eval_loss" in state.log_history[-1]:
            log_entry["eval_loss"] = state.log_history[-1]["eval_loss"]

        self.metrics_history.append(log_entry)


def get_early_stopping_callback(patience: int = 3, threshold: float = 0.0) -> EarlyStoppingCallback:
    """Creates an EarlyStoppingCallback that halts training if validation loss fails to decrease."""
    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )
