import pytest
import torch
from unittest.mock import MagicMock
from viemmo.training.trainer import VRAMAndGradientCallback

def test_vram_and_gradient_callback_initialization():
    callback = VRAMAndGradientCallback()
    assert callback.metrics_history == []
    assert callback.step_start_time is None

def test_vram_and_gradient_callback_step_end():
    callback = VRAMAndGradientCallback()
    
    # Mock state and args
    args = MagicMock()
    state = MagicMock()
    state.global_step = 1
    state.log_history = [{"loss": 0.4215}]

    # Mock PyTorch model with finite gradients
    model = MagicMock()
    param1 = torch.tensor([1.0, 2.0], requires_grad=True)
    param1.grad = torch.tensor([0.1, 0.2])
    model.named_parameters.return_value = [("layer.weight", param1)]

    callback.on_step_begin(args, state, MagicMock())
    callback.on_step_end(args, state, MagicMock(), model=model)

    assert len(callback.metrics_history) == 1
    entry = callback.metrics_history[0]
    assert entry["step"] == 1
    assert entry["loss"] == 0.4215
    assert entry["non_finite_grad_detected"] is False
    assert "peak_vram_mb" in entry
    assert "step_duration_sec" in entry

def test_vram_and_gradient_callback_detects_nan_gradient():
    callback = VRAMAndGradientCallback()
    
    args = MagicMock()
    state = MagicMock()
    state.global_step = 2
    state.log_history = []

    # Mock PyTorch model with NaN gradient
    model = MagicMock()
    param_nan = torch.tensor([1.0, 2.0], requires_grad=True)
    param_nan.grad = torch.tensor([float("nan"), 0.2])
    model.named_parameters.return_value = [("layer.weight", param_nan)]

    callback.on_step_begin(args, state, MagicMock())
    callback.on_step_end(args, state, MagicMock(), model=model)

    assert len(callback.metrics_history) == 1
    assert callback.metrics_history[0]["non_finite_grad_detected"] is True
