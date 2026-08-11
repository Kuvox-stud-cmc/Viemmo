import pytest
from unittest.mock import MagicMock
from transformers import EarlyStoppingCallback
from viemmo.training.callbacks import VRAMAndGradientCallback, get_early_stopping_callback

def test_vram_and_gradient_callback_eval_loss_logging():
    cb = VRAMAndGradientCallback()
    args = MagicMock()
    state = MagicMock()
    state.global_step = 50
    state.log_history = [{"loss": 1.25, "eval_loss": 1.10}]

    cb.on_step_begin(args, state, MagicMock())
    cb.on_step_end(args, state, MagicMock(), model=None)

    assert len(cb.metrics_history) == 1
    entry = cb.metrics_history[0]
    assert entry["step"] == 50
    assert entry["loss"] == 1.25
    assert entry["eval_loss"] == 1.10

def test_get_early_stopping_callback():
    cb = get_early_stopping_callback(patience=5, threshold=0.01)
    assert isinstance(cb, EarlyStoppingCallback)
    assert cb.early_stopping_patience == 5
    assert cb.early_stopping_threshold == 0.01
