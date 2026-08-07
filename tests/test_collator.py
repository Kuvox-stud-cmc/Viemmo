import pytest
import torch
from unittest.mock import MagicMock
from viemmo.training.collator import DataCollatorForVietnameseCompletionLM

class DummyTokenizer:
    """Mock tokenizer for fast unit testing without loading large model weights."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        # Token mapping dictionary
        self.vocab = {
            "<|padding|>": 0,
            "<|user|>": 1,
            "<|endoftext|>": 2,
            "<|assistant|>": 3,
            "\n": 4,
            "Xin": 10,
            "chào": 11,
            "Tôi": 12,
            "là": 13,
            "AI": 14,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<|assistant|>\n":
            return [3, 4]
        elif text == "<|user|>\n":
            return [1, 4]
        return [self.vocab.get(tok, 99) for tok in text.split()]

def test_prompt_tokens_masked_with_minus_100():
    tokenizer = DummyTokenizer()
    collator = DataCollatorForVietnameseCompletionLM(
        response_template="<|assistant|>\n",
        tokenizer=tokenizer,
        ignore_index=-100
    )

    # Prompt: "<|user|>\n Xin chào \n <|assistant|>\n" -> tokens: [1, 4, 10, 11, 4, 3, 4]
    # Response: "Tôi là AI <|endoftext|>" -> tokens: [12, 13, 14, 2]
    # Full input_ids: [1, 4, 10, 11, 4, 3, 4, 12, 13, 14, 2]
    full_input_ids = [1, 4, 10, 11, 4, 3, 4, 12, 13, 14, 2]

    examples = [{"input_ids": full_input_ids}]
    batch = collator(examples)

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # Delimiter '<|assistant|>\n' ends at index 6 (0-indexed: tokens 3, 4 are at indices 5, 6)
    # Indices 0..6 must all be -100
    for i in range(7):
        assert labels[i].item() == -100, f"Token at index {i} should be masked with -100"

    # Response tokens (indices 7..10) must match input_ids
    for i in range(7, 11):
        assert labels[i].item() == input_ids[i].item(), f"Token at index {i} should equal input_id"

def test_padding_and_batching_masked():
    tokenizer = DummyTokenizer()
    collator = DataCollatorForVietnameseCompletionLM(
        response_template="<|assistant|>\n",
        tokenizer=tokenizer,
        ignore_index=-100
    )

    seq_short = [1, 4, 3, 4, 12, 2]  # len 6
    seq_long = [1, 4, 10, 11, 4, 3, 4, 12, 13, 14, 2]  # len 11

    batch = collator([{"input_ids": seq_short}, {"input_ids": seq_long}])

    assert batch["input_ids"].shape == (2, 11)
    assert batch["labels"].shape == (2, 11)
    assert batch["attention_mask"].shape == (2, 11)

    # Check sequence 1 padding (indices 6..10)
    assert torch.all(batch["attention_mask"][0, 6:] == 0)
    assert torch.all(batch["labels"][0, 6:] == -100)

def test_missing_template_fallback():
    tokenizer = DummyTokenizer()
    collator = DataCollatorForVietnameseCompletionLM(
        response_template="<|assistant|>\n",
        tokenizer=tokenizer,
        ignore_index=-100
    )

    # Sequence missing response template
    bad_seq = [1, 4, 10, 11, 2]
    batch = collator([{"input_ids": bad_seq}])

    # All labels should be masked with -100
    assert torch.all(batch["labels"] == -100)

def test_cross_entropy_loss_calculation():
    tokenizer = DummyTokenizer()
    collator = DataCollatorForVietnameseCompletionLM(
        response_template="<|assistant|>\n",
        tokenizer=tokenizer,
        ignore_index=-100
    )

    full_input_ids = [1, 4, 3, 4, 12, 13, 2] # len 7: [1, 4, 3, 4] is prompt, [12, 13, 2] is response
    batch = collator([{"input_ids": full_input_ids}])

    labels = batch["labels"]
    
    # Create fake logits: shape (batch_size, seq_len, vocab_size)
    vocab_size = 50
    logits = torch.randn(1, 7, vocab_size)

    # PyTorch CrossEntropyLoss with ignore_index=-100
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

    # Loss must be a valid non-zero scalar computed strictly over unmasked tokens
    assert not torch.isnan(loss)
    assert loss.item() > 0
