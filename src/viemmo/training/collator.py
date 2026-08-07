from typing import Any, Dict, List, Optional, Union
import torch
from transformers import PreTrainedTokenizerBase

class DataCollatorForVietnameseCompletionLM:
    """Data collator that masks prompt tokens with -100 so Cross-Entropy loss 
    is computed ONLY on the assistant's completion tokens.
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        tokenizer: PreTrainedTokenizerBase,
        instruction_template: Optional[Union[str, List[int]]] = None,
        ignore_index: int = -100,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.pad_to_multiple_of = pad_to_multiple_of

        # Convert string templates to token IDs if necessary
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = response_template

        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(
                instruction_template, add_special_tokens=False
            )
        else:
            self.instruction_token_ids = instruction_template

    def _find_subsequence(self, sequence: List[int], pattern: List[int]) -> int:
        """Find the start index of pattern in sequence. Returns -1 if not found."""
        pattern_len = len(pattern)
        if pattern_len == 0:
            return -1
        for i in range(len(sequence) - pattern_len + 1):
            if sequence[i : i + pattern_len] == pattern:
                return i
        return -1

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Handle input_ids list from examples
        batch_input_ids = [
            ex["input_ids"] if isinstance(ex["input_ids"], torch.Tensor) else torch.tensor(ex["input_ids"], dtype=torch.long)
            for ex in examples
        ]

        # Calculate max length in batch
        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for input_ids_tensor in batch_input_ids:
            ids_list = input_ids_tensor.tolist()
            seq_len = len(ids_list)

            # Create base labels tensor (copy of input_ids)
            labels_list = list(ids_list)

            # Find the end of the response_template in the input_ids
            response_start = self._find_subsequence(ids_list, self.response_token_ids)

            if response_start != -1:
                # Calculate index where assistant content starts (after response_template)
                response_content_start = response_start + len(self.response_token_ids)
                # Mask out all prompt tokens up to response_content_start
                for i in range(response_content_start):
                    labels_list[i] = self.ignore_index
            else:
                # If template is not found, mask the entire sequence to prevent training on unformatted data
                labels_list = [self.ignore_index] * seq_len

            # Pad sequence
            pad_len = max_len - seq_len
            if pad_len > 0:
                padded_ids = ids_list + [pad_token_id] * pad_len
                padded_mask = [1] * seq_len + [0] * pad_len
                padded_lbls = labels_list + [self.ignore_index] * pad_len
            else:
                padded_ids = ids_list
                padded_mask = [1] * seq_len
                padded_lbls = labels_list

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
            padded_labels.append(padded_lbls)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.torch_call(examples)
