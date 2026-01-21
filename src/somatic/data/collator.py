"""Batch collation for antibody sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..tokenizer import tokenizer


class AntibodyCollator:
    """
    Collates antibody sequences into padded batches.

    Format: [CLS] heavy light [EOS]
    Chain IDs: 0 for CLS/heavy, 1 for light/EOS
    Coordinates: (optional) CA atom 3D coordinates for each position
    """

    def __init__(self, max_length: int = 320, pad_to_max: bool = False) -> None:
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def _encode_pair(
        self,
        heavy: str,
        light: str,
        heavy_cdr: list[int] | None,
        light_cdr: list[int] | None,
        heavy_nt: list[int] | None,
        light_nt: list[int] | None,
    ) -> dict[str, list[int]]:
        heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
        light_ids = tokenizer.encode(light, add_special_tokens=False)

        token_ids = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
        chain_ids = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)
        special_mask = [1] + [0] * len(heavy_ids) + [0] * len(light_ids) + [1]

        if heavy_cdr is not None and light_cdr is not None:
            cdr_mask = [0] + heavy_cdr + light_cdr + [0]
        else:
            cdr_mask = None

        if heavy_nt is not None and light_nt is not None:
            nt_mask = [0] + heavy_nt + light_nt + [0]
        else:
            nt_mask = None

        return {
            "token_ids": token_ids,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
            "nt_mask": nt_mask,
            "special_mask": special_mask,
        }

    def _pad_sequence(self, seq: list[int], target_len: int, pad_value: int) -> list[int]:
        if len(seq) >= target_len:
            return seq[:target_len]
        return seq + [pad_value] * (target_len - len(seq))

    def _concatenate_coords(
        self,
        heavy_coords: np.ndarray | None,
        light_coords: np.ndarray | None,
        heavy_len: int,
        light_len: int,
    ) -> np.ndarray | None:
        """Concatenate heavy and light chain coordinates.

        Adds placeholder coordinates for CLS and EOS tokens.

        Args:
            heavy_coords: Heavy chain coordinates (N_heavy, 3) or None.
            light_coords: Light chain coordinates (N_light, 3) or None.
            heavy_len: Length of heavy chain sequence.
            light_len: Length of light chain sequence.

        Returns:
            Combined coordinates (1 + N_heavy + N_light + 1, 3) or None.
        """
        if heavy_coords is None or light_coords is None:
            return None

        # Validate coordinate lengths match sequence lengths
        if len(heavy_coords) != heavy_len or len(light_coords) != light_len:
            return None

        # Create placeholder coordinates for special tokens (zeros)
        cls_coord = np.zeros((1, 3), dtype=np.float32)
        eos_coord = np.zeros((1, 3), dtype=np.float32)

        # Concatenate: [CLS] + heavy + light + [EOS]
        combined = np.concatenate([cls_coord, heavy_coords, light_coords, eos_coord], axis=0)
        return combined

    def _pad_coords(
        self,
        coords: np.ndarray,
        target_len: int,
    ) -> np.ndarray:
        """Pad or truncate coordinate array to target length.

        Args:
            coords: Coordinate array (seq_len, 3).
            target_len: Target sequence length.

        Returns:
            Padded/truncated coordinates (target_len, 3).
        """
        current_len = len(coords)

        if current_len >= target_len:
            return coords[:target_len]

        # Pad with zeros
        padding = np.zeros((target_len - current_len, 3), dtype=np.float32)
        return np.concatenate([coords, padding], axis=0)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        encoded = []
        coords_list = []

        for example in batch:
            enc = self._encode_pair(
                heavy=example["heavy_chain"],
                light=example["light_chain"],
                heavy_cdr=example.get("heavy_cdr_mask"),
                light_cdr=example.get("light_cdr_mask"),
                heavy_nt=example.get("heavy_non_templated_mask"),
                light_nt=example.get("light_non_templated_mask"),
            )
            encoded.append(enc)

            # Handle coordinates if present
            heavy_coords = example.get("heavy_coords")
            light_coords = example.get("light_coords")
            heavy_len = len(example["heavy_chain"])
            light_len = len(example["light_chain"])

            combined_coords = self._concatenate_coords(
                heavy_coords, light_coords, heavy_len, light_len
            )
            coords_list.append(combined_coords)

        lengths = [len(e["token_ids"]) for e in encoded]
        pad_len = self.max_length if self.pad_to_max else min(max(lengths), self.max_length)

        token_ids, chain_ids, attention_mask, special_masks = [], [], [], []
        cdr_masks, nt_masks = [], []
        padded_coords = []

        has_cdr = encoded[0]["cdr_mask"] is not None
        has_nt = encoded[0]["nt_mask"] is not None
        has_coords = coords_list[0] is not None

        for i, enc in enumerate(encoded):
            seq_len = min(len(enc["token_ids"]), pad_len)

            token_ids.append(self._pad_sequence(enc["token_ids"], pad_len, tokenizer.pad_token_id))
            chain_ids.append(self._pad_sequence(enc["chain_ids"], pad_len, 0))
            attention_mask.append([1] * seq_len + [0] * (pad_len - seq_len))
            special_masks.append(self._pad_sequence(enc["special_mask"], pad_len, 1))

            if has_cdr:
                cdr_masks.append(self._pad_sequence(enc["cdr_mask"], pad_len, 0))
            if has_nt:
                nt_masks.append(self._pad_sequence(enc["nt_mask"], pad_len, 0))
            if has_coords and coords_list[i] is not None:
                padded_coords.append(self._pad_coords(coords_list[i], pad_len))

        result = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "chain_ids": torch.tensor(chain_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_masks, dtype=torch.bool),
            "cdr_mask": torch.tensor(cdr_masks, dtype=torch.long) if has_cdr else None,
            "non_templated_mask": (torch.tensor(nt_masks, dtype=torch.long) if has_nt else None),
            "coords": (
                torch.tensor(np.stack(padded_coords), dtype=torch.float32)
                if has_coords and padded_coords
                else None
            ),
        }

        return result
