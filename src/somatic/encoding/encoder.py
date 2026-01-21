"""Encoding API for extracting embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..data.collator import AntibodyCollator
from ..model import SomaticModel
from ..tokenizer import tokenizer
from .pooling import MeanMaxPooling, PoolingStrategy, create_pooling


class SomaticEncoder:
    """High-level API for encoding antibody sequences.

    Parameters
    ----------
    model
        Trained SomaticModel instance.
    device
        Device to run inference on.
    pooling
        Pooling strategy. Can be a string ("mean", "cls", "max", "mean_max")
        or a PoolingStrategy instance. If None, returns full sequence embeddings.

    Examples
    --------
    >>> encoder = SomaticEncoder.from_pretrained("model.pt", pooling="mean")
    >>> embedding = encoder.encode("EVQLV...", "DIQMT...")
    >>> embeddings = encoder.encode_batch(heavy_list, light_list)
    """

    def __init__(
        self,
        model: SomaticModel,
        device: str | torch.device = "cpu",
        pooling: str | PoolingStrategy | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)

        if pooling is None:
            self.pooling = None
        elif isinstance(pooling, str):
            self.pooling = create_pooling(pooling)
        else:
            self.pooling = pooling

        self.collator = AntibodyCollator(
            max_length=model.config.max_seq_len, pad_to_max=False
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "cpu",
        pooling: str | None = None,
    ) -> "SomaticEncoder":
        """Load an encoder from a pretrained checkpoint.

        Parameters
        ----------
        model_path
            Path to the model checkpoint.
        device
            Device to load the model on.
        pooling
            Pooling strategy to use.

        Returns
        -------
        SomaticEncoder
            Encoder instance.
        """
        model = SomaticModel.from_pretrained(str(model_path), map_location=device)
        return cls(model, device=device, pooling=pooling)

    def _prepare_input(self, heavy_chain: str, light_chain: str) -> dict[str, Tensor]:
        """Prepare a single sequence pair for encoding."""
        example = {
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "heavy_cdr_mask": None,
            "light_cdr_mask": None,
            "heavy_non_templated_mask": None,
            "light_non_templated_mask": None,
        }
        batch = self.collator([example])
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def _prepare_batch(
        self, heavy_chains: list[str], light_chains: list[str]
    ) -> dict[str, Tensor]:
        """Prepare a batch of sequence pairs for encoding."""
        examples = [
            {
                "heavy_chain": h,
                "light_chain": l,
                "heavy_cdr_mask": None,
                "light_cdr_mask": None,
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            }
            for h, l in zip(heavy_chains, light_chains)
        ]
        batch = self.collator(examples)
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def encode(
        self,
        heavy_chain: str,
        light_chain: str,
        return_numpy: bool = False,
    ) -> Tensor | np.ndarray:
        """Encode a single antibody sequence pair.

        Parameters
        ----------
        heavy_chain
            Heavy chain amino acid sequence.
        light_chain
            Light chain amino acid sequence.
        return_numpy
            If True, return numpy array instead of tensor.

        Returns
        -------
        Tensor or np.ndarray
            If pooling is set, returns shape (hidden_dim,) or (hidden_dim*2,) for mean_max.
            If pooling is None, returns shape (seq_len, hidden_dim).
        """
        batch = self._prepare_input(heavy_chain, light_chain)

        outputs = self.model(
            token_ids=batch["token_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        hidden_states = outputs["hidden_states"]

        if self.pooling is not None:
            embeddings = self.pooling(hidden_states, batch["attention_mask"])
            embeddings = embeddings.squeeze(0)
        else:
            seq_len = int(batch["attention_mask"].sum().item())
            embeddings = hidden_states[0, :seq_len, :]

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    @torch.no_grad()
    def encode_batch(
        self,
        heavy_chains: list[str],
        light_chains: list[str],
        return_numpy: bool = False,
        batch_size: int = 32,
    ) -> Tensor | np.ndarray | list:
        """Encode a batch of antibody sequence pairs.

        Parameters
        ----------
        heavy_chains
            List of heavy chain sequences.
        light_chains
            List of light chain sequences.
        return_numpy
            If True, return numpy arrays instead of tensors.
        batch_size
            Batch size for processing.

        Returns
        -------
        Tensor, np.ndarray, or list
            If pooling is set, returns stacked embeddings of shape (n, hidden_dim).
            If pooling is None, returns a list of variable-length embeddings.
        """
        if len(heavy_chains) != len(light_chains):
            raise ValueError(
                f"Number of heavy chains ({len(heavy_chains)}) must match "
                f"number of light chains ({len(light_chains)})"
            )

        all_embeddings = []

        for i in range(0, len(heavy_chains), batch_size):
            batch_heavy = heavy_chains[i : i + batch_size]
            batch_light = light_chains[i : i + batch_size]

            batch = self._prepare_batch(batch_heavy, batch_light)

            outputs = self.model(
                token_ids=batch["token_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            hidden_states = outputs["hidden_states"]

            if self.pooling is not None:
                embeddings = self.pooling(hidden_states, batch["attention_mask"])
                all_embeddings.append(embeddings)
            else:
                for j in range(hidden_states.shape[0]):
                    seq_len = int(batch["attention_mask"][j].sum().item())
                    emb = hidden_states[j, :seq_len, :]
                    if return_numpy:
                        emb = emb.cpu().numpy()
                    all_embeddings.append(emb)

        if self.pooling is not None:
            result = torch.cat(all_embeddings, dim=0)
            if return_numpy:
                return result.cpu().numpy()
            return result
        else:
            return all_embeddings

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension.

        Returns
        -------
        int
            The dimension of the output embeddings.
        """
        dim = self.model.config.d_model
        if isinstance(self.pooling, MeanMaxPooling):
            return dim * 2
        return dim

    @torch.no_grad()
    def get_logits(
        self,
        heavy_chain: str,
        light_chain: str,
    ) -> dict[str, Tensor]:
        """Get raw logits for all positions.

        Useful for likelihood scoring, perplexity computation, etc.

        Parameters
        ----------
        heavy_chain
            Heavy chain amino acid sequence.
        light_chain
            Light chain amino acid sequence.

        Returns
        -------
        dict
            Dictionary containing:
            - 'logits': Full logits tensor (seq_len, vocab_size)
            - 'heavy_logits': Heavy chain logits (heavy_len, vocab_size)
            - 'light_logits': Light chain logits (light_len, vocab_size)
        """
        batch = self._prepare_input(heavy_chain, light_chain)

        outputs = self.model(
            token_ids=batch["token_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs["logits"][0]  # Remove batch dimension
        chain_ids = batch["chain_ids"][0]
        attention_mask = batch["attention_mask"][0]

        # Find chain boundaries (excluding CLS at start and EOS at end)
        seq_len = int(attention_mask.sum().item())
        logits = logits[:seq_len]
        chain_ids = chain_ids[:seq_len]

        # Heavy chain: positions where chain_id == 0, excluding CLS (position 0)
        heavy_mask = (chain_ids == 0)
        heavy_mask[0] = False  # Exclude CLS
        heavy_logits = logits[heavy_mask]

        # Light chain: positions where chain_id == 1, excluding EOS (last position)
        light_mask = (chain_ids == 1)
        light_mask[seq_len - 1] = False  # Exclude EOS
        light_logits = logits[light_mask]

        return {
            "logits": logits,
            "heavy_logits": heavy_logits,
            "light_logits": light_logits,
        }

    @torch.no_grad()
    def log_likelihood(
        self,
        heavy_chain: str,
        light_chain: str,
    ) -> dict[str, Tensor]:
        """Compute log-likelihood for antibody sequences.

        Computes the sum of log probabilities for each token given the
        model's predictions. Useful for scoring sequence quality or
        comparing sequence variants.

        Parameters
        ----------
        heavy_chain
            Heavy chain amino acid sequence.
        light_chain
            Light chain amino acid sequence.

        Returns
        -------
        dict
            Dictionary containing:
            - 'log_likelihood': Total log-likelihood (scalar tensor)
            - 'heavy_log_likelihood': Heavy chain log-likelihood (scalar tensor)
            - 'light_log_likelihood': Light chain log-likelihood (scalar tensor)
        """
        batch = self._prepare_input(heavy_chain, light_chain)

        outputs = self.model(
            token_ids=batch["token_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs["logits"][0]  # Remove batch dimension
        token_ids = batch["token_ids"][0]
        chain_ids = batch["chain_ids"][0]
        attention_mask = batch["attention_mask"][0]

        # Find chain boundaries (excluding CLS at start and EOS at end)
        seq_len = int(attention_mask.sum().item())
        logits = logits[:seq_len]
        token_ids = token_ids[:seq_len]
        chain_ids = chain_ids[:seq_len]

        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Heavy chain: positions where chain_id == 0, excluding CLS (position 0)
        heavy_mask = chain_ids == 0
        heavy_mask[0] = False  # Exclude CLS
        heavy_log_probs = log_probs[heavy_mask]
        heavy_targets = token_ids[heavy_mask]
        heavy_ll = (
            heavy_log_probs.gather(dim=-1, index=heavy_targets.unsqueeze(-1))
            .squeeze(-1)
            .sum()
        )

        # Light chain: positions where chain_id == 1, excluding EOS (last position)
        light_mask = chain_ids == 1
        light_mask[seq_len - 1] = False  # Exclude EOS
        light_log_probs = log_probs[light_mask]
        light_targets = token_ids[light_mask]
        light_ll = (
            light_log_probs.gather(dim=-1, index=light_targets.unsqueeze(-1))
            .squeeze(-1)
            .sum()
        )

        total_ll = heavy_ll + light_ll

        return {
            "log_likelihood": total_ll,
            "heavy_log_likelihood": heavy_ll,
            "light_log_likelihood": light_ll,
        }

    @torch.no_grad()
    def perplexity(
        self,
        heavy_chain: str,
        light_chain: str,
    ) -> dict[str, Tensor]:
        """Compute perplexity for antibody sequences.

        Perplexity is computed as exp(-log_likelihood / num_tokens).
        Lower perplexity indicates the model assigns higher probability
        to the sequence.

        Parameters
        ----------
        heavy_chain
            Heavy chain amino acid sequence.
        light_chain
            Light chain amino acid sequence.

        Returns
        -------
        dict
            Dictionary containing:
            - 'perplexity': Combined perplexity (scalar tensor)
            - 'heavy_perplexity': Heavy chain perplexity (scalar tensor)
            - 'light_perplexity': Light chain perplexity (scalar tensor)
        """
        ll_result = self.log_likelihood(heavy_chain, light_chain)

        heavy_len = len(heavy_chain)
        light_len = len(light_chain)
        total_len = heavy_len + light_len

        heavy_ppl = torch.exp(-ll_result["heavy_log_likelihood"] / heavy_len)
        light_ppl = torch.exp(-ll_result["light_log_likelihood"] / light_len)
        total_ppl = torch.exp(-ll_result["log_likelihood"] / total_len)

        return {
            "perplexity": total_ppl,
            "heavy_perplexity": heavy_ppl,
            "light_perplexity": light_ppl,
        }

    @torch.no_grad()
    def predict(
        self,
        heavy_chain: str,
        light_chain: str,
        return_probs: bool = False,
    ) -> dict[str, str | Tensor]:
        """Predict tokens at masked positions.

        Takes sequences that may contain <mask> tokens and predicts
        the most likely amino acid at each masked position.

        Parameters
        ----------
        heavy_chain
            Heavy chain sequence. May contain <mask> tokens.
        light_chain
            Light chain sequence. May contain <mask> tokens.
        return_probs
            If True, also return probabilities for the predictions.

        Returns
        -------
        dict
            Dictionary containing:
            - 'heavy_chain': Predicted heavy chain sequence (str)
            - 'light_chain': Predicted light chain sequence (str)
            - 'heavy_probs': (if return_probs) Prediction probabilities for heavy chain
            - 'light_probs': (if return_probs) Prediction probabilities for light chain

        Examples
        --------
        >>> encoder = SomaticEncoder.from_pretrained("model.pt")
        >>> result = encoder.predict(
        ...     heavy_chain="EVQLV<mask><mask>SGGG",
        ...     light_chain="DIQMT"
        ... )
        >>> result['heavy_chain']
        'EVQLVQSGGG'
        """
        batch = self._prepare_input(heavy_chain, light_chain)

        outputs = self.model(
            token_ids=batch["token_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs["logits"][0]  # (seq_len, vocab_size)
        token_ids = batch["token_ids"][0].clone()
        chain_ids = batch["chain_ids"][0]
        attention_mask = batch["attention_mask"][0]

        seq_len = int(attention_mask.sum().item())

        # Find masked positions and replace with predictions
        mask_positions = token_ids[:seq_len] == tokenizer.mask_token_id
        if mask_positions.any():
            predictions = logits[:seq_len].argmax(dim=-1)
            token_ids[:seq_len] = torch.where(
                mask_positions, predictions, token_ids[:seq_len]
            )

        # Split into heavy and light chains
        # Heavy: chain_id == 0, excluding CLS (position 0)
        # Light: chain_id == 1, excluding EOS (last position)
        heavy_mask = (chain_ids[:seq_len] == 0)
        heavy_mask[0] = False  # Exclude CLS

        light_mask = (chain_ids[:seq_len] == 1)
        light_mask[seq_len - 1] = False  # Exclude EOS

        heavy_ids = token_ids[:seq_len][heavy_mask].tolist()
        light_ids = token_ids[:seq_len][light_mask].tolist()

        # Decode to strings (join tokens without spaces)
        heavy_seq = "".join(tokenizer.convert_ids_to_tokens(heavy_ids))
        light_seq = "".join(tokenizer.convert_ids_to_tokens(light_ids))

        result: dict[str, str | Tensor] = {
            "heavy_chain": heavy_seq,
            "light_chain": light_seq,
        }

        if return_probs:
            probs = torch.softmax(logits[:seq_len], dim=-1)
            pred_probs = probs.max(dim=-1).values

            result["heavy_probs"] = pred_probs[heavy_mask]
            result["light_probs"] = pred_probs[light_mask]

        return result
