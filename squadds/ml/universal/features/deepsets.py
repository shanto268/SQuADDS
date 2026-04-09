"""Permutation-invariant DeepSets encoder for design parameter dictionaries.

Maps an arbitrary-length ``{param_name: value}`` dictionary to a fixed
ℝ^32 vector, ensuring that the output is independent of dictionary
insertion order.

Architecture::

    For each (key, value) pair:
        x_i = Embedding(key_index) * value      # (embed_dim,)
        h_i = SharedMLP(x_i)                    # (hidden_dim,)

    agg = MaskedSum(h_i for all i)              # (hidden_dim,)
    out = ρ_MLP(agg)                            # (out_dim,)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant encoder for design parameter dictionaries.

    Args:
        vocab_size: Number of distinct parameter keys in the vocabulary.
        embed_dim: Embedding dimension for each parameter key.
        hidden_dim: Hidden dimension of the shared φ-MLP.
        out_dim: Output embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Key embedding
        self.key_embedding = nn.Embedding(vocab_size, embed_dim)

        # Shared φ-MLP: processes each (key_emb * value) independently
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ρ-MLP: processes the aggregated representation
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        key_indices: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            key_indices: Integer tensor of parameter key indices,
                shape ``(B, max_params)``.
            values: Float tensor of parameter values,
                shape ``(B, max_params)``.
            mask: Boolean tensor, ``True`` for valid entries,
                shape ``(B, max_params)``.  If ``None``, all entries
                are assumed valid.

        Returns:
            Embedding tensor of shape ``(B, out_dim)``.
        """
        # (B, max_params, embed_dim)
        key_embs = self.key_embedding(key_indices)

        # Scale by parameter value: (B, max_params, embed_dim)
        scaled = key_embs * values.unsqueeze(-1)

        # Apply shared MLP: (B, max_params, hidden_dim)
        h = self.phi(scaled)

        # Masked sum aggregation
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()

        agg = h.sum(dim=1)  # (B, hidden_dim)

        return self.rho(agg)  # (B, out_dim)

    @staticmethod
    def build_vocab(param_names: list[str]) -> dict[str, int]:
        """Build a parameter name → index vocabulary.

        Args:
            param_names: List of all possible parameter names across
                all component types.

        Returns:
            Dictionary mapping parameter name to integer index.
        """
        return {name: idx for idx, name in enumerate(sorted(set(param_names)))}

    @staticmethod
    def encode_params(
        params: dict[str, float],
        vocab: dict[str, int],
        max_params: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a design parameter dict to tensors.

        Args:
            params: ``{param_name: value}`` dictionary.
            vocab: Parameter name → index mapping.
            max_params: Pad/truncate to this length.  If ``None``,
                uses ``len(vocab)``.

        Returns:
            Tuple of ``(key_indices, values, mask)`` tensors, each of
            shape ``(max_params,)``.
        """
        if max_params is None:
            max_params = len(vocab)

        keys = torch.zeros(max_params, dtype=torch.long)
        vals = torch.zeros(max_params, dtype=torch.float32)
        mask = torch.zeros(max_params, dtype=torch.bool)

        for i, (name, value) in enumerate(params.items()):
            if i >= max_params:
                break
            if name in vocab:
                keys[i] = vocab[name]
                vals[i] = float(value)
                mask[i] = True

        return keys, vals, mask
