"""EncoderInterface — abstract base class for encoders in icefall."""

import torch
from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor


class EncoderInterface(ABC, torch.nn.Module):
    """Interface for encoders used in icefall models.

    The encoder takes in features x and feature lengths x_lens, and returns
    encoded output and output lengths.
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x: (seq_len, batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
          x_lens: (batch_size,)
        Returns:
          (output, output_lens)
        """
        ...
