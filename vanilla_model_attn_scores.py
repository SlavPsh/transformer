import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from typing import Optional

from torch import Tensor


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,):

        self.attn_scores = None

        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)

    
    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x , attn_scores = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )
        self.attn_scores = attn_scores
        return self.dropout1(x)
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x
    
    def return_attn_scores(self):
        return self.attn_scores

class TransformerRegressor(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, wandb_logger=None):
        super(TransformerRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, output_size)
        self.num_heads = n_head

        # Initialize the wandb logger
        self.wandb_logger = wandb_logger
        if self.wandb_logger is not None:
            if self.wandb_logger.initialized == False:
                self.wandb_logger.initialize()

    def forward(self, input, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)

        layer1 = self.encoder.layers[0]
        attn_scores = layer1.return_attn_scores()
        
        out = self.decoder(memory)
        #if torch.isnan(memory).any(): 
        #    logging.error("Memory contains NaN values. Check attention mask.")
        #out = self.decoder(memory)
        return out, attn_scores
    
    def attach_wandb_logger(self, wandb_logger):
        self.wandb_logger = wandb_logger
    