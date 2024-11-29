import torch
import torch.nn as nn

import logging
import copy
from typing import Optional, Union, Callable, Tuple

from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from torch import Tensor
import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)


_compiled_flex_attention = None

def get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention



class MultiheadFlexAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        batch_first=False,
        score_mod=None,
        mask_mod=None,
        **factory_kwargs,
    ):
        super(MultiheadFlexAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.score_mod = score_mod  # Custom scoring modification function
        self.mask_mod = mask_mod  # Custom mask modification function

        # Learnable linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, query_seq_length=None):
        # Shape checks
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)
       
        seq_len, src_len, batch_size, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)
        
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, but got {embed_dim}")

        # Linear projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        assert query.size(-1) % self.num_heads == 0, "query's embed_dim must be divisible by the number of heads"
        assert key.size(-1) % self.num_heads == 0, "key's embed_dim must be divisible by the number of heads"
        assert value.size(-1) % self.num_heads == 0, "value's embed_dim must be divisible by the number of heads"

        query = query.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)  # [batch_size,  num_heads, seq_len, head_dim]
        key = key.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        value = value.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        #num_of_padded_elements = (~key_padding_mask).sum(dim=1) 
        true_seq_length_ex_padding = query_seq_length

        # TODO: Move this mask outside of the forward function
        def sliding_window_and_padding_mask(b, h, q_idx, kv_idx):

            row_mask = q_idx <= true_seq_length_ex_padding[b]
            column_mask = kv_idx <= true_seq_length_ex_padding[b]
            windowed_mask = ((q_idx - kv_idx).abs() <= 50)  

            return row_mask & column_mask & windowed_mask

        def padding_score_mod(score, b, h, q_idx, kv_idx):
            return score

        # Apply custom flex attention
        padding_block_mask = create_block_mask(sliding_window_and_padding_mask, batch_size, None, seq_len, seq_len, device=query.device)
        compiled_flex_attention = get_compiled_flex_attention()
        attn_output = compiled_flex_attention(query, key, value, block_mask=padding_block_mask)

        # Reorder dimensions
        attn_output = attn_output.permute(2, 0, 1, 3)  # Shape: [seq_len, batch_size, num_heads, head_dim]

        # Only make contiguous if necessary
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()

        # Reshape to combine num_heads and head_dim
        attn_output = attn_output.view(seq_len, batch_size, embed_dim)  # Shape: [seq_len, batch_size, embed_dim]

        # Final linear projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(-3, -2)

        return attn_output

class FlexTransformerEncoder(Module):
    """
    TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of TransformerEncoderLayer (required).
        num_layers: number of sub-encoder layers in the encoder (required).
        norm: optional layer normalization at the output (default=None).
    """

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        """
        Passes the input through each encoder layer.

        Args:
            src: input sequence (batch_size, seq_len, d_model).
            mask: optional attention mask (seq_len, seq_len).
            src_key_padding_mask: optional padding mask (batch_size, seq_len).
            is_causal: if True, applies causal masking (default=None).

        Returns:
            Encoded output (batch_size, seq_len, d_model).
        """
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerRegressor(Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, wandb_logger=None):
        super(TransformerRegressor, self).__init__()
        self.input_layer = Linear(input_size, d_model)

        encoder_layers = FlexTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = FlexTransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = Linear(d_model, output_size)
        self.num_heads = n_head
        self.att_mask_used = False

        # Initialize the wandb logger
        self.wandb_logger = wandb_logger
        if self.wandb_logger is not None:
            if self.wandb_logger.initialized == False:
                self.wandb_logger.initialize()

  
    def attach_wandb_logger(self, wandb_logger):
        self.wandb_logger = wandb_logger

    
    def forward(self, input_coord, padding_mask):
        # Here we use only 3 coordinates x,y,z as input to the model
        x = self.input_layer(input_coord)  # Transform coordinates part of the input into d_model space

        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
       
        if torch.isnan(memory).any(): 
            logging.error("Memory contains NaN values. Check attention mask.")
        out = self.decoder(memory)
        return out
    
    
class FlexTransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadFlexAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.return_attention = False
        
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        # Normalization and fast path checks removed for brevity
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

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:


        output = self.self_attn(
            x,
            x,
            x,
            key_padding_mask,
        )

        return self.dropout1(output[0])

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])