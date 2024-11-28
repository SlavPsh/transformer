import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from hdbscan import HDBSCAN
import logging
import copy
from typing import Optional, Union, Callable, Tuple
#from torch.nn import TransformerEncoder
from torch.nn.modules.activation import MultiheadAttention
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


class MultiheadFlexAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        batch_first=False,
        score_mod=None,
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

        # Learnable linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Shape checks
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        batch_size, seq_len, embed_dim = query.size()
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, but got {embed_dim}")

        # Linear projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Reshape for multi-head
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply custom flex attention
        attn_output = flex_attention(query, key, value, score_mod=self.score_mod)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final linear projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # If batch_first, convert back
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output

class CustomTransformerEncoder(Module):
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

class CustomTransformerRegressor(Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, use_att_mask=False, wandb_logger=None, use_flash_attention=False):
        super(CustomTransformerRegressor, self).__init__()
        self.input_layer = Linear(input_size, d_model)

        
        encoder_layers = CustomTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True, use_flash_attention=use_flash_attention)
        self.encoder = CustomTransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = Linear(d_model, output_size)
        self.num_heads = n_head
        self.att_mask_used = use_att_mask
        self.flash_attention = use_flash_attention

        # Initialize the wandb logger
        self.wandb_logger = wandb_logger
        if self.wandb_logger is not None:
            if self.wandb_logger.initialized == False:
                self.wandb_logger.initialize()

    """
    def forward(self, input, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        out = self.decoder(memory)
        return out
    """
    
    def attach_wandb_logger(self, wandb_logger):
        self.wandb_logger = wandb_logger
        
    def set_use_att_mask(self, use_att_mask):
        self.att_mask_used = use_att_mask
    
    def forward(self, input_coord, input_for_mask, padding_mask, return_attention_from_layer=None):
        # Here we use only 3 coordinates x,y,z as input to the model
        x = self.input_layer(input_coord)  # Transform coordinates part of the input into d_model space

        expanded_mask = None
        attention_matrix = None

        if self.att_mask_used:
            batch_size, seq_len, _ = x.size()
            num_heads = self.num_heads
            # Calculate the distance mask using the raw input
            distance_mask = self.calculate_distance_mask(input_for_mask, padding_mask)  # input used for mask calculation
                
            # Expand the mask for all heads without duplicating the data

            # Calculate efficiency and purity
            efficiency, purity = calc_efficiency_purity(distance_mask, input_for_mask, padding_mask)


            # Log the mask efficiency and purity

            self.wandb_logger.log({'Mask Efficiency': efficiency, 'Mask Purity': purity})
        
            # Save the mask and input for mask to file for debugging
            #if  self.save_to_file == True:
            #    torch.save(distance_mask, '/data/atlas/users/spshenov/temp/distance_mask.pt')
            #    torch.save(input_coord, '/data/atlas/users/spshenov/temp/input_coord.pt')
            #    self.save_to_file = False
            
            expanded_mask = distance_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len).reshape(batch_size * num_heads, seq_len, seq_len)
        
        if self.flash_attention:
            # Cannot use attn mask and return attention from layer with Flash Attention
            memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        else:
            if return_attention_from_layer is not None:
                self.encoder.layers[return_attention_from_layer].set_return_attention(True)

            memory = self.encoder(src=x, src_key_padding_mask=padding_mask, mask=expanded_mask)

            if return_attention_from_layer is not None:
                attention_matrix = self.encoder.layers[return_attention_from_layer].get_attention_matrix()
                self.encoder.layers[return_attention_from_layer].set_return_attention(False)    
        # Regularization of the output for stability of clustering algorithm
        if torch.isnan(memory).any(): 
            logging.error("Memory contains NaN values. Check attention mask.")
        out = self.decoder(memory)
        return out
    
    
    def calculate_distance_mask(self, input_for_mask, padding_mask, z_0_limit = 197.4, phi_r_ratio_limit = 0.001825, angular_separation_limit = 1.797):
        # Calculate the distance mask based on the input

        points = input_for_mask.detach()  # Shape: [batch_size, seq_len, num_features]


        # Reshape points for broadcasting
        point1 = points.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, num_features]
        point2 = points.unsqueeze(1)  # Shape: [batch_size, 1, seq_len, num_features]

        # Expand the padding mask to both sequence dimensions
        expanded_padding_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)  # Shape: [batch_size, seq_len, seq_len]
        expanded_padding_mask.diagonal(dim1=-2, dim2=-1).fill_(True)  # Set diagonal elements to True in order to keep attention to itself

        # Avoid computing metrics for padding and diagonal elements by setting combined_mask
        non_padding_mask = ~expanded_padding_mask
        large_value = torch.full((1,), 1e6, device=points.device)


        # Extract required coordinates for metric computations 
        z1, z2 = point1[..., 0], point2[..., 0]
        r1, r2 = point1[..., 1], point2[..., 1]
        phi1, phi2 = point1[..., 2], point2[..., 2]
        eta1, eta2 = point1[..., 3], point2[..., 3]

        # Compute r_diff, avoiding division by zero for non-relevant pairs
        r_diff = torch.where(non_padding_mask, r2 - r1, large_value)
        z_0 = torch.where(
            (r_diff != 0) & non_padding_mask,
            torch.abs(z1 - r1 * (z2 - z1) / r_diff),
            large_value
        )

        # Compute phi_diff and apply adjustments for values > π
        phi_diff = torch.where(non_padding_mask, torch.abs(phi2 - phi1), large_value)
        phi_diff = torch.where((phi_diff > np.pi) & non_padding_mask, 2 * np.pi - phi_diff, phi_diff)
        
        # Calculate phi_r_ratio and angular_separation only for non-padding pairs
        phi_r_ratio = torch.where(non_padding_mask, phi_diff / (torch.abs(r_diff) + 1e-8), large_value)
        angular_separation = torch.where(non_padding_mask, torch.sqrt((eta2 - eta1) ** 2 + phi_diff ** 2), large_value)

        # Create the mask based on metric limits
        mask = (non_padding_mask & 
                (z_0 > z_0_limit) & 
                (phi_r_ratio > phi_r_ratio_limit) & 
                (angular_separation > angular_separation_limit))

        """
        # For debugging
        # Calculate the number of True values
        num_true = torch.sum(mask).item()

        # Calculate the total number of values
        total_values = mask.numel()

        # Calculate the ratio of True values to total values
        ratio_true_to_total = num_true / total_values

        print(f"Number of True values: {num_true}")
        print(f"Total number of values: {total_values}")
        print(f"Ratio of True values to total values: {ratio_true_to_total:.2f}")

        """


        return mask  # Shape: [batch_size, seq_len, seq_len]
    
def custom_clustering(pred_params, min_cl_size, min_samples):
    '''
    Function to perform HDBSCAN on the predicted track parameters, with specified
    HDBSCAN hyperparameters. Returns the associated cluster IDs.
    '''
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    cluster_labels = []
    for _, event_prediction in enumerate(pred_params):
        regressed_params = np.array(event_prediction.tolist())
        event_cluster_labels = clustering_algorithm.fit_predict(regressed_params)
        cluster_labels.append(event_cluster_labels)

    cluster_labels = [torch.from_numpy(cl_lbl).int() for cl_lbl in cluster_labels]
    return cluster_labels

def calc_efficiency_purity(distance_mask, input_for_mask, padding_mask):
    # Calculate the efficiency and purity of the distance mask
    points = input_for_mask.detach()  # Shape: [batch_size, seq_len, num_features]
    # Reshape points for broadcasting
    point1 = points.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, num_features]
    point2 = points.unsqueeze(1)  # Shape: [batch_size, 1, seq_len, num_features]
    
    # Make padding 1 dimension extra on the number of points
    padding_mask_expanded = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
    particle_id1 = point1[..., 4]
    particle_id2 = point2[..., 4]

    # Remove padding from the distance mask
    mask_no_padding = ~distance_mask & ~padding_mask_expanded
    mask_true_edges = torch.where((particle_id1 == particle_id2) & (particle_id1 != 0), True, False).to(torch.bool)
    # Remove padding from the true edges mask
    mask_true_edges_no_padding = mask_true_edges & ~padding_mask_expanded

    # Create an upper triangular mask, excluding the diagonal
    batch_size, seq_len, _ = mask_no_padding.shape
    upper_triangular_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    upper_triangular_mask = upper_triangular_mask.to(mask_no_padding.device)
    upper_triangular_mask = upper_triangular_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply the upper triangular mask to count unique pairs
    mask_no_padding = mask_no_padding & upper_triangular_mask
    mask_true_edges_no_padding = mask_true_edges_no_padding & upper_triangular_mask

    # Calculate the overlap between the distance mask and the true edges mask
    overlap = (mask_no_padding & mask_true_edges_no_padding).sum().item()
    attention_count = mask_no_padding.sum().item()
    true_count = mask_true_edges_no_padding.sum().item()

    # Calculate the efficiency and purity, with a small epsilon to avoid division by zero
    epsilon = 1e-8
    efficiency = overlap / (true_count + epsilon)
    purity = overlap / (attention_count + epsilon)

    return efficiency, purity


class CustomTransformerEncoderLayer(Module):
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
        use_flash_attention: bool = False,
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
        self.use_flash_attention = use_flash_attention # Flag to use Flash Attention
        self.attention_matrix = None
        
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
    
    def get_attention_matrix(self):
        return self.attention_matrix
    
    def set_return_attention(self, return_attention: bool):
        self.return_attention = return_attention

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
        need_att_weights = self.return_attention

        if self.use_flash_attention:
            # Utilize flash attention 
            # Note: Flash Attention is not supported on all devices
            # If the device does not support Flash Attention, the code will fall back to the default PyTorch implementation
            # need_weights is set to False for Flash Attention to avoid errors
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = self.self_attn(
                        x,
                        x,
                        x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        need_weights=False, 
                        is_causal=is_causal,
                    )

            except RuntimeError as e:
                logging.error(f"An error occurred when applying Flash attention : {e}")
                output = self.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False, 
                    is_causal=is_causal,
                )
        else:
            output = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_att_weights, 
                average_attn_weights = True,
                is_causal=is_causal,
            )

        self.attention_matrix = output[1]

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