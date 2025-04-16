import torch
import torch.nn as nn
from typing import Optional
#from functools import lru_cache

from torch import Tensor
import logging
import torch.nn.functional as F

#from torch.nn.modules.activation import MultiheadAttention
from custom_mha import CustomMultiHeadAttention
from torch.nn.attention.flex_attention import (
    create_block_mask,
    create_mask,
)


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,):

        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = CustomMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )   

             
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        flex_mask = None,
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

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        elif any(
            len(getattr(m, "_forward_hooks", {}))
            + len(getattr(m, "_forward_pre_hooks", {}))
            for m in self.modules()
        ):
            why_not_sparsity_fast_path = "forward pre-/hooks are attached to the module"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                (x.device.type in _supported_device_type) for x in tensor_args
            ):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
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
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, flex_mask=flex_mask)
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
        flex_mask = None
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
            flex_mask=flex_mask
        )[0]
        return self.dropout1(x)
    
class CustomTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm, enable_nested_tensor=enable_nested_tensor)
        
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        flex_mask = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask_for_layers,
                flex_mask=flex_mask
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerRegressor(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout):
        super(TransformerRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)

        encoder_layers = CustomTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = CustomTransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, output_size)
        self.num_heads = n_head
        self.mask_cache = {}


    def forward(self, input, batch_name, flex_padding_mask, timer=None):
        # TODO: exclude input and output layer compute for padding tokens
        if timer:
            timer.start('input_layer')
        x = self.input_layer(input)
        if timer:
            torch.cuda.synchronize()
            timer.stop()
        B , S = input.size(0), input.size(1)
        #S = input.size(0)

        # Do caching HERE 
        if timer:
            timer.start('block_mask')

        #if batch_name not in self.mask_cache:
        #    self.mask_cache[batch_name] = create_block_mask_cached(flex_padding_mask, B, None, S, S, device=input.device)
        mask = create_block_mask_cached(flex_padding_mask, B, None, S, S, device=input.device)

        if timer:
            torch.cuda.synchronize()
            timer.stop()
        
        if timer:
            timer.start('encoder')
        #memory = self.encoder(src=x, flex_mask=self.mask_cache[batch_name])
        memory = self.encoder(src=x, flex_mask=mask)
        if timer:
            torch.cuda.synchronize()
            timer.stop()
        if timer:
            timer.start('decoder')
        out = self.decoder(memory)
        if timer:
            torch.cuda.synchronize()
            timer.stop()
        #if torch.isnan(memory).any(): 
        #    logging.error("Memory contains NaN values. Check attention mask.")
        #out = self.decoder(memory)
        return out

#@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=True)
    return block_mask

def generate_padding_mask(lengths):
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked

    """
    def padding_mask(b, h, q_idx, kv_idx):
        #rows_mask = q_idx < lengths[b]
        cols_mask = kv_idx < lengths[b]

        return cols_mask

    return padding_mask

def generate_sliding_window_padding_mask(lengths, SLIDING_WINDOW=1024):
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked

    """
    def padding_mask(b, h, q_idx, kv_idx):
        length = lengths[b]
        # Can we pad query here as well?
        padding_mask = (kv_idx < length)
        half_L = length // 2
        d = (kv_idx - q_idx) % length
        d = torch.where(d > half_L, length - d, d)

        #eta_mask1 = (index1[q_idx] == index1[kv_idx])
        #eta_mask2 = (index2[q_idx] == index2[kv_idx])

        return (d <= SLIDING_WINDOW) & padding_mask
    return padding_mask

def generate_cluster_padding_mask(lengths, cluster_id: Tensor):

    def doc_mask_mod(b, h, q_idx, kv_idx):
        # Can we pad query here as well?
        padding_mask = (kv_idx < lengths[b])
        same_doc = (cluster_id[b, q_idx] == cluster_id[b, kv_idx]) 
        #q_logical = q_idx - offsets[document_id[q_idx]]
        #kv_logical = kv_idx - offsets[document_id[kv_idx]]
        #inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return padding_mask & same_doc

    return doc_mask_mod


def generate_only_cluster_mask(cluster_id: Tensor):

    def doc_mask_mod(b, h, q_idx, kv_idx):
        # Can we pad query here as well?
        same_doc = (cluster_id[q_idx] == cluster_id[kv_idx]) 
        #q_logical = q_idx - offsets[document_id[q_idx]]
        #kv_logical = kv_idx - offsets[document_id[kv_idx]]
        #inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc

    return doc_mask_mod

def generate_doc_event_cluster_padding_mask( length_tensor, cluster_tensor: Tensor):

    def doc_mask_mod(b, h, q_idx, kv_idx):
        padding_mask = (kv_idx < length_tensor[b])
        #same_event = (event_tensor[q_idx] == event_tensor[kv_idx])
        same_cluster = (cluster_tensor[b][q_idx] == cluster_tensor[b][kv_idx]) 

        return same_cluster & padding_mask

    return doc_mask_mod
    
    