import random
import numpy as np
from functools import lru_cache, partial
from model import TransformerRegressor as TransformerStandard
from flex_attn_model import TransformerRegressor as TransformerFlex

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from triton.testing import do_bench

torch.set_default_device("cuda")
seed = 6
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
#flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
    # generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return torch.tensor(sentence_lengths)

# Generate a batch of semi-realistic data using Zipf distribution for sentence lengths
# in the form of nested tensors with the jagged layout.
def gen_batch(N, E_q, E_k, E_v, device, dtype=torch.float32, query_seq_len_1=False):
    # generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=N)
    print("Sentence lengths:")
    print(type(sentence_lengths))
    print(sentence_lengths)

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
    if query_seq_len_1:
        query = torch.nested.nested_tensor([
            torch.randn(1, E_q, dtype=dtype, device=device)
            for l in sentence_lengths
        ], layout=torch.jagged)
    else:
        query = torch.nested.nested_tensor([
            torch.randn(l.item(), E_q, dtype=dtype, device=device)
            for l in sentence_lengths
        ], layout=torch.jagged)

    key = torch.nested.nested_tensor([
        torch.randn(s.item(), E_k, dtype=dtype, device=device)
        for s in sentence_lengths
    ], layout=torch.jagged)

    value = torch.nested.nested_tensor([
        torch.randn(s.item(), E_v, dtype=dtype, device=device)
        for s in sentence_lengths
    ], layout=torch.jagged)

    return query, key, value, sentence_lengths

import timeit
import math

def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end - begin), torch.cuda.max_memory_allocated()

data_type = torch.float32

# The kernels will utilize block sparisty to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def test_mask(
    score_mod=None,
    mask_mod=None,
    B=1,
    H=1,
    S=5,
    D=16,
    skip_correctness=False,
    print_mask=True,
):
    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide a score_mod or mask_mod"
    query = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float32, requires_grad=True
    )
    gradOut = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=query.device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=query.device)

    causal_fa2 = lambda: F.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    xformers_mask = lambda: F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    flex_attention_call = lambda: flex_attention(
        query, key, value, score_mod=score_mod, block_mask=block_mask
    )

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # Forward pass
    causal_fa2_time = do_bench(causal_fa2)
    xformers_mask_time = do_bench(xformers_mask)
    flex_ms = do_bench(flex_attention_call)

    # Backward pass
    causal_fa2_out = causal_fa2()
    xformers_out = xformers_mask()
    flex_out = flex_attention_call()

    causal_fa2_bw_time = do_bench(
        lambda: causal_fa2_out.backward(gradOut, retain_graph=True)
    )
    xformers_mask_bw_time = do_bench(
        lambda: xformers_out.backward(gradOut, retain_graph=True)
    )
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    # Inline correctness check
    if not skip_correctness:
        xformers_outs = []
        flex_outs = []

        query.grad = None
        key.grad = None
        value.grad = None

        out1 = xformers_mask()
        xformers_outs.append(out1)
        out1.backward(gradOut)
        xformers_outs += [query.grad, key.grad, value.grad]

        query.grad = None
        key.grad = None
        value.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [query.grad, key.grad, value.grad]
        for flex, xformer in zip(flex_outs, xformers_outs):
            print(flex)
            print("---------")
            print(xformer)
            torch.testing.assert_close(flex, xformer, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    # Usage in your results formatting:
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{xformers_mask_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_time, 4):.2f}",
            f"{xformers_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        f"\nResults for {score_mod.__name__ if score_mod is not None else mask_mod.__name__}:"
    )
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
    )
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    # Clean up to save memory
    del query, key, value, gradOut, causal_fa2_out, xformers_out, flex_out
    torch.cuda.empty_cache()

    
def noop(score, b, h, q_idx, kv_idx):
    return score


#test_mask(noop, print_mask=True)

SLIDING_WINDOW = 10


def sliding_window_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    windowed_mask = (
        q_idx - kv_idx <= SLIDING_WINDOW
    )  # We dont need to check the right side of the sliding window since we are applying the causal mask

    return causal_mask & windowed_mask


#test_mask(mask_mod=sliding_window_causal_mask)
batch_sz = 1
N, E_q, E_k, E_v, E_total = 5, 16, 16, 16, 16
E_out = E_q
d_model = E_q
nheads = 1
dropout = 0.0
bias = True
device='cuda'

input_data = torch.randn(1, N, 3, device="cuda", dtype=torch.float32, requires_grad=True)

query = torch.randn(
batch_sz, N, E_q, device="cuda", dtype=torch.float32, requires_grad=True
)
key = torch.randn(
batch_sz, N, E_k, device="cuda", dtype=torch.float32, requires_grad=True
)
value = torch.randn(
batch_sz, N, E_v, device="cuda", dtype=torch.float32, requires_grad=True
)

torch.manual_seed(42)
standard_model = TransformerStandard(
num_encoder_layers = 1,
d_model = d_model,
n_head=nheads,
input_size = 3,
output_size = 2,
dim_feedforward=16,
dropout=0
).to('cuda')

torch.manual_seed(42)
flex_model = TransformerFlex(
num_encoder_layers = 1,
d_model = d_model,
n_head=nheads,
input_size = 3,
output_size = 2,
dim_feedforward=16,
dropout=0
).to('cuda')



padding_mask = torch.zeros(input_data.size(0), input_data.size(1), dtype=torch.bool)
padding_mask[:, -2:] = True  

#output_standard = standard_model(input_data, padding_mask=padding_mask)
#output_flex = flex_model(input_data, seq_lengths=torch.tensor([3]))
#output_standard = standard_model(input_data)
#output_flex = flex_model(input_data)

import flex_attn_model as flex
"""
from nested_mha import NestedMultiHeadAttention

query, key, value, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device)

S = sentence_lengths.max().item()
print(f"Total sequence length in nested query {sentence_lengths.sum().item()}, max sequence length {S}")
padded_query, padded_key, padded_value = (
    t.to_padded_tensor(0.0) for t in (query, key, value)
)

vanilla_mha_layer = nn.MultiheadAttention(E_q, nheads, dropout=dropout, batch_first=True, bias=bias, device='cuda')
mha_layer = NestedMultiHeadAttention(E_q, E_k, E_v, E_total, nheads, dropout=dropout, bias=bias, device='cuda')

# ``nn.MultiheadAttention`` uses a non conventional initialization for layers, so do this for exact parity :(
mha_layer.out_proj.weight = nn.Parameter(vanilla_mha_layer.out_proj.weight.clone().detach())
mha_layer.packed_proj.weight = nn.Parameter(vanilla_mha_layer.in_proj_weight.clone().detach())
mha_layer.out_proj.bias = nn.Parameter(vanilla_mha_layer.out_proj.bias.clone().detach())
mha_layer.packed_proj.bias = nn.Parameter(vanilla_mha_layer.in_proj_bias.clone().detach())

#new_mha_layer = torch.compile(mha_layer)

new_result = mha_layer(query, query, query, is_causal=False)

standard_result = vanilla_mha_layer(padded_query,
                                          padded_query,
                                          padded_query,
                                          attn_mask=None,
                                          key_padding_mask=None,
                                          need_weights=False,
                                          is_causal=False)


from torch.nn.attention.flex_attention import flex_attention

def generate_alibi_bias(H: int):
    ""Returns an alibi bias score_mod given the number of heads H
    Args:
        H: number of heads
    Returns:
        alibi_bias: alibi bias score_mod
    ""
    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (q_idx - kv_idx) * scale
        return score + bias
    return alibi_mod

query, key, value, _ = gen_batch(N, E_q, E_k, E_v, device)
n_heads, D = 8, E_q // 8
alibi_score_mod = generate_alibi_bias(n_heads)
query = (
    query.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
)
key = key.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
value = (
    value.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
)
out_flex2 = flex_attention(query, query, query, score_mod=alibi_score_mod)

"""

print("Test of the transformer encoder layer\n")
query = torch.randn(
batch_sz, N, E_q, device="cuda", dtype=torch.float32, requires_grad=True
)
query1 = query.clone().detach()

torch.manual_seed(42)
vanilla_encoder_layer = nn.TransformerEncoderLayer(d_model, nheads, 16, dropout=0, batch_first=True)
torch.manual_seed(42)
encoder_layer = flex.TransformerEncoderLayer(d_model, nheads, 16, dropout=0, batch_first=True)

# ``nn.MultiheadAttention`` uses a non conventional initialization for layers, so do this for exact parity :(
#encoder_layer.self_attn.in_proj_weight = nn.Parameter(vanilla_encoder_layer.self_attn.in_proj_weight.clone().detach())
#encoder_layer.self_attn.out_proj.weight = nn.Parameter(vanilla_encoder_layer.self_attn.out_proj.weight.clone().detach())
#encoder_layer.self_attn.in_proj_bias = nn.Parameter(vanilla_encoder_layer.self_attn.in_proj_bias.clone().detach())
#encoder_layer.self_attn.out_proj.bias = nn.Parameter(vanilla_encoder_layer.self_attn.out_proj.bias.clone().detach())

torch.manual_seed(42)
standard_result = vanilla_encoder_layer(query1)
torch.manual_seed(42)
new_result = encoder_layer(query)


print(new_result)
print(standard_result)


