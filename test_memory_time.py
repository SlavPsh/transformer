import torch
import torch.nn as nn
import logging
from vanilla_model import TransformerRegressor as VanillaTransformerRegressor
from custom_model import TransformerRegressor as FlexTransformerRegressor
from custom_model import generate_padding_mask
import time
import csv


def generate_random_padded_input(batch_size, max_hits, feature_dim):
    lengths = torch.randint(low=int(0.8 * max_hits), high=max_hits + 1, size=(batch_size,))
    inputs = torch.zeros(batch_size, max_hits, feature_dim)
    padding_mask = torch.ones(batch_size, max_hits, dtype=torch.bool)

    for i, length in enumerate(lengths):
        inputs[i, :length] = torch.randn(length, feature_dim)
        padding_mask[i, :length] = False

    return inputs, lengths, padding_mask


def test_model_memory_and_time(model, model_name, device, n_hits_list):
    results = []

    for max_hits in n_hits_list:
        torch.cuda.empty_cache()
        mem0 = torch.cuda.memory_allocated(device)

        input_cpu, lengths_cpu, padding_mask_cpu = generate_random_padded_input(1, max_hits, 3)

        mem1 = torch.cuda.memory_allocated(device)

        torch.cuda.synchronize()
        start_transfer_time = time.perf_counter()

        input_gpu = input_cpu.to(device)
        lengths_gpu = lengths_cpu.to(device)

        if model_name == "FlexTransformer":
            padding_mask_gpu = generate_padding_mask(lengths_gpu)
        else:
            padding_mask_gpu = padding_mask_cpu.to(device)

        torch.cuda.synchronize()
        end_transfer_time = time.perf_counter()
        transfer_time = end_transfer_time - start_transfer_time

        mem2 = torch.cuda.memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_forward_time = time.perf_counter()

        if model_name == "FlexTransformer":
            _ = model(input_gpu, "const_string",padding_mask_gpu)
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                _ = model(input_gpu, padding_mask_gpu)

        

        torch.cuda.synchronize()
        end_forward_time = time.perf_counter()
        forward_time = end_forward_time - start_forward_time

        mem3 = torch.cuda.memory_allocated(device)
        peak_mem_forward = torch.cuda.max_memory_allocated(device)

        results.append((
            model_name,
            max_hits,
            mem1 - mem0,
            mem2 - mem0,
            mem3 - mem0,
            transfer_time,
            forward_time,
            peak_mem_forward
        ))

        print(
            f"{model_name} max_hits={max_hits}: "
            f"mem_before_transfer={mem1 - mem0}, "
            f"mem_after_transfer={mem2 - mem0}, "
            f"mem_after_forward={mem3 - mem0}, "
            f"transfer_time={transfer_time:.6f}s, "
            f"forward_time={forward_time:.6f}s, "
            f"peak_mem_forward={peak_mem_forward} bytes"
        )

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vanilla_model = VanillaTransformerRegressor(
        num_encoder_layers=6,
        d_model=64,
        n_head=4,
        input_size=3,
        output_size=5,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)

    flex_model = FlexTransformerRegressor(
        num_encoder_layers=6,
        d_model=64,
        n_head=4,
        input_size=3,
        output_size=5,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)

    n_hits_list = [10, 20, 30, 100, 300,  1000,  3000, 5000, 7000, 10000, 12000]

    vanilla_results = test_model_memory_and_time(vanilla_model, "VanillaTransformer", device, n_hits_list)
    flex_results = test_model_memory_and_time(flex_model, "FlexTransformer", device, n_hits_list)

    all_results = vanilla_results + flex_results

    with open("memory_time_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "max_hits",
            "mem_before_transfer",
            "mem_after_transfer",
            "mem_after_forward",
            "transfer_time_s",
            "forward_time_s",
            "peak_mem_forward"
        ])
        writer.writerows(all_results)

    print("Memory usage and timing data saved to memory_time_results.csv.")


if __name__ == "__main__":
    main()
