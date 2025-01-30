import os
import glob
import re
import random
import logging
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def pad_tensor(tensor, pad_length, pad_value=-1):
    """Pad a tensor to the specified length with the given pad value."""
    if tensor.dim() == 1:
        padding = pad_length - tensor.size(0)
        if padding > 0:
            return torch.nn.functional.pad(tensor, (0, padding), value=pad_value)
    elif tensor.dim() == 2:
        padding = pad_length - tensor.size(0)
        if padding > 0:
            return torch.nn.functional.pad(tensor, (0, 0, 0, padding), value=pad_value)
    return tensor



def load_and_process_batch(batch_csv, len_csv):

    if isinstance(batch_csv, list) and isinstance(len_csv, list):
        # Read and concatenate dataframes
        batch_dfs = [pd.read_csv(csv) for csv in batch_csv]
        len_dfs = [pd.read_csv(csv) for csv in len_csv]
        df_data = pd.concat(batch_dfs, ignore_index=True)
        df_len = pd.concat(len_dfs, ignore_index=True)
    else:
        # Read single dataframes
        df_data = pd.read_csv(batch_csv)
        df_len = pd.read_csv(len_csv)

    # Determine the maximum event length
    lengths = df_len['length'].to_numpy()
    max_length_raw = lengths.max()
    # Make it multiples of 128 for flex_attention block masking to work
    max_length = ((max_length_raw + 128 - 1) // 128) * 128

    # Initialize lists to store padded tensors
    input_data_tensors = []
    output_data_tensors = []
    cluster_tensors = []

    # Group by event_id and pad each group
    grouped = df_data.groupby('event_id')
    for event_id, event_data in grouped:
        input_data_tensor = torch.tensor(event_data[['x', 'y', 'z']].values, dtype=torch.float32)
        output_data_tensor = torch.tensor(event_data[['theta', 'sin_phi', 'cos_phi', 'q', 'log_p']].values, dtype=torch.float32)
        cluster_tensor = torch.tensor(event_data['cluster_id'].values)
        

        # Pad the tensors
        input_data_tensor = pad_tensor(input_data_tensor, max_length)
        cluster_tensor = pad_tensor(cluster_tensor, max_length)
        
        # Append to lists
        input_data_tensors.append(input_data_tensor)
        output_data_tensors.append(output_data_tensor)
        cluster_tensors.append(cluster_tensor)
    
    # Stack the tensors to form the batch
    input_data_tensor = torch.stack(input_data_tensors)
    output_data_tensor = torch.cat(output_data_tensors, dim=0)
    cluster_tensor = torch.stack(cluster_tensors)
    original_lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    
    return input_data_tensor, output_data_tensor, cluster_tensor, original_lengths_tensor


def gather_batch_files(folder_path, pattern_batch="batch_*.csv", pattern_lengths="event_lengths_*.csv",
                       train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    1) Finds all 'batch_*.csv' in folder_path.
    2) For each one, looks for the matching 'event_lengths_{k}.csv'.
    3) Pairs them up as (batch_csv, lengths_csv).
    4) Shuffles them, splits into train/val/test sets by ratio.
    5) Returns (train_pairs, val_pairs, test_pairs), each a list of (batch_csv, lengths_csv).
    """
    assert abs(train_ratio+val_ratio+test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"

    random.seed(37) # to have the same output from shuffle

    pattern_b = os.path.join(folder_path, pattern_batch)
    batch_files = glob.glob(pattern_b)

    # We'll parse out the integer from 'batch_{k}.csv'
    regex_batch = re.compile(r"batch_(\d+)\.csv$")

    # Build dict: key=k, value=full path to batch
    batch_dict = {}
    for bf in batch_files:
        base = os.path.basename(bf)
        m = regex_batch.match(base)
        if m:
            k_str = m.group(1)
            k = int(k_str)
            batch_dict[k] = bf

    # now find lengths files
    pattern_l = os.path.join(folder_path, pattern_lengths)
    length_files = glob.glob(pattern_l)

    # Build dict: key=k, value=full path to event_lengths
    regex_len = re.compile(r"event_lengths_(\d+)\.csv$")
    lengths_dict = {}
    for lf in length_files:
        base = os.path.basename(lf)
        m2 = regex_len.match(base)
        if m2:
            k_str = m2.group(1)
            k = int(k_str)
            lengths_dict[k] = lf

    # We'll build list of (k, batch_file, lengths_file) for all k that appear in both
    pairs = []
    for k, bf in batch_dict.items():
        if k in lengths_dict:
            lf = lengths_dict[k]
            pairs.append((bf, lf))
        else:
            logging.warning(f"Warning: no matching event_lengths_{k}.csv for {bf}")

    if not pairs:
        raise FileNotFoundError("No matching pairs of (batch_k, event_lengths_k) found.")

    # Shuffle
    random.shuffle(pairs)

    # Split
    total = len(pairs)
    train_end = int(total*train_ratio)
    val_end   = train_end + int(total*val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs   = pairs[train_end:val_end]
    test_pairs  = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


class ConcatBatchDataset(Dataset):
    """
    Each sample = combination of 'num_batches' pairs of CSVs:
      - batch_{k}.csv   -> data rows
      - event_lengths_{k}.csv -> length of each event
    We concatenate them all.
    """
    def __init__(self, pairs_list, use_double_batches = False):
        """
        pairs_list: list of (batch_csv, length_csv)
        """
        super().__init__()
        self.pairs_list = pairs_list
        self.use_double_batches = use_double_batches

    def __len__(self):
        if self.use_double_batches:
            return len(self.pairs_list) // 2
        else:
            return len(self.pairs_list)

    def __getitem__(self, idx):
        
        if self.use_double_batches:
            batch_csv, len_csv = self.pairs_list[2*idx]
            batch_csv2, len_csv2 = self.pairs_list[2*idx + 1]
            batch_csv = [batch_csv, batch_csv2]
            len_csv = [len_csv, len_csv2]
        else:
            batch_csv, len_csv = self.pairs_list[idx]
            
        input_data_tensor, output_data_tensor, cluster_tensor, lengths_tensor = load_and_process_batch(batch_csv, len_csv)
        return input_data_tensor, output_data_tensor, cluster_tensor, lengths_tensor


def create_dataloaders(folder_path, use_double_batches = False):
    """
    Each loader returns (data_tensor, lengths_tensor) with batch_size=1.
    """
    logging.info(f"Loading data from {folder_path}")
    train_pairs, val_pairs, test_pairs = gather_batch_files(folder_path)
    logging.info(f"Train has {len(train_pairs)} pairs, Val {len(val_pairs)}, Test {len(test_pairs)}")

    train_ds = ConcatBatchDataset(train_pairs, use_double_batches)
    val_ds = ConcatBatchDataset(val_pairs, use_double_batches)
    test_ds = ConcatBatchDataset(test_pairs, use_double_batches)

    return train_ds, val_ds, test_ds

def main():
    folder_path = "/projects/0/nisei0750/slava/data/ml_input_data/batched_clustered_data_1_752"
    # Step A: gather + split
    # Step B: create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(folder_path)

    # Step C: usage
    for i, (data, event_tensor, cluster_tensor) in enumerate(train_loader):
        # data shape e.g. [1, bigN, M] or [bigN, M] depending on collate
        # lengths shape e.g. [1, some_num] or [some_num]
        # We'll see how default collation wraps them.
        print(f"Train iteration {i}: data={data.shape}, lengths={event_tensor.shape}")
        # do your training step...
        if i>=1: 
            break

    # Similarly for val, test
    for i, (data, event_tensor, cluster_tensor) in enumerate(val_loader):
        print("Val iteration:", i, data.shape, event_tensor.shape)
        break

    for i, (data, event_tensor, cluster_tensor) in enumerate(test_loader):
        print("Test iteration:", i, data.shape, event_tensor.shape)
        break

if __name__ == "__main__":
    main()