import os
import glob
import re
import random

import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
            print(f"Warning: no matching event_lengths_{k}.csv for {bf}")

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
    def __init__(self, pairs_list, num_batches):
        """
        pairs_list: list of (batch_csv, length_csv)
        num_batches: how many pairs to combine per sample
        """
        super().__init__()
        self.pairs_list = pairs_list
        self.num_batches = num_batches
        # how many samples can we produce
        self._max_index = math.floor(len(self.pairs_list)/self.num_batches)

    def __len__(self):
        return self._max_index

    def __getitem__(self, idx):
        start_i = idx*self.num_batches
        end_i   = start_i + self.num_batches
        chunk = self.pairs_list[start_i:end_i]  # sub-list

        # We'll store data as a list of DataFrames, lengths as list of arrays
        data_dfs   = []
        lengths_arr = []

        for (batch_csv, len_csv) in chunk:
            df_data = pd.read_csv(batch_csv)
            # optional preprocessing if needed
            data_dfs.append(df_data)

            df_len = pd.read_csv(len_csv)
            # e.g. df_len has columns: [event_id, length], or just [length]?
            # We'll assume it has a 'length' column
            # Convert to a simple numpy array
            length_vals = df_len["length"].to_numpy()
            lengths_arr.append(length_vals)

        # 1) Concat the data frames (ignore_index -> we just stack them)
        big_df = pd.concat(data_dfs, ignore_index=True)

        # 2) Concat the lengths arrays
        # We'll produce one big array
        big_lengths = np.concatenate(lengths_arr, axis=0)

        # Convert to Torch Tensors
        data_tensor    = torch.tensor(big_df.values, dtype=torch.float32)
        lengths_tensor = torch.tensor(big_lengths,  dtype=torch.long)

        return data_tensor, lengths_tensor

def create_dataloaders(train_pairs, val_pairs, test_pairs, 
                       num_batches_train=4, num_batches_val=2, num_batches_test=2):
    """
    Each loader returns (data_tensor, lengths_tensor) with batch_size=1.
    """
    train_ds = ConcatBatchDataset(train_pairs, num_batches_train)
    val_ds   = ConcatBatchDataset(val_pairs,   num_batches_val)
    test_ds  = ConcatBatchDataset(test_pairs,  num_batches_test)

    # We set batch_size=1 => each iteration yields exactly one (data, lengths)
    # shuffle train, not shuffle val/test
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def main():
    folder_path = "/projects/0/nisei0750/slava/data/ml_input_data/batched_clustered_data_2_5"
    # Step A: gather + split
    train_pairs, val_pairs, test_pairs = gather_batch_files(folder_path)
    print(f"Train has {len(train_pairs)} pairs, Val {len(val_pairs)}, Test {len(test_pairs)}")

    # Step B: create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs, val_pairs, test_pairs,
        num_batches_train=4,  # combine 4 pairs in each sample for train
        num_batches_val=2,
        num_batches_test=2
    )

    # Step C: usage
    for i, (data, lengths) in enumerate(train_loader):
        # data shape e.g. [1, bigN, M] or [bigN, M] depending on collate
        # lengths shape e.g. [1, some_num] or [some_num]
        # We'll see how default collation wraps them.
        print(f"Train iteration {i}: data={data.shape}, lengths={lengths.shape}")
        # do your training step...
        if i>=1: 
            break

    # Similarly for val, test
    for i, (data, lengths) in enumerate(val_loader):
        print("Val iteration:", i, data.shape, lengths.shape)
        break

    for i, (data, lengths) in enumerate(test_loader):
        print("Test iteration:", i, data.shape, lengths.shape)
        break

if __name__ == "__main__":
    main()