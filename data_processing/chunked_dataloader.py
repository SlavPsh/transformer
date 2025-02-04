from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import torch
import numpy as np

def read_chunks_preserve_events(
    file_path,
    chunksize=200_000,
    max_length=300_000,
    event_id_col="event_id"
):
    """
    A generator that yields DataFrames, each no bigger than max_length rows,
    ensuring no partial events across chunk boundaries.
    
    See the previous snippet for the full implementation details.
    """
    df_iter = pd.read_csv(file_path, chunksize=chunksize)
    leftover_df = pd.DataFrame()

    for df_chunk in df_iter:
        # 1) Merge leftover
        if not leftover_df.empty:
            df_chunk = pd.concat([leftover_df, df_chunk], ignore_index=True)
            leftover_df = pd.DataFrame()

        if len(df_chunk) == 0:
            continue

        # 2) Identify partial event at the end
        last_event_id = df_chunk[event_id_col].iloc[-1]
        idx = len(df_chunk) - 1
        while idx >= 0 and df_chunk[event_id_col].iloc[idx] == last_event_id:
            idx -= 1
        incomplete_start = idx + 1

        leftover_partial = df_chunk.iloc[incomplete_start:].copy()
        df_chunk_complete = df_chunk.iloc[:incomplete_start]

        leftover_df = pd.concat([leftover_df, leftover_partial], ignore_index=True)

        # 3) Ensure chunk does not exceed max_length
        df_chunk_complete = trim_to_max_length(df_chunk_complete, leftover_df, max_length, event_id_col)

        if len(df_chunk_complete) > 0:
            yield df_chunk_complete

    # After iteration, leftover might remain
    while len(leftover_df) > 0:
        # Attempt final chunk
        df_chunk_complete = trim_to_max_length(leftover_df, pd.DataFrame(), max_length, event_id_col)
        leftover_df = leftover_df.iloc[len(df_chunk_complete):]  # remove what's used
        if len(df_chunk_complete) > 0:
            yield df_chunk_complete
        else:
            # If we can't trim enough, break or handle
            break

def trim_to_max_length(df_chunk, leftover_df, max_length, event_id_col):
    """
    Repeatedly remove the last event if the chunk is too big.
    Adds removed events to leftover_df at the front, so they
    can be used in next iteration. (Or store them differently.)
    """
    while len(df_chunk) > max_length:
        # find last event
        last_eid = df_chunk[event_id_col].iloc[-1]
        idx = len(df_chunk) - 1
        while idx >= 0 and df_chunk[event_id_col].iloc[idx] == last_eid:
            idx -= 1
        last_event_start = idx + 1
        # remove last event from chunk
        last_evt_df = df_chunk.iloc[last_event_start:].copy()
        df_chunk = df_chunk.iloc[:last_event_start]
        # prepend to leftover
        leftover_df = pd.concat([last_evt_df, leftover_df], ignore_index=True)

    return df_chunk

def make_tensors_from_chunk(df_chunk, pad_val=-1, multiple_of=128):
    """
    Given a DataFrame chunk with columns:
      - x_norm, y_norm, z_norm
      - log_p_norm, q, sin_phi, cos_phi, theta_norm
      - event_id, cluster_id
    produce 5 Tensors:
      1) shape (padded_len, 3) for [x_norm, y_norm, z_norm]
      2) shape (padded_len, 5) for [log_p_norm, q, sin_phi, cos_phi, theta_norm]
      3) a single scalar for original length
      4) shape (padded_len,) for event_id
      5) shape (padded_len,) for cluster_id

    The DataFrame is padded so that padded_len is multiple_of. The pad value is -1 by default.
    """
    # original length
    orig_len = len(df_chunk)

    # round up to multiple_of
    padded_len = ((orig_len + multiple_of - 1) // multiple_of) * multiple_of

    # Prepare columns
    # If your CSV has these columns named exactly:
    x = df_chunk["x_norm"].values.astype(np.float32)
    y = df_chunk["y_norm"].values.astype(np.float32)
    z = df_chunk["z_norm"].values.astype(np.float32)

    log_p = df_chunk["log_p_norm"].values.astype(np.float32)
    q = df_chunk["q"].values.astype(np.float32)
    sin_phi = df_chunk["sin_phi"].values.astype(np.float32)
    cos_phi = df_chunk["cos_phi"].values.astype(np.float32)
    theta = df_chunk["theta_norm"].values.astype(np.float32)

    event_ids = df_chunk["event_id"].values.astype(np.int64)   
    num_unique_events = len(np.unique(event_ids))

    
    cluster_ids = df_chunk["cluster_id"].values.astype(np.int64)

    # We'll build NumPy arrays for each set, then pad them
    # 1) x_norm, y_norm, z_norm => shape (orig_len, 3)
    xyz = np.column_stack([x, y, z])  # shape (orig_len, 3)
    # 2) log_p_norm, q, sin_phi, cos_phi, theta_norm => shape (orig_len,5)
    other5 = np.column_stack([theta,  sin_phi, cos_phi, q, log_p]) # (orig_len, 5)

    # function to pad 2D
    def pad_2d(arr, final_len, pad_val):
        if len(arr) == 0:
            # edge case: empty
            return np.full((final_len, arr.shape[1]), pad_val, dtype=arr.dtype)
        padded_arr = np.full((final_len, arr.shape[1]), pad_val, dtype=arr.dtype)
        padded_arr[:len(arr)] = arr
        return padded_arr

    # function to pad 1D
    def pad_1d(arr, final_len, pad_val):
        if len(arr) == 0:
            return np.full((final_len,), pad_val, dtype=arr.dtype)
        out = np.full((final_len,), pad_val, dtype=arr.dtype)
        out[:len(arr)] = arr
        return out

    xyz_padded = pad_2d(xyz, padded_len, pad_val)
    other5_padded = pad_2d(other5, padded_len, pad_val)
    event_ids_padded = pad_1d(event_ids, padded_len, pad_val)
    

    # convert to Torch Tensors
    xyz_tensor = torch.from_numpy(xyz_padded)
    other5_tensor = torch.from_numpy(other5_padded)
    length_tensor = torch.tensor([orig_len], dtype=torch.int64)
    event_id_tensor = torch.from_numpy(event_ids_padded)


    cluster_ids_padded = pad_1d(cluster_ids, padded_len, pad_val)
    cluster_id_tensor = torch.from_numpy(cluster_ids_padded)

    return num_unique_events, xyz_tensor, other5_tensor, length_tensor, event_id_tensor, cluster_id_tensor

class ChunkedIterableDataset(IterableDataset):
    """
    An IterableDataset that yields data from large CSV in chunk-based manner,
    ensuring no partial event overlap, and each chunk <= max_length.
    
    Each chunk is a DataFrame.
      - yield entire chunk as one item
    """

    def __init__(self, file_path, chunksize=200_000, max_length=300_000):
        super().__init__()
        self.file_path = file_path
        self.chunksize = chunksize
        self.max_length = max_length

    def __iter__(self):
        # We'll create the generator for chunk reading
        chunk_gen = read_chunks_preserve_events(
            file_path=self.file_path,
            chunksize=self.chunksize,
            max_length=self.max_length,
            event_id_col="event_id"
        )

        # Now we can yield each chunk in form of tensors
        for chunk_df in chunk_gen:
            num_events, hit_input_tensor, param_output_tensor, length_tensor, evt_tensor, clus_tensor = make_tensors_from_chunk(chunk_df)
            # yield them as a tuple
            yield  num_events, hit_input_tensor, param_output_tensor, length_tensor, evt_tensor, clus_tensor

    
def get_dataloader(chunk_dataset):
    loader = DataLoader(chunk_dataset, batch_size=1)
    return loader
