import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os


class EventDataset(Dataset):
    """
    A Dataset that holds:
      - data_tensor: shape (num_events, max_len, num_features)
      - lengths_tensor: shape (num_events,)
    Each __getitem__(i) returns (data_tensor[i], lengths_tensor[i]).
    """
    def __init__(self, data_tensor, lengths_tensor, folder_path, 
                    feature_cols, normalize=False, normalization_features=None):
            super().__init__()
            self.data = data_tensor
            self.lengths = lengths_tensor
            assert self.data.shape[0] == self.lengths.shape[0]

            self.normalize = normalize
            if self.normalize and normalization_features is not None:
                # explicitly get normalization parameters directly from normalization_features
                self.norm_indices = {}
                for feat_name, stats in normalization_features.items():
                    if feat_name not in feature_cols:
                        raise ValueError(f"Feature {feat_name} not in feature_cols!")
                    idx = feature_cols.index(feat_name)
                    mean = stats['mean']
                    std = stats['std']
                    self.norm_indices[idx] = {'mean': mean, 'std': std}

                # Apply normalization explicitly and efficiently here:
                for idx, stat in self.norm_indices.items():
                    mean, std = stat['mean'], stat['std']
                    feature_data = self.data[..., idx].float()
                    nan_mask = torch.isnan(feature_data)
                    feature_data[~nan_mask] = (feature_data[~nan_mask] - mean) / std
                    self.data[..., idx] = feature_data  # assign normalized data explicitly once


    def __len__(self):
        # The dataset length = number of events
        return self.data.shape[0]

    def __getitem__(self, idx):
        # lengths[i] = a scalar (or 1D with shape ()).
        event_data = self.data[idx]     
        event_length = self.lengths[idx] # int
        return event_data, event_length


def load_split_dataset(folder_path, split="train", feature_cols=None,
                       normalize=False, normalization_features=None):
    """
    Loads data_tensor and lengths_tensor from:
       {folder_path}/{split}_data.pt
       {folder_path}/{split}_lengths.pt
    and returns an EventDataset with them.
    """
    data_file = f"{folder_path}/{split}_data.pt"
    length_file = f"{folder_path}/{split}_lengths.pt"

    # Load the saved PyTorch tensors
    data_tensor = torch.load(data_file, map_location="cpu")     
    lengths_tensor = torch.load(length_file, map_location="cpu") # shape (N,)

    dataset = EventDataset(data_tensor, lengths_tensor, folder_path, 
                           feature_cols, normalize, normalization_features)
    return dataset


def get_train_valid_dataloaders(config, batch_size=8, num_workers=0, train_fraction=1.0):
    """
    Returns 2 DataLoaders: train, valid
    Each will yield batches of shape (batch_size, max_len, 14)
    plus a length tensor shape (batch_size,).

    folder_path: str, e.g. "my_folder/" where the *.pt files exist
    batch_size: int
    num_workers: used by DataLoader for multi-process data loading
    """
    folder_path = config['data']['data_dir']
    feature_cols = config['data']['feature_cols']
    normalize = config['data'].get('normalize', False)
    normalization_features = config['data'].get('normalization_features', {})
    
    gen = torch.Generator()
    gen.manual_seed(37)  


    train_ds = load_split_dataset(folder_path, "train", feature_cols, normalize, normalization_features)
    valid_ds = load_split_dataset(folder_path, "valid", feature_cols, normalize, normalization_features)
    
    if train_fraction < 1.0:
        num_train = int(len(train_ds) * train_fraction)
        train_ds, _ = random_split(train_ds, [num_train, len(train_ds) - num_train], generator= gen)
    # Typically we shuffle train, not valid/test:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator = gen)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, valid_loader


def get_test_dataloader(config, batch_size=8, num_workers=0):
    """
    Returns test DataLoader
    It will yield batches of shape (batch_size, max_len, 14)
    plus a length tensor shape (batch_size,).

    folder_path: str, e.g. "my_folder/" where the *.pt files exist
    batch_size: int
    num_workers: used by DataLoader for multi-process data loading
    """
    folder_path = config['data']['data_dir']
    feature_cols = config['data']['feature_cols']
    normalize = config['data'].get('normalize', False)
    normalization_features = config['data'].get('normalization_features', {})

    test_ds  = load_split_dataset(folder_path, "test", feature_cols, normalize, normalization_features)

    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return test_loader