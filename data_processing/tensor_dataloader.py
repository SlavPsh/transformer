import torch
from torch.utils.data import Dataset, DataLoader

class EventDataset(Dataset):
    """
    A Dataset that holds:
      - data_tensor: shape (num_events, max_len, 14)
      - lengths_tensor: shape (num_events,)
    Each __getitem__(i) returns (data_tensor[i], lengths_tensor[i]).
    """
    def __init__(self, data_tensor, lengths_tensor):
        super().__init__()
        self.data = data_tensor      # shape (N, max_len, 14)
        self.lengths = lengths_tensor  # shape (N,)
        assert self.data.shape[0] == self.lengths.shape[0], \
            "data and lengths must have the same number of events"

    def __len__(self):
        # The dataset length = number of events
        return self.data.shape[0]

    def __getitem__(self, idx):
        # data[i] = shape (max_len, 14)
        # lengths[i] = a scalar (or 1D with shape ()).
        event_data = self.data[idx]      # shape (max_len, 14)
        event_length = self.lengths[idx] # int
        return event_data, event_length


def load_split_dataset(folder_path, split="train"):
    """
    Loads data_tensor and lengths_tensor from:
       {folder_path}/{split}_data.pt
       {folder_path}/{split}_lengths.pt
    and returns an EventDataset with them.
    """
    data_file = f"{folder_path}/{split}_data.pt"
    length_file = f"{folder_path}/{split}_lengths.pt"

    # Load the saved PyTorch tensors
    data_tensor = torch.load(data_file)      # shape (N, max_len, 14)
    lengths_tensor = torch.load(length_file) # shape (N,)

    dataset = EventDataset(data_tensor, lengths_tensor)
    return dataset


def get_train_valid_dataloaders(folder_path, batch_size=8, num_workers=0):
    """
    Returns 2 DataLoaders: train, valid
    Each will yield batches of shape (batch_size, max_len, 14)
    plus a length tensor shape (batch_size,).

    folder_path: str, e.g. "my_folder/" where the *.pt files exist
    batch_size: int
    num_workers: used by DataLoader for multi-process data loading
    """
    train_ds = load_split_dataset(folder_path, split="train")
    valid_ds = load_split_dataset(folder_path, split="valid")
    # Typically we shuffle train, not valid/test:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, valid_loader


def get_test_dataloader(folder_path, batch_size=8, num_workers=0):
    """
    Returns test DataLoader
    It will yield batches of shape (batch_size, max_len, 14)
    plus a length tensor shape (batch_size,).

    folder_path: str, e.g. "my_folder/" where the *.pt files exist
    batch_size: int
    num_workers: used by DataLoader for multi-process data loading
    """
    test_ds  = load_split_dataset(folder_path, split="test")

    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return test_loader