import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd

PAD_TOKEN = -1

class HitsDataset(Dataset):

    def __init__(self, device, hits_data, track_params_data=None, class_data=None):
        self.hits_data = hits_data.to(device)
        self.track_params_data = track_params_data.to(device)
        self.class_data = class_data.to(device)
        self.total_events = self.__len__()

    def __len__(self):
        return self.hits_data.shape[0]

    def __getitem__(self, idx):
        return idx, self.hits_data[idx], self.track_params_data[idx], self.class_data[idx]

def get_dataloaders(dataset, train_frac, valid_frac, test_frac, batch_size):
    train_set, valid_set, test_set = random_split(dataset, [train_frac, valid_frac, test_frac], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

def load_trackml_data(data, normalize=False, chunking=False):
    """
    Function for reading .csv file with TrackML data and creating tensors
    containing the hits and ground truth information from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to. normalize decides whether the data will be normalized first. 
    chunking allows for reading .csv files in chunks.
    """

    if not chunking:
        data = pd.read_csv(data)

    # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event = shuffled_data.groupby("event_id")
    max_num_hits = data_grouped_by_event.size().max()

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,0]**2 + event_track_params_data[:,1]**2 + event_track_params_data[:,2]**2)
        log_p = np.log(p)
        log_pt = np.log(np.sqrt(event_track_params_data[:,0]**2 + event_track_params_data[:,1]**2))
        log_pz = np.log(np.abs(event_track_params_data[:,2]))
        q = event_track_params_data[:,3]
        theta = np.arccos(event_track_params_data[:,2]/p)
        phi = np.arctan2(event_track_params_data[:,1], event_track_params_data[:,0])
        processed_event_track_params_data = np.column_stack([theta, np.sin(phi), np.cos(phi), q, log_p])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data
