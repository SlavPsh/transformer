import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd

PAD_TOKEN = -1

class HitsDataset(Dataset):

    def __init__(self, hits_data, hits_seq_length, hits_masking, track_params_data=None, particle_data=None):
        self.hits_data = hits_data
        self.hits_seq_length = hits_seq_length
        self.hits_masking = hits_masking
        self.track_params_data = track_params_data
        self.particle_data = particle_data
        self.total_events = self.__len__()

    def __len__(self):
        return self.hits_data.shape[0]

    def __getitem__(self, idx):
        return idx, self.hits_data[idx], self.hits_seq_length[idx], self.hits_masking[idx], self.track_params_data[idx], self.particle_data[idx]

def get_dataloaders(dataset, train_frac, valid_frac, test_frac, batch_size, drop_last=False):
    train_set, valid_set, test_set = random_split(dataset, [train_frac, valid_frac, test_frac], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

def load_trackml_data(data, normalize=True, chunking=False):
    """
    Function for reading .csv file with TrackML data and creating tensors
    containing the hits and ground truth information from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to. normalize decides whether the data will be normalized first. 
    chunking allows for reading .csv files in chunks.
    """

    if not chunking:
        data = pd.read_csv(data)



    # Shuffling the data and grouping by event ID, add random state for reproducibility
    shuffled_data = data.sample(frac=1, random_state=37)
    #shuffled_data = data
    # Add extra colums to the data
    shuffled_data["p"] = np.sqrt(shuffled_data["px"]**2 + shuffled_data["py"]**2 + shuffled_data["pz"]**2)
    shuffled_data["log_p"] = np.log(shuffled_data["p"])
    shuffled_data["pt"] = np.sqrt(shuffled_data["px"]**2 + shuffled_data["py"]**2)
    #shuffled_data["log_pt"] = np.log(shuffled_data["pt"])
    shuffled_data["theta"] = np.arccos(shuffled_data["pz"]/shuffled_data["p"])
    shuffled_data["phi"] = np.arctan2(shuffled_data["py"], shuffled_data["px"])
    shuffled_data["sin_phi"] = np.sin(shuffled_data["phi"])
    shuffled_data["cos_phi"] = np.cos(shuffled_data["phi"])
    shuffled_data['eta'] = -np.log(np.tan(shuffled_data['theta']/2.))

        # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "theta", "log_p"]:
            mean = shuffled_data[col].mean()
            std = shuffled_data[col].std()
            shuffled_data[col] = (shuffled_data[col] - mean)/std

    data_grouped_by_event = shuffled_data.groupby("event_id")
    max_num_hits = data_grouped_by_event.size().max() + 1
    # Round up to the next multiple of 128 for flex attention
    max_num_hits = ((max_num_hits + 127) // 128) * 128

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        sequence_length = len(event_rows)
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        padded_hit_data = np.pad(event_hit_data, [(0, max_num_hits-sequence_length), (0, 0)], "constant", constant_values=PAD_TOKEN)
        return padded_hit_data, sequence_length
    
    def extract_hits_data_for_masking(event_rows):
        # Returns tensors with cluster ids for hits 
        """
        event_hit_data = event_rows[["x", "y", "z", "particle_id"]].to_numpy(dtype=np.float32)
        r_cyl = np.sqrt(event_hit_data[:,0]**2 + event_hit_data[:,1]**2)
        rho = np.sqrt(event_hit_data[:,0]**2 + event_hit_data[:,1]**2 + event_hit_data[:,2]**2)
        phi_cyl = np.arctan2(event_hit_data[:,1], event_hit_data[:,0])
        theta_coord = np.arccos(event_hit_data[:,2]/rho)
        eta_coord = -np.log(np.tan(theta_coord/2.))
        
        hits_data_for_masking = np.column_stack([event_hit_data[:,2], r_cyl, phi_cyl, eta_coord, event_hit_data[:,3]])
        """
        sequence_length = len(event_rows)
        #hits_data_for_masking = event_rows["cluster_id"].to_numpy(dtype=np.int32)
        hits_data_for_masking = event_rows["event_id"].to_numpy(dtype=np.int32)
        
        hits_data_for_masking_padded = np.pad(hits_data_for_masking, (0, max_num_hits-sequence_length), "constant", constant_values=-2)
        
        return hits_data_for_masking_padded

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["theta","sin_phi","cos_phi", "q", "log_p"]].to_numpy(dtype=np.float32)

        theta = event_track_params_data[:,0]
        sin_phi = event_track_params_data[:,1]
        cos_phi = event_track_params_data[:,2]
        q = event_track_params_data[:,3]
        log_p = event_track_params_data[:,4]
        #vz = event_track_params_data[:,5]
        processed_event_track_params_data = np.column_stack([theta, sin_phi, cos_phi, q, log_p])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_particle_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        event_hit_classes_data = event_rows[["particle_id","weight", "theta", "sin_phi", "q", "pt", "eta"]].to_numpy(dtype=np.float32)
        
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    results = data_grouped_by_event.apply(extract_hits_data)
    grouped_hits_data, sequence_lengths = zip(*results)

    grouped_masking_data = data_grouped_by_event.apply(extract_hits_data_for_masking)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_particle_data = data_grouped_by_event.apply(extract_particle_data)
    

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data))
    hits_data_seq_lengths = torch.tensor(sequence_lengths, dtype=torch.long) 
    hits_data_for_masking = torch.tensor(np.stack(grouped_masking_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_particle_data = torch.tensor(np.stack(grouped_particle_data.values))

    return hits_data, hits_data_seq_lengths, hits_data_for_masking, track_params_data, hit_particle_data
