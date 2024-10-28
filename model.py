import torch
import torch.nn as nn
import numpy as np
from hdbscan import HDBSCAN

class TransformerRegressor(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, use_att_mask=False):
        super(TransformerRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, output_size)
        self.num_heads = n_head
        self.att_mask_used = use_att_mask

    """
    def forward(self, input, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        out = self.decoder(memory)
        return out
    """

    def set_use_att_mask(self, use_att_mask):
        self.att_mask_used = use_att_mask
    
    def forward(self, input_coord, input_for_mask, padding_mask):
        # Here we use only 3 coordinates x,y,z as input to the model
        x = self.input_layer(input_coord)  # Transform coordinates part of the input into d_model space

        if self.att_mask_used:
            batch_size, seq_len, _ = x.size()
            num_heads = self.num_heads
            # Calculate the distance mask using the raw input
            distance_mask = self.calculate_distance_mask(input_for_mask)  # input used for mask calculation
            # Expand the mask for all heads without duplicating the data
        
            # Save the mask and input for mask to file for debugging
            #if  self.save_to_file == True:
            #    torch.save(distance_mask, '/data/atlas/users/spshenov/temp/distance_mask.pt')
            #    torch.save(input_coord, '/data/atlas/users/spshenov/temp/input_coord.pt')
            #    self.save_to_file = False
        
            expanded_mask = distance_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len).reshape(batch_size * num_heads, seq_len, seq_len)
            # Apply the distance mask to the attention mechanism
            memory = self.encoder(src=x, src_key_padding_mask=padding_mask, mask=expanded_mask)
        else:
            memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        
        out = self.decoder(memory)
        return out
    
    def calculate_distance_mask(self, input_for_mask, z_0_limit = 197.4, phi_r_ratio_limit = 0.001825, angular_separation_limit = 1.797):
        # Calculate the distance mask based on the input

        points = input_for_mask.detach()  # Shape: [batch_size, seq_len, num_features]


        # Reshape points for broadcasting
        point1 = points.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, num_features]
        point2 = points.unsqueeze(1)  # Shape: [batch_size, 1, seq_len, num_features]

        # Extract required coordinates for metric computations 
        z1 = point1[..., 0]
        z2 = point2[..., 0]
        r1 = point1[..., 1]
        r2 = point2[..., 1]
        phi1 = point1[..., 2]
        phi2 = point2[..., 2]
        eta1 = point1[..., 3]
        eta2 = point2[..., 3]

        # Compute the metrics from coordinates
        r_diff = r2 - r1
        z_0 = torch.where(
        r_diff != 0,  # check for 0 in denominator
        torch.abs(z1 - r1 * (z2 - z1) / r_diff),  # Normal calculation
        torch.full_like(r_diff, 1e6)  # Assign a very large value when denominator is 0
        )
        phi_diff = torch.abs(phi2 - phi1)
        phi_diff = torch.where(phi_diff > np.pi, 2 * np.pi - phi_diff, phi_diff)
        phi_r_ratio = phi_diff / (torch.abs(r2 - r1) + 1e-8)
        angular_separation = torch.sqrt((eta2 - eta1) ** 2 + phi_diff ** 2)

        # Create mask based on metric limits
        mask = torch.where(
            (z_0 < z_0_limit) & (phi_r_ratio < phi_r_ratio_limit) & (angular_separation < angular_separation_limit),
            True, False).to(torch.bool)
        
        """
        # For debugging
        # Calculate the number of True values
        num_true = torch.sum(mask).item()

        # Calculate the total number of values
        total_values = mask.numel()

        # Calculate the ratio of True values to total values
        ratio_true_to_total = num_true / total_values

        print(f"Number of True values: {num_true}")
        print(f"Total number of values: {total_values}")
        print(f"Ratio of True values to total values: {ratio_true_to_total:.2f}")

        """


        return mask  # Shape: [batch_size, seq_len, seq_len]
    
def clustering(pred_params, min_cl_size, min_samples):
    '''
    Function to perform HDBSCAN on the predicted track parameters, with specified
    HDBSCAN hyperparameters. Returns the associated cluster IDs.
    '''
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    cluster_labels = []
    for _, event_prediction in enumerate(pred_params):
        regressed_params = np.array(event_prediction.tolist())
        event_cluster_labels = clustering_algorithm.fit_predict(regressed_params)
        cluster_labels.append(event_cluster_labels)

    cluster_labels = [torch.from_numpy(cl_lbl).int() for cl_lbl in cluster_labels]
    return cluster_labels