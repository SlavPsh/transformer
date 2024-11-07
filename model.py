import torch
import torch.nn as nn
import numpy as np
from hdbscan import HDBSCAN
import logging

class TransformerRegressor(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, use_att_mask=False, wandb_logger=None, use_flash_attention=False):
        super(TransformerRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, output_size)
        self.num_heads = n_head
        self.att_mask_used = use_att_mask
        self.flash_attention = use_flash_attention

        # Initialize the wandb logger
        self.wandb_logger = wandb_logger
        if self.wandb_logger is not None:
            if self.wandb_logger.initialized == False:
                self.wandb_logger.initialize()

    """
    def forward(self, input, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        out = self.decoder(memory)
        return out
    """
    def attach_wandb_logger(self, wandb_logger):
        self.wandb_logger = wandb_logger
        
    def set_use_att_mask(self, use_att_mask):
        self.att_mask_used = use_att_mask
    
    def forward(self, input_coord, input_for_mask, padding_mask):
        # Here we use only 3 coordinates x,y,z as input to the model
        x = self.input_layer(input_coord)  # Transform coordinates part of the input into d_model space

        if self.att_mask_used:
            batch_size, seq_len, _ = x.size()
            num_heads = self.num_heads
            # Calculate the distance mask using the raw input
            distance_mask = self.calculate_distance_mask(input_for_mask, padding_mask)  # input used for mask calculation
                
            # Expand the mask for all heads without duplicating the data

            # Calculate efficiency and purity
            efficiency, purity = calc_efficiency_purity(distance_mask, input_for_mask, padding_mask)


            # Log the mask efficiency and purity

            self.wandb_logger.log({'Mask Efficiency': efficiency, 'Mask Purity': purity})
        
            # Save the mask and input for mask to file for debugging
            #if  self.save_to_file == True:
            #    torch.save(distance_mask, '/data/atlas/users/spshenov/temp/distance_mask.pt')
            #    torch.save(input_coord, '/data/atlas/users/spshenov/temp/input_coord.pt')
            #    self.save_to_file = False
            
            expanded_mask = distance_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len).reshape(batch_size * num_heads, seq_len, seq_len)

            # Apply the distance mask to the attention mechanism
             # Enable Flash Attention and apply to encoder
            if self.flash_attention:
                with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    memory = self.encoder(x, mask=expanded_mask, src_key_padding_mask=padding_mask)
            else:
                memory = self.encoder(src=x, src_key_padding_mask=padding_mask, mask=expanded_mask)
        else:
            if self.flash_attention:
                with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
            else:
                memory = self.encoder(src=x, src_key_padding_mask=padding_mask)

        # Regularization of the output for stability of clustering algorithm
        if torch.isnan(memory).any(): 
            logging.error("Memory contains NaN values. Check attention mask.")
        out = self.decoder(memory)
        return out
    
    def calculate_distance_mask(self, input_for_mask, padding_mask, z_0_limit = 197.4, phi_r_ratio_limit = 0.001825, angular_separation_limit = 1.797):
        # Calculate the distance mask based on the input

        points = input_for_mask.detach()  # Shape: [batch_size, seq_len, num_features]


        # Reshape points for broadcasting
        point1 = points.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, num_features]
        point2 = points.unsqueeze(1)  # Shape: [batch_size, 1, seq_len, num_features]

        # Expand the padding mask to both sequence dimensions
        expanded_padding_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)  # Shape: [batch_size, seq_len, seq_len]
        expanded_padding_mask.diagonal(dim1=-2, dim2=-1).fill_(True)  # Set diagonal elements to True in order to keep attention to itself

        # Avoid computing metrics for padding and diagonal elements by setting combined_mask
        non_padding_mask = ~expanded_padding_mask
        large_value = torch.full((1,), 1e6, device=points.device)


        # Extract required coordinates for metric computations 
        z1, z2 = point1[..., 0], point2[..., 0]
        r1, r2 = point1[..., 1], point2[..., 1]
        phi1, phi2 = point1[..., 2], point2[..., 2]
        eta1, eta2 = point1[..., 3], point2[..., 3]

        # Compute r_diff, avoiding division by zero for non-relevant pairs
        r_diff = torch.where(non_padding_mask, r2 - r1, large_value)
        z_0 = torch.where(
            (r_diff != 0) & non_padding_mask,
            torch.abs(z1 - r1 * (z2 - z1) / r_diff),
            large_value
        )

        # Compute phi_diff and apply adjustments for values > π
        phi_diff = torch.where(non_padding_mask, torch.abs(phi2 - phi1), large_value)
        phi_diff = torch.where((phi_diff > np.pi) & non_padding_mask, 2 * np.pi - phi_diff, phi_diff)
        
        # Calculate phi_r_ratio and angular_separation only for non-padding pairs
        phi_r_ratio = torch.where(non_padding_mask, phi_diff / (torch.abs(r_diff) + 1e-8), large_value)
        angular_separation = torch.where(non_padding_mask, torch.sqrt((eta2 - eta1) ** 2 + phi_diff ** 2), large_value)

        # Create the mask based on metric limits
        mask = (non_padding_mask & 
                (z_0 > z_0_limit) & 
                (phi_r_ratio > phi_r_ratio_limit) & 
                (angular_separation > angular_separation_limit))

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

def calc_efficiency_purity(distance_mask, input_for_mask, padding_mask):
    # Calculate the efficiency and purity of the distance mask
    points = input_for_mask.detach()  # Shape: [batch_size, seq_len, num_features]
    # Reshape points for broadcasting
    point1 = points.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, num_features]
    point2 = points.unsqueeze(1)  # Shape: [batch_size, 1, seq_len, num_features]
    
    # Make padding 1 dimension extra on the number of points
    padding_mask_expanded = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
    particle_id1 = point1[..., 4]
    particle_id2 = point2[..., 4]

    # Remove padding from the distance mask
    mask_no_padding = ~distance_mask & ~padding_mask_expanded
    mask_true_edges = torch.where((particle_id1 == particle_id2) & (particle_id1 != 0), True, False).to(torch.bool)
    # Remove padding from the true edges mask
    mask_true_edges_no_padding = mask_true_edges & ~padding_mask_expanded

    # Create an upper triangular mask, excluding the diagonal
    batch_size, seq_len, _ = mask_no_padding.shape
    upper_triangular_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    upper_triangular_mask = upper_triangular_mask.to(mask_no_padding.device)
    upper_triangular_mask = upper_triangular_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply the upper triangular mask to count unique pairs
    mask_no_padding = mask_no_padding & upper_triangular_mask
    mask_true_edges_no_padding = mask_true_edges_no_padding & upper_triangular_mask

    # Calculate the overlap between the distance mask and the true edges mask
    overlap = (mask_no_padding & mask_true_edges_no_padding).sum().item()
    attention_count = mask_no_padding.sum().item()
    true_count = mask_true_edges_no_padding.sum().item()

    # Calculate the efficiency and purity, with a small epsilon to avoid division by zero
    epsilon = 1e-8
    efficiency = overlap / (true_count + epsilon)
    purity = overlap / (attention_count + epsilon)

    return efficiency, purity