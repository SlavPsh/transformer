import torch
import torch.nn as nn
import numpy as np

from custom_encoder import TransformerEncoderLayer as custom_encoder

'''
class AngleDifferenceLoss(nn.Module):
    """
    Implementation of Angle Difference Loss, which assumes the predictions and targets
    are angles (dealing with the rotational invariance problem).
    """
    def __init__(self):
        super(AngleDifferenceLoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((((predictions - targets) + np.pi) % 2*np.pi) - np.pi)
'''

class TransformerRegressor(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, use_flashattn=False):
        super(TransformerRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        if use_flashattn:
            encoder_layers = custom_encoder(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        else:
            encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, input, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        out = self.decoder(memory)
        return out


def save_model(model, optim, type, val_losses, train_losses, epoch, count, file_name):
    print(f"Saving {type} model")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'count': count,
    }, f"{file_name}_{type}")