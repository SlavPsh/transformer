import torch
import torch.nn as nn
import threading
#from torch.nn.modules.activation import MultiheadAttention
from custom_mha import CustomMultiHeadAttention
from torch.nn.attention.flex_attention import (
    create_block_mask,
    create_mask,
    flex_attention,
)


class FlexAttentionSingleton:
    """Singleton class to compile FlexAttention function once and reuse it."""

    _instance = None
    _lock = threading.Lock()  # Ensures thread safety

    def __new__(cls):
        # Ensure only one instance is created
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._compiled_function = None  # Initialize compiled function
        return cls._instance

    def get_compiled_function(self, function):
        """Compile the function if not already compiled and return it."""
        if self._compiled_function is None:
            print("Compiling the function for the first time...")
            self._compiled_function = torch.compile(function)
        return self._compiled_function

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,):

        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = CustomMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )   

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, block_mask=None):
        if block_mask is not None:
            src_mask = block_mask
            self.self_attn.use_block_mask = True
             
        return super().forward(src, src_mask, src_key_padding_mask, is_causal)  # Calls the same logic


class TransformerRegressor(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout, wandb_logger=None):
        super(TransformerRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)

        encoder_layers = CustomTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, output_size)
        self.num_heads = n_head

        # Initialize the wandb logger
        self.wandb_logger = wandb_logger
        if self.wandb_logger is not None:
            if self.wandb_logger.initialized == False:
                self.wandb_logger.initialize()

    def forward(self, input, padding_mask, flex_padding_mask):
        # TODO: exclude input layer compute for padding tokens
        x = self.input_layer(input)
        S = x.size(1)
        block_mask = create_block_mask(flex_padding_mask, None, None, S, S)   
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask, block_mask=block_mask)
        out = self.decoder(memory)
        #if torch.isnan(memory).any(): 
        #    logging.error("Memory contains NaN values. Check attention mask.")
        #out = self.decoder(memory)
        return out
    
    def attach_wandb_logger(self, wandb_logger):
        self.wandb_logger = wandb_logger
    