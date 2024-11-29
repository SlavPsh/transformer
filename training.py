from data_processing.dataset import load_trackml_data
import torch
import torch.nn as nn
import numpy as np

# Import supporting tools
import wandb
from utils.io_utils import load_config, setup_logging, unique_output_dir, copy_config_to_output, get_file_path
from utils.wandb_utils import WandbLogger
import argparse
import logging
import os, sys
from coolname import generate_slug
from data_processing.dataset import HitsDataset, PAD_TOKEN, get_dataloaders


def setup_training(config, device):
    '''
    Sets up the model, optimizer, and loss function for training. Returns the model,
    optimizer, loss function, and the starting epoch.
    '''
    config_flash_attention = config['model']['use_flash_attention']
    config_att_mask = config['model']['use_att_mask']
    config_lr = config['training']['default_lr']
    config_model_type = config['model']['type']

    sweep_learning_rate = wandb.config.learning_rate if 'learning_rate' in wandb.config else config_lr
    sweep_att_mask = wandb.config.use_att_mask if 'use_att_mask' in wandb.config else config_att_mask
    sweep_flash_attention = wandb.config.use_flash_attention if 'use_flash_attention' in wandb.config else config_flash_attention

    
    if config_model_type == 'flex_attention':
        from flex_attn_model import TransformerRegressor
        
        logging.info(f"Using flex attention model")
        # model
        model = TransformerRegressor(
            num_encoder_layers = config['model']['num_encoder_layers'],
            d_model = config['model']['d_model'],
            n_head=config['model']['n_head'],
            input_size = config['model']['input_size'],
            output_size = config['model']['output_size'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ).to(device)
    else:
        from model import TransformerRegressor
        # model
        model = TransformerRegressor(
            num_encoder_layers = config['model']['num_encoder_layers'],
            d_model = config['model']['d_model'],
            n_head=config['model']['n_head'],
            input_size = config['model']['input_size'],
            output_size = config['model']['output_size'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            use_att_mask=sweep_att_mask
        ).to(device)

    # optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=sweep_learning_rate)

    # criterion/loss function
    loss_fn = nn.MSELoss()

    # check whether to load from checkpoint
    if not config['training']['start_from_scratch']:
        if 'checkpoint_path' not in config['training'] or not config['training']['checkpoint_path']:
            logging.error("Checkpoint path must be provided when resuming from a checkpoint.")
            sys.exit("Error: Checkpoint path not provided but required for resuming training.")
        elif not os.path.exists(config['training']['checkpoint_path']):
            logging.error(f"Checkpoint file not found: {config['training']['checkpoint_path']}")
            sys.exit("Error: Checkpoint file does not exist.")
        else:
            checkpoint = torch.load(config['training']['checkpoint_path'])
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            
            if 'use_att_mask' in checkpoint:
                model.set_use_att_mask(checkpoint['use_att_mask'])
                logging.info(f"Using attention mask is set to  {checkpoint['use_att_mask']}")
            else:
                model.set_use_att_mask(False)
                logging.info(f'Using attention mask is set to False')
            
            logging.info(f"Resuming training from checkpoint: {config['training']['checkpoint_path']}. Starting from epoch {start_epoch}.")
    else:
        start_epoch = 0
        if 'checkpoint_path' in config['training'] and config['training']['checkpoint_path']:
            logging.warning("Checkpoint path provided but will not be used since training starts from scratch.")
    
    return model, optimizer, loss_fn, start_epoch

def train_epoch(model, optim, train_loader, loss_fn, device):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.

    for data in train_loader:
        _, hits, hits_seq_length, hits_masking, track_params, _ = data
        # Zero the gradients

        optim.zero_grad()
        # Transfer batch to GPU
        hits, hits_seq_length, hits_masking, track_params = hits.to(device),hits_seq_length.to(device), hits_masking.to(device), track_params.to(device)
        # Make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        
        pred = model(hits, hits_seq_length)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        # Get the weights for the loss function
        #classes = torch.unsqueeze(classes[~padding_mask], 0)
        #weights = classes[...,1]
        #weights = weights.unsqueeze(-1)
        # Calculate loss and use it to update weights
        
        loss = loss_fn(pred, track_params)
        loss.backward()
        optim.step()
        losses += loss.item()

        # Free up memory explicitely
        del hits, hits_masking, track_params, padding_mask, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return losses / len(train_loader)

def evaluate(model, validation_loader, loss_fn, device):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses = 0.
    with torch.no_grad():
        for data in validation_loader:
            _, hits, hits_seq_length, hits_masking, track_params, _ = data
            hits, hits_seq_length, hits_masking, track_params = hits.to(device), hits_seq_length.to(device), hits_masking.to(device), track_params.to(device)
            # Make prediction
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            pred = model(hits, hits_seq_length)

            pred = torch.unsqueeze(pred[~padding_mask], 0)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)

            # Get the weights for the loss function
            #classes = torch.unsqueeze(classes[~padding_mask], 0)
            #weights = classes[...,1]
            #weights = weights.unsqueeze(-1)
            
            loss = loss_fn(pred, track_params)
            losses += loss.item()

            # Free up memory explicitely
            del hits, hits_masking, track_params, padding_mask, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
    return losses / len(validation_loader)

def custom_mse_loss(predictions, targets, weights):

    # Ensure the weights are normalized
    normalized_weights = weights / weights.sum()

    # Compute the squared difference between predictions and targets
    squared_diff = (predictions - targets) ** 2  
    # Apply the extracted weights
    weighted_squared_diff = normalized_weights * squared_diff  
    # Compute the mean of the weighted squared differences
    loss = weighted_squared_diff.mean()

    return loss

def main(config_path):
    #Create unique run name
    run_name = generate_slug(3)+"_train"
    # Load the configuration file
    config = load_config(config_path)
    # Create the output directory
    output_dir = unique_output_dir(config, run_name) # with time stamp and run name
    copy_config_to_output(config_path, output_dir)
    # Set up logging
    setup_logging(config, output_dir)

    # Set up wandb
    wandb_logger = WandbLogger(config=config["wandb"],
                                output_dir=output_dir,
                                run_name=run_name,
                                job_type="training")
    wandb_logger.initialize()
    # Log the configuration
    logging.info(f'Loading config from {config_path} ')
    logging.info(f'Output_dir: {output_dir}')
    logging.info(f"Description: {config['experiment']['description']}")
    early_stopping_epoch = config['training']['early_stopping']['patience']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Device: {device}')
    logging.info(f'Torch cuda version: {torch.version.cuda}')

    torch.manual_seed(37)  # for reproducibility
    data_path = get_file_path(config['data']['data_dir'], config['data']['data_file'])
    logging.info(f'Loading data from {data_path} ...')
    hits_data, hits_data_seq_lengths, hits_masking, track_params_data, track_classes_data = load_trackml_data(data=data_path)
    dataset = HitsDataset(hits_data, hits_data_seq_lengths, hits_masking, track_params_data, track_classes_data)

    batch_size = config['training']['batch_size']
    train_loader, valid_loader, _ = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=batch_size, drop_last=True)
    logging.info(f'Data loaded.')

    config

    # Set up the model, optimizer, and loss function
    model, optimizer, loss_fn, start_epoch = setup_training(config, device)
    model.attach_wandb_logger(wandb_logger)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    memory_stats = wandb_logger.get_system_memory_stats()
    logging.info(f"Memory stats: {memory_stats}")

    logging.info("Started training and validation")
    if 'watch_interval' in config['wandb']:
        watch_interval = config['wandb']['watch_interval']
        wandb_logger.run.watch(model, log_freq=watch_interval)
        logging.info(f"wandb started watching at interval {watch_interval} ")

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(start_epoch, config['training']['total_epochs']):
        # Train the model
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn, device)

        # Evaluate using validation split
        val_loss = evaluate(model, valid_loader, loss_fn, device)

        # Print info to the cluster logging
        logging.info(f"Epoch: {epoch}\nVal loss: {val_loss:.10f}, Train loss: {train_loss:.10f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        memory_stats = wandb_logger.get_system_memory_stats()
        wandb_logger.log({'train/train_loss' : train_loss, 'train/epoch' : epoch, 'train/validation loss' : val_loss, **memory_stats})

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            wandb_logger.save_model(model, 'model_best.pth', optimizer, epoch, output_dir)
            logging.info(f"Checkpoint saved to output_dir. Best of run. Epoch: {epoch}")
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            wandb_logger.save_model(model, f'model_last.pth', optimizer, epoch, output_dir)
            logging.info(f"Checkpoint saved to output_dir. Last of run. Epoch: {epoch}")
            count += 1

        if count >= early_stopping_epoch:
            logging.info("Early stopping triggered")
            break
    
    logging.info("Finished training")
    wandb_logger.alert("Finished training", "Finished training")

    wandb_logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a given config file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration TOML file.')
    
    # Parse arguments
    args = parser.parse_args()

    full_config = load_config(args.config_path)
    sweep_config = full_config['sweep']

    # Initialize the sweep and start the sweep agent
    
    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=lambda: main(args.config_path))