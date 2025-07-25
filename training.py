from data_processing.dataset import load_trackml_data
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

# Import supporting tools
import wandb
from utils.io_utils import load_config, setup_logging, unique_output_dir, copy_config_to_output, get_file_path, get_total_grad_norm
from utils.wandb_utils import WandbLogger, get_system_memory_stats
from utils.timing_utils import StepTimer
import argparse
import logging
import os, sys
from coolname import generate_slug
from data_processing.dataset import HitsDataset, PAD_TOKEN, get_dataloaders, flatten_and_pad
from data_processing.tensor_dataloader import get_train_valid_dataloaders
from custom_model import generate_cluster_padding_mask, generate_padding_mask, generate_sliding_window_padding_mask

import torch.profiler
from evaluation.scoring import calc_score_trackml
from evaluation.clustering import clustering, clustering_inception, clustering_HDBSCAN

from evaluation.loss import supcon_loss_flat


def setup_training(config, device, train_loader):
    '''
    Sets up the model, optimizer, and loss function for training. Returns the model,
    optimizer, loss function, and the starting epoch.
    '''
    config_model_type = config['model']['type']
    logging.info(f"Model type: {config_model_type}")

    num_encoder_layers = wandb.config.num_encoder_layers if hasattr(wandb.config, 'num_encoder_layers') else config['model']['num_encoder_layers']
    d_model = wandb.config.d_model if hasattr(wandb.config, 'd_model') else config['model']['d_model']
    n_head = wandb.config.n_head if hasattr(wandb.config, 'n_head') else config['model']['n_head']
    

    input_size = len(config['model']['input_features'])

    output_size_set = config['model']['output_size'] if 'output_size' in config['model'] else len(config['model']['output_features'])
    output_size = wandb.config.output_size if hasattr(wandb.config, 'output_size') else output_size_set

    if hasattr(wandb.config, 'dim_feedforward'):
        dim_feedforward = wandb.config.dim_feedforward
    elif hasattr(config['model'], 'dim_feedforward'):
        dim_feedforward = config['model']['dim_feedforward']
    else:
        dim_feedforward = 2 * d_model

    logging.info("=== Model ===")
    logging.info(f"num_encoder_layers: {num_encoder_layers}")
    logging.info(f"d_model: {d_model}")
    logging.info(f"n_head: {n_head}")
    logging.info(f"dim_feedforward: {dim_feedforward}")
    logging.info(f"dropout: {config['model']['dropout']}")
    logging.info(f"input_features: {config['model']['input_features']}")
    logging.info(f"output_features: {config['model']['output_features']}")
    


    
    if config_model_type == 'vanilla':
        from vanilla_model import TransformerRegressor

        model = TransformerRegressor(
            num_encoder_layers = config['model']['num_encoder_layers'],
            d_model = config['model']['d_model'],
            n_head=config['model']['n_head'],
            input_size = input_size,
            output_size = output_size,
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ).to(device)

    elif config_model_type == 'flex_attention':
        from custom_model import TransformerRegressor
        


        model = TransformerRegressor(
            num_encoder_layers = num_encoder_layers,
            d_model = d_model,
            n_head=n_head,
            input_size = input_size,
            output_size = output_size,
            dim_feedforward=dim_feedforward,
            dropout=config['model']['dropout']
        ).to(device)

    # Parameters for LR scheduler
    mode = config['training']['scheduler']['mode']
    factor = config['training']['scheduler']['factor']
    patience = config['training']['scheduler']['patience']
    initial_lr =  wandb.config.initial_lr if hasattr(wandb.config, 'initial_lr') else config['training']['scheduler']['initial_lr']
    min_lr =  wandb.config.min_lr if hasattr(wandb.config, 'min_lr') else config['training']['scheduler']['min_lr']
    
    # Parameters for OneCycleLR
    max_lr =  config['training']['scheduler']['max_lr']
    warmup_factor = config['training']['scheduler']['warmup_factor']
    accumulation_steps = config['training']['accumulation_steps']
    div_factor = max_lr / initial_lr
    final_div_factor = initial_lr / min_lr
    updates_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    total_updates = updates_per_epoch * config['training']['total_epochs']


    # Change lr based on batch size and model size 
    batch_size = config['training']['batch_size']
    lr_scale = (batch_size / 8) * np.sqrt(128 / d_model)

    initial_lr = initial_lr * lr_scale
    min_lr = min_lr * lr_scale

    logging.info(f"initial_lr: {initial_lr}")


    
    #optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)


    #lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,  eta_min=min_lr)

    """
    lr_scheduler = OneCycleLR(optimizer,
                    max_lr=max_lr,
                    total_steps=total_updates,
                    pct_start=warmup_factor,
                    anneal_strategy='cos',
                    div_factor=div_factor,
                    final_div_factor=final_div_factor)
    """
    

    # criterion/loss function
    #loss_fn = nn.MSELoss()

    output_features = config['model']['output_features']
    loss_weighting = config['training']['loss_weighting']

    """
    loss_fn = partial(
    mse_per_feature_loss_weighted,   # the full loss implementation
    feature_names=output_features,   # fixed positional arg
    pt_threshold=0.9,
    pt_weighting=loss_weighting,
    exp_beta=5.0)
    """
    temperature = wandb.config.temperature if hasattr(wandb.config, 'temperature') else 0.05
    loss_fn = partial(supcon_loss_flat, 
                      pt_threshold=0.9,
                      temperature=temperature)

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
            lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
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
    
    return model, optimizer, lr_scheduler, loss_fn, start_epoch

def train_epoch(model, optim, train_loader, loss_fn, scaler, lr_scheduler, device, config, wandb_logger, epoch = 0, timer=None):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    config_model_type = config['model']['type']

    input_features = config['model']['input_features']
    output_features = config['model']['output_features']

    input_size = len(input_features)
    output_size = len(output_features)
    accumulation_steps = config['training']['accumulation_steps']

    feature_cols = config['data']['feature_cols']

    cluster_idx = feature_cols.index("cluster_id") 

    input_feature_indices = [feature_cols.index(feat) for feat in input_features]
    output_feature_indices = [feature_cols.index(feat) for feat in output_features]

    pt_idx = feature_cols.index("pt")
    pid_idx = feature_cols.index("particle_id")
    event_idx = feature_cols.index("event_id")
    
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    total_loss_sum = 0.
    total_count_sum = 0


    if config_model_type == 'flex_attention':
        
        optim.zero_grad()

        """
        # MEMORY PROFILING

        WAIT, WARMUP, ACTIVE, REPEAT = 50, 40, 50, 1

        # Create a torch.profiler.profile object, and call it as the last part of the training loop
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=WAIT,
                warmup=WARMUP,
                active=ACTIVE,
                repeat=REPEAT
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("/projects/nisei0750/slava/data/memory_profiling", worker_name='worker0'),
            record_shapes=True,
            profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
            with_stack=True
        ) as p:
        """

        for i, (data_tensor, length_tensor) in enumerate(train_loader):  
            #optim.zero_grad() 

            # For logging
            step_number = i + epoch * len(train_loader) 

            if i % 100 == 0:
                logging.info(f"Starting batch {i}")

                if timer:
                    stats = timer.get_stats(reset=False)
                    logging.info(f"Accum Timer stats: {stats}")
            
            # Slice the data tensor
            length_tensor = length_tensor.squeeze(0)
            in_data_tensor_cpu = data_tensor[..., input_feature_indices]
            #out_data_tensor_cpu = data_tensor[..., output_feature_indices]
            pt_tensor_cpu = data_tensor[..., pt_idx].squeeze(-1)

            pid_tensor_cpu = data_tensor[..., pid_idx].squeeze(-1)
            event_tensor_cpu = data_tensor[..., event_idx].squeeze(-1)  # Assuming the first feature is event ID

            cluster_tensor_cpu = data_tensor[..., cluster_idx].squeeze(-1)
            cluster_tensor_cpu = cluster_tensor_cpu.long()
            length_tensor = length_tensor.long()

            length_tensor = length_tensor.to(device)
            in_data_tensor = in_data_tensor_cpu.to(device)
            #out_data_tensor = out_data_tensor_cpu.to(device)
            cluster_tensor = cluster_tensor_cpu.to(device)
            pt_tensor = pt_tensor_cpu.to(device)
            pid_tensor = pid_tensor_cpu.to(device)
            event_tensor = event_tensor_cpu.to(device)
            
            #phi_tensor = phi_tensor_cpu.to(device)
            #eta_coord_tensor = eta_coord_tensor_cpu.to(device)

            #flex_padding_mask = generate_padding_mask(length_tensor)
            flex_padding_mask = generate_cluster_padding_mask(length_tensor, cluster_tensor)
            #flex_padding_mask = generate_sliding_window_padding_mask(length_tensor)
            
                
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  
                
                pred = model(in_data_tensor, f'train_{i}', flex_padding_mask, timer)
                if i % 100 == 0:
                    memory_stats = get_system_memory_stats()
                    logging.info(f"Memory stats after forward: {memory_stats}")
                # Create a mask to select the first L[batch] elements in the seq dimension

                if timer:
                    timer.start('unmasking_pred')

                batched_target = []

                batched_pred, batched_pid, batched_evt, batched_pt = [], [], [], []

                B = len(length_tensor)

                for b_idx in range(B):
                    if B == 1:
                        seq_len = length_tensor.item()
                    else:
                        seq_len = length_tensor[b_idx].item()
                    # unpad just [0..seq_len) for pred
                    this_pred = pred[b_idx, :seq_len, :]
                    #this_target = out_data_tensor[b_idx, :seq_len, :]
                    this_pt = pt_tensor[b_idx, :seq_len]
                    this_pid = pid_tensor[b_idx, :seq_len]
                    this_event = event_tensor[b_idx, :seq_len]
                    batched_pred.append(this_pred)
                    #batched_target.append(this_target)
                    batched_pt.append(this_pt)
                    batched_pid.append(this_pid)
                    batched_evt.append(this_event)  # Event ID for each batch

                
                pred = torch.cat(batched_pred, dim=0)
                #out_data_tensor = torch.cat(batched_target, dim=0)
                pt = torch.cat(batched_pt, dim=0)
                pt = pt.to(dtype=pred.dtype)
                loss_per_feature = None

                pid_flat = torch.cat(batched_pid, dim=0)         # (N,)
                evt_flat = torch.cat(batched_evt, dim=0)         # (N,)


                if timer:
                    timer.stop()

                if timer:
                    timer.start('loss_calc')
                #loss, loss_per_feature = loss_fn(pred, out_data_tensor, pt=pt)
                loss = loss_fn(pred, pid_flat, evt_flat, pt)
                if timer:
                    timer.stop()


            if (loss is None ) or (loss.item() is None):
                logging.info(f"Loss is None for batch {i}")
                raise RuntimeError("Loss is None")
            
            if loss_per_feature is not None:
                wandb_logger.log({f'train/{k}': v.item() for k, v in loss_per_feature.items()}, step=step_number)
                
            loss = loss / accumulation_steps  
            if timer:
                timer.start('backprop')

            scaler.scale(loss).backward()
            #loss.backward()

            if timer:
                timer.stop()

            # Check for non-finite gradients
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        logging.warning(f"[Non-finite Gradient] {name}  !!!")
                        #raise RuntimeError("NaN/Inf gradient detected!")
            
            grad_norm = get_total_grad_norm(model)
            wandb_logger.log({"train/grad_norm": grad_norm}, step=step_number)
            # Add gradient clipping , but before unscale gradients
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            if (i + 1) % accumulation_steps == 0:
                if timer:
                    timer.start('optimizer_step')
                scaler.step(optim)
                #optim.step()
                if timer:
                    timer.stop()
                

                scaler.update()
                optim.zero_grad()

            # for logging
            total_loss_sum += loss.item()*accumulation_steps
            total_count_sum += 1
                
            """
                p.step()
                
                # Allow to break early for the purpose of shorter profiling
                if i == (WAIT + WARMUP + ACTIVE) * REPEAT:
                    break
            """
        
        
        remainder = len(train_loader) % accumulation_steps
        if remainder != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            
        # Other model types        
    else:
        for i, data in enumerate(train_loader):
            _, hits, hits_seq_length, hits_masking, track_params, _ = data
            
            # Zero the gradients
            optim.zero_grad()
            # Transfer batch to GPU
            hits, hits_seq_length, hits_masking, track_params = hits.to(device), hits_seq_length.to(device), hits_masking.to(device), track_params.to(device)
            
            
            # Make prediction
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)

            pred = model(hits, padding_mask=padding_mask)
            pred = torch.unsqueeze(pred[~padding_mask], 0)
        
            loss, _  = loss_fn(pred, track_params)
            loss.backward()
            optim.step()
            total_loss_sum += loss.item()
            total_count_sum += 1

    return total_loss_sum / total_count_sum

def evaluate(model, validation_loader, loss_fn, device, config, final_epoch_csv_path=None):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    config_model_type = config['model']['type']

    feature_cols = config['data']['feature_cols']
    input_features = config['model']['input_features']
    output_features = config['model']['output_features']
        
    output_size = len(output_features)
    input_size = len(input_features)

    pt_idx = feature_cols.index("pt")                
    eta_idx = feature_cols.index("eta")              
    particle_id_idx = feature_cols.index("particle_id")  
    event_idx = feature_cols.index("event_id")
    weight_idx = feature_cols.index("weight") 
    input_feature_indices = [feature_cols.index(feat) for feat in input_features]
    output_feature_indices = [feature_cols.index(feat) for feat in output_features]

    cluster_idx = feature_cols.index("cluster_id") 

    # Get the network in evaluation mode
    model.eval()
    total_loss_sum = 0.
    total_count_sum = 0
    mean_dm_score = 0
    mean_track_ml_score = 0

    with torch.no_grad():
        if config_model_type == 'flex_attention':
            for i,  (data_tensor, length_tensor) in enumerate(validation_loader):    
                length_tensor = length_tensor.squeeze(0)

                in_data_tensor_cpu   = data_tensor[..., input_feature_indices]     
                #out_data_tensor_cpu  = data_tensor[..., output_feature_indices] 

                pt_tensor_cpu = data_tensor[..., pt_idx].squeeze(-1)
                pid_tensor_cpu = data_tensor[..., particle_id_idx].squeeze(-1)
                event_tensor_cpu = data_tensor[..., event_idx].squeeze(-1) 
                cluster_tensor_cpu   = data_tensor[..., cluster_idx].squeeze(-1)
                cluster_tensor_cpu = cluster_tensor_cpu.long()
                length_tensor = length_tensor.long()

                length_tensor = length_tensor.to(device)
                in_data_tensor = in_data_tensor_cpu.to(device)
                #out_data_tensor = out_data_tensor_cpu.to(device)
                cluster_tensor = cluster_tensor_cpu.to(device)
                pt_tensor = pt_tensor_cpu.to(device)
                pid_tensor = pid_tensor_cpu.to(device)
                event_tensor = event_tensor_cpu.to(device)
                
                #flex_padding_mask = generate_padding_mask(length_tensor)
                flex_padding_mask = generate_cluster_padding_mask(length_tensor, cluster_tensor)
                #flex_padding_mask = generate_sliding_window_padding_mask(length_tensor)

                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(in_data_tensor, f'valid_{i}', flex_padding_mask)
                    
                    batched_pred = []
                    batched_target = []
                    batched_pt = []
                    batched_pid = []
                    batched_event = []
                    

                    num_batches = len(length_tensor)

                    for batch_idx in range(num_batches):

                        seq_len = length_tensor[batch_idx].item()
                        # unpad just [0..seq_len) for pred
                        this_pred = pred[batch_idx, :seq_len, :]
                        #this_target = out_data_tensor[batch_idx, :seq_len, :]
                        this_pt = pt_tensor[batch_idx, :seq_len]
                        this_pid = pid_tensor[batch_idx, :seq_len]
                        this_event = event_tensor[batch_idx, :seq_len]
                        batched_pred.append(this_pred)
                        #batched_target.append(this_target)
                        batched_pt.append(this_pt)
                        batched_pid.append(this_pid)
                        batched_event.append(this_event)  # Event ID for each batch
                    
                    pred = torch.cat(batched_pred, dim=0)
                    pid_flat = torch.cat(batched_pid, dim=0)
                    evt_flat = torch.cat(batched_event, dim=0)
                    #out_data_tensor = torch.cat(batched_target, dim=0)
                    pt = torch.cat(batched_pt, dim=0)
                    pt = pt.to(dtype=pred.dtype)

                    #loss, _ = loss_fn(pred, out_data_tensor, pt=pt)
                    loss = loss_fn(pred, pid_flat, evt_flat, pt)

                if i == 0:
                    scores_list = []
                    track_ml_scores_list = []
                    epsilon = 2
                    min_samples = 1
                    
                    length_tensor_cpu = length_tensor.to('cpu')
                    #existing_cluster_ids = [cluster_tensor_cpu[i, :length_tensor_cpu[i]] for i in range(len(length_tensor_cpu))]
                    # Cluster predictions
                    cluster_labels_list = clustering_HDBSCAN(batched_pred, min_cl_size=epsilon, min_samples=min_samples)
                    #cluster_labels_list = clustering_inception(batched_pred, existing_cluster_ids, epsilon=epsilon, min_samples=min_samples,cluster_noise=True )
                    
                    # Check the best posible score by passing true data : 
                    #cluster_labels_list = clustering(batched_target, min_cl_size=min_cl_size, min_samples=min_samples)
                    
                    metrics_data_tensor_cpu  = data_tensor[:, :, [pt_idx, eta_idx, particle_id_idx, weight_idx]]
                    
                    metrics_data_tensor_cpu = [metrics_data_tensor_cpu[i, :length_tensor_cpu[i], :] for i in range(len(length_tensor_cpu))]

                    # Check the score if we use clusters from before transformer stage
                    #cluster_labels_list = [cluster_tensor_cpu[i, :length_tensor_cpu[i]] for i in range(len(length_tensor_cpu))]
                    
                    for cluster_labels, track_labels in zip(cluster_labels_list, metrics_data_tensor_cpu):
                        event_score, scores, nr_particles, predicted_tracks, true_tracks = calc_score_trackml(cluster_labels, track_labels, pt_threshold=0.9)
                        scores_list.append(scores[1])
                        track_ml_scores_list.append(event_score)
                    
                    mean_dm_score = sum(scores_list) / len(scores_list)
                    mean_track_ml_score = sum(track_ml_scores_list) / len(track_ml_scores_list)

                    logging.info(f"Eval Batch {i} . Mean DM score {mean_dm_score} with epsilon {epsilon} and min_samples {min_samples}")
                    logging.info(f"Eval Batch {i} . Mean trackML score {mean_track_ml_score}")

                    
                # for validation and logging
                total_loss_sum += loss.item()
                total_count_sum += 1



                
        else:
            for i, data in enumerate(validation_loader):
                _, hits, hits_seq_length, hits_masking, track_params, _ = data
                hits, hits_seq_length, hits_masking, track_params = hits.to(device), hits_seq_length.to(device), hits_masking.to(device), track_params.to(device)
                
                padding_mask = (hits == PAD_TOKEN).all(dim=2)
                track_params = torch.unsqueeze(track_params[~padding_mask], 0)

                pred = model(hits, padding_mask=padding_mask)
                pred = torch.unsqueeze(pred[~padding_mask], 0)
            
                loss, _ = loss_fn(pred, track_params)
                total_loss_sum += loss.item()
                total_count_sum += 1
   
    return total_loss_sum / total_count_sum, mean_dm_score, mean_track_ml_score

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
    for key, value in config.items():
        logging.info(f'{key}: {value}')
    logging.info(f'Output_dir: {output_dir}')


    early_stopping_epoch = config['training']['early_stopping']['patience']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Device: {device}')
    logging.info(f'Torch cuda version: {torch.version.cuda}')

    torch.manual_seed(37)  
    
    config_model_type = config['model']['type']
    batch_size = config['training']['batch_size']
    scaler = None
    train_loader = None
    valid_loader = None



    if config_model_type == 'flex_attention':
        logging.info(f"FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}")
        scaler = torch.GradScaler("cuda")

        data_folder = config['data']['data_dir']
        logging.info(f'Loading data from {data_folder} ...')

        sweep_train_fraction = wandb.config.train_fraction if hasattr(wandb.config, 'train_fraction') else 1.0

        train_loader, valid_loader = get_train_valid_dataloaders(config, batch_size=batch_size, train_fraction=sweep_train_fraction) 
    else:
        data_path = get_file_path(config['data']['data_dir'], config['data']['data_file'])
    
        logging.info(f'Loading data from {data_path} ...')
        
        hits_data, hits_data_seq_lengths, hits_masking, track_params_data, track_classes_data = load_trackml_data(data=data_path)
        dataset = HitsDataset(hits_data, hits_data_seq_lengths, hits_masking, track_params_data, track_classes_data)
        
        train_loader, valid_loader, _ = get_dataloaders(dataset,
                                                                train_frac=0.7,
                                                                valid_frac=0.15,
                                                                test_frac=0.15,
                                                                batch_size=batch_size, drop_last=True)
    logging.info(f'Data loaded.')

    # Set up the model, optimizer, and loss function
    model, optimizer,lr_scheduler, loss_fn, start_epoch = setup_training(config, device, train_loader)
    
    timer = StepTimer(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    memory_stats = get_system_memory_stats()
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

    # To check for spikes in validation loss
    min_epoch_for_spike_check = 5
    spike_factor = 1.2
    
    spike_count = 0

    for epoch in range(start_epoch, config['training']['total_epochs']):
        # Train the model
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn, scaler, lr_scheduler, device, config, wandb_logger, epoch, timer)

        # Evaluate using validation split
        val_loss, mean_dm_score, mean_track_ml_score = evaluate(model, valid_loader, loss_fn, device, config)
        
        # adjust learning rate based on validation loss
        lr_scheduler.step(val_loss)
        #lr_scheduler.step()
        if config['training']['scheduler']['verbose']:
            current_lr = optimizer.param_groups[0]['lr'] # get last lr
            logging.info(f"lr: {current_lr}")
            wandb_logger.log({"train/learning rate": current_lr}, step = (epoch + 1) *  len(train_loader) )

        # Print info to the cluster logging
        logging.info(f"Epoch: {epoch}\nVal loss: {val_loss:.10f}, Train loss: {train_loss:.10f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)


        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            wandb_logger.save_model(model, 'model_best.pth', optimizer, lr_scheduler, epoch, output_dir)
            logging.info(f"Checkpoint saved to output_dir. Best of run. Epoch: {epoch}")
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            wandb_logger.save_model(model, f'model_last.pth', optimizer, lr_scheduler, epoch, output_dir)
            logging.info(f"Checkpoint saved to output_dir. Last of run. Epoch: {epoch}")
            count += 1
        
        time_epoch_stats = timer.get_stats(reset=True) 
        wandb_logger.log({'train/train_loss' : train_loss, 'train/epoch' : epoch,
                        'train/validation loss' : val_loss, 'train/min_val_loss' : min_val_loss, 'train/mean_dm_score' : mean_dm_score, 
                        'train/mean_track_ml_score' : mean_track_ml_score,
                        **time_epoch_stats}, step = (epoch + 1) * len(train_loader))

        
        #  If the model's val loss spikes to more than spike_factor from best
        #    and we are past some min_epoch_for_spike_check
        if (spike_count >= min_epoch_for_spike_check) and (val_loss >= spike_factor * min_val_loss):
            logging.info(f"Val loss spiked: {val_loss:.4f} >= {spike_factor} * {min_val_loss:.4f}.")
            logging.info("Reloading 'model_best.pth' and lowering LR.")
            # Reload from best
            best_ckpt_path = os.path.join(output_dir, "model_best.pth")
            if os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, weights_only=False)
                #Get the current LR from  optimizer param groups
                current_lrs = [pg["lr"] for pg in optimizer.param_groups]

                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                lr_scheduler.load_state_dict(ckpt['scheduler_state'])
                
                # reduce LR manually
                #  Overwrite the LR in each param group using the *old* LR, scaled by 0.6
                for group_idx, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = current_lrs[group_idx] * 0.6

                logging.info(f"LR manually changed after spike. New LR: {optimizer.param_groups[0]['lr']}")
                spike_count = 0
            else:
                logging.warning("No model_best.pth found to reload from!")
                spike_count += 1
        else:
            # increment the spike count
            spike_count += 1
        
        if count >= early_stopping_epoch:
            logging.info("Early stopping triggered")
            break

    # Add final evaluate for the stats
    final_epoch_csv_path = os.path.join(output_dir, "final_epoch_preds.csv")
    if os.path.exists(final_epoch_csv_path):
        os.remove(final_epoch_csv_path)

    _, _, _ = evaluate(model, valid_loader, loss_fn, device, config, final_epoch_csv_path=final_epoch_csv_path)

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