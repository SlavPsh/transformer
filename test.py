import torch
from data_processing.dataset import HitsDataset, get_dataloaders
from data_processing.dataset import load_trackml_data, PAD_TOKEN
from evaluation.scoring import calc_score_trackml, calculate_bined_scores
from evaluation.clustering import clustering
#from evaluation.plotting import plot_heatmap

# Import supporting tools
from utils.io_utils import load_config, setup_logging, unique_output_dir, copy_config_to_output, get_file_path
from utils.wandb_utils import WandbLogger
import argparse
import logging
from coolname import generate_slug
import wandb
from data_processing.tensor_dataloader import get_test_dataloader
from utils.timing_utils import StepTimer


def load_model(config, device):

    config_model_type = config['model']['type']
    logging.info(f"Model type: {config_model_type}")
    
    #sweep_att_mask = wandb.config.use_att_mask if 'use_att_mask' in wandb.config else config_att_mask
    #sweep_flash_attention = wandb.config.use_flash_attention if 'use_flash_attention' in wandb.config else config_flash_attention

    if config_model_type == 'vanilla':
            from vanilla_model import TransformerRegressor

            model = TransformerRegressor(
                num_encoder_layers = config['model']['num_encoder_layers'],
                d_model = config['model']['d_model'],
                n_head=config['model']['n_head'],
                input_size = config['model']['input_size'],
                output_size = config['model']['output_size'],
                dim_feedforward=config['model']['dim_feedforward'],
                dropout=config['model']['dropout']
            ).to(device)
    elif config_model_type == 'flash_attention':
        from flash_model import TransformerRegressor
        print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

        model = TransformerRegressor(
            num_encoder_layers = config['model']['num_encoder_layers'],
            d_model = config['model']['d_model'],
            n_head=config['model']['n_head'],
            input_size = config['model']['input_size'],
            output_size = config['model']['output_size'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ).to(device)
    elif config_model_type == 'flex_attention':
        from custom_model import TransformerRegressor

        # For better performance, you can use:
        # flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

        model = TransformerRegressor(
            num_encoder_layers = config['model']['num_encoder_layers'],
            d_model = config['model']['d_model'],
            n_head=config['model']['n_head'],
            input_size = config['model']['input_size'],
            output_size = config['model']['output_size'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif config_model_type == 'vanilla_attn_scores':
        from vanilla_model_attn_scores import TransformerRegressor

        model = TransformerRegressor(
            num_encoder_layers = config['model']['num_encoder_layers'],
            d_model = config['model']['d_model'],
            n_head=config['model']['n_head'],
            input_size = config['model']['input_size'],
            output_size = config['model']['output_size'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ).to(device)


    if 'checkpoint_path' not in config['model'] or not config['model']['checkpoint_path']:
        logging.error('Checkpoint path must be provided for evaluation.')
    else:
        if device.type == 'cpu':
            checkpoint = torch.load(config['model']['checkpoint_path'], map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(config['model']['checkpoint_path'])

        logging.info(f"Checkpoint :  {checkpoint.keys()}")
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] + 1
        
        """
        # Logic to set the attention mask the same as the one used during training
        if 'att_mask_used' in checkpoint:
            model.set_use_att_mask(checkpoint['att_mask_used'])
            logging.info(f"Using attention mask is set to {checkpoint['att_mask_used']}")
        else:
            model.set_use_att_mask(False)
            logging.info(f"Using attention mask is set to False")
        """ 
        logging.info(f"Loaded checkpoint from {config['model']['checkpoint_path']}")
        logging.info(f"Loaded model_state of epoch {epoch}. Ignoring optimizer_state. Starting evaluation from checkpoint.")

    model.eval()
    return model 


def test_one_event(model, data_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    one_event_loader = get_test_dataloader(data_folder, batch_size=1)
    for i,  (data_tensor, length_tensor) in enumerate(one_event_loader):
        in_data_tensor_cpu = data_tensor[..., :3]
        out_data_tensor_cpu =  torch.cat(( data_tensor[..., 9:11],data_tensor[..., 12:14]), dim=-1)
        in_data_tensor = in_data_tensor_cpu.to(device)
        out_data_tensor = out_data_tensor_cpu.to(device)
        length_tensor = length_tensor.to(device)
        

        from custom_model import generate_padding_mask
        flex_padding_mask = generate_padding_mask(length_tensor)

        with torch.amp.autocast('cuda'):
            pred = model(in_data_tensor, f'test_one_{i}', flex_padding_mask) 

        pred_list = [pred[i, :length_tensor[i], :] for i in range(len(length_tensor))]
        out_data_tensor = [out_data_tensor[i, :length_tensor[i], :] for i in range(len(length_tensor))]
        cluster_labels_list = clustering(pred_list, 5, 3)
        for cluster_labels, track_labels in zip(cluster_labels_list, out_data_tensor):
            event_score, scores, nr_particles, predicted_tracks, true_tracks = calc_score_trackml(cluster_labels, track_labels, pt_threshold=0.9)
            logging.info(f"TrackML score: {event_score}")
            logging.info(f"Eff Scores: {scores}")
        
        break


def test_main(model, test_loader, min_cl_size, min_samples, bin_ranges, device, config, wandb_logger=None, timer=None):
    '''
    Evaluates the network on the test data. Returns the predictions and scores.
    '''
    # Get the network in evaluation mode
    config_model_type = config['model']['type']
    torch.set_grad_enabled(False)
    model.eval()
    pred_list = []
    track_labels_list = []
    scores_list = []
    perfects, doubles, lhcs =  0., 0., 0.


    # Initialize a dictionary to store bin scores for all events
    counter = 0
    combined_bin_scores = {param: [] for param in bin_ranges.keys()}
    
    if config_model_type == 'flex_attention':
        for i,  (data_tensor, length_tensor) in enumerate(test_loader):
            
            in_data_tensor_cpu = data_tensor[..., :3]
            #pd.DataFrame(truth_rows, columns=['hit_id',  THESE COLUMNS :  'pt', 'eta', 'particle_id', 'weight'])
            out_data_tensor_cpu =  torch.cat(( data_tensor[..., 9:11],data_tensor[..., 12:14]), dim=-1)
            
            #out_data_tensor_cpu = torch.cat(( data_tensor[..., 8:10],data_tensor[..., 11:13]), dim=-1)
            #cluster_tensor_cpu = data_tensor[..., 13:].squeeze(-1)
            
            in_data_tensor = in_data_tensor_cpu.to(device)
            out_data_tensor = out_data_tensor_cpu.to(device)
            #cluster_tensor = cluster_tensor_cpu.to(device)

            length_tensor = length_tensor.to(device)
            
            from custom_model import generate_padding_mask

            flex_padding_mask = generate_padding_mask(length_tensor)
                
            with torch.amp.autocast('cuda'):
                pred = model(in_data_tensor, f'test_{i}', flex_padding_mask, timer)

            if timer:
                timer.start('unmasking')
            pred_list = [pred[i, :length_tensor[i], :] for i in range(len(length_tensor))]
            out_data_tensor = [out_data_tensor[i, :length_tensor[i], :] for i in range(len(length_tensor))]
            if timer:
                timer.stop()
            
            if timer:
                timer.start('clustering')

            """

            cluster_labels_list = []
            for evt_pred in pred_list:
                # evt_pred.shape => (N, 6)
                # The 6th dimension (evt_pred[:, 5]) is predicted pT.

                # 1) Identify hits with pT>0.9
                mask = (evt_pred[:, 5] > 0.9)
                if not mask.any():
                    # If no hits exceed 0.9 => entire event = -1
                    cluster_labels = -1 * torch.ones(
                        evt_pred.size(0),
                        dtype=torch.long,
                        device=evt_pred.device
                    )
                else:
                    # 2) Take only subset of features (e.g. first 5 if your clustering uses them)
                    subset_pred = evt_pred[mask, :5]

                    # 3) Run your existing clustering function on this subset
                    #    If your clustering(...) expects a list of events, pass [subset_pred]
                    #    which returns a list of cluster labels (just 1 entry).
                    sub_labels_list = clustering([subset_pred], min_cl_size, min_samples)
                    sub_labels = sub_labels_list[0]  # shape (#hits_above_threshold,)

                    # 4) Map subset labels back to full event => -1 for excluded hits
                    cluster_labels = -1 * torch.ones(
                        evt_pred.size(0),
                        dtype=torch.long,
                        device=evt_pred.device
                    )
                    sub_labels = sub_labels.to(evt_pred.device) 
                    sub_labels = sub_labels.to(torch.long)
                    cluster_labels[mask] = sub_labels

                cluster_labels_list.append(cluster_labels)
            """

            cluster_labels_list = clustering(pred_list, min_cl_size, min_samples)
            if timer:
                timer.stop()

            for cluster_labels, track_labels in zip(cluster_labels_list, out_data_tensor):
                event_score, scores, nr_particles, predicted_tracks, true_tracks = calc_score_trackml(cluster_labels, track_labels, pt_threshold=0.9)
                scores_list.append(event_score)
                if wandb_logger != None:
                    metrics = {'batch/event score' : event_score, 
                            'batch/num_hits_per_event' : len(cluster_labels),
                            'batch/num_particles_per_event' : nr_particles,
                            'batch/event_double_maj_score' : scores[1]
                            }
                    wandb_logger.log(metrics)


            if i % 100 == 0:
                logging.info(f"Processed {i} batches. Last score: {scores_list[-1]} Mean score so far {sum(scores_list) / len(scores_list)}")


    else:
        for data in test_loader:
            # data is per event (becasue batch_size = 1)
            # Split the data for this event
            event_id, hits, hits_seq_length, hits_masking, track_params, track_labels = data
            hits, hits_seq_length, hits_masking, track_params, track_labels = hits.to(device), hits_seq_length.to(device), hits_masking.to(device), track_params.to(device), track_labels.to(device)
    
            # Make prediction
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            attn_scores = None

            if config_model_type == 'vanilla':
                pred = model(hits, padding_mask=padding_mask)
                pred = torch.unsqueeze(pred[~padding_mask], 0)
            elif config_model_type == 'vanilla_attn_scores':
                pred, attn_scores = model(hits, padding_mask=padding_mask)
                pred = torch.unsqueeze(pred[~padding_mask], 0)

                hits_to_save = hits.detach().cpu()
                track_labels_to_save = track_labels.detach().cpu()
                padding_to_save = padding_mask.detach().cpu()
                attn_scores_to_save = attn_scores.detach().cpu()
                torch.save(hits_to_save, f"/projects/0/nisei0750/slava/data/attn_scores_200_500/hits_{event_id}.pt")
                torch.save(track_labels_to_save, f"/projects/0/nisei0750/slava/data/attn_scores_200_500/track_labels_{event_id}.pt")
                torch.save(padding_to_save, f"/projects/0/nisei0750/slava/data/attn_scores_200_500/padding_{event_id}.pt")
                torch.save(attn_scores_to_save, f"/projects/0/nisei0750/slava/data/attn_scores_200_500/attn_scores_{event_id}.pt")
        
            hits = torch.unsqueeze(hits[~padding_mask], 0)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)
            track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)
        

        #bin_scores = calculate_bined_scores(predicted_tracks, true_tracks, bin_ranges)
        #for param, scores in bin_scores.items():
        #    combined_bin_scores[param].append(scores)
            

    #wandb_logger.plot_binned_scores(combined_bin_scores, total_average_score)
    total_average_score = sum(scores_list) / len(scores_list)

    return total_average_score

def main(config_path):

    
    # Load the configuration file
    config = load_config(config_path)


    if 'checkpoint_path' not in config['model'] or not config['model']['checkpoint_path']:
        print('Checkpoint path must be provided for evaluation.')
    else:
        #Create unique run name from the checkpoint path
        import re
        match = re.search(r'\d{8}_\d{6}_([a-zA-Z0-9-]+)_train', config['model']['checkpoint_path'])
        if match:
            run_name = match.group(1) + "_" + generate_slug(2) + "_eval"
        else:
            run_name = generate_slug(3) + "_eval"

    # Create the output directory
    output_dir = unique_output_dir(config, run_name) # with time stamp and run name
    copy_config_to_output(config_path, output_dir)
    # Set up logging in the output directory
    setup_logging(config, output_dir, job="evaluation")

    # Set up wandb
    wandb_logger = WandbLogger(config=config["wandb"],
                                output_dir=output_dir,
                                run_name=run_name,
                                job_type="evaluation")
    wandb_logger.initialize()
    logging.info(f"Loading config from {config_path} ")
    for key, value in config.items():
        logging.info(f'{key}: {value}')
    logging.info(f"Description: {config['experiment']['description']}")
    logging.info(f"Output_dir: {output_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    logging.info(f" ==== Model : ====")
    logging.info(config['model'])

    """
    torch.manual_seed(37)  # for reproducibility
    data_path = get_file_path(config['data']['data_dir'], config['data']['data_file'])
    logging.info(f"Loading data from {data_path} ...")
    hits_data, hits_data_seq_lengths, hits_masking, track_params_data, track_particle_data = load_trackml_data(data=data_path)
    dataset = HitsDataset(hits_data, hits_data_seq_lengths, hits_masking, track_params_data, track_particle_data)
    # Test loader has batch size 1 in defintion
    _, _, test_loader = get_dataloaders(dataset,
                                        train_frac=0.7,
                                        valid_frac=0.15,
                                        test_frac=0.15,
                                        batch_size=1)

    """
    data_folder = config['data']['data_dir']
    logging.info(f'Loading data from {data_folder} ...')
    batch_size = config['training']['batch_size']
    test_loader = get_test_dataloader(data_folder, batch_size=batch_size) 

    logging.info("Data loaded")

    model = load_model(config, device)  
    #model.attach_wandb_logger(wandb_logger)
    timer = StepTimer(device)

    logging.info("Started evaluation")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")

    cl_size = wandb.config.min_cl_size if 'min_cl_size' in wandb.config else 5
    min_sam = wandb.config.min_samples if 'min_samples' in wandb.config else 3
    bin_ranges = config['bin_ranges']

    avg_score = test_main(model, test_loader, cl_size, min_sam, bin_ranges, device, config, wandb_logger, timer)
    print(f'cluster size {cl_size}, min samples {min_sam}, TrackML score {avg_score}', flush=True)
    logging.info(f'cluster size {cl_size}, min samples {min_sam}, TrackML score {avg_score}')
    #print(perfect, double_maj, lhc, flush=True)
    timer_stats = timer.get_stats(reset=True)
    wandb_logger.log({'total/cluster size' : cl_size, 'total/min sample size' : min_sam,'total/trackML score': avg_score, **timer_stats})

    wandb_logger.alert("Finished test", "Finished test")
    
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a given config file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration TOML file.')
    
    # Parse arguments
    args = parser.parse_args()
    full_config = load_config(args.config_path)
    sweep_config = full_config['sweep']

    # Initialize the sweep and start the sweep agent
    
    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=lambda: main(args.config_path))