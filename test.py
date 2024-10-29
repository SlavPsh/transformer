import torch
from model import TransformerRegressor, clustering
from data_processing.dataset import HitsDataset, get_dataloaders
from data_processing.dataset import load_trackml_data, PAD_TOKEN
from evaluation.scoring import calc_score_trackml, calculate_bined_scores
#from evaluation.plotting import plot_heatmap

# Import supporting tools
from utils.io_utils import load_config, setup_logging, unique_output_dir, copy_config_to_output, get_file_path
from utils.wandb_utils import WandbLogger
import argparse
import logging
from time import gmtime, strftime
from coolname import generate_slug
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def load_model(config, device):
    # model
    model = TransformerRegressor(
        num_encoder_layers = config['model']['num_encoder_layers'],
        d_model = config['model']['d_model'],
        n_head=config['model']['n_head'],
        input_size = config['model']['input_size'],
        output_size = config['model']['output_size'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        use_att_mask=config['model']['use_att_mask']
    ).to(device)

    if 'checkpoint_path' not in config['model'] or not config['model']['checkpoint_path']:
        logging.error('Checkpoint path must be provided for evaluation.')
    else:
        if device.type == 'cpu':
            checkpoint = torch.load(config['model']['checkpoint_path'], map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(config['model']['checkpoint_path'])

        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] + 1
        if 'att_mask_used' in checkpoint:
            model.set_use_att_mask(checkpoint['att_mask_used'])
            logging.info(f'Using attention mask is set to {checkpoint['att_mask_used']}')
        else:
            model.set_use_att_mask(False)
            logging.info(f'Using attention mask is set to False')

        
        logging.info(f'Loaded checkpoint from {config["model"]["checkpoint_path"]}')
        logging.info(f'Loaded model_state of epoch {epoch}. Ignoring optimizer_state. Starting evaluation from checkpoint.')

    model.eval()
    return model 

def test_main(model, test_loader, min_cl_size, min_samples, bin_ranges, wandb_logger=None):
    '''
    Evaluates the network on the test data. Returns the predictions and scores.
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.

    # Initialize a dictionary to store bin scores for all events
    combined_bin_scores = {param: [] for param in bin_ranges.keys()}

    for data in test_loader:
        # data is per event
        # Split the data for this event
        event_id, hits, hits_masking, track_params, track_labels = data
 
        # Make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, hits_masking, padding_mask)

        hits = torch.unsqueeze(hits[~padding_mask], 0)
        pred = torch.unsqueeze(pred[~padding_mask], 0)

        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)


        cluster_labels = clustering(pred, min_cl_size, min_samples)

        event_score, scores, nr_particles, predicted_tracks, true_tracks = calc_score_trackml(cluster_labels[0], track_labels[0])

        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        bin_scores = calculate_bined_scores(predicted_tracks, true_tracks, bin_ranges)
        for param, scores in bin_scores.items():
            combined_bin_scores[param].append(scores)
           

        if wandb_logger != None:
            metrics = {'test/event_id' : event_id[0],
                       'test/event score' : event_score, 
                       'test/num_hits_per_event' : len(hits[0]),
                       'test/num_particles_per_event' : nr_particles
                       }
            wandb_logger.log(metrics)

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)

    # Aggregate the bin scores across all events for each parameter
    aggregated_bin_scores = {}
    for param in bin_ranges.keys():
        all_bin_scores_df = pd.concat(combined_bin_scores[param])
        aggregated_bin_scores[param] = all_bin_scores_df.groupby(f'{param}_bin', observed=False).sum().reset_index()

    total_average_score = score/len(test_loader)

    wandb_logger.plot_binned_scores(aggregated_bin_scores, total_average_score)

    return predictions, total_average_score, perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)

def main(config_path):
        #Create unique run name
    run_name = generate_slug(3)+"_eval"
    # Load the configuration file
    config = load_config(config_path)
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
    logging.info(f'Loading config from {config_path} ')
    logging.info(f'Output_dir: {output_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    torch.manual_seed(37)  # for reproducibility
    data_path = get_file_path(config['data']['data_dir'], config['data']['data_file'])
    logging.info(f'Loading data from {data_path} ...')
    hits_data, hits_masking, track_params_data, track_particle_data = load_trackml_data(data=data_path)
    dataset = HitsDataset(device, hits_data, hits_masking, track_params_data, track_particle_data)
    _, _, test_loader = get_dataloaders(dataset,
                                        train_frac=0.7,
                                        valid_frac=0.15,
                                        test_frac=0.15,
                                        batch_size=64)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Data loaded")
    logging.info("Data loaded")

    model = load_model(config, device)  
    model.attach_wandb_logger(wandb_logger)

    logging.info("Started evaluation")

    cl_size = 5
    min_sam = 4
    bin_ranges = config['bin_ranges']
    preds, score, perfect, double_maj, lhc = test_main(model, test_loader, cl_size, min_sam, bin_ranges, wandb_logger)
    print(f'cluster size {cl_size}, min samples {min_sam}, TrackML score {score}', flush=True)
    logging.info(f'cluster size {cl_size}, min samples {min_sam}, TrackML score {score}')
    #print(perfect, double_maj, lhc, flush=True)

    wandb_logger.log({'test/cluster size' : cl_size, 'test/min sample size' : min_sam,'test/trackML score': score})

    if cl_size == 5 and min_sam == 3:
        preds = list(preds.values())
        #for param in params:
        #    plot_heatmap(preds, param, args.model_name)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a given config file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration TOML file.')
    
    # Parse arguments
    args = parser.parse_args()
    main(args.config_path)