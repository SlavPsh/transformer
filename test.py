import torch
from model import TransformerRegressor
from data_processing.dataset import HitsDataset, get_dataloaders
from data_processing.dataset import load_trackml_data, PAD_TOKEN
from evaluation.scoring import calc_score_trackml, calculate_bined_scores
from training import clustering
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
        dropout=config['model']['dropout']
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
        logging.info(f'Loaded checkpoint from {config["model"]["checkpoint_path"]}')
        logging.info(f'Loaded model_state of epoch {epoch}. Ignoring optimizer_state. Starting evaluation from checkpoint.')

    model.eval()
    return model 

def predict(model, test_loader, min_cl_size, min_samples, wandb_logger=None):
    '''
    Evaluates the network on the test data. Returns the predictions and scores.
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.
    # Define the min, max, and step for each parameter
    # MOve this to the config file
    bin_ranges = {
        'log_p': {'min': -3, 'max': 5, 'step': 0.5},
        'theta': {'min': 0, 'max': np.pi, 'step': np.pi/10},
        'q': {'min': -2, 'max': 1, 'step': 1},
        'sin_phi': {'min': -1, 'max': 1, 'step': 0.2}
    }

    # Initialize a dictionary to store bin scores for all events
    combined_bin_scores = {param: [] for param in bin_ranges.keys()}

    for data in test_loader:
        # data is per event
        # Split the data for this event
        event_id, hits, track_params, track_labels = data
 
        # Make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        hits = torch.unsqueeze(hits[~padding_mask], 0)
        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        
        # For evaluating the clustering performance on the (noisy) ground truth
        # noise = np.random.laplace(0, 0.05, size=(track_params.shape[0], track_params.shape[1], track_params.shape[2]))
        # track_params += noise
        # cluster_labels = clustering(track_params, min_cl_size, min_samples)


        cluster_labels = clustering(pred, min_cl_size, min_samples)

        event_score, scores, nr_particles, event_tracks = calc_score_trackml(cluster_labels[0], track_labels[0])

        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        # ToDo : create bins on log_p , go through the bins and calculate maj_weight of each track that falls in the bin
        # and maj_weight of each track that falls in the bin and has a 'good' set to 1
        # Eventually move the bins to the config file
        bin_scores = calculate_bined_scores(event_tracks, bin_ranges)
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

    # Plot the percentage of good_major_weight over total_major_weight per bin and log to wandb
    for param, df in aggregated_bin_scores.items():
        df['percentage_good_major_weight'] = (df['good_major_weight'] / df['total_major_weight']) * 100
        plt.figure()
        x = df[f'{param}_bin'].astype(str)
        y = df['percentage_good_major_weight']
        plt.plot(x, y, marker='o', color='black')
        plt.fill_between(x, y, 0, where=(y >= 0), facecolor='blue', alpha=0.8)
        plt.fill_between(x, y, y.max(), where=(y >= 0), facecolor='red', alpha=0.3)
        plt.ylim(90, 100)  # Set y-axis range from 60% to 100%
        plt.title(f'Percentage of Good Major Weight vs Total Major Weight for {param}')
        plt.xlabel(f'{param} Bins')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if wandb_logger is not None:
            wandb_logger.log({f'aggregated_bin_scores_{param}': wandb.Image(plt)})
        plt.close()

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)

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
    logging.info(f'Loading config from {config_path} ')
    logging.info(f'Output_dir: {output_dir}')
    # Set up wandb
    wandb_logger = WandbLogger(config=config["wandb"],
                                output_dir=output_dir,
                                run_name=run_name,
                                job_type="evaluation")
    wandb_logger.initialize()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    torch.manual_seed(37)  # for reproducibility
    data_path = get_file_path(config['data']['data_dir'], config['data']['data_file'])
    logging.info(f'Loading data from {data_path} ...')
    hits_data, track_params_data, track_particle_data = load_trackml_data(data=data_path)
    dataset = HitsDataset(device, hits_data, track_params_data, track_particle_data)
    _, _, test_loader = get_dataloaders(dataset,
                                        train_frac=0.7,
                                        valid_frac=0.15,
                                        test_frac=0.15,
                                        batch_size=64)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Data loaded")
    logging.info("Data loaded")

    model = load_model(config, device)  

    logging.info("Started evaluation")

    cl_size = 5
    min_sam = 4
    preds, score, perfect, double_maj, lhc = predict(model, test_loader, cl_size, min_sam, wandb_logger)
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