import torch
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import TransformerRegressor
from training import predict
from data_processing.dataset import HitsDataset, get_dataloaders
from data_processing.dataset import load_trackml_data
#from evaluation.plotting import plot_heatmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nr_hits', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_type', type=str, choices=['2d', 'linear', 'curved', 'trackml'])

    parser.add_argument('--nr_enc_layers', type=int, default=6) 
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--nr_heads', type=int, default=4) 
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_func = None
    in_size = 3
    out_size = 3
    params = []
    if args.data_type == '2d':
        data_func = load_linear_2d_data
        in_size = 2
        out_size = 1
        params = ['slope']
    elif args.data_type == 'linear':
        data_func = load_linear_3d_data
        params = ["theta", "sinphi", "cosphi"]
    elif args.data_type == 'curved':
        data_func = load_curved_3d_data
        params = ["radial_coeff", "pitch_coeff", "azimuthal_coeff"]
    elif args.data_type == 'trackml':
        data_func = load_trackml_data
        out_size = 5
        params = ["theta", "sinphi", "cosphi", "q", 'log_p']

    transformer = TransformerRegressor(num_encoder_layers=args.nr_enc_layers,
                                        d_model=args.embedding_size,
                                        n_head=args.nr_heads,
                                        input_size=in_size,
                                        output_size=out_size,
                                        dim_feedforward=args.hidden_dim,
                                        dropout=args.dropout)
    transformer = transformer.to(DEVICE)

    checkpoint = torch.load(args.model_name, map_location=torch.device('cpu'))
    transformer.load_state_dict(checkpoint['model_state_dict'])
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))


    hits_data, track_params_data, track_classes_data = data_func(data=args.data_path, max_num_hits=args.max_nr_hits)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                                train_frac=0.7,
                                                                valid_frac=0.15,
                                                                test_frac=0.15,
                                                                batch_size=1)
    print('data loaded')

    for cl_size in [3, 4, 5, 6, 7]:
        for min_sam in [2, 3, 4, 5, 6]:
            preds, score, perfect, double_maj, lhc = predict(transformer, test_loader, cl_size, min_sam, args.data_type)
            print(f'cluster size {cl_size}, min samples {min_sam}, score {score}', flush=True)
            print(perfect, double_maj, lhc, flush=True)

            if cl_size == 5 and min_sam == 2:
                preds = list(preds.values())
                #for param in params:
                #    plot_heatmap(preds, param, args.model_name)