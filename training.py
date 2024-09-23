from data_processing.dataset import load_trackml_data
import torch
import torch.nn as nn
import numpy as np
from hdbscan import HDBSCAN

from model import TransformerRegressor, save_model
from data_processing.dataset import  PAD_TOKEN, get_dataloaders
from evaluation.scoring import calc_score, calc_score_trackml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    DATA_PATH = '/data/atlas/users/spshenov/trackml_10to50tracks_40kevents.csv'
    MAX_NUM_HITS = 100
    hits_data, track_params_data, track_classes_data = load_trackml_data(data=DATA_PATH, max_num_hits=MAX_NUM_HITS)
    print(DEVICE)
    print('finished loading the trackML data')