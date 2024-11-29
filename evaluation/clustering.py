import torch
import numpy as np
from hdbscan import HDBSCAN


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