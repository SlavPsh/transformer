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



def clustering_inception(pred_params, existing_cluster_ids, min_cl_size, min_samples, cluster_noise=False):
    '''
    Performs clustering separately within each existing cluster.
    
    Parameters:
    - pred_params: List of tensors, each containing predictions for an event.
    - existing_cluster_ids: List of tensors, each containing existing cluster IDs for an event.
    - min_cl_size: Minimum cluster size for HDBSCAN.
    - min_samples: Minimum samples per cluster for HDBSCAN.
    - cluster_noise: If True, re-cluster original noise points (existing label = -1).
    
    Returns:
    - cluster_labels: List of tensors with new cluster labels per event.
    '''
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    cluster_labels = []

    for event_preds, event_existing_clusters in zip(pred_params, existing_cluster_ids):
        event_preds_np = event_preds.cpu().float().numpy()
        existing_clusters_np = event_existing_clusters.cpu().numpy()

        unique_existing_labels = np.unique(existing_clusters_np)
        new_event_labels = -np.ones_like(existing_clusters_np, dtype=int)
        next_label = 0

        for existing_label in unique_existing_labels:
            if existing_label == -1 and not cluster_noise:
                continue  # Skip clustering original noise points unless explicitly specified

            indices = np.where(existing_clusters_np == existing_label)[0]
            sub_params = event_preds_np[indices]

            # Check if enough points for HDBSCAN
            if len(sub_params) < max(min_cl_size, min_samples):
                continue  # Too few points to cluster, remain labeled as noise

            sub_labels = clustering_algorithm.fit_predict(sub_params)

            # Assign new labels, excluding HDBSCAN-detected noise
            valid_sub_labels = np.unique(sub_labels[sub_labels != -1])
            for sub_label in valid_sub_labels:
                mask = (sub_labels == sub_label)
                new_event_labels[indices[mask]] = next_label
                next_label += 1

        cluster_labels.append(torch.from_numpy(new_event_labels).int())

    return cluster_labels
