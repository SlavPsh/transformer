import numpy as np
import pandas as pd
import torch


# Following two functions are directly taken from the official TrackML github repository:
# https://github.com/LAL/trackml-library/tree/master
def _analyze_tracks(truth, submission):
    """Compute the majority particle, hit counts, and weight for each track.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.

    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    event = pd.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)

    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weight = 0 
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight

    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pd.DataFrame.from_records(tracks, columns=cols)

def get_good_tracks(tracks, purity_threshold=0.5):
    """Get the good tracks from the tracks dataframe.

    Parameters
    ----------
    tracks : pandas dataframe containing reconstructed tracks and their associated particle IDs,
                together with the number of hits in the track and the number of hits in the particle,
                as well as the weight of the hits.
    purity_threshold : float, optional
        The purity threshold for a track to be considered good. The default is 0.5.

    Returns
    -------
    pandas dataframe
        A dataframe containing the good tracks.
    """
    # Calculate the purity of the tracks
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (purity_rec > purity_threshold) & (purity_maj > purity_threshold)

    return tracks[good_track]

def score_event(tracks):
    """Compute the TrackML event score for a single event.

    Parameters
    ----------
    tracks : pandas dataframe containing reconstructed tracks and their associated particle IDs,
                together with the number of hits in the track and the number of hits in the particle,
                as well as the weight of the hits.
    hit_truth_track_params : pandas dataframe containing the truth information of the track paramters
                for each particle. Default is None.
    """
    # Compute the total score
    good_tracks = get_good_tracks(tracks)
    total_score = good_tracks['major_weight'].sum()

    return total_score



def efficiency_scores(tracks, n_particles, predicted_count_thld=3):
    """
    Function to calculate the perfect match efficiency, double majority match
    efficiency and LHC-style efficiency of tracks. 
    Code adapted from https://github.com/gnn-tracking/gnn_tracking/blob/main/src/gnn_tracking/metrics/cluster_metrics.py
    """

    tracks['maj_frac'] = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    tracks['maj_pid_frac'] = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])

    tracks['valid'] = tracks['nhits'] >= predicted_count_thld

    tracks['perfect_match'] = (tracks['major_nhits'] == tracks['major_particle_nhits']) & (tracks['maj_frac'] > 0.99) & tracks['valid']
    tracks['double_majority'] = (tracks['maj_pid_frac'] > 0.5) & (tracks['maj_frac'] > 0.5) & tracks['valid']
    tracks['lhc_match'] = (tracks['maj_frac'] > 0.75) & tracks['valid']

    n_clusters = len(tracks['track_id'])

    n_perfect_match = sum(tracks["perfect_match"])
    n_double_majority = sum(tracks["double_majority"])
    n_lhc_match = sum(tracks["lhc_match"])

    # Calculate and return perfect match efficiency, LHC-style match efficiency, 
    # and double majority match efficiency
    return n_perfect_match/n_particles, n_double_majority/n_particles, n_lhc_match/n_clusters


def calc_score(pred_lbl, true_lbl):
    """
    Function for calculating the TrackML score and efficiency scores of REDVID data, based 
    on the predicted cluster labels pred_lbl and true particle IDs true_lbl from a single
    event. Every hit is given weight of 1.
    """
    truth_rows, pred_rows = [], []
    for ind, part in enumerate(true_lbl):
        truth_rows.append((ind, part[0].item(), 1))

    for ind, pred in enumerate(pred_lbl):
        pred_rows.append((ind, pred.item()))

    truth = pd.DataFrame(truth_rows)
    truth.columns = ['hit_id', 'particle_id', 'weight']
    submission = pd.DataFrame(pred_rows)
    submission.columns = ['hit_id', 'track_id']

    nr_particles = len(truth['particle_id'].unique().tolist())
    
    tracks = _analyze_tracks(truth, submission) 
    return score_event(tracks), efficiency_scores(tracks, nr_particles)

def calculate_bined_scores(predicted_tracks, true_tracks, bin_ranges):
    # Extract bin_params from the keys of bin_ranges
    bin_params = list(bin_ranges.keys())

    # Generate bin edges using numpy.arange
    # Create bins with -inf and inf for outer bins
    bins = {
        param: np.arange(bin_ranges[param]['min'], bin_ranges[param]['max'] + bin_ranges[param]['step'], bin_ranges[param]['step'])
        for param in bin_params
    }
    #bins = {param: np.arange(bin_ranges[param]['min'], bin_ranges[param]['max'] + bin_ranges[param]['step'], bin_ranges[param]['step']) for param in bin_params}

    # Initialize a dictionary to store bin scores for each parameter
    all_bin_scores = {}

    for param in bin_params:
        # Create bins for the parameter
        predicted_tracks[f'{param}_bin'] = pd.cut(predicted_tracks[param], bins=bins[param], right=False)
        true_tracks[f'{param}_bin'] = pd.cut(true_tracks[param], bins=bins[param], right=False)

        good_tracks = get_good_tracks(predicted_tracks)

        # Group by the bins
        predicted_grouped = predicted_tracks.groupby(f'{param}_bin', observed=False)
        true_grouped = true_tracks.groupby(f'{param}_bin', observed=False)
        good_grouped = good_tracks.groupby(f'{param}_bin', observed=False)

        # Calculate the total major_weight for all tracks in each bin
        total_major_weight = predicted_grouped['major_weight'].sum()
        total_true_weight = true_grouped['weight'].sum()

        # Calculate the total major_weight for tracks with 'good' set to 1 in each bin
        good_major_weight = good_grouped['major_weight'].sum()

        # Calculate the total count of major tracks in predicted_grouped
        total_predicted_count = predicted_grouped.size()

        # Calculate the total count of true tracks in true_grouped
        total_true_count = true_grouped.size()

        # Calculate the count of 'good' tracks in each bin
        good_predicted_count = good_grouped.size()

        event_efficiency = good_predicted_count / total_true_count
        event_fake_rate = (total_predicted_count - good_predicted_count) / total_predicted_count

        # Combine the results into a single DataFrame
        bin_scores = pd.DataFrame({
            'total_major_weight': total_major_weight,
            'good_major_weight': good_major_weight,
            'total_true_weight': total_true_weight,
            'total_predicted_count': total_predicted_count,
            'good_predicted_count': good_predicted_count,
            'total_true_count': total_true_count,
            'event_efficiency': event_efficiency,
            'event_fake_rate': event_fake_rate
        }).reset_index()

        # Store the bin scores in the dictionary
        all_bin_scores[param] = bin_scores

    return all_bin_scores

def calc_score_trackml(pred_lbl, true_lbl, pt_threshold=0.9):
    """
    Calculates the TrackML score and efficiency scores for a single event, 
    but only counting tracks with pt > pt_threshold for the efficiency
    (including the count of 'n_particles').

    Parameters
    ----------
    pred_lbl : array-like
        Predicted cluster labels for each hit (same order as 'true_lbl').
    true_lbl : array-like
        True info for each hit, shaped e.g. (N,4): [pt, eta, particle_id, weight]
        or your arrangement. The 1st entry is pt, 2nd is eta, 3rd is particle_id, 4th is weight.
    pt_threshold : float, optional
        Only tracks above this pt are included in efficiency calculations,
        including the count of n_particles.

    Returns
    -------
    event_score : float
        Overall event score (all hits) from `score_event(...)`.
    eff_scores  : object
        Efficiency-related metrics. Here only the subset with pt > pt_threshold is included.
    n_particles_cut : int
        Number of unique particle_ids with pt>pt_threshold (the denominator for efficiency).
    tracks      : pd.DataFrame
        Detailed table merging predicted cluster info with major_particle_id plus 'pt','eta'.
    true_tracks : pd.DataFrame
        Aggregated truth table (grouped by particle_id with columns [pt, eta, weight]).
    """
    # Build data rows from `true_lbl` => each element ~ [pt, eta, particle_id, weight]
    truth_rows = []
    for ind, part in enumerate(true_lbl):
        # part => [pt, eta, particle_id, weight]
        truth_rows.append((ind, part[0].item(), part[1].item(), part[2].item(), part[3].item()))
        
    # Build data rows from predicted cluster labels => one label per hit
    pred_rows = []
    for ind, pred in enumerate(pred_lbl):
        pred_rows.append((ind, pred.item()))
    
    # Construct dataframes
    truth = pd.DataFrame(truth_rows, columns=['hit_id', 'pt', 'eta', 'particle_id', 'weight'])
    submission = pd.DataFrame(pred_rows, columns=['hit_id', 'track_id'])
    
    # Merge predicted cluster IDs with the truth info
    tracks = _analyze_tracks(
        truth[['hit_id', 'particle_id', 'weight']], 
        submission
    )

    # Normalize total weight
    total_weight = truth['weight'].sum()
    truth['weight'] = truth['weight'] / total_weight

    # Compute "event_score" on all hits/tracks (unchanged logic)
    event_score = score_event(tracks)

    # Build 'true_tracks' with aggregated info by particle_id
    true_tracks = truth.groupby('particle_id', as_index=False).agg({
        'hit_id':  'count',
        'pt':      'first',
        'eta':     'first',
        'weight':  'sum'
    })

    # Attach 'pt','eta' from the major_particle_id to the `tracks` DataFrame
    tracks = tracks.merge(
        true_tracks[['particle_id','pt','eta']], 
        left_on='major_particle_id', 
        right_on='particle_id', 
        how='left'
    )

    # Filter out only high-pt tracks for efficiency
    highpt_tracks = tracks[tracks['pt'] > pt_threshold].copy()

    # Also count how many unique particles had pt>pt_threshold
    # => for the denominator in efficiency
    true_tracks_cut = true_tracks[true_tracks['pt'] > pt_threshold]
    n_particles_cut = len(true_tracks_cut['particle_id'].unique())

    # Compute efficiency on the subset of tracks with pt>pt_threshold
    eff_scores = efficiency_scores(highpt_tracks, n_particles_cut)

    return event_score, eff_scores, n_particles_cut, tracks, true_tracks



def calc_edge_efficiency(pred_lbl, true_lbl):
    """
    Function for calculating reconstucted edge efficiency
    Input: pred_lbl: predicted track labels, true_lbl: true particle IDs
    Output: edge_efficiency: reconstructed edge efficiency
    """
    truth_rows, pred_rows = [], []
    for ind, part in enumerate(true_lbl):
        truth_rows.append((ind, part[0].item(), part[1].item(), part[2].item(), part[3].item(), part[4].item(), part[5].item()))

    for ind, pred in enumerate(pred_lbl):
        pred_rows.append((ind, pred.item()))
    
    truth = pd.DataFrame(truth_rows)
    truth.columns = ['hit_id', 'particle_id', 'weight', 'theta', 'sin_phi', 'q', 'log_p']
    submission = pd.DataFrame(pred_rows)
    submission.columns = ['hit_id', 'track_id']

    # Create adjacency matrices for true and predicted edges
    # True adjacency matrix
    particle_ids = truth['particle_id'].values
    true_adj = (particle_ids[:, None] == particle_ids[None, :]) & (particle_ids[:, None] != 0)

    # Predicted adjacency matrix
    track_ids = submission['track_id'].values
    pred_adj = (track_ids[:, None] == track_ids[None, :]) & (track_ids[:, None] != 0)

    # Calculate overlap between true and predicted edges (upper triangular part only)
    upper_tri = torch.triu_indices(len(particle_ids), len(particle_ids), offset=1)
    true_edges = true_adj[upper_tri[0], upper_tri[1]]
    pred_edges = pred_adj[upper_tri[0], upper_tri[1]]
    
    # Count true and predicted edges
    overlap_edges = (true_edges & pred_edges).sum()
    true_edge_count = true_edges.sum()
    edge_efficiency = overlap_edges / true_edge_count if true_edge_count > 0 else 0

    return edge_efficiency


import os

def append_predictions_to_csv(preds, targets, batch_idx, csv_path, param_names=None):
    """
    Appends model predictions and corresponding targets to a CSV file.
    Each row corresponds to one 'hit' or sample in the batch.

    Args:
        preds       : Tensor of shape (N, out_dim)
        targets     : Tensor of shape (N, out_dim)
        batch_idx   : The current batch index (integer)
        csv_path    : Path to the CSV file to append
        param_names : Optional list of parameter names for each dimension
                      (like ['dx','dy','dz','dphi',...]) 
    """
    # Convert to CPU numpy for DataFrame creation
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    N, out_dim = preds_np.shape

    # If param_names not specified, auto-generate
    if (not param_names) or (len(param_names) < out_dim):
        param_names = [f"param_{i}" for i in range(out_dim)]

    # Build a dictionary for the DataFrame
    # Each dimension => "pred_{name}" and "true_{name}"
    data_dict = {
        "batch_idx": [batch_idx]*N  # repeated for each row in this batch
    }

    for i, pname in enumerate(param_names):
        data_dict[f"pred_{pname}"] = preds_np[:, i]
        data_dict[f"true_{pname}"] = targets_np[:, i]

    df_batch = pd.DataFrame(data_dict)

    # Append to CSV (no header if file exists)
    file_exists = os.path.isfile(csv_path)
    df_batch.to_csv(csv_path, mode='a', header=not file_exists, index=False)
