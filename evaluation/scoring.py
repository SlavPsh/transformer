import numpy as np
import pandas as pd
import torch
import os


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

    # Calculate and return perfect match efficiency, LHC-style match efficiency, 
    # and double majority match efficiency

    n_perfect_match = sum(tracks["perfect_match"])
    n_double_majority = sum(tracks["double_majority"])
    n_lhc_match = sum(tracks["lhc_match"])

    perfect_ratio = n_perfect_match / n_particles if n_particles > 0 else 0
    double_majority_ratio = n_double_majority / n_particles if n_particles > 0 else 0
    lhc_ratio = n_lhc_match / n_clusters if n_clusters > 0 else 0

    return perfect_ratio, double_majority_ratio, lhc_ratio


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

def calculate_bined_scores(predicted_tracks, true_tracks, bin_ranges, pt_threshold=0.9):
    bin_params = list(bin_ranges.keys())

    predicted_tracks = predicted_tracks[predicted_tracks["pt"] > pt_threshold].copy()
    true_tracks = true_tracks[true_tracks["pt"] > pt_threshold].copy()
    true_tracks.rename(columns={'hit_id': 'nhits_truth'}, inplace=True)

    bins = {
        param: np.arange(bin_ranges[param]['min'], 
                         bin_ranges[param]['max'] + bin_ranges[param]['step'], 
                         bin_ranges[param]['step'])
        for param in bin_params
    }

    all_bin_scores = {}

    # Define reconstructable truth particles explicitly (≥3 hits & pt>threshold)
    reconstructable_pids = set(
        true_tracks[
            (true_tracks['nhits_truth'] >= 3) & 
            (true_tracks['pt'] > pt_threshold)
        ]['particle_id']
    )

    for param in bin_params:
        predicted_tracks[f'{param}_bin'] = pd.cut(predicted_tracks[param], bins=bins[param], right=False)
        true_tracks[f'{param}_bin'] = pd.cut(true_tracks[param], bins=bins[param], right=False)

        good_tracks = get_good_tracks(predicted_tracks)

        # Explicit marking of tracks matched to reconstructable particles
        predicted_tracks['is_reconstructable'] = predicted_tracks['major_particle_id'].isin(reconstructable_pids)
        predicted_tracks['is_good'] = predicted_tracks.index.isin(good_tracks.index)

        # Define explicitly reconstructed tracks ≥3 hits matched to reconstructable particles
        reco_predicted_tracks = predicted_tracks[
            (predicted_tracks['nhits'] >= 3) &
            (predicted_tracks['is_reconstructable'])
        ].copy()

        # Numerator: reconstructed tracks failing double majority criterion
        num_fake_DM = (~reco_predicted_tracks['is_good']).groupby(reco_predicted_tracks[f'{param}_bin'], observed = True).sum()

        # Denominator: all reconstructed tracks ≥3 hits matched to reconstructable particles
        denom_fake_DM = reco_predicted_tracks.groupby(f'{param}_bin', observed = True).size()

        # Calculate explicitly your DM fake rate
        fake_rate_DM = num_fake_DM / denom_fake_DM.replace(0, np.nan)

        # Efficiency calculation remains unchanged (as defined previously)
        # Numerator: reconstructable particles matched by ≥1 good track
        good_tracks_reco = good_tracks[good_tracks['major_particle_id'].isin(reconstructable_pids)]
        matched_pids = good_tracks_reco['major_particle_id'].unique()

        # Count reconstructable truth particles matched by good track per bin
        true_reco_grouped = true_tracks[(true_tracks['nhits_truth'] >= 3)].groupby(f'{param}_bin', observed = True)
        total_reco_true_particles_bin = true_reco_grouped['particle_id'].nunique()

        matched_reco_true_particles_bin = true_tracks[
            true_tracks['particle_id'].isin(matched_pids)
        ].groupby(f'{param}_bin', observed = True)['particle_id'].nunique()

        efficiency = matched_reco_true_particles_bin / total_reco_true_particles_bin.replace(0, np.nan)

        bin_scores = pd.DataFrame({
            'total_predicted_tracks': predicted_tracks.groupby(f'{param}_bin', observed = True).size(),
            'reco_predicted_tracks': denom_fake_DM,
            'num_fake_DM_tracks': num_fake_DM,
            'total_reco_true_particles': total_reco_true_particles_bin,
            'matched_reco_true_particles': matched_reco_true_particles_bin,
            'event_efficiency': efficiency,
            'event_fake_rate': fake_rate_DM
        }).reset_index()

        all_bin_scores[param] = bin_scores

    return all_bin_scores


def calculate_bined_scores_old(predicted_tracks, true_tracks, bin_ranges, pt_threshold = 0.9):
    # Extract bin_params from the keys of bin_ranges
    bin_params = list(bin_ranges.keys())

    predicted_tracks = predicted_tracks[predicted_tracks["pt"]>pt_threshold].copy()
    true_tracks = true_tracks[true_tracks["pt"]>pt_threshold].copy()

    true_tracks.rename(columns={'hit_id': 'nhits_truth'}, inplace=True)

    # Generate bin edges using numpy.arange
    # Create bins with -inf and inf for outer bins
    bins = {
        param: np.arange(bin_ranges[param]['min'], bin_ranges[param]['max'] + bin_ranges[param]['step'], bin_ranges[param]['step'])
        for param in bin_params
    }
    #bins = {param: np.arange(bin_ranges[param]['min'], bin_ranges[param]['max'] + bin_ranges[param]['step'], bin_ranges[param]['step']) for param in bin_params}

    # Initialize a dictionary to store bin scores for each parameter
    all_bin_scores = {}

    
    # Reconstructable truth particles (≥3 hits)
    reconstructable_particles = set(true_tracks[true_tracks['nhits_truth'] >= 3]['particle_id'].unique())

    for param in bin_params:
        # Create bins for the parameter
        predicted_tracks[f'{param}_bin'] = pd.cut(predicted_tracks[param], bins=bins[param], right=False)
        true_tracks[f'{param}_bin'] = pd.cut(true_tracks[param], bins=bins[param], right=False)

        good_tracks = get_good_tracks(predicted_tracks)

        # Efficiency Calculation:
        # Count how many reconstructable truth particles have at least one good reconstructed track
        good_tracks_reco = good_tracks[good_tracks['major_particle_id'].isin(reconstructable_particles)]
        particles_matched_to_good = good_tracks_reco.groupby('major_particle_id').size().index.unique()

        # Count total reconstructable truth particles per bin
        true_reco_grouped = true_tracks[true_tracks['nhits_truth'] >= 3].groupby(f'{param}_bin', observed=False)
        total_reco_true_particles_bin = true_reco_grouped['particle_id'].nunique()

        # Count reconstructable truth particles matched by at least one good track per bin
        good_tracks_reco_grouped = true_tracks[
            true_tracks['particle_id'].isin(particles_matched_to_good)
        ].groupby(f'{param}_bin', observed=False)
        matched_reco_true_particles_bin = good_tracks_reco_grouped['particle_id'].nunique()

        event_efficiency = matched_reco_true_particles_bin / total_reco_true_particles_bin.replace(0, np.nan)

        # Fake Rate Calculation:
        # Reconstructed tracks NOT matched to any reconstructable particle
        predicted_tracks['matched_to_reco'] = predicted_tracks['major_particle_id'].isin(reconstructable_particles)
        predicted_grouped = predicted_tracks.groupby(f'{param}_bin', observed=False)
        
        total_predicted_tracks = predicted_grouped.size()
        unmatched_predicted_tracks = predicted_grouped.apply(lambda df: (~df['matched_to_reco']).sum())

        event_fake_rate = unmatched_predicted_tracks / total_predicted_tracks.replace(0, np.nan)


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
     The 1st entry is pt, 2nd is eta, 3rd is particle_id, 4th is weight.
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
    truth['particle_id'] = truth['particle_id'].astype('int64')

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



def append_predictions_to_csv(preds_list, targets_list, out_data_list, cluster_list, input_list, csv_path, param_names=None):
    import torch
    import pandas as pd
    import os

    # Concatenate once on GPU, then move to CPU
    input_tensor = torch.cat(input_list, dim=0).float().detach().cpu()
    preds_tensor = torch.cat(preds_list, dim=0).float().detach().cpu()
    targets_tensor = torch.cat(targets_list, dim=0).float().detach().cpu()
    particle_id_tensor = torch.cat([out[:, 2] for out in out_data_list], dim=0).long().detach().cpu()
    cluster_id_tensor = torch.cat(cluster_list, dim=0).long().detach().cpu()
    pt_tensor = torch.cat([out[:, 0] for out in out_data_list], dim=0).float().detach().cpu()

    # batch_id creation with integer type
    batch_ids = torch.cat([
        torch.full((pred.shape[0],), idx, dtype=torch.long)
        for idx, pred in enumerate(preds_list)
    ]).cpu()

    # Convert tensors to numpy arrays
    input_np = input_tensor.numpy()
    preds_np = preds_tensor.numpy()
    targets_np = targets_tensor.numpy()
    cluster_np = cluster_id_tensor.numpy()
    particle_ids_np = particle_id_tensor.numpy()
    pt_np = pt_tensor.numpy()
    batch_ids_np = batch_ids.numpy()

    # Handle parameter names
    N, out_dim = preds_np.shape
    if (not param_names) or (len(param_names) < out_dim):
        param_names = [f"param_{i}" for i in range(out_dim)]

    # Build data dictionary
    data_dict = {
        "batch_id": batch_ids_np,
        "particle_id": particle_ids_np,
        "pt" : pt_np,
        "cluster_id" : cluster_np,
    }
    for i, pname in enumerate(param_names):
        data_dict[f"pred_{pname}"] = preds_np[:, i]
        data_dict[f"true_{pname}"] = targets_np[:, i]

    data_dict["x"] = input_np[:, 0]
    data_dict["y"] = input_np[:, 1]
    data_dict["z"] = input_np[:, 2]

    # Write to CSV 
    df_batch = pd.DataFrame(data_dict)

    file_exists = os.path.isfile(csv_path)
    df_batch.to_csv(csv_path, mode='a', header=not file_exists, index=False)

