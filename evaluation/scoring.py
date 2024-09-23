import numpy as np
import pandas as pd


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
            maj_weights = 0
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


def score_event(tracks):
    """Compute the TrackML event score for a single event.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    return tracks['major_weight'][good_track].sum()

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


def calc_score_trackml(pred_lbl, true_lbl):
    """
    Function for calculating the TrackML score and efficiency scores of TrackML data, based 
    on the predicted cluster labels pred_lbl and true particle IDs true_lbl from a single
    event. 
    """
    truth_rows, pred_rows = [], []
    for ind, part in enumerate(true_lbl):
        truth_rows.append((ind, part[0].item(), part[1].item()))

    for ind, pred in enumerate(pred_lbl):
        pred_rows.append((ind, pred.item()))

    truth = pd.DataFrame(truth_rows)
    truth.columns = ['hit_id', 'particle_id', 'weight']
    submission = pd.DataFrame(pred_rows)
    submission.columns = ['hit_id', 'track_id']

    nr_particles = len(truth['particle_id'].unique().tolist())

    tracks = _analyze_tracks(truth, submission) 
    return score_event(tracks), efficiency_scores(tracks, nr_particles)