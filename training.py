from data_processing.dataset import load_trackml_data

if __name__ == '__main__':
    DATA_PATH = '/data/atlas/users/spshenov/trackml_10to50tracks_40kevents.csv'
    MAX_NUM_HITS = 100
    hits_data, track_params_data, track_classes_data = load_trackml_data(data=DATA_PATH, max_num_hits=MAX_NUM_HITS)
    print(hits_data)
    print('finished loading the trackML data')