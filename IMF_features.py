import pickle
import numpy as np
import mne
import pandas as pd
import os
from scipy.stats import kurtosis, skew
import emd
from antropy import hjorth_params
from multiprocessing import Pool, cpu_count

def extract_features_from_imf(imf):
    mob, com = hjorth_params(imf)
    return {
        'mean': np.mean(imf),
        'variance': np.var(imf),
        'skewness': skew(imf),
        'kurtosis': kurtosis(imf),
        'mean_frequency': np.mean(emd.spectra.frequency_transform(imf, 256, 'nht')[1]),
        'mobility': mob,
        'complexity': com
    }

def extract_IMF_features(subject_id, epoch_id, epochs, stages, num_imfs, ensemble_size, label):
    imf_opts = {'sd_thresh': 0.05}
    subject_features = []
    subject_imfs = []

    for epoch_data, sleep_stage in zip(epochs.get_data(copy=True), stages):  
        imfs = emd.sift.ensemble_sift(
            epoch_data.flatten(), max_imfs=num_imfs,
            nensembles=ensemble_size, nprocesses=1,
            ensemble_noise=1, imf_opts=imf_opts
        )
        for i in range(imfs.shape[1]):
            imf = imfs[:, i]
            imf_features = extract_features_from_imf(imf)
            subject_features.append({
                'subject ID': subject_id,
                'epoch ID': epoch_id,
                'imf index': i,
                'sleep stage': sleep_stage,
                'diagnose': label,
                'features': imf_features
            })
            subject_imfs.append({
                'subject ID': subject_id,
                'epoch ID': epoch_id,
                'imf index': i,
                'sleep stage': sleep_stage,
                'diagnose': label,
                'imf': imf
            })
    return subject_features, subject_imfs

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def process_file(args):
    file, folder, output_features, output_imfs, num_imfs, ensemble_size = args

    #incase a file exists
    subject_id = os.path.splitext(file)[0].split("_")[0]
    features_path = os.path.join(output_features, f"features_{subject_id}.pkl")
    if os.path.exists(features_path):
        print(f"Skipped (already processed): {subject_id}")
        return
    print(f"Processing: {file}") 

    data = np.load(os.path.join(folder, file), allow_pickle=True)
    all_features = []
    all_imfs = []

    for epoch in data:
        epoch_data = np.array(epoch["epoch_data"])
        events = np.array(epoch["event"]).reshape(1, 3)
        info = mne.create_info(ch_names=['F3'], sfreq=epoch["sfreq"], ch_types=['eeg'])

        if epoch_data.ndim == 2:
            epoch_data = epoch_data[np.newaxis, :, :]

        epochs = mne.EpochsArray(epoch_data, info, events=events)
        epoch_id = epoch["epoch_id"]
        sleep_stage = events[:, -1]
        label = epoch.get("label", None)

        features, imfs = extract_IMF_features(subject_id, epoch_id, epochs, sleep_stage, num_imfs, ensemble_size, label)
        all_features += features
        all_imfs += imfs

    save_to_pickle(all_features, features_path)
    save_to_pickle(all_imfs, os.path.join(output_imfs, f"imfs_{subject_id}.pkl"))
    print(f"Finished: {file}")

def get_IMF_features(folder, output_features, output_imfs, num_imfs, ensemble_size):
    os.makedirs(output_features, exist_ok=True)
    os.makedirs(output_imfs, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    args = [(f, folder, output_features, output_imfs, num_imfs, ensemble_size) for f in files]

    num_workers = min(cpu_count(), len(files))
    print(f"Using {num_workers} parallel workers")

    with Pool(num_workers) as pool:
        pool.map(process_file, args)

# === Run ===
npy_30sec_files = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/30sec_npys"
output_imfs = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/imf_features"
output_features = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/imf_set_features"

get_IMF_features(npy_30sec_files, output_features, output_imfs, num_imfs=10, ensemble_size=20)
