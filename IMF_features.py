# %%
import pickle
import numpy as np
import mne
import pandas as pd
import os
from scipy.stats import kurtosis
from scipy.stats import skew
import emd
from antropy import hjorth_params

# %%
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

def get_IMF_features(folder, output_features, output_imfs, num_imfs, ensemble_size):
    os.makedirs(output_features, exist_ok=True)
    os.makedirs(output_imfs, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file), allow_pickle=True)
            if len(data) == 0:
                continue

            #incase file exists
            subject_id = data[0]["subject_id"]
            features_path = os.path.join(output_features, f"features_{subject_id}.pkl")
            imfs_path = os.path.join(output_imfs, f"imfs_{subject_id}.pkl")
            if os.path.exists(features_path) and os.path.exists(imfs_path):
                print(f"Skipped (already processed): {subject_id}")
                continue

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

                features, imfs = extract_IMF_features(
                    subject_id, epoch_id, epochs, sleep_stage,
                    num_imfs=num_imfs, ensemble_size=ensemble_size, label=label
                )
                all_features += features
                all_imfs += imfs

            save_to_pickle(all_features, features_path)
            save_to_pickle(all_imfs, imfs_path)
            print(f"Processed: {subject_id}")

    
def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:  
        pickle.dump(data, file)


# %%
def get_IMF_features(folder, output_features, output_imfs, num_imfs, ensemble_size):
    os.makedirs(output_features, exist_ok=True)
    os.makedirs(output_imfs, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
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
                subject_id = epoch["subject_id"]
                epoch_id = epoch["epoch_id"]
                sleep_stage = events[:, -1]
                label = epoch.get("label", None)

                features, imfs = extract_IMF_features(
                    subject_id, epoch_id, epochs, sleep_stage,
                    num_imfs=10, ensemble_size=10, label=label
                )
                all_features += features
                all_imfs += imfs

            save_to_pickle(all_features, f"{output_features}/features_{subject_id}.pkl")
            save_to_pickle(all_imfs, f"{output_imfs}/imfs_{subject_id}.pkl")


# %%
# npy_30sec_files = "/Users/estreya/Desktop/Narcolepsy Project/project source code/results"
# output_imfs = r"/Users/estreya/Desktop/imfs_features"
# output_features = r"/Users/estreya/Desktop/imfs_set_features"

# get_IMF_features(npy_30sec_files, output_features, output_imfs, num_imfs=10, ensemble_size=10)


