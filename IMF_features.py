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

def extract_IMF_features(subject_id, epoch_id, epochs, stages, num_imfs, ensemble_size, label):
    imf_opts = {'sd_thresh': 0.05} #when to stop sifting
    subject_features = []
    subject_imfs = []
    
    for epoch_data, sleep_stage in zip(epochs.get_data(), stages):
        imfs = emd.sift.ensemble_sift(epoch_data.flatten(), max_imfs=num_imfs, nensembles=ensemble_size, nprocesses=-1, ensemble_noise=1, imf_opts=imf_opts)
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
            #print(f"saved {epoch_id}")
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


