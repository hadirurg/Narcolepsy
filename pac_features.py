# %%
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import os
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort
import pickle

# %%


def extract_PAC_features(subject_id, epoch_id, epoch, stage, label, sf , pac_object):
    print(epoch.get_data().shape)
    signal = epoch.get_data()[0][0]
    xpac = pac_object.filterfit(sf, signal[np.newaxis, :])
    print(signal.shape)
    all_pac = []
    all_pac.append({
                'subject ID': subject_id,
                'epoch ID': epoch_id,
                'sleep stage': stage,
                'features': xpac,
                'diagnose': label
            })
    return all_pac

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:  
        pickle.dump(data, file)
    

# %%
def get_pac_features(folder, output_features, pac_object):
    os.makedirs(output_features, exist_ok=True)
    all_epochs = []

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            subject_id = os.path.splitext(file)[0].split("_")[0]
            pac_features_path = os.path.join(output_features, f"pac_features_{subject_id}.pkl")

            if os.path.exists(pac_features_path):
                print(f"Skipped (already processed): {subject_id}")
                return

            print(f"Processing: {file}") 
            data = np.load(os.path.join(folder, file), allow_pickle=True)

            for epoch in data:
                epoch_data = np.array(epoch["epoch_data"])
                events = np.array(epoch["event"]).reshape(1, 3)
                sfreq= epoch["sfreq"]
                info = mne.create_info(ch_names=['F3'], sfreq=sfreq, ch_types=['eeg'])
                if epoch_data.ndim == 2:
                    epoch_data = epoch_data[np.newaxis, :, :]

                epochs = mne.EpochsArray(epoch_data, info, events=events)
                subject_id = epoch["subject_id"]
                epoch_id = epoch["epoch_id"]
                sleep_stage = events[:, -1]
                label = epoch.get("label", None)
                
    
                pac_data = extract_PAC_features(subject_id, epoch_id, epochs, sleep_stage, label,  sfreq , pac_object)
                all_epochs += pac_data

            save_to_pickle(all_epochs, pac_features_path)

pac_object = Pac(idpac=(2, 0, 4), f_pha=(0.5, 20, 1, 0.5), f_amp=(20, 40, 5, 1)) 
npy_30sec_files = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/30sec_npys"
output_pac = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/pac_features"

get_pac_features(npy_30sec_files, output_pac, pac_object)


