# %%
import mne
import pandas as pd
import os
import numpy as np
import seaborn as sns
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import convolve
from matplotlib.colors import ListedColormap
import pickle
from mne.time_frequency import psd_array_multitaper
import mne
mne.set_log_level('WARNING')

# %%
def compute_frequency_ratios(subject_id, epoch_id, epochs, stages, label):
    stage_mapping = {1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}
    ratios_data = []

    for epoch_data, sleep_stage in zip(epochs.get_data(copy=False), stages):
        psd, freqs = psd_array_multitaper( #[1,2,...30] $power in every frequency bin
            epoch_data, fmin=0.5, fmax=32.0, sfreq=epochs.info['sfreq'], normalization='full'
        )
        psd = np.mean(psd, axis=0) #averages the power values across all channels
        psd /= np.sum(psd) #normalizes the psd

        numerator1 = psd[(freqs >= 8.6) & (freqs < 19.3)]
        denominator1 = psd[(freqs >= 1.0) & (freqs < 10.9)]
        numerator2 = psd[(freqs >= 11.5) & (freqs < 20.3)]
        denominator2 = psd[(freqs >= 17.9) & (freqs < 31.5)]
        
         # Frequency bands, psd for every band range
        delta_band = psd[(freqs >= 0.5) & (freqs < 4.0)]
        theta_band = psd[(freqs >= 4.0) & (freqs < 8.0)]
        alpha_band = psd[(freqs >= 8.0) & (freqs < 12.0)]

        # Power in each band, the sum of the power in every band to cal total energy
        delta_power = np.sum(delta_band)
        theta_power = np.sum(theta_band)
        alpha_power = np.sum(alpha_band)

        # Ratios
        ratio_delta_theta = delta_power / theta_power if theta_power > 0 else np.nan
        ratio_delta_alpha = delta_power / alpha_power if alpha_power > 0 else np.nan
        ratio_theta_alpha = theta_power / alpha_power if alpha_power > 0 else np.nan

        ratio1 = np.sum(numerator1) / np.sum(denominator1)
        ratio2 = np.sum(numerator2) / np.sum(denominator2)

        ratios_data.append({
            'subject ID': subject_id,
            'epoch ID': epoch_id,
            'sleep stage': sleep_stage,
            'diagnose': label,
            'ratio1': ratio1,
            'ratio2': ratio2,
            'Delta Power': delta_power,
            'Theta Power': theta_power,
            'Alpha Power': alpha_power,
            'Delta/Theta': ratio_delta_theta,
            'Delta/Alpha': ratio_delta_alpha,
            'Theta/Alpha': ratio_theta_alpha,
        })
    return ratios_data

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:  
        pickle.dump(data, file)

# %%
def get_ratios_features(folder, output_features):
    os.makedirs(output_features, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            
            subject_id = os.path.splitext(file)[0].split("_")[0]
            output_path = os.path.join(output_features, f"ratios_features_{subject_id}.pkl")
            if os.path.exists(output_path):
                print(f"Skipped (already processed): {subject_id}")
                continue
                
            print(f"Processing: {file}") 
            
            all_epochs = []
            data = np.load(os.path.join(folder, file), allow_pickle=True)

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
                

                ratios_data = compute_frequency_ratios(subject_id, epoch_id, epochs, sleep_stage, label)
                all_epochs += ratios_data

            save_to_pickle(all_epochs, output_path)

npy_5sec_files = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/5sec_npys"
output_specteral = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/specteral_features"
get_ratios_features(npy_5sec_files, output_specteral) 

# %%
import numpy as np
import matplotlib.pyplot as plt

folder = r"S:\Research\baladat1\narcolepsy_ai\ho00322_lab\specteral_features"
all_data = []
for file in os.listdir(folder):
    if file.endswith(".pkl"):
        file_path = os.path.join(folder, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            all_data.extend(data)  
df = pd.DataFrame(all_data)

def smooth(data, window_len=50):
    if window_len < 3:
        return data
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]

def oplot2d_50(ax, ratio1, ratio2, A, label, subject_id):
    sratio1 = smooth(ratio1, window_len=50)
    sratio2 = smooth(ratio2, window_len=50)
    color_mapping = {1: 'red', 2: 'yellow', 3: 'green', 4: 'blue', 5: 'magenta'}
    colors = [color_mapping[stage] for stage in A]
    scatter = ax.scatter(np.log(sratio1), np.log(sratio2), c=colors, s=6, alpha=0.2)
    ax.set_xlabel('log(ratio1)')
    ax.set_ylabel('log(ratio2)')
    label_mapping = {0: 'Narcolepsy Patient', 1: 'Healthy Control', 2: 'Other Hypersomnia'}
    ax.set_title(f'Subject: {subject_id},\n {label_mapping[label]}', fontsize=10)
    return scatter

def plot_subjects_ratios(df, oplot2d_func, nrows=5, ncols=5, max_figures_per_label=20):
    subject_ids = df['subject ID'].unique()
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    label_counts = {0: 0, 1: 0, 2: 0}
    plot_index = 0
    for subject_id in subject_ids:
        subject_df = df[df['subject ID'] == subject_id]
        label = subject_df['diagnose'].iloc[0]
        if label_counts[label] < max_figures_per_label:
            A = subject_df['sleep stage'].values.tolist()
            ratio1 = subject_df['ratio1'].values
            ratio2 = subject_df['ratio2'].values
            ax = axs[plot_index // ncols, plot_index % ncols]
            oplot2d_func(ax, ratio1, ratio2, A, label, subject_id)
            label_counts[label] += 1
            plot_index += 1
        if plot_index >= nrows * ncols:
            break

    stage_labels = ['W', 'N1', 'N2', 'N3', 'REM']
    colors = ['red', 'yellow', 'green', 'blue', 'magenta']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in colors]
    fig.legend(handles, stage_labels, loc='lower center', ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

plot_subjects_ratios(df, oplot2d_50, nrows=5, ncols=5, max_figures_per_label=20)



