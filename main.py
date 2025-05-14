# %%
import numpy as np
import mne
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/estreya/Desktop/Narcolepsy Project/codes')
from get_npy_files import generate_npy
sys.path.append('/Users/estreya/Desktop/')
from spectral_features import get_ratios_features
from IMF_features import get_IMF_features
from pac_features import get_pac_features

# %% [markdown]
# Generate npy files..
# 
# by converting 8 hours recording into epochs of either 30 or 3 seconds

# %%
# data_dirs = [r"/Users/estreya/Desktop/Narcolepsy Project/subject_recording"]
# output_dir = r"/Users/estreya/Desktop/Narcolepsy Project/project source code/results"

# edf_files = generate_npy(data_dirs, output_dir,chunk_duration=5, max_time=5.0)

# %%
# import numpy as np
# import matplotlib.pyplot as plt

# npy_file_path = "/Users/estreya/Desktop/Narcolepsy Project/project source code/test results/chc001_epochs.npy"
# data = np.load(npy_file_path, allow_pickle=True)
# epoch = data[400]

# epoch_data = np.array(epoch['epoch_data'])  # Convert to NumPy array
# subject_id = epoch['subject_id']
# epoch_id = epoch['epoch_id']
# sampling_freq = epoch['sfreq']

# std_dev = np.std(epoch_data, axis=1)  # Compute std for each channel

# time = np.arange(epoch_data.shape[1]) / sampling_freq

# # Plot the EEG signal
# plt.figure(figsize=(10, 4))
# plt.plot(time, epoch_data[0], label="EEG Signal")  # Assuming 1 channel, otherwise change indexing
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (V)")
# plt.title(f"Epoch {epoch_id} - Subject {subject_id} (Std Dev: {std_dev[0]:.2e} V)")
# plt.legend()
# plt.show()

# # Print noise statistics
# print(f"Epoch {epoch_id} - Subject {subject_id}")
# print(f"Signal Standard Deviation: {std_dev[0]:.2e} V")
# print(f"Max Amplitude: {np.max(epoch_data[0]):.2e} V")
# print(f"Min Amplitude: {np.min(epoch_data[0]):.2e} V")


# npy_file_path = "/Users/estreya/Desktop/Narcolepsy Project/project source code/results/chc001_epochs.npy"

# data = np.load(npy_file_path, allow_pickle=True)

# print(f"Data Type: {type(data)}")
# print(f"Total Epochs Loaded: {len(data)}\n")

# for i, epoch in enumerate(data[:5]): 
#     print(f"Epoch {i+1}:")
#     print(f"  Subject ID: {epoch['subject_id']}")
#     print(f"  Epoch ID: {epoch['epoch_id']}")
#     print(f"  Sampling Frequency: {epoch['sfreq']}")
#     print(f"  Event: {epoch['event']}")
#     print(f"  Label: {epoch['label']}")
#     print(f"  Epoch Data Shape: {np.array(epoch['epoch_data']).shape}\n")


# %% [markdown]
# Extract features

# %%
npy_30sec_files = "/Users/estreya/Desktop/Narcolepsy Project/project source code/results"
npy_5sec_files = "/Users/estreya/Desktop/Narcolepsy Project/project source code/results"

output_imfs = r"/Users/estreya/Desktop/imfs_features"
output_features = r"/Users/estreya/Desktop/imfs_set_features"
output_specteral = r"/Users/estreya/Desktop/ratios_features"
output_pac = r"/Users/estreya/Desktop/pac_features"

# get_IMF_features(npy_30sec_files, output_features, output_imfs, num_imfs=10, ensemble_size=10)
# get_pac_features(npy_30sec_files, output_pac)
# get_ratios_features(npy_5sec_files, output_specteral) 

# %%
# import pickle
# import pandas as pd

# with open("/Users/estreya/Desktop/ratios_features/ratios_features_chc001.pkl", "rb") as f:
#     data = pickle.load(f)

# df = pd.DataFrame(data)

# print(df.head(10))
# print(df.columns)

# plot_subjects_ratios(df, oplot2d_50)


