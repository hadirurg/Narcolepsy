import numpy as np
import mne
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_Raw(edf_path):
    raw_train = mne.io.read_raw_edf(
       edf_path,
       stim_channel="Event marker",
       infer_types=True,
       preload=True,
       verbose="error",
    )
    raw_data = mne.io.read_raw_edf(edf_path, stim_channel="Event marker", infer_types=True, preload=True, verbose="error",)
    raw_data.pick_channels(['F3']) 
    return raw_data

def get_annotation(input_path, output_path, duration=30):
    annotations = []
    onset = 0
    with open(input_path, 'r') as file:
        for line in file:
            description = line.strip()
            annotations.append({
                "onset": onset,
                "duration": duration,
                "description": description
            })
            onset += duration
    df = pd.DataFrame(annotations)
    df.to_csv(output_path, index=False)
    df = pd.read_csv(output_path)
    annot_train = mne.Annotations(onset=df['onset'].values, duration=df['duration'].values, description=df['description'].values)
    return annot_train

event_id = {"wake": 1, "N1": 2, "N2": 3, "N3": 4, "N4": 4, "REM": 5}
def data_preprocess(raw_eeg, data_annotate,chunk_duration):

    # 1. apply bandpass filter
    raw_eeg.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin')
    raw_eeg.set_annotations(data_annotate, emit_warning=False)

    # 2. segment into epochs and create events
    annotation_desc_2_event_id = {"wake": 1, "N1": 2, "N2": 3, "N3": 4, "N4": 4, "REM": 5}
    events_train, _ = mne.events_from_annotations(
        raw_eeg, event_id=annotation_desc_2_event_id, chunk_duration=chunk_duration
    )
    return events_train

def get_epochs(raw_data, events, max_time):
    tmax = max_time - 1.0 / raw_data.info["sfreq"]
    epochs = mne.Epochs(
        raw=raw_data,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True
    )
    print(f"Remaining epochs: {len(epochs)}")
    del raw_data
    return epochs


def generate_npy(data_dirs, output_dir, chunk_duration, max_time):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels_path = r"/user/HS402/ho00322/labels.xlsx"
    labels_df = pd.read_excel(labels_path, header=None)
    label_dict = dict(zip(labels_df[0].astype(str), labels_df[1]))

    edf_files = []
    annotation_files_dict = {}

    for data_dir in data_dirs:
        edf_files.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.edf')])
        annotation_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.eannot')]
        annotation_files_dict.update({os.path.splitext(os.path.basename(f))[0]: f for f in annotation_files})

    for edf_path in edf_files:
        edf_filename = os.path.splitext(os.path.basename(edf_path))[0]
        subject_id = os.path.basename(edf_path).split('-')[0]
        output_path = os.path.join(output_dir, f"{subject_id}_epochs.npy")

        if os.path.exists(output_path):
            print(f"Skipped (already exists): {subject_id}_epochs.npy")
            continue

        if edf_filename in annotation_files_dict:
            try:
                # Save CSV in same folder as .eannot input
                csv_output_path = os.path.join(os.path.dirname(annotation_files_dict[edf_filename]), f"{edf_filename}.csv")

                raw_data = get_Raw(edf_path)
                annotations = get_annotation(annotation_files_dict[edf_filename], csv_output_path)
                events = data_preprocess(raw_data, annotations, chunk_duration)
                epochs = get_epochs(raw_data, events, max_time)

                label = label_dict.get(subject_id, None)
                epoch_data = epochs.get_data()

                epochs_list = [{
                    "subject_id": subject_id,
                    "epoch_id": f"{subject_id}_epoch_{i}",
                    "epoch_data": epoch.tolist(),
                    "event": epochs.events[i].tolist() if i < len(epochs.events) else None,
                    "sfreq": epochs.info['sfreq'],
                    "label": label
                } for i, epoch in enumerate(epoch_data)]

                np.save(output_path, epochs_list)
                print(f"Saved: {subject_id}_epochs.npy")

            except Exception as e:
                print(f"Error processing {edf_path}: {e}")

    return edf_files


# Example usage
data_dirs = [
    r"/vol/research/baladat1/data/mnc/cnc",
    r"/vol/research/baladat1/data/mnc/dhc/training",
    r"/vol/research/baladat1/data/mnc/dhc/test/controls",  
    r"/vol/research/baladat1/data/mnc/dhc/test/nc-lh",
    r"/vol/research/baladat1/data/mnc/dhc/test/nc-nh"
]
output_dir = r"/vol/research/baladat1/narcolepsy_ai/ho00322_lab/5sec_npys"
edf_files = generate_npy(data_dirs, output_dir, chunk_duration=5, max_time=5.0)

