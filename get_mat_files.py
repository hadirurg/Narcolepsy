# %%
import numpy as np
import scipy.io
import os

def get_mat(input_folder, output_folder):
    stage_mapping = {1: 'W', 2: 'N1', 3: 'N2', 4: 'N3', 5: 'REM'}

    for file in sorted(os.listdir(input_folder))[:1]: 
        if file.endswith('.npy'):
            file_path = os.path.join(input_folder, file)
            data = np.load(file_path, allow_pickle=True)

            # Take only first 10 epochs
            data = data[:10]

            epoch_data = [epoch["epoch_data"] for epoch in data]
            events = [epoch["event"] for epoch in data]
            epoch_ids = [epoch["epoch_id"] for epoch in data]
            subject_id = data[0]["subject_id"]

            labels = []
            timeSeriesData = np.empty((len(epoch_data), 1), dtype=object)
            keywords = []

            for i, epoch_array in enumerate(epoch_data):
                reshaped_epoch = np.array(epoch_array).reshape(-1, 1)
                timeSeriesData[i, 0] = reshaped_epoch

                sleep_stage = events[i][2]
                stage_label = stage_mapping.get(sleep_stage, 'Unknown')
                
                label = f"{epoch_ids[i]}"
                labels.append(label)
                keywords.append(stage_label)

            intermediate_data = {
                "labels": np.array(labels, dtype=object).reshape(-1, 1),
                "timeSeriesData": timeSeriesData,
                "keywords": np.array(keywords, dtype=object).reshape(-1, 1)
            }

            output_path = os.path.join(output_folder, f"{subject_id}.mat")
            scipy.io.savemat(output_path, intermediate_data)
            print(f"Saved: {output_path}")

input = r"/vol/research/baladat1/narcolepsy_ai/hadir_features/30sec_npys"
output = r"/vol/research/baladat1/narcolepsy_ai/hadir_features/mat_files"
os.makedirs(output, exist_ok=True)

get_mat(input, output)



