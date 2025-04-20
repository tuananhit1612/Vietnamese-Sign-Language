import numpy as np
import os
import pickle
import pandas as pd



def load_npy_data(input_dir, df):
    X = []
    y = []
    max_length = 225

    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for npy_file in npy_files:
        keypoints = np.load(os.path.join(input_dir, npy_file))

        video_name = os.path.splitext(npy_file)[0]
        video_name = video_name.rsplit('_', 1)[0]
        id_text = video_name

        if id_text is not None:
            if keypoints.shape[0] < max_length:
                padding = np.zeros((max_length - keypoints.shape[0], keypoints.shape[1]))  # Padding các dòng còn thiếu
                keypoints = np.vstack([keypoints, padding])

            y.append(id_text)
            X.append(keypoints)
        else:
            print(f"ID for video {video_name} not found!")

    return np.array(X), np.array(y)


def save_dataset_to_pickle(X, y, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"dataset has been saved to {output_file}")


def main():
    excel_file_path = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/video_to_text_mapping.xlsx'
    df = pd.read_excel(excel_file_path)

    input_dir = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/npy/test'

    X, y = load_npy_data(input_dir, df)

    output_file = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/test.p'
    save_dataset_to_pickle(X, y, output_file)


if __name__ == '__main__':
    main()
