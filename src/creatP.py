import numpy as np
import os
import pickle
import pandas as pd


def mapping_id_num(df, id_text):
    # Check if id_text exists in the dataframe
    match = df[df['id_video'] == id_text]
    if not match.empty:
        return match['id_sequence'].values[0]
    else:
        print(f"Warning: id_video '{id_text}' not found in the dataframe.")
        return None
def load_npy_data(input_dir, df):
    X = []
    y = []

    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for npy_file in npy_files:
        keypoints = np.load(os.path.join(input_dir, npy_file))
        video_name = os.path.splitext(npy_file)[0]
        video_name = video_name.rsplit('_', 1)[0]
        id_num = mapping_id_num(df, video_name)
        y.append(id_num)
        X.append(keypoints)

    return np.array(X), np.array(y)


def save_dataset_to_pickle(X, y, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"Dataset has been saved to {output_file}")


def main():
    excel_file_path = 'D:/tmp/test/video_to_text_mapping_basic.xlsx'
    df = pd.read_excel(excel_file_path)

    input_dir = 'D:/tmp/test/train_landmark_files'

    X, y = load_npy_data(input_dir, df)

    output_file = 'D:/tmp/test/data.p'
    save_dataset_to_pickle(X, y, output_file)

if __name__ == '__main__':
    main()
