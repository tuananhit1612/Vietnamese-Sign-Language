from itertools import product
import pandas as pd
import numpy as np
import os

from augment_pipeline import augment_pipeline

df_csv = pd.read_excel("D:/tmp/test/train.xlsx")
for i in range(0,3):
    for _, row in df_csv.iterrows():
        file_id = row['file_id']
        df = pd.read_parquet(f"D:/tmp/test/train_landmark_files/{file_id}.parquet")

        techniques = ['mirror', 'offset', 'noise', 'scale', 'jitter']

        combinations = list(product([False, True], repeat=len(techniques)))[1:]


        output_folder = "D:/tmp/test/train_landmark_files/augmented_combinations"
        os.makedirs(output_folder, exist_ok=True)

        for idx, combo in enumerate(combinations, 1):
            combo_dict = dict(zip(techniques, combo))
            df_aug = augment_pipeline(df, apply=combo_dict)
            combo_name = "_".join([k for k, v in combo_dict.items() if v])
            output_path = os.path.join(output_folder, f"{file_id}_aug_{i}_{idx:02d}.parquet")

            df_aug.to_parquet(output_path, index=False)
        print(f"hoan tất tăng cường {file_id}")

print("Done.")
