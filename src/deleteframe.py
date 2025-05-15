import pandas as pd
import os

df_csv = pd.read_excel("D:/tmp/test/train.xlsx")

for _, row in df_csv.iterrows():
    file_id = row['file_id']
    parquet_path = f"D:/tmp/test/train_landmark_files/{file_id}.parquet"

    if not os.path.exists(parquet_path):
        print(f"Không tìm thấy: {parquet_path}")
        continue

    df_landmarks = pd.read_parquet(parquet_path)

    if len(df_landmarks) <= 20:
        print(f"{file_id} có ít hơn 20 frame, bỏ qua.")
        continue

    df_trimmed = df_landmarks.iloc[10:-5].reset_index(drop=True)

    df_trimmed.to_parquet(parquet_path, index=False)
    print(f"{file_id}: giữ {len(df_trimmed)} frame")
