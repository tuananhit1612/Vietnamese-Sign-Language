import pandas as pd
import os


def export_parquet_to_excel(parquet_path, excel_path=None):
    if not os.path.exists(parquet_path):
        print(f"File không tồn tại: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)

    if excel_path is None:
        excel_path = parquet_path.replace(".parquet", ".xlsx")

    df.to_excel(excel_path, index=False)
    print(f"Đã lưu Excel tại: {excel_path}")


if __name__ == "__main__":
    parquet_file = r"D:\train_test\train_landmark_files\D0492.parquet"
    export_parquet_to_excel(parquet_file)
