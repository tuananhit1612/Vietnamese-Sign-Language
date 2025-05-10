import pandas as pd

# Đọc tệp Parquet
df = pd.read_parquet("D:/tmp/test/train_landmark_files/D0489.parquet")

# Hiển thị nội dung của DataFrame
print(df)
df.to_excel("D:/tmp/test/train_landmark_files/output.xlsx", index=False)
# Xem thông tin tổng quan về dữ liệu (số dòng, cột, kiểu dữ liệu)
print(df.info())

# Xem 5 dòng đầu tiên của DataFrame
print(df.head())
