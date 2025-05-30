import pickle
import numpy as np

cutpad_path = "C:/Users/ADMIN/Desktop/Dataset/cutpad_data.pkl"
resample_path = "C:/Users/ADMIN/Desktop/Dataset/resample_data.pkl"
output_path = "C:/Users/ADMIN/Desktop/Dataset/train_data.pkl"

with open(cutpad_path, "rb") as f:
    data1 = pickle.load(f)
X1 = data1["X"]
y1 = data1["y"]

with open(resample_path, "rb") as f:
    data2 = pickle.load(f)
X2 = data2["X"]
y2 = data2["y"]

print(len(y1))
print(len(y2))
X2_array = np.array(X2)
y2_array = np.array(y2)


X_combined = np.concatenate((X1, X2_array), axis=0)
y_combined = np.concatenate((y1, y2_array), axis=0)
label_map = data1["label_map"]
with open(output_path, "wb") as f:
    pickle.dump({
        "X": X_combined,
        "y": y_combined,
        "label_map": label_map
    }, f)

print("Đã gộp X và y, lưu vào:", output_path)
print(f"Tổng số mẫu: {len(X_combined)}")
print(f"Tổng số nhãn: {len(y_combined)}")

