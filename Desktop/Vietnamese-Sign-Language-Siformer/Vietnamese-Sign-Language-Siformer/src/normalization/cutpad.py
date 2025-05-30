import numpy as np
import pickle

with open("C:/Users/ADMIN/Desktop/Dataset/packed_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X_raw = data["X"]
y = data["y"]
label_map = data["label_map"]

target_seq_len = 50
X_fixed = []

for arr in X_raw:
    if arr.shape[0] > target_seq_len:
        mid = arr.shape[0] // 2
        start = max(0, mid - target_seq_len // 2)
        arr = arr[start:start + target_seq_len]
    elif arr.shape[0] < target_seq_len:
        pad_len = target_seq_len - arr.shape[0]
        pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    X_fixed.append(arr)

X = np.array(X_fixed)

with open("C:/Users/ADMIN/Desktop/Dataset/cutpad_data.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y, "label_map": label_map}, f)

