import pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

file_path = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/data.p'
data = load_pickle_file(file_path)
print(data)
x = data[0]
y = data[1]
print(f"X: {len(x)}")
print(f"Y: {len(y)}")
