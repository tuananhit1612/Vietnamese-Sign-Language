import os
import pandas as pd
import cv2

# Đường dẫn thư mục chứa các video con
video_base_path = r"C:\Users\ADMIN\Downloads\MySQL_Video (1)\MySQL_Video\static\videos"

# Đường dẫn file Excel đầu vào
excel_file_path = r"C:\Users\ADMIN\Desktop\Dataset\train.xlsx"

# Đọc dữ liệu từ Excel
df = pd.read_excel(excel_file_path)
df['file_id'] = df['file_id'].astype(str).str.strip()

# Hàm tìm video path và tạo đường dẫn web dạng /static/videos/...
def find_video_path(file_id, base_folder):
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file_id.lower() in file.lower():
                full_path = os.path.join(root, file)
                relative_path = full_path.split("static" + os.sep, 1)[-1].replace("\\", "/")
                return f"/static/{relative_path}", full_path  # trả cả đường dẫn web và đầy đủ để kiểm tra
    return None, None

# Hàm kiểm tra video có mở được hay không
def is_video_readable(video_full_path):
    if not video_full_path or not os.path.exists(video_full_path):
        return False
    try:
        cap = cv2.VideoCapture(video_full_path)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except:
        return False

# Áp dụng tìm đường dẫn và kiểm tra
video_paths = []
readable_flags = []

for file_id in df['file_id']:
    web_path, full_path = find_video_path(file_id, video_base_path)
    video_paths.append(web_path)
    readable_flags.append(is_video_readable(full_path))

df['video_path'] = video_paths
df['is_readable'] = readable_flags

# Ghi kết quả ra Excel mới
output_file = r"C:\Users\ADMIN\Desktop\Dataset\train_with_video_paths_checked.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Đã hoàn tất! File kết quả lưu tại: {output_file}")
