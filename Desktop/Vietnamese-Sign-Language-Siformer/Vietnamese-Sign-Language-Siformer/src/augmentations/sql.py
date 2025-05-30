import pandas as pd

excel_path = r"C:\Users\ADMIN\Desktop\Dataset\train_with_video_paths_checked.xlsx"
df = pd.read_excel(excel_path)

# Xóa dòng nào thiếu dữ liệu
df = df.dropna(subset=['gloss', 'category_id', 'video_path'])

values_list = []
for _, row in df.iterrows():
    video_id = row['video_path'].split('/')[-1].replace('.mp4', '')  # Lấy tên file làm ID
    title = str(row['gloss']).replace("'", "''")                      # Escape dấu nháy
    video_url = str(row['video_path']).strip()
    category_id = int(row['category_id'])

    values_list.append(f"('{video_id}', '{title}', '{video_url}', {category_id})")

# Gộp thành 1 câu INSERT duy nhất
sql = "INSERT INTO sign_videos (id, title, video_url, category_id) VALUES\n" + ",\n".join(values_list) + ";"

# Lưu ra file
output_sql_path = r"C:\Users\ADMIN\Desktop\Dataset\insert_sign_videos_single.sql"
with open(output_sql_path, 'w', encoding='utf-8') as f:
    f.write(sql)

print(f"✅ Đã tạo file SQL gộp tại: {output_sql_path}")
