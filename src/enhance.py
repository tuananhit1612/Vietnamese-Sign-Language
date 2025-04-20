import cv2
import numpy as np
import os
import random

def random_crop(frame, crop_size=(200, 200)):
    h, w = frame.shape[:2]
    top = random.randint(0, h - crop_size[1])
    left = random.randint(0, w - crop_size[0])
    cropped_frame = frame[top:top+crop_size[1], left:left+crop_size[0]]
    return cv2.resize(cropped_frame, (w, h))

def flip_video(frame):
    return cv2.flip(frame, 1)  # Lật ngang

def rotate_frame(frame, angle=30):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, matrix, (w, h))
    return rotated_frame

def add_motion_blur(frame, ksize=15):
    """
    Thêm hiệu ứng motion blur vào video
    :param frame: Khung hình video
    :param ksize: Kích thước kernel (càng lớn blur càng mạnh)
    :return: Khung hình đã thêm hiệu ứng mờ
    """
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1)/2), :] = np.ones(ksize)
    kernel /= ksize
    blurred_frame = cv2.filter2D(frame, -1, kernel)
    return blurred_frame

def add_light_and_shadow(frame):
    """
    Thêm ánh sáng và bóng vào khung hình.
    :param frame: Khung hình video
    :return: Khung hình với hiệu ứng ánh sáng và bóng
    """
    shadow = frame.copy()
    alpha = random.uniform(0.5, 1.5)  # Random ánh sáng
    beta = random.randint(-50, 50)    # Random bóng
    shadow = cv2.convertScaleAbs(shadow, alpha=alpha, beta=beta)
    return shadow

def process_video(input_video_path, output_dir):
    cap = cv2.VideoCapture(input_video_path)

    # Video information
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.basename(input_video_path).split('.')[0]
    out_flip = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_flip.mp4"), fourcc, fps, (width, height))
    out_rotate = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_rotate.mp4"), fourcc, fps, (width, height))
    out_light_shadow = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_light_shadow.mp4"), fourcc, fps, (width, height))
    out_motion_blur = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_motion_blur.mp4"), fourcc, fps,
                                      (width, height))
    rotate_angle = random.randint(-30, 30)
    ksize = random.randint(10, 23)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply effects
        flipped_frame = flip_video(frame)
        rotated_frame_rs = rotate_frame(frame, rotate_angle)

        # Thêm hiệu ứng ánh sáng và bóng
        light_shadow_frame = add_light_and_shadow(frame)
        blurred_frame = add_motion_blur(frame,ksize)
        # Write frames to video
        out_flip.write(flipped_frame)
        out_rotate.write(rotated_frame_rs)
        out_light_shadow.write(light_shadow_frame)
        out_motion_blur.write(blurred_frame)

    cap.release()
    out_flip.release()
    out_rotate.release()
    out_light_shadow.release()
    out_motion_blur.release()
    os.remove(input_video_path)
    print(f"4 augmented videos saved to {output_dir}")

def process_all_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("No videos found in the input directory.")
        return

    for i,video_file in video_files:
        print(f"Processing video {i+1}/{len(video_files)}")
        input_video_path = os.path.join(input_dir, video_file)
        process_video(input_video_path, output_dir)

    print("Completed augmenting all videos.")

if __name__ == "__main__":
    INPUT_VIDEO_DIR = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/videos'
    OUTPUT_VIDEO_DIR = 'D:/Dev/DoAnCoSo_NCKH/Vietnamese-Sign-Language/data/videos/output/tangcuong'

    process_all_videos(INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR)
