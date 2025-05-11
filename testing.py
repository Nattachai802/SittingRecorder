import cv2
import mediapipe as mp
import os
import numpy as np
import math
import csv
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import glob

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def upscale_image(image, scale=2.0):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def create_image_grid(images, grid_size=(12, 12), image_size=(64, 64)):
    rows, cols = grid_size
    canvas = np.zeros((rows * image_size[1], cols * image_size[0], 3), dtype=np.uint8)
    for idx, img in enumerate(images[:rows * cols]):
        resized = cv2.resize(img, image_size)
        r = idx // cols
        c = idx % cols
        canvas[r*image_size[1]:(r+1)*image_size[1], c*image_size[0]:(c+1)*image_size[0]] = resized
    return canvas

def is_similar(img1, img2, threshold=0.95):
    img1_gray = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (256, 256))
    img2_gray = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (256, 256))
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score > threshold

def remove_similar_frames(frames, threshold=0.95):
    result_frames = [frames[0]]
    for i in tqdm(range(1, len(frames)), desc=f"SSIM @ {threshold:.2f}"):
        if is_similar(frames[i-1], frames[i], threshold):
            blank = np.zeros_like(frames[i])
            result_frames.append(blank)
        else:
            result_frames.append(frames[i])
    return result_frames

def ensure_csv_header(csv_path):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "parent_folder", "subfolder", "filename", "total_frames",
                "frames_ssim_100", "frames_ssim_098", "frames_ssim_095",
                "frames_ssim_090", "frames_ssim_085", "frames_ssim_080", "coverage"
            ])

def process_video(video_path, csv_path):
    VIS_TH = 0.5
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    path_obj = Path(video_path)
    parent = path_obj.parts[-3]
    sub = path_obj.parts[-2]
    file_stem = path_obj.stem
    prefix = f"{parent}_{sub}"

    grid_before_path = os.path.join(output_dir, f"{prefix}_grid_before_ssim_{file_stem}.jpg")
    final_output_path = os.path.join(output_dir, f"{prefix}_final_grid_{file_stem}.jpg")

    pose = mp.solutions.pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    valid_keypoint_match = 0
    all_frames = []
    ref_visible = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"{file_stem}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        upscaled = upscale_image(frame, scale=2.0)
        sharpened = sharpen_image(upscaled)
        frame_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                sharpened,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            if frame_count == 0:
                ref_visible = [pt.visibility > VIS_TH for pt in results.pose_landmarks.landmark]
            else:
                lm = results.pose_landmarks.landmark
                match_cnt = sum((ref and (pt.visibility > VIS_TH)) for ref, pt in zip(ref_visible, lm))
                if sum(ref_visible) > 0 and match_cnt / sum(ref_visible) >= 0.9:
                    valid_keypoint_match += 1

        all_frames.append(sharpened)
        frame_count += 1
        pbar.update(1)

    cap.release()
    pose.close()
    pbar.close()

    if frame_count == 0:
        print("❌ No frames found:", video_path)
        return

    coverage = valid_keypoint_match / (frame_count - 1) if frame_count > 1 else 0
    grid_cols = 12
    grid_rows = math.ceil(len(all_frames) / grid_cols)

    grid_before = create_image_grid(all_frames, grid_size=(grid_rows, grid_cols), image_size=(128, 128))
    cv2.imwrite(grid_before_path, grid_before)

    ssims = {}
    thresholds = [1.00, 0.98, 0.95, 0.90, 0.85, 0.80]
    for th in thresholds:
        cleaned = remove_similar_frames(all_frames, threshold=th)
        ssims[th] = sum(1 for f in cleaned if not np.all(f == 0))
        grid_th = create_image_grid(cleaned, grid_size=(grid_rows, grid_cols), image_size=(128, 128))
        suffix = f"{int(th * 100):03d}"
        grid_path = os.path.join(output_dir, f"{prefix}_grid_after_ssim{suffix}_{file_stem}.jpg")
        cv2.imwrite(grid_path, grid_th)

    cv2.imwrite(final_output_path, grid_th)  # last one as final

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            parent, sub, file_stem, frame_count,
            ssims[1.00], ssims[0.98], ssims[0.95],
            ssims[0.90], ssims[0.85], ssims[0.80], round(coverage, 4)
        ])

# === MAIN LOOP ===
csv_summary_path = 'result/summary.csv'
ensure_csv_header(csv_summary_path)
video_folders = glob.glob('recordings/*/*')
for folder in video_folders:
    for video in glob.glob(os.path.join(folder, '*.mp4')):
        print(f"\n▶ Processing: {video}")
        process_video(video, csv_summary_path)
