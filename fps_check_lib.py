
import cv2
import math
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import mediapipe as mp
import pandas as pd

def downsample_video(
        video_path: str,
        target_fps_list: list[int] | None = None,
        out_dir: str = "downsampled"
    ) -> list[str]:
    """
    รับไฟล์วิดีโอ 1 คลิปแล้วสร้างไฟล์ที่ fps ต่ำลงตาม target_fps_list
    ----------------------------------------------------------------------
    Parameters
    ----------
    video_path : str
        path ของวิดีโอต้นฉบับ
    target_fps_list : list[int] | None
        fps ที่ต้องการ (ถ้า None จะสร้าง 5 ค่า: (fps_orig-5,…,-25) ขั้นละ-5)
    out_dir : str
        โฟลเดอร์เก็บผลลัพธ์
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    fps_orig   = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -------- 1) กำหนดชุด fps เป้าหมาย --------
    if target_fps_list is None:
        step = max(1, int(fps_orig // 6))          # ให้ได้ ~5 ค่า
        target_fps_list = [int(fps_orig - i*step)  # 30→[25,20,15,10,5]
                           for i in range(1, 6)
                           if fps_orig - i*step > 0]

    # ป้องกัน target เกิน fps ต้นฉบับ
    target_fps_list = [f for f in target_fps_list if f < fps_orig]

    # -------- 2) เตรียม VideoWriter ทุกตัว --------
    out_paths = []
    writers   = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")       # .mp4

    Path(out_dir).mkdir(exist_ok=True)

    for f in target_fps_list:
        out_path = Path(out_dir) / f"{Path(video_path).stem}_{f}fps.mp4"
        writers[f] = cv2.VideoWriter(str(out_path), fourcc, f, (width, height))
        out_paths.append(str(out_path))

    # -------- 3) วนอ่านเฟรมแล้วเขียนแบบ skip --------
    acc = {f:0.0 for f in target_fps_list}         # accumulator ต่อ fps
    ratio = {f: fps_orig / f for f in target_fps_list}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for f in target_fps_list:
            acc[f] += 1
            if acc[f] >= ratio[f]:                 # ถึงคิวเขียนเฟรม
                writers[f].write(frame)
                acc[f] -= ratio[f]

        frame_idx += 1

    # -------- 4) ปิดไฟล์ทั้งหมด --------
    cap.release()
    for w in writers.values():
        w.release()

    return out_paths


def redundancy_stats(video_path: str,
                     resize_to: tuple[int, int] = (256, 256),
                     threshold: float = 0.95,
                     use_gray: bool = True) -> dict:
    """
    คำนวณ SSIM ระหว่างเฟรมติดกันของวิดีโอ
    คืน dict: {mean_ssim, redundant_ratio, total_frames}
    """
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        raise ValueError(f"ไม่สามารถเปิดไฟล์ {video_path}")

    total, redundant, ssim_sum = 0, 0, 0.0
    
    # เตรียมเฟรมแรก
    if use_gray:
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if resize_to:
        prev = cv2.resize(prev, resize_to)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                desc=f"Scanning {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if use_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize_to:
            frame = cv2.resize(frame, resize_to)

        # คำนวณ SSIM ระหว่าง prev–curr
        score = ssim(prev, frame)
        ssim_sum += score
        if score > threshold:
            redundant += 1
        total += 1

        prev = frame
        pbar.update(1)
    cap.release(); pbar.close()

    # เฟรมแรกไม่มีคู่ เปรียบเทียบไม่ได้
    mean_ssim = ssim_sum / max(total, 1)
    redundant_ratio = redundant / max(total, 1)

    return dict(mean_ssim=mean_ssim,
                redundant_ratio=redundant_ratio,
                total_frames=total+1)  # +1 รวมเฟรมแรก



# ---------- CONFIG ----------
VIS_TH      = 0.5          # threshold visibility
SSIM_TH     = 0.95         # two frames “ซ้ำ” ถ้า SSIM > 0.95
JOINTS_IDX  = [            # ข้อต่อหลักที่ติดตาม Jitter
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST
]

# ---------- HELPER -----------
def calc_ssim(prev, curr):
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(prev_g, curr_g, full=True)
    return score

def analyse_clip(path: str):
    cap = cv2.VideoCapture(str(path))
    mp_pose = mp.solutions.pose.Pose()

    total, dup, full_landmark = 0, 0, 0
    traj = {idx: [] for idx in JOINTS_IDX}

    ret, prev = cap.read()        # เฟรมแรก
    if not ret:
        raise ValueError(f"Cannot read {path}")
    total += 1

    # ----- process first frame -----
    res = mp_pose.process(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        visible_cnt = sum(pt.visibility > VIS_TH for pt in lm)
        if visible_cnt == len(lm):
            full_landmark += 1
        for idx in JOINTS_IDX:
            traj[idx].append([lm[idx].x, lm[idx].y])

    # ----- loop rest frames -----
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        # SSIM
        if calc_ssim(prev, frame) > SSIM_TH:
            dup += 1
        prev = frame

        # Pose
        res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            visible_cnt = sum(pt.visibility > VIS_TH for pt in lm)
            if visible_cnt == len(lm):
                full_landmark += 1
            for idx in JOINTS_IDX:
                traj[idx].append([lm[idx].x, lm[idx].y])
        else:
            for idx in JOINTS_IDX:
                traj[idx].append([np.nan, np.nan])

    cap.release()
    mp_pose.close()

    # ----- metrics -----
    coverage = full_landmark / total
    dup_pct  = dup / (total-1) if total > 1 else 0

    jitter_vals, std_vals = [], []
    for idx, pts in traj.items():
        pts = np.array(pts)
        # เอาเฉพาะจุดที่ไม่ NaN
        mask = ~np.isnan(pts).any(axis=1)
        pts_valid = pts[mask]
        if len(pts_valid) > 1:
            deltas = np.linalg.norm(np.diff(pts_valid, axis=0), axis=1)
            jitter_vals.append(deltas.mean())
            std_vals.append(pts_valid.std(axis=0).mean())
        else:
            jitter_vals.append(np.nan)
            std_vals.append(np.nan)

    jitter_avg = np.nanmean(jitter_vals)
    std_avg    = np.nanmean(std_vals)

    return dict(
        frames      = total,
        dup_pct     = dup_pct,
        coverage    = coverage,
        jitter      = jitter_avg,
        stability   = std_avg
    )

# ---------- MAIN -------------
def analyse_set(baseline_path: str, others: list[str]):
    """
    baseline_path : คลิป fps สูงสุด (เช่น 30 fps)
    others        : list คลิปที่ down-sample แล้ว
    """
    results = []
    print("=== Baseline ===")
    base_metrics = analyse_clip(baseline_path)
    base_metrics['clip'] = Path(baseline_path).name
    results.append(base_metrics)

    print("\n=== Down-sampled clips ===")
    for p in tqdm(others):
        m = analyse_clip(p)
        m['clip'] = Path(p).name
        # สร้าง delta เทียบ baseline (MAE แนวคิดง่าย ๆ)
        m['Δcoverage']  = m['coverage']  - base_metrics['coverage']
        m['Δjitter']    = m['jitter']    - base_metrics['jitter']
        m['Δstability'] = m['stability'] - base_metrics['stability']
        m['dup_diff']   = m['dup_pct']   - base_metrics['dup_pct']
        results.append(m)

    return pd.DataFrame(results)