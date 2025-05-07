from fps_check_lib import downsample_video, analyse_set
from fps_result import run_stats, _extract_fps  # เพิ่มตรงนี้
import pandas as pd
from pathlib import Path

def run_pipeline(video_path: str):
    """
    รับวิดีโอ baseline 1 ไฟล์ แล้วรันทุก P แบบอัตโนมัติ
    คืนค่าที่เป็น FPS ที่ดีที่สุด (ไม่ต่างจาก baseline + redundancy ต่ำ)
    """
    # ----- P2: Downsample -----
    print(">>> Step P2: Downsampling")
    ds_videos = downsample_video(video_path)

    # ----- P3–P4: Metrics -----
    print("\n>>> Step P3–P4: Analyse clips")
    df_metrics = analyse_set(video_path, ds_videos)

    # Save เผื่อไว้ใช้ซ้ำ
    out_csv = Path("metrics_all_fps.csv")
    df_metrics.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")
    print('นำข้อมูลที่ได้ไปวิเคราะห์ต่อในการทำEDA')

    """

    # ----- P5: Statistical Analysis -----
    print("\n>>> Step P5: Statistical Analysis")
    run_stats(df_metrics)

    # ----- P6: เลือก FPS ที่ดีสุด -----
    print("\n>>> Step P6: Selecting best FPS")
    try:
        base_fps = _extract_fps(video_path)  # ใช้ _extract_fps แทน split("_")
    except ValueError:
        print(f"Warning: Filename does not contain a valid FPS value. Using default FPS of 30.")
        base_fps = 30

    df_metrics["fps"] = df_metrics["clip"].apply(lambda clip: _extract_fps(clip))  # ใช้ _extract_fps กับ clip

    # เงื่อนไขที่ผ่าน:
    # 1) coverage ไม่ต่ำเกิน
    # 2) jitter/stability ไม่แย่กว่ามาก
    # 3) redundancy (dup_pct) ไม่เกิน 0.95
    # 4) ไม่แตกต่างจาก baseline (พิจารณาจาก Δ ไม่เยอะ)

    fps_f1_scores = []
    for p in ds_videos:
        try:
            fps_val = _extract_fps(p)  # ใช้ _extract_fps
            f1 = 1 - df_metrics[df_metrics["clip"] == Path(p).name]["dup_pct"].values[0]
            fps_f1_scores.append((fps_val, f1))
        except KeyError:
            print(f"Warning: Missing data for clip {p}. Skipping.")

    fps_f1_scores.sort(key=lambda x: x[1], reverse=True)

    if fps_f1_scores:
        best_fps = fps_f1_scores[0][0]
        print(f"\n>>> Recommended FPS: {best_fps} fps")
        return best_fps
    else:
        print("ไม่พบ FPS ที่ผ่านเกณฑ์ทั้งหมด ใช้ baseline ต่อไป")
        return base_fps
    """


run_pipeline("recordings/forward_20250506_162415_camera2.mp4")
