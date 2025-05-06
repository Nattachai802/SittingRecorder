from fps_check_lib import downsample_video, analyse_set
from fps_result import run_stats
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

    # ----- P5: Statistical Analysis -----
    print("\n>>> Step P5: Statistical Analysis")
    run_stats(df_metrics)

    # ----- P6: เลือก FPS ที่ดีสุด -----
    print("\n>>> Step P6: Selecting best FPS")
    base_fps = max([int(Path(video_path).stem.split("_")[-1].replace("fps", ""))])
    df_metrics["fps"] = df_metrics["clip"].str.extract(r"(\d+)").astype(int)
    
    # เงื่อนไขที่ผ่าน:
    # 1) coverage ไม่ต่ำเกิน
    # 2) jitter/stability ไม่แย่กว่ามาก
    # 3) redundancy (dup_pct) ไม่เกิน 0.95
    # 4) ไม่แตกต่างจาก baseline (พิจารณาจาก Δ ไม่เยอะ)

    good_fps = df_metrics.loc[
        (df_metrics["fps"] != base_fps) &
        (df_metrics["Δcoverage"] >= -0.02) &
        (df_metrics["Δjitter"] <= 0.001) &
        (df_metrics["Δstability"] <= 0.001) &
        (df_metrics["dup_pct"] <= 0.95)
    ].sort_values("fps", ascending=False)

    if not good_fps.empty:
        best_fps = good_fps.iloc[0]["fps"]
        print(f"\n>>> Recommended FPS: {best_fps} fps")
        return best_fps
    else:
        print("ไม่พบ FPS ที่ผ่านเกณฑ์ทั้งหมด ใช้ baseline ต่อไป")
        return base_fps


run_pipeline("recordings/forward_20250506_162415_camera2.mp4")
