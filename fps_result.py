
from __future__ import annotations
import pandas as pd
import re, itertools, math
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- Stats ----------
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# ----------- CONFIG ----------
ALPHA      = 0.05                     # ค่าตัดสิน
METRICS    = ["coverage", "jitter", "stability", "dup_pct"]  # field ที่วิเคราะห์
SUBJECT_ID = "S0"                     # ถ้ามีคลิปชุดเดียวให้ fix รหัส subject ไว้

# ----------------------------------------------------------------
def _extract_fps(name: str) -> int:
    """
    'baseline_30fps.mp4' → 30
    'clip_25fps.mp4'     → 25
    """
    m = re.search(r"(\d+)\s*fps", name.lower())
    if m:
        return int(m.group(1))
    # กรณี baseline_30 / baseline30
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    raise ValueError(f"หา fps ไม่เจอจากชื่อไฟล์: {name}")

def reshape_long(df_in: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    คืน DataFrame long-format: [subject,fps,value]
    """
    rows = []
    for _, r in df_in.iterrows():
        fps_val = _extract_fps(r["clip"])
        rows.append(dict(subject=SUBJECT_ID, fps=fps_val, value=r[metric]))
    out = pd.DataFrame(rows).sort_values("fps").reset_index(drop=True)
    return out

# ----------------- PARAMETRIC -----------------
def rm_anova(long_df: pd.DataFrame) -> tuple[float,float]:
    """
    Repeated-Measures (within subject) ANOVA
    คืน (F-stat, p-value)
    """
    aov = AnovaRM(long_df,
                  depvar="value",
                  subject="subject",
                  within=["fps"]).fit()
    F   = aov.anova_table["F Value"][0]
    p   = aov.anova_table["Pr > F"][0]
    return F, p

def tukey(long_df: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc Tukey HSD ทั้งหมด"""
    res = pairwise_tukeyhsd(long_df["value"],
                            groups=long_df["fps"],
                            alpha=ALPHA)
    return pd.DataFrame(data=res._results_table.data[1:],   # กำจัด header
                        columns=res._results_table.data[0])

# -------------- NON-PARAMETRIC ----------------
def friedman(long_df: pd.DataFrame) -> tuple[float,float]:
    """
    Friedman test (alternative to RM-ANOVA when data non-normal / n=1 subj.)
    """
    pivot = long_df.pivot(index="subject", columns="fps", values="value")
    stat, p = friedmanchisquare(*pivot.values.T)
    return stat, p

def pairwise_wilcoxon(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon signed-rank (pairwise vs. baseline fps สูงสุด)
    ใช้ Holm correction
    """
    pivot = long_df.pivot(index="subject", columns="fps", values="value")
    base_fps = pivot.columns.max()            # baseline = fps สูงสุด
    p_raw, comp = [], []                      # เก็บ p-value เดิม / คู่ทดสอบ
    for f in pivot.columns:
        if f == base_fps: 
            continue
        stat,p = wilcoxon(pivot[base_fps], pivot[f])
        p_raw.append(p)
        comp.append((f, base_fps))
    # Holm correction
    reject, p_adj, *_ = multipletests(p_raw, alpha=ALPHA, method="holm")
    out = pd.DataFrame(dict(fps=comp, p_raw=p_raw, p_adj=p_adj, reject=reject))
    return out

# -------------------- MAIN --------------------
def run_stats(df_metrics: pd.DataFrame,
              metrics: list[str] = METRICS,
              alpha: float      = ALPHA,
              force_nonparam: bool=False) -> None:
    """
    df_metrics : DataFrame ที่ได้จาก analyse_set()
    """
    for m in metrics:
        print(f"\n======================  {m.upper()}  ======================")
        long_df = reshape_long(df_metrics, m)

        # --- เลือกวิธีทดสอบ ---
        use_nonparam = force_nonparam or (long_df["subject"].nunique() < 2)
        # (ถ้ามี subject เดียว AnovaRM จะไม่ทำงาน → บังคับใช้ friedman)

        if not use_nonparam:
            # ---------- RM-ANOVA ----------
            F,p = rm_anova(long_df)
            print(f"[RM-ANOVA]  F = {F:.3f},  p = {p:.5f}")
            if p < alpha:
                print("  ↳ ต่างอย่างมีนัยฯ → Post-hoc Tukey (α={:.3f})".format(alpha))
                tuk = tukey(long_df)
                # โชว์เฉพาะคู่ baseline เท่านั้น
                base_fps = long_df["fps"].max()
                sel = tuk[((tuk.group1==base_fps)|(tuk.group2==base_fps))]
                print(sel.to_string(index=False))
            else:
                print("  ↳ ไม่พบความแตกต่าง (p > α)")
        else:
            # ---------- Friedman ----------
            stat,p = friedman(long_df)
            print(f"[Friedman χ²]  χ² = {stat:.3f},  p = {p:.5f}")
            if p < alpha:
                print("  ↳ ต่างอย่างมีนัยฯ → Wilcoxon pairwise+Holm")
                pw = pairwise_wilcoxon(long_df)
                print(pw.to_string(index=False))
            else:
                print("  ↳ ไม่พบความแตกต่าง (p > α)")


# -------------------- EXAMPLE -------------------
if __name__ == "__main__":
    """
    1) รัน analyse_set() ได้ df แล้วเซฟเป็น CSV (หรือส่งตรงก็ได้)
    2) โหลด df แล้วเรียก run_stats(df)
    """
    # df = analyse_set(baseline_path, others)
    df  = pd.read_csv("metrics_all_fps.csv")       # <-- ตัวอย่าง

    run_stats(df, metrics=["coverage","jitter","stability","dup_pct"])


# ---------------- CONFIG (ปรับได้) -----------------
THRESH = {                       # ค่าสูงสุดที่ยอมให้แย่ลง (±)
    "coverage" : -0.02,          # ห้ามลดลง > 2 %
    "jitter"   :  0.0005,        # ห้ามเพิ่มเกิน 0.0005
    "stability":  0.0003,
    "dup_pct"  :  0.05,          # ห้ามเพิ่มเกิน 5 %
}
OUT_DIR = Path("report")         # โฟลเดอร์ผลลัพธ์
OUT_DIR.mkdir(exist_ok=True)

# -------- util : p-value ของคู่ fps vs baseline ---------------
def pvals_vs_base(long_df: pd.DataFrame, alpha=ALPHA) -> dict[int,float]:
    """
    คืน dict {fps : p_val}  (baseline เองไม่คืน)
    ใช้ Tukey ถ้าอนุกรม parametric, else Wilcoxon
    """
    base = long_df["fps"].max()
    if long_df["subject"].nunique() > 1:       # ใช้ Tukey
        tk = tukey(long_df)
        sel = tk[(tk.group1==base)|(tk.group2==base)]
        return { int(r.group1 if r.group2==base else r.group2) : r.pvalue
                 for _,r in sel.iterrows() }
    # ----- Wilcoxon (single-subject / non-param) -----
    pv = {}
    pivot = long_df.pivot(index="subject", columns="fps", values="value")
    for f in pivot.columns:
        if f==base: continue
        _,p = wilcoxon(pivot[base], pivot[f])
        pv[int(f)] = p
    return pv

# ----------- MAIN : สรุป + เลือก FPS ----------------
def choose_fps(df_metrics: pd.DataFrame,
               metrics: list[str]=METRICS,
               alpha: float=ALPHA,
               thresh: dict[str,float]=THRESH) -> int:
    baseline_fps = df_metrics["clip"].apply(_extract_fps).max()
    baseline_row = df_metrics[df_metrics["clip"].str.contains(str(baseline_fps))]
    if baseline_row.empty:
        raise ValueError("หา baseline ไม่เจอในตาราง")

    summary_rows = []
    pass_fps = set()  # fps ที่ผ่านทั้ง metric+stat จะคัดเลือกท้ายสุด

    for m in metrics:
        long_df = reshape_long(df_metrics, m)
        pvals   = pvals_vs_base(long_df, alpha)   # {fps: p}
        base_val = baseline_row.iloc[0][m]

        for _,r in df_metrics.iterrows():
            fps = _extract_fps(r["clip"])
            if fps==baseline_fps: continue
            val   = r[m]
            delta = val - base_val
            # ----- ตรวจทิศของ metric -----
            if m=="coverage":                   # coverage สูงกว่าดี
                ok_delta = delta >= thresh[m]   # ยอมให้ลดได้ไม่เกิน |thresh|
            else:                               # metric ยิ่งน้อยยิ่งดี
                ok_delta = delta <= thresh[m]
            ok_p   = (pvals.get(fps,1) > alpha)

            summary_rows.append(dict(fps=fps, metric=m, value=val,
                                      delta=delta, p=pvals.get(fps,1),
                                      ok_delta=ok_delta, ok_p=ok_p))
    # --------- Summary DF ----------
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR/"metrics_summary.csv", index=False)

    # --------- เลือก fps -------------
    for fps in sorted(df_sum.fps.unique()):
        df_sub = df_sum[df_sum.fps==fps]
        if df_sub["ok_delta"].all() and df_sub["ok_p"].all():
            pass_fps.add(fps)

    best = max(pass_fps) if pass_fps else baseline_fps
    print("\n🎯  Recommended FPS =", best)
    # --------- วาดกราฟ ----------
    for m in metrics:
        plt.figure()
        plt.title(f"{m} vs FPS")
        xs = df_metrics["clip"].apply(_extract_fps)
        ys = df_metrics[m]
        plt.plot(xs, ys, marker="o")
        plt.axvline(best, ls="--", label=f"chosen {best}fps")
        plt.xlabel("FPS"); plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR/f"{m}.png", dpi=120)
        plt.close()
    return best

# ----------------- DEMO -----------------
if __name__ == "__main__":
    df = pd.read_csv("metrics_all_fps.csv")
    run_stats(df)                   # ← P5 (พิมพ์ผลให้ดู)
    choose_fps(df)                  # ← P6