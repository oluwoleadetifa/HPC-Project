# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_ROOT = Path("../results")
CSV_FILE = RESULTS_ROOT / "timings.csv"
PLOTS_DIR = RESULTS_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CSV_FILE)
df['workers'] = df['workers'].astype(int)
df['time_sec'] = df['time_sec'].astype(float)
df['images_per_sec'] = df['images_per_sec'].astype(float)

# Helper: speedup relative to sequential w=1
def compute_speedup(sub_df):
    baseline_time = sub_df[(sub_df['method']=='sequential') & (sub_df['workers']==1)]['time_sec'].mean()
    sub_df['speedup'] = baseline_time / sub_df['time_sec']
    return sub_df

df = df.groupby(['size','count']).apply(compute_speedup)

# ===== 1) Runtime vs workers =====
for size in df['size'].unique():
    for count in df['count'].unique():
        sub = df[(df['size']==size) & (df['count']==count)]
        plt.figure()
        for method in sub['method'].unique():
            msub = sub[sub['method']==method].sort_values('workers')
            plt.errorbar(msub['workers'], msub['time_sec'], yerr=0, label=method, marker='o')
        plt.xlabel("Workers")
        plt.ylabel("Wall-clock Time (s)")
        plt.title(f"Runtime vs Workers | Size={size}, Count={count}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = PLOTS_DIR / f"runtime_w{size}_c{count}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print("Saved:", fname)

# ===== 2) Throughput vs workers =====
for size in df['size'].unique():
    for count in df['count'].unique():
        sub = df[(df['size']==size) & (df['count']==count)]
        plt.figure()
        for method in sub['method'].unique():
            msub = sub[sub['method']==method].sort_values('workers')
            plt.plot(msub['workers'], msub['images_per_sec'], label=method, marker='o')
        plt.xlabel("Workers")
        plt.ylabel("Throughput (images/sec)")
        plt.title(f"Throughput vs Workers | Size={size}, Count={count}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = PLOTS_DIR / f"throughput_w{size}_c{count}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print("Saved:", fname)

# ===== 3) Speedup vs workers =====
for size in df['size'].unique():
    for count in df['count'].unique():
        sub = df[(df['size']==size) & (df['count']==count)]
        plt.figure()
        for method in sub['method'].unique():
            msub = sub[sub['method']==method].sort_values('workers')
            plt.plot(msub['workers'], msub['speedup'], label=method, marker='o')
        plt.xlabel("Workers")
        plt.ylabel("Speedup (baseline: sequential w=1)")
        plt.title(f"Speedup vs Workers | Size={size}, Count={count}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = PLOTS_DIR / f"speedup_w{size}_c{count}.png"
        plt.sav
