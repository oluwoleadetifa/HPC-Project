# analyze_results_combined.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Configure seaborn for nicer plots
sns.set(style="whitegrid")

RESULTS_ROOT = Path("../results")
CSV_FILE = RESULTS_ROOT / "timings.csv"
PLOTS_DIR = RESULTS_ROOT / "plots_combined"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CSV_FILE)
df['workers'] = df['workers'].astype(int)
df['time_sec'] = df['time_sec'].astype(float)
df['images_per_sec'] = df['images_per_sec'].astype(float)

# Compute speedup relative to sequential w=1
def compute_speedup(sub_df):
    baseline_time = sub_df[(sub_df['method']=='sequential') & (sub_df['workers']==1)]['time_sec'].mean()
    sub_df['speedup'] = baseline_time / sub_df['time_sec']
    return sub_df

df = df.groupby(['size','count']).apply(compute_speedup)

# Metrics to plot
metrics = [
    ("time_sec", "Wall-clock Time (s)"),
    ("images_per_sec", "Throughput (images/sec)"),
    ("speedup", "Speedup (baseline: sequential w=1)")
]

for size in df['size'].unique():
    sub = df[df['size']==size]
    for metric_col, metric_label in metrics:
        plt.figure(figsize=(10,6))
        for method in sub['method'].unique():
            method_sub = sub[sub['method']==method]
            # Plot each dataset count separately
            for count in method_sub['count'].unique():
                count_sub = method_sub[method_sub['count']==count].sort_values('workers')
                label = f"{method} | {count} imgs"
                plt.plot(count_sub['workers'], count_sub[metric_col], marker='o', label=label)

        plt.xlabel("Workers",  fontsize=18)
        plt.ylabel(metric_label, fontsize=18)
        plt.title(f"{metric_label} vs Workers | Image Size={size}", fontsize=18)
        plt.xticks(sorted(sub['workers'].unique()))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = PLOTS_DIR / f"{metric_col}_size{size}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print("Saved:", fname)

print("All combined plots saved in:", PLOTS_DIR.resolve())
