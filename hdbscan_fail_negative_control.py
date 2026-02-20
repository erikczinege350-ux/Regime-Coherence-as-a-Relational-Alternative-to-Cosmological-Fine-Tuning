import sys, subprocess, numpy as np
from pathlib import Path

# --- input / output ---
INP = Path("/content/toy_v033_baseline_maintext_bw/diag_pca_dump_baseline.npz")
OUTDIR = Path("/content/outputs/controls")
OUTDIR.mkdir(parents=True, exist_ok=True)

assert INP.exists(), f"Missing: {INP}"

# --- ensure hdbscan installed ---
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "hdbscan"])

import hdbscan
import pandas as pd

# --- load FAIL points in PASS-fit PCA space ---
dat = np.load(INP, allow_pickle=True)
T_fail = dat["T_fail"]  # shape (n_fail, 6)
n = T_fail.shape[0]
print("Loaded T_fail:", T_fail.shape)

# --- subsample exactly n=60,000 (as Appendix) ---
rng = np.random.default_rng(12345)
m = 60_000
idx = rng.choice(n, size=m, replace=False) if n > m else np.arange(n)
X = T_fail[idx].astype(np.float32, copy=False)
print("Subsample X:", X.shape)

# --- configs matching Appendix D.4 table ---
configs = [
    {"mcs": 100, "ms": None},
    {"mcs": 200, "ms": None},
    {"mcs": 50,  "ms": 10},
]

rows = []
for cfg in configs:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["mcs"],
        min_samples=cfg["ms"],
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    noise = int(np.sum(labels == -1))
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    frac = float((m - noise) / m)
    rows.append([cfg["mcs"], cfg["ms"] if cfg["ms"] is not None else "None", n_clusters, f"{noise}/{m}", frac])

df = pd.DataFrame(rows, columns=["mcs", "ms", "clusters", "noise / n", "clustered fraction"])
print("\nHDBSCAN FAIL negative-control table:")
print(df.to_string(index=False))

# --- save CSV for repo + manuscript copy-paste ---
out_csv = OUTDIR / "hdbscan_fail_negative_control.csv"
df.to_csv(out_csv, index=False)
print("\nSaved:", out_csv)