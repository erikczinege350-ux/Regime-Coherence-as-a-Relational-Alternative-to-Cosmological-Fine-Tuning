# scripts/sweeps/n_sweep_baseline_v033.py
# Baseline v0.33 N-sweep (3M, 30M, 70M, 90M) -> one CSV summary
# Uses the same logic as your baseline: sampling + gates + PASS-fit PCA in global z-space + ID diagnostics.

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Fixed config (match manuscript)
# -----------------------------
P = 0.5
eps = 0.05
cL = 0.3
N_DIAG = 300_000
SEED = 12345

N_LIST = [3_000_000, 30_000_000, 70_000_000, 90_000_000]

# Outputs
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../repoblokk
OUTDIR = REPO_ROOT / "outputs" / "sweeps"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "n_sweep_baseline_v033.csv"

# -----------------------------
# Helpers: ID estimators
# -----------------------------
def id_twonn(Z):
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(Z)
    d, _ = nn.kneighbors(Z, return_distance=True)
    r1 = d[:, 1] + 1e-30
    r2 = d[:, 2] + 1e-30
    mu_ratio = r2 / r1
    mu_ratio = mu_ratio[np.isfinite(mu_ratio) & (mu_ratio > 1.0)]
    y = np.log(mu_ratio)
    y = y[np.isfinite(y)]
    if y.size < 200:
        return np.nan
    return float(1.0 / np.mean(y))

def id_mle_levina_bickel(Z, k=20):
    Z = np.asarray(Z, float)
    if Z.shape[0] <= k + 5:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
    dists, _ = nn.kneighbors(Z, return_distance=True)
    dists = dists[:, 1:]  # drop self
    dists = np.maximum(dists, 1e-15)
    rk = dists[:, -1]
    logs = np.log(rk[:, None] / dists[:, :-1])
    denom = np.sum(logs, axis=1)
    good = np.isfinite(denom) & (denom > 0)
    if good.sum() < max(200, int(0.2 * Z.shape[0])):
        return np.nan
    d_hat = (k - 1) / denom[good]
    d_hat = d_hat[np.isfinite(d_hat) & (d_hat > 0)]
    return float(np.median(d_hat)) if d_hat.size else np.nan

def subsample_rows(Z, nmax, seed):
    if Z.shape[0] <= nmax:
        return Z
    rr = np.random.default_rng(seed)
    idx = rr.choice(Z.shape[0], size=nmax, replace=False)
    return Z[idx]

# -----------------------------
# Core: one run
# -----------------------------
def run_one(N, seed=SEED):
    rng = np.random.default_rng(seed)

    # Sampling domain (log10 ranges)
    ranges = {
        "g": (-6, +6),
        "a": (-3, +3),
        "b": (-12, +2),
        "q": (-4, +4),
        "l": (-8, +8),
        "u": (-3, +3),
    }

    # Draw samples
    log = {k: rng.uniform(*v, size=N) for k, v in ranges.items()}
    val = {k: 10 ** log[k] for k in log}
    g = val["g"]; a = val["a"]; b = val["b"]; q = val["q"]; l = val["l"]; u = val["u"]

    # Auxiliary gates (v0.33)
    pass_b = (b >= 1e-6) & (b <= 1e2)
    pass_a = (a >= 0.2) & (a <= 5.0)
    pass_qbox = (q >= 1e-2) & (q <= 1e2)
    S_atom = (a ** 2) / (u + 1e-300)
    pass_Satom = (S_atom >= 0.03) & (S_atom <= 30.0)

    # Timescale proxies (baseline)
    t_grow = 1.0 / np.sqrt(g * b + 1e-300)
    t_H = 1.0 / np.sqrt(1.0 + cL * np.abs(l))
    t_eff = t_grow / (np.power(q + 1e-300, P))

    # Primary relational band
    ratio = t_eff / (t_H + 1e-300)
    pass_primary = (ratio > eps) & (ratio < 1 / eps)

    PASS = pass_b & pass_a & pass_qbox & pass_Satom & pass_primary
    f_pass = float(PASS.mean())

    # Diagnostic subsample
    idx_all = np.arange(N)
    idx_diag = rng.choice(idx_all, size=N_DIAG, replace=False) if (N_DIAG < N) else idx_all

    log_diag = {k: log[k][idx_diag] for k in log}
    PASS_d = PASS[idx_diag]
    FAIL_d = ~PASS_d

    # PCA in global z-score (fit on PASS)
    X_all = np.vstack([
        log_diag["g"], log_diag["a"], log_diag["b"],
        log_diag["q"], log_diag["l"], log_diag["u"]
    ]).T

    mu_all = X_all.mean(axis=0)
    sd_all = X_all.std(axis=0, ddof=0) + 1e-30
    Z_all = (X_all - mu_all) / sd_all
    Z_pass = Z_all[PASS_d]
    Z_fail = Z_all[FAIL_d]

    if Z_pass.shape[0] <= 200:
        raise RuntimeError("Too few PASS in diagnostic subsample for stable PCA/ID. Increase N or N_DIAG.")

    pca = PCA(n_components=6).fit(Z_pass)
    evr = pca.explained_variance_ratio_
    PCA4 = float(np.sum(evr[:4]))
    PCA6 = float(evr[-1])

    T_all = pca.transform(Z_all)
    T_pass = T_all[PASS_d]
    T_fail = T_all[FAIL_d]

    var_pass = np.var(T_pass, axis=0)
    var_fail = np.var(T_fail, axis=0) + 1e-30
    var_ratio_PC6 = float((var_pass / var_fail)[5])

    # ID (subsample for speed)
    MAX_N_ID_PASS = 50_000
    MAX_N_ID_FAIL = 50_000
    Zp_id = subsample_rows(Z_pass, MAX_N_ID_PASS, seed=seed + 10)
    Zf_id = subsample_rows(Z_fail, MAX_N_ID_FAIL, seed=seed + 20)

    twonn_pass = id_twonn(Zp_id)
    twonn_fail = id_twonn(Zf_id)

    # Levina–Bickel MLE sweep (k=5..50 step 5), 10 seeds bands
    k_values = np.arange(5, 51, 5)
    n_seeds = 10
    D_pass = np.zeros((n_seeds, len(k_values)))
    D_fail = np.zeros((n_seeds, len(k_values)))

    for si in range(n_seeds):
        Zp = subsample_rows(Z_pass, MAX_N_ID_PASS, seed=seed + 1000 + si)
        Zf = subsample_rows(Z_fail, MAX_N_ID_FAIL, seed=seed + 2000 + si)
        for ki, k in enumerate(k_values):
            D_pass[si, ki] = id_mle_levina_bickel(Zp, k=int(k))
            D_fail[si, ki] = id_mle_levina_bickel(Zf, k=int(k))

    pass_mean = np.nanmean(D_pass, axis=0)
    fail_mean = np.nanmean(D_fail, axis=0)
    delta = fail_mean - pass_mean
    delta_d_min = float(np.nanmin(delta))
    delta_d_max = float(np.nanmax(delta))

    return {
        "N": int(N),
        "N_DIAG": int(N_DIAG),
        "f_PASS": f_pass,
        "PCA4": PCA4,
        "PCA6": PCA6,
        "var_ratio_PC6": var_ratio_PC6,
        "twonn_PASS": float(twonn_pass),
        "twonn_FAIL": float(twonn_fail),
        "delta_d_min": delta_d_min,
        "delta_d_max": delta_d_max,
    }

# -----------------------------
# Run sweep
# -----------------------------
rows = []
for N in N_LIST:
    print(f"\n--- Running N={N:,} ---")
    out = run_one(N)
    print("f_PASS =", out["f_PASS"])
    print("PCA4  =", out["PCA4"], " | PCA6 =", out["PCA6"])
    print("var_ratio_PC6 =", out["var_ratio_PC6"])
    print("Δd range =", out["delta_d_min"], out["delta_d_max"])
    rows.append(out)

df = pd.DataFrame(rows)
df.to_csv(OUTCSV, index=False)
print("\nSaved:", OUTCSV.resolve())
print("\nDONE.")