# scripts/sweeps/eps_sweep_baseline_v033.py
# === Regime Coherence toy model: EPS sweep (baseline v0.33) ===
# Runs baseline pipeline for eps in {0.02,0.03,0.05,0.10,0.20}
# Outputs:
#   outputs/sweeps/eps_sweep_baseline_v033.csv
#
# Notes:
# - Diagnostics computed on a retained diagnostic subsample (N_DIAG).
# - PCA is fit on PASS in global z-score space (within the diagnostic subsample).
# - Var-ratio columns are Var_PASS/Var_FAIL along PC5 and PC6 (diagnostic subsample).
# - Δd is Levina–Bickel MLE at k=20, reported only if PASS_diag count is sufficient.

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Config (keep aligned with manuscript)
# -----------------------------
SEED = 12345
rng = np.random.default_rng(SEED)

N = 3_000_000          # full Monte Carlo per eps (keep manageable in Colab)
N_DIAG = 300_000       # retained subsample for PCA/ID
P = 0.5                # baseline exponent
cL = 0.3               # lambda scale in t_H

EPS_LIST = [0.02, 0.03, 0.05, 0.10, 0.20]

# ID settings
K_ID = 20
MIN_PASS_FOR_ID = 2500      # matches your table pattern (1914,1082 -> no ID)

OUTDIR = Path("outputs/sweeps")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "eps_sweep_baseline_v033.csv"

# -----------------------------
# Sampling domain (log10 ranges)
# -----------------------------
ranges = {
    "g": (-6, +6),     # log10(G/G0)
    "a": (-3, +3),     # log10(alpha/alpha0)
    "b": (-12, +2),    # log10(etaB/etaB0)
    "q": (-4, +4),     # log10(Q/Q0)
    "l": (-8, +8),     # log10(Lambda/Lambda0)
    "u": (-3, +3),     # log10(mu/mu0)
}

def id_mle_levina_bickel(Z, k=20):
    Z = np.asarray(Z, float)
    if Z.shape[0] <= k + 5:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(Z)
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

def run_one_eps(eps: float):
    # 1) Draw samples
    log = {k: rng.uniform(*v, size=N) for k, v in ranges.items()}
    val = {k: 10**log[k] for k in log}
    g = val["g"]; a = val["a"]; b = val["b"]; q = val["q"]; l = val["l"]; u = val["u"]

    # 2) Auxiliary gates (v0.33)
    pass_b = (b >= 1e-6) & (b <= 1e2)
    pass_a = (a >= 0.2) & (a <= 5.0)
    pass_qbox = (q >= 1e-2) & (q <= 1e2)

    S_atom = (a**2) / (u + 1e-300)
    pass_Satom = (S_atom >= 0.03) & (S_atom <= 30.0)

    # 3) Timescale proxies (baseline)
    t_grow = 1.0 / np.sqrt(g * b + 1e-300)
    t_H    = 1.0 / np.sqrt(1.0 + cL * np.abs(l))
    t_eff  = t_grow / (np.power(q + 1e-300, P))

    # 4) Primary band
    ratio = t_eff / (t_H + 1e-300)
    pass_primary = (ratio > eps) & (ratio < 1/eps)

    PASS = pass_b & pass_a & pass_qbox & pass_Satom & pass_primary

    # 5) Diagnostic subsample
    idx_all = np.arange(N)
    idx_diag = rng.choice(idx_all, size=N_DIAG, replace=False) if (N_DIAG < N) else idx_all

    PASS_d = PASS[idx_diag]
    FAIL_d = ~PASS_d

    n_pass_diag = int(PASS_d.sum())
    f_pass_diag = float(n_pass_diag) / float(N_DIAG)

    # 6) Global z-score space on diagnostic subsample
    X_all = np.vstack([
        log["g"][idx_diag], log["a"][idx_diag], log["b"][idx_diag],
        log["q"][idx_diag], log["l"][idx_diag], log["u"][idx_diag]
    ]).T

    mu_all = X_all.mean(axis=0)
    sd_all = X_all.std(axis=0, ddof=0) + 1e-30
    Z_all  = (X_all - mu_all) / sd_all

    Z_pass = Z_all[PASS_d]
    Z_fail = Z_all[FAIL_d]

    # PCA fit on PASS
    if Z_pass.shape[0] < 300:
        # too few PASS in diag -> cannot stably fit PCA
        return dict(
            eps=eps,
            fPASS_diag=f_pass_diag,
            var_ratio_PC5=np.nan,
            var_ratio_PC6=np.nan,
            Delta_d_k20=np.nan,
            N_PASS_diag=n_pass_diag,
        )

    pca = PCA(n_components=6).fit(Z_pass)
    T_all  = pca.transform(Z_all)
    T_pass = T_all[PASS_d]
    T_fail = T_all[FAIL_d]

    var_pass = np.var(T_pass, axis=0)
    var_fail = np.var(T_fail, axis=0) + 1e-30
    var_ratio = var_pass / var_fail

    # ID (k=20) only if PASS count sufficient
    if n_pass_diag >= MIN_PASS_FOR_ID:
        d_pass = id_mle_levina_bickel(Z_pass, k=K_ID)
        d_fail = id_mle_levina_bickel(Z_fail, k=K_ID)
        delta_d = (d_fail - d_pass) if (np.isfinite(d_pass) and np.isfinite(d_fail)) else np.nan
    else:
        delta_d = np.nan

    return dict(
        eps=eps,
        fPASS_diag=f_pass_diag,
        var_ratio_PC5=float(var_ratio[4]),
        var_ratio_PC6=float(var_ratio[5]),
        Delta_d_k20=float(delta_d) if np.isfinite(delta_d) else np.nan,
        N_PASS_diag=n_pass_diag,
    )

# -----------------------------
# Main
# -----------------------------
rows = []
for eps in EPS_LIST:
    print(f"\n--- Running eps={eps:.2f} ---")
    row = run_one_eps(eps)
    rows.append(row)
    print("fPASS_diag =", row["fPASS_diag"])
    print("Var-ratio PC5 =", row["var_ratio_PC5"], " | PC6 =", row["var_ratio_PC6"])
    print("Delta_d(k=20) =", row["Delta_d_k20"])
    print("N_PASS_diag =", row["N_PASS_diag"])

df = pd.DataFrame(rows, columns=[
    "eps", "fPASS_diag", "var_ratio_PC5", "var_ratio_PC6", "Delta_d_k20", "N_PASS_diag"
])

df.to_csv(OUTCSV, index=False)
print("\nSaved:", str(OUTCSV.resolve()))
print("DONE.")