# === Baseline v0.33: Δd(k) band plot + gate-effect control (PASS_primary vs PASS_full) ===
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

OUTDIR = Path("outputs/controls")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Config (match your manuscript scripts)
# -----------------------------
SEED = 12345
rng = np.random.default_rng(SEED)

N = 3_000_000
N_DIAG = 300_000

P = 0.5
eps = 0.05
cL = 0.3

k_values = np.arange(5, 51, 5)
n_seeds = 10

# Sampling domain (log10 ranges) (from your sweep scripts)
ranges = {
    "g": (-6, +6),
    "a": (-3, +3),
    "b": (-12, +2),
    "q": (-4, +4),
    "l": (-8, +8),
    "u": (-3, +3),
}

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
# 1) Draw samples
# -----------------------------
log = {k: rng.uniform(*v, size=N) for k, v in ranges.items()}
val = {k: 10**log[k] for k in log}
g = val["g"]; a = val["a"]; b = val["b"]; q = val["q"]; l = val["l"]; u = val["u"]

# -----------------------------
# 2) Gates (from eps_sweep_baseline_v033.py)
# -----------------------------
pass_b = (b >= 1e-6) & (b <= 1e2)
pass_a = (a >= 0.2) & (a <= 5.0)
pass_qbox = (q >= 1e-2) & (q <= 1e2)
S_atom = (a**2) / (u + 1e-300)
pass_Satom = (S_atom >= 0.03) & (S_atom <= 30.0)
pass_gates = pass_b & pass_a & pass_qbox & pass_Satom

# -----------------------------
# 3) Timescale proxies (baseline)
# -----------------------------
t_grow = 1.0 / np.sqrt(g * b + 1e-300)
t_H    = 1.0 / np.sqrt(1.0 + cL * np.abs(l))
t_eff  = t_grow / (np.power(q + 1e-300, P))

ratio = t_eff / (t_H + 1e-300)
PASS_primary = (ratio > eps) & (ratio < 1/eps)
PASS_full    = PASS_primary & pass_gates

# -----------------------------
# 4) Diagnostic subsample
# -----------------------------
idx_all = np.arange(N)
idx_diag = rng.choice(idx_all, size=N_DIAG, replace=False) if (N_DIAG < N) else idx_all

# Global z-score space on diagnostic subsample (log10 parameters)
X_all = np.vstack([
    log["g"][idx_diag], log["a"][idx_diag], log["b"][idx_diag],
    log["q"][idx_diag], log["l"][idx_diag], log["u"][idx_diag]
]).T
mu_all = X_all.mean(axis=0)
sd_all = X_all.std(axis=0, ddof=0) + 1e-30
Z_all  = (X_all - mu_all) / sd_all

# -----------------------------
# 5) Helper: compute PC6 var-ratio and Δd(k=20) for a PASS mask within DIAG
# -----------------------------
def pca_varratio_pc6_and_dd20(PASS_d_mask, seedbase=0):
    FAIL_d_mask = ~PASS_d_mask
    Z_pass = Z_all[PASS_d_mask]
    Z_fail = Z_all[FAIL_d_mask]

    if Z_pass.shape[0] < 300:
        return np.nan, np.nan

    pca = PCA(n_components=6).fit(Z_pass)
    T_all  = pca.transform(Z_all)
    T_pass = T_all[PASS_d_mask]
    T_fail = T_all[FAIL_d_mask]
    var_pass = np.var(T_pass, axis=0)
    var_fail = np.var(T_fail, axis=0) + 1e-30
    var_ratio_pc6 = float((var_pass / var_fail)[5])

    # Δd at k=20 (single estimate; stable enough for gate-control table)
    MAXN = 50_000
    Zp = subsample_rows(Z_pass, MAXN, seed=SEED + 100 + seedbase)
    Zf = subsample_rows(Z_fail, MAXN, seed=SEED + 200 + seedbase)
    d_pass = id_mle_levina_bickel(Zp, k=20)
    d_fail = id_mle_levina_bickel(Zf, k=20)
    delta_d = (d_fail - d_pass) if (np.isfinite(d_pass) and np.isfinite(d_fail)) else np.nan
    return var_ratio_pc6, float(delta_d) if np.isfinite(delta_d) else np.nan

# Gate-effect summary (DIAG-level, consistent with your appendix tables)
PASS_primary_d = PASS_primary[idx_diag]
PASS_full_d    = PASS_full[idx_diag]

gate_rows = []
for name, mask in [("PASS_primary", PASS_primary_d), ("PASS_full", PASS_full_d)]:
    f = float(mask.mean())
    n = int(mask.sum())
    vr6, dd20 = pca_varratio_pc6_and_dd20(mask, seedbase=0 if name=="PASS_primary" else 1)
    gate_rows.append(dict(label=name, fPASS_diag=f, N_PASS_diag=n, var_ratio_PC6=vr6, Delta_d_k20=dd20))

gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(OUTDIR / "gate_effect_baseline.csv", index=False)

# -----------------------------
# 6) Δd(k) band for PASS_full (the reported baseline)
# -----------------------------
Z_pass_full = Z_all[PASS_full_d]
Z_fail_full = Z_all[~PASS_full_d]

MAX_N_ID = 50_000
D_pass = np.zeros((n_seeds, len(k_values)))
D_fail = np.zeros((n_seeds, len(k_values)))

for si in range(n_seeds):
    Zp = subsample_rows(Z_pass_full, MAX_N_ID, seed=SEED + 1000 + si)
    Zf = subsample_rows(Z_fail_full, MAX_N_ID, seed=SEED + 2000 + si)
    for ki, kk in enumerate(k_values):
        D_pass[si, ki] = id_mle_levina_bickel(Zp, k=int(kk))
        D_fail[si, ki] = id_mle_levina_bickel(Zf, k=int(kk))

delta = D_fail - D_pass
dd_mean = np.nanmean(delta, axis=0)
dd_p16  = np.nanquantile(delta, 0.16, axis=0)
dd_p84  = np.nanquantile(delta, 0.84, axis=0)

band_df = pd.DataFrame({"k": k_values, "delta_d_mean": dd_mean, "delta_d_p16": dd_p16, "delta_d_p84": dd_p84})
band_df.to_csv(OUTDIR / "id_vs_k_baseline_band.csv", index=False)

# Plot
plt.figure(figsize=(6.2, 4.2))
plt.plot(k_values, dd_mean, marker="o", linewidth=1.5)
plt.fill_between(k_values, dd_p16, dd_p84, alpha=0.25)
plt.xlabel("k")
plt.ylabel(r"$\Delta d \equiv d_{\mathrm{FAIL}}-d_{\mathrm{PASS}}$")
plt.tight_layout()
plt.savefig(OUTDIR / "fig_id_vs_k_baseline_band.png", dpi=300)
plt.close()

print("OK: wrote")
print(" -", (OUTDIR / "fig_id_vs_k_baseline_band.png"))
print(" -", (OUTDIR / "id_vs_k_baseline_band.csv"))
print(" -", (OUTDIR / "gate_effect_baseline.csv"))
print("\nGate-effect summary:")
print(gate_df)