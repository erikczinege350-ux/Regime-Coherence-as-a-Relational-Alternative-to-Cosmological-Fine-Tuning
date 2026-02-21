# === Regime Coherence toy model: BASELINE v0.33 (main-text figures: PCA twopanel, PC5–PC6, ID vs k) ===
# One-cell Colab: sampling + gates + PCA diagnostics + Figure outputs (B/W) + ID table
#
# Outputs saved into: ./toy_v033_baseline_maintext_bw/
# - fig_pca_twopanel_baseline.png/.pdf   (PC1–PC2 scatter + PC6 histogram)
# - fig_pc5_pc6_baseline.png/.pdf       (PC5–PC6 transverse view)
# - fig_id_vs_k_baseline.png/.pdf       (Intrinsic dimension vs k)
# - ID_summary_baseline.csv
#
# Notes:
# - All plots are black/white (grayscale) and titles contain NO "Figure X" strings.
# - MODE fixed to baseline (2-timescale). No 3τ here.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 0) Repro + settings
# -----------------------------
rng = np.random.default_rng(12345)

N = 30000000        # full Monte Carlo
N_DIAG = 300_000       # retained subsample for PCA/ID
P = 0.5                # baseline exponent
eps = 0.05             # relational band: eps < t_eff/t_H < 1/eps
cL = 0.3               # lambda scale in t_H

MAX_PLOT = 50_000
BINS = 140

OUTDIR = Path("./toy_v033_baseline_maintext_bw")
OUTDIR.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Global B/W plotting style
# -----------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.grid": False,
    "font.size": 11,
})

# -----------------------------
# 1) Sampling domain (log10 ranges)
# -----------------------------
ranges = {
    "g": (-6, +6),     # log10(G/G0)
    "a": (-3, +3),     # log10(alpha/alpha0)
    "b": (-12, +2),    # log10(etaB/etaB0)
    "q": (-4, +4),     # log10(Q/Q0)
    "l": (-8, +8),     # log10(Lambda/Lambda0)
    "u": (-3, +3),     # log10(mu/mu0)
}

# -----------------------------
# 2) Draw samples
# -----------------------------
log = {k: rng.uniform(*v, size=N) for k, v in ranges.items()}
val = {k: 10**log[k] for k in log}
g = val["g"]; a = val["a"]; b = val["b"]; q = val["q"]; l = val["l"]; u = val["u"]

# -----------------------------
# 3) Auxiliary gates (v0.33)
# -----------------------------
b_min, b_max = 1e-6, 1e2
pass_b = (b >= b_min) & (b <= b_max)

a_min, a_max = 0.2, 5.0
pass_a = (a >= a_min) & (a <= a_max)

q_min, q_max = 1e-2, 1e2
pass_qbox = (q >= q_min) & (q <= q_max)

S_atom = (a**2) / (u + 1e-300)
Smin, Smax = 0.03, 30.0
pass_Satom = (S_atom >= Smin) & (S_atom <= Smax)

# -----------------------------
# 4) Timescale proxies (baseline)
# -----------------------------
t_grow = 1.0 / np.sqrt(g * b + 1e-300)
t_H    = 1.0 / np.sqrt(1.0 + cL * np.abs(l))
t_eff  = t_grow / (np.power(q + 1e-300, P))

# -----------------------------
# 5) Primary relational constraint (baseline)
# -----------------------------
ratio = t_eff / (t_H + 1e-300)
pass_primary = (ratio > eps) & (ratio < 1/eps)

PASS = pass_b & pass_a & pass_qbox & pass_Satom & pass_primary
FAIL = ~PASS

# -----------------------------
# 6) Report
# -----------------------------
n_pass = int(PASS.sum())
f_pass = float(PASS.mean())

print("=== CONFIG (BASELINE) ===")
print(f"N = {N:,} | N_DIAG = {N_DIAG:,} | P = {P} | eps = {eps} | cL = {cL}")
print("\n=== CORE ===")
print(f"PASS = {n_pass:,}  (f_PASS = {f_pass:.6f})")
print("\nGate pass rates (p-independent):")
print(f"  pass_b     = {pass_b.mean():.4f}")
print(f"  pass_a     = {pass_a.mean():.4f}")
print(f"  pass_qbox  = {pass_qbox.mean():.4f}")
print(f"  pass_Satom = {pass_Satom.mean():.4f}")
print(f"Primary pass rate (t_eff/t_H band) = {pass_primary.mean():.6f}")

# -----------------------------
# 7) Retain diagnostic subsample
# -----------------------------
idx_all = np.arange(N)
idx_diag = rng.choice(idx_all, size=N_DIAG, replace=False) if (N_DIAG < N) else idx_all

log_diag = {k: log[k][idx_diag] for k in log}
PASS_d = PASS[idx_diag]
FAIL_d = ~PASS_d

# -----------------------------
# 8) PCA in GLOBAL z-score space (fit on PASS)
# -----------------------------
cols_order = ["g","a","b","q","l","u"]
X_all = np.vstack([log_diag["g"], log_diag["a"], log_diag["b"], log_diag["q"], log_diag["l"], log_diag["u"]]).T

mu_all = X_all.mean(axis=0)
sd_all = X_all.std(axis=0, ddof=0) + 1e-30
Z_all  = (X_all - mu_all) / sd_all

Z_pass = Z_all[PASS_d]
Z_fail = Z_all[FAIL_d]

assert Z_pass.shape[0] > 200, "Too few PASS in diagnostic subsample — relax eps or increase N_DIAG."

pca = PCA(n_components=6).fit(Z_pass)
evr = pca.explained_variance_ratio_
print("\n=== PCA (PASS-fit, global z-score) ===")
print("EVR:", np.round(evr, 4))
print("PCA4 =", float(np.sum(evr[:4])))
print("PC6  =", float(evr[-1]))

T_all  = pca.transform(Z_all)
T_pass = T_all[PASS_d]
T_fail = T_all[FAIL_d]

# -----------------------------
# 8b) SAVE diagnostics for clustering (FAIL-side)
# -----------------------------
DUMP = OUTDIR / "diag_pca_dump_baseline.npz"
np.savez(
    DUMP,
    T_fail=T_fail.astype(np.float32),   # FAIL in PASS-fit PCA space (PC1..PC6)
    T_pass=T_pass.astype(np.float32),   # optional
    PASS_d=PASS_d.astype(np.uint8),
    mu_all=mu_all.astype(np.float64),
    sd_all=sd_all.astype(np.float64),
    evr=evr.astype(np.float64),
    meta=np.array([N, N_DIAG, P, eps, cL], dtype=np.float64),
)
print("Saved:", DUMP.resolve())


var_pass = np.var(T_pass, axis=0)
var_fail = np.var(T_fail, axis=0) + 1e-30
var_ratio = var_pass / var_fail

print("\n=== Variance ratios (PASS/FAIL) by PC ===")
for i in range(6):
    print(f"PC{i+1}: {var_ratio[i]:.4f}")

# downsample for plotting
idxP = np.where(PASS_d)[0]
idxF = np.where(FAIL_d)[0]
k_pass = min(MAX_PLOT, idxP.size)
k_fail = min(MAX_PLOT, idxF.size)
plot_pass = rng.choice(idxP, size=k_pass, replace=False)
plot_fail = rng.choice(idxF, size=k_fail, replace=False)

pc1 = T_all[:,0]; pc2 = T_all[:,1]; pc5 = T_all[:,4]; pc6 = T_all[:,5]

# -----------------------------
# 9) Figure 1 (main text): PCA twopanel (PC1–PC2 + PC6 histogram)
# -----------------------------
fig = plt.figure(figsize=(13.2, 5.4))

ax1 = fig.add_subplot(1,2,1)
ax1.scatter(pc1[plot_fail], pc2[plot_fail], s=2, c="0.80", alpha=0.10, marker=".", linewidths=0, rasterized=True)
ax1.scatter(pc1[plot_pass], pc2[plot_pass], s=6, c="0.05", alpha=0.85, marker="o", linewidths=0, rasterized=True)
ax1.set_xlabel("PC1 (PASS-fit PCA; global z-score)")
ax1.set_ylabel("PC2 (PASS-fit PCA; global z-score)")
ax1.set_title("PASS vs FAIL in PASS-fitted PCA basis")

ax2 = fig.add_subplot(1,2,2)
lo, hi = np.quantile(pc6[idxF], [0.005, 0.995])
ax2.hist(pc6[idxF], bins=BINS, range=(lo,hi), density=True, color="0.80", alpha=0.85, label="FAIL")
ax2.hist(pc6[idxP], bins=BINS, range=(lo,hi), density=True, color="0.10", alpha=0.75, label="PASS")
ax2.set_xlabel("PC6 projection (normal direction)")
ax2.set_ylabel("density")
ax2.set_title("Transverse compression along PC6")
ax2.legend(frameon=False)

fig.tight_layout()
f1_png = OUTDIR / "fig_pca_twopanel_baseline.png"
f1_pdf = OUTDIR / "fig_pca_twopanel_baseline.pdf"
fig.savefig(f1_png, dpi=320, bbox_inches="tight")
fig.savefig(f1_pdf, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", f1_png.resolve())

# -----------------------------
# 10) Figure 2 (main text): Transverse compression view (PC5–PC6 scatter)
# -----------------------------
fig = plt.figure(figsize=(6.4, 5.6))
ax = fig.add_subplot(1,1,1)
ax.scatter(pc5[plot_fail], pc6[plot_fail], s=2, c="0.80", alpha=0.12, marker=".", linewidths=0, rasterized=True)
ax.scatter(pc5[plot_pass], pc6[plot_pass], s=6, c="0.05", alpha=0.85, marker="o", linewidths=0, rasterized=True)
ax.set_xlabel("PC5 (PASS-fit PCA; global z-score)")
ax.set_ylabel("PC6 (PASS-fit PCA; global z-score)")
ax.set_title("Transverse structure (PC5–PC6)")
fig.tight_layout()
f2_png = OUTDIR / "fig_pc5_pc6_baseline.png"
f2_pdf = OUTDIR / "fig_pc5_pc6_baseline.pdf"
fig.savefig(f2_png, dpi=320, bbox_inches="tight")
fig.savefig(f2_pdf, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", f2_png.resolve())

# -----------------------------
# 11) Intrinsic dimension: TwoNN + Levina–Bickel MLE sweep (ID vs k)
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
    nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(Z)
    dists, _ = nn.kneighbors(Z, return_distance=True)
    dists = dists[:, 1:]  # drop self
    dists = np.maximum(dists, 1e-15)
    rk = dists[:, -1]
    logs = np.log(rk[:, None] / dists[:, :-1])
    denom = np.sum(logs, axis=1)
    good = np.isfinite(denom) & (denom > 0)
    if good.sum() < max(200, 0.2 * Z.shape[0]):
        return np.nan
    d_hat = (k - 1) / denom[good]
    d_hat = d_hat[np.isfinite(d_hat) & (d_hat > 0)]
    return float(np.median(d_hat)) if d_hat.size else np.nan

MAX_N_ID_PASS = 50_000
MAX_N_ID_FAIL = 50_000

def subsample_rows(Z, nmax, seed=0):
    if Z.shape[0] <= nmax:
        return Z
    rr = np.random.default_rng(seed)
    idx = rr.choice(Z.shape[0], size=nmax, replace=False)
    return Z[idx]

Zp_id = subsample_rows(Z_pass, MAX_N_ID_PASS, seed=0)
Zf_id = subsample_rows(Z_fail, MAX_N_ID_FAIL, seed=1)

twonn_pass = id_twonn(Zp_id)
twonn_fail = id_twonn(Zf_id)
print("\n=== TwoNN (global z-space; subsampled) ===")
print(f"TwoNN_PASS = {twonn_pass:.4f}")
print(f"TwoNN_FAIL = {twonn_fail:.4f}")

k_values = np.arange(5, 51, 5)
n_seeds = 10
seeds = np.arange(n_seeds)

D_pass = np.zeros((n_seeds, len(k_values)))
D_fail = np.zeros((n_seeds, len(k_values)))

for si, s in enumerate(seeds):
    Zp = subsample_rows(Z_pass, MAX_N_ID_PASS, seed=int(1000 + s))
    Zf = subsample_rows(Z_fail, MAX_N_ID_FAIL, seed=int(2000 + s))
    for ki, k in enumerate(k_values):
        D_pass[si, ki] = id_mle_levina_bickel(Zp, k=int(k))
        D_fail[si, ki] = id_mle_levina_bickel(Zf, k=int(k))

pass_mean = np.nanmean(D_pass, axis=0)
pass_std  = np.nanstd(D_pass, axis=0)
fail_mean = np.nanmean(D_fail, axis=0)
fail_std  = np.nanstd(D_fail, axis=0)

# -----------------------------
# 12) Figure 3 (main text): ID vs k
# -----------------------------
plt.figure(figsize=(6.8, 5.0))
plt.plot(k_values, pass_mean, marker="o", color="0.0", label="PASS")
plt.fill_between(k_values, pass_mean - pass_std, pass_mean + pass_std, color="0.0", alpha=0.12)
plt.plot(k_values, fail_mean, marker="o", color="0.35", label="FAIL")
plt.fill_between(k_values, fail_mean - fail_std, fail_mean + fail_std, color="0.35", alpha=0.12)
plt.xlabel("k (neighbourhood size)")
plt.ylabel("Intrinsic dimension estimate d (Levina–Bickel MLE)")
plt.title("Intrinsic dimension vs neighbourhood scale")
plt.legend(frameon=False)
plt.tight_layout()

f3_png = OUTDIR / "fig_id_vs_k_baseline.png"
f3_pdf = OUTDIR / "fig_id_vs_k_baseline.pdf"
plt.savefig(f3_png, dpi=320, bbox_inches="tight")
plt.savefig(f3_pdf, bbox_inches="tight")
plt.show()
plt.close()
print("Saved:", f3_png.resolve())

# Save ID summary table
tab = pd.DataFrame({
    "k": k_values,
    "d_PASS_mean": pass_mean,
    "d_PASS_std": pass_std,
    "d_FAIL_mean": fail_mean,
    "d_FAIL_std": fail_std,
})
tab["Delta_d(FAIL-PASS)"] = tab["d_FAIL_mean"] - tab["d_PASS_mean"]
id_csv = OUTDIR / "ID_summary_baseline.csv"
tab.to_csv(id_csv, index=False)
print("Saved:", id_csv.resolve())

print("\nDONE.")
print("Key numbers:")
print("f_PASS =", f_pass)
print("Var ratios PC5/PC6 (PASS/FAIL):", float(var_ratio[4]), float(var_ratio[5]))
print("Δd range (FAIL-PASS) over k:", float(np.nanmin(tab['Delta_d(FAIL-PASS)'])), float(np.nanmax(tab['Delta_d(FAIL-PASS)'])))
