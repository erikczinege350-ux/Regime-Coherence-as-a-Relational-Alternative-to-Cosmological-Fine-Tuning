# === Regime Coherence toy model: THREE-TIMESCALE (3τ) v0.33 — Appendix D.5 ===
# Reproducible: fixed seed, fixed gates, PASS-fit PCA in global z-score space, ID vs k
# Outputs -> /content/outputs/three-tau/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 0) Fixed config (Appendix D.5)
# -----------------------------
SEED = 12345
rng = np.random.default_rng(SEED)

N = 3_000_000
N_DIAG = 300_000

P  = 0.5
PN = 1.0
cL = 0.3

eps1_lo, eps1_hi = 0.05, 20.0   # t_grow / t_nl
eps2_lo, eps2_hi = 0.05, 20.0   # t_nl / t_H

MAX_PLOT = 50_000
BINS = 140

OUTDIR = Path("./outputs/three-tau")
OUTDIR.mkdir(exist_ok=True, parents=True)

# B/W plotting style (journal-safe)
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
    "g": (-6, +6),
    "a": (-3, +3),
    "b": (-12, +2),
    "q": (-4, +4),
    "l": (-8, +8),
    "u": (-3, +3),
}

log = {k: rng.uniform(*v, size=N) for k, v in ranges.items()}
val = {k: 10**log[k] for k in log}

g = val["g"]; a = val["a"]; b = val["b"]; q = val["q"]; l = val["l"]; u = val["u"]

# -----------------------------
# 2) Auxiliary gates (v0.33)
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
# 3) Timescale proxies (3τ)
# -----------------------------
t_grow = 1.0 / np.sqrt(g * b + 1e-300)
t_H    = 1.0 / np.sqrt(1.0 + cL * np.abs(l))

# middle scale (explicitly distinct from baseline t_eff exponent if PN != P)
t_nl = t_grow / (np.power(q + 1e-300, PN))

ratio1 = t_grow / (t_nl + 1e-300)
ratio2 = t_nl   / (t_H  + 1e-300)

pass_band1 = (ratio1 > eps1_lo) & (ratio1 < eps1_hi)
pass_band2 = (ratio2 > eps2_lo) & (ratio2 < eps2_hi)
pass_primary = pass_band1 & pass_band2

PASS = pass_b & pass_a & pass_qbox & pass_Satom & pass_primary
FAIL = ~PASS

n_pass = int(PASS.sum())
f_pass = float(PASS.mean())

print("=== CONFIG (3τ) ===")
print(f"SEED={SEED} | N={N:,} | N_DIAG={N_DIAG:,} | P={P} | PN={PN} | cL={cL}")
print(f"bands: (t_grow/t_nl) in [{eps1_lo},{eps1_hi}] AND (t_nl/t_H) in [{eps2_lo},{eps2_hi}]")
print("=== CORE ===")
print(f"PASS = {n_pass:,}  (f_PASS = {f_pass:.5f})")
print("Gate pass rates:")
print(f"  pass_b     = {pass_b.mean():.4f}")
print(f"  pass_a     = {pass_a.mean():.4f}")
print(f"  pass_qbox  = {pass_qbox.mean():.4f}")
print(f"  pass_Satom = {pass_Satom.mean():.4f}")
print(f"  pass_primary (3τ bands) = {pass_primary.mean():.6f}")

# -----------------------------
# 4) Diagnostic subsample
# -----------------------------
idx_all = np.arange(N)
idx_diag = rng.choice(idx_all, size=N_DIAG, replace=False) if (N_DIAG < N) else idx_all

PASS_d = PASS[idx_diag]
FAIL_d = ~PASS_d

log_diag = {k: log[k][idx_diag] for k in log}

# -----------------------------
# 5) PCA in GLOBAL z-score space (fit on PASS)
# -----------------------------
X_all = np.vstack([
    log_diag["g"], log_diag["a"], log_diag["b"],
    log_diag["q"], log_diag["l"], log_diag["u"]
]).T

mu_all = X_all.mean(axis=0)
sd_all = X_all.std(axis=0, ddof=0) + 1e-30
Z_all  = (X_all - mu_all) / sd_all

Z_pass = Z_all[PASS_d]
Z_fail = Z_all[FAIL_d]
assert Z_pass.shape[0] > 200, "Too few PASS in diagnostic subsample."

pca = PCA(n_components=6).fit(Z_pass)
evr = pca.explained_variance_ratio_
PCA4 = float(np.sum(evr[:4]))
PCA6 = float(evr[-1])

T_all  = pca.transform(Z_all)
T_pass = T_all[PASS_d]
T_fail = T_all[FAIL_d]

var_pass = np.var(T_pass, axis=0)
var_fail = np.var(T_fail, axis=0) + 1e-30
var_ratio = var_pass / var_fail

print("\n=== PCA (PASS-fit, global z-score) ===")
print("EVR:", np.round(evr, 4))
print("PCA4 =", PCA4, " | PCA6 =", PCA6)
print("Var ratios (PASS/FAIL): PC5 =", float(var_ratio[4]), " PC6 =", float(var_ratio[5]))

# Save dump (optional but useful)
DUMP = OUTDIR / "diag_pca_dump_three.npz"
np.savez(
    DUMP,
    T_fail=T_fail.astype(np.float32),
    T_pass=T_pass.astype(np.float32),
    PASS_d=PASS_d.astype(np.uint8),
    mu_all=mu_all.astype(np.float64),
    sd_all=sd_all.astype(np.float64),
    evr=evr.astype(np.float64),
    meta=np.array([N, N_DIAG, P, PN, cL, eps1_lo, eps1_hi, eps2_lo, eps2_hi, SEED], dtype=np.float64),
)
print("Saved:", DUMP.resolve())

# -----------------------------
# 6) Figures (same style as baseline)
# -----------------------------
idxP = np.where(PASS_d)[0]
idxF = np.where(FAIL_d)[0]
k_pass = min(MAX_PLOT, idxP.size)
k_fail = min(MAX_PLOT, idxF.size)
plot_pass = rng.choice(idxP, size=k_pass, replace=False)
plot_fail = rng.choice(idxF, size=k_fail, replace=False)

pc1 = T_all[:,0]; pc2 = T_all[:,1]; pc5 = T_all[:,4]; pc6 = T_all[:,5]

# (a) twopanel: PC1–PC2 + PC6 hist
fig = plt.figure(figsize=(13.2, 5.4))
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(pc1[plot_fail], pc2[plot_fail], s=2, c="0.80", alpha=0.10, marker=".", linewidths=0, rasterized=True)
ax1.scatter(pc1[plot_pass], pc2[plot_pass], s=6, c="0.05", alpha=0.85, marker="o", linewidths=0, rasterized=True)
ax1.set_xlabel("PC1 (PASS-fit PCA; global z-score)")
ax1.set_ylabel("PC2 (PASS-fit PCA; global z-score)")
ax1.set_title("PASS vs FAIL in PASS-fitted PCA basis (3τ)")

ax2 = fig.add_subplot(1,2,2)
lo, hi = np.quantile(pc6[idxF], [0.005, 0.995])
ax2.hist(pc6[idxF], bins=BINS, range=(lo,hi), density=True, color="0.80", alpha=0.85, label="FAIL")
ax2.hist(pc6[idxP], bins=BINS, range=(lo,hi), density=True, color="0.10", alpha=0.75, label="PASS")
ax2.set_xlabel("PC6 projection (normal direction)")
ax2.set_ylabel("density")
ax2.set_title("Transverse compression along PC6 (3τ)")
ax2.legend(frameon=False)

fig.tight_layout()
f1_png = OUTDIR / "fig_pca_twopanel_three.png"
f1_pdf = OUTDIR / "fig_pca_twopanel_three.pdf"
fig.savefig(f1_png, dpi=320, bbox_inches="tight")
fig.savefig(f1_pdf, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", f1_png.resolve())

# (b) PC5–PC6 scatter
fig = plt.figure(figsize=(6.4, 5.6))
ax = fig.add_subplot(1,1,1)
ax.scatter(pc5[plot_fail], pc6[plot_fail], s=2, c="0.80", alpha=0.12, marker=".", linewidths=0, rasterized=True)
ax.scatter(pc5[plot_pass], pc6[plot_pass], s=6, c="0.05", alpha=0.85, marker="o", linewidths=0, rasterized=True)
ax.set_xlabel("PC5 (PASS-fit PCA; global z-score)")
ax.set_ylabel("PC6 (PASS-fit PCA; global z-score)")
ax.set_title("Transverse structure (PC5–PC6, 3τ)")
fig.tight_layout()
f2_png = OUTDIR / "fig_pc5_pc6_three.png"
f2_pdf = OUTDIR / "fig_pc5_pc6_three.pdf"
fig.savefig(f2_png, dpi=320, bbox_inches="tight")
fig.savefig(f2_pdf, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", f2_png.resolve())

# -----------------------------
# 7) Intrinsic dimension: TwoNN + Levina–Bickel MLE (k=5..50)
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
    dists = dists[:, 1:]
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

def subsample_rows(Z, nmax, seed=0):
    if Z.shape[0] <= nmax:
        return Z
    rr = np.random.default_rng(seed)
    idx = rr.choice(Z.shape[0], size=nmax, replace=False)
    return Z[idx]

MAX_N_ID_PASS = 50_000
MAX_N_ID_FAIL = 50_000

Zp_id = subsample_rows(Z_pass, MAX_N_ID_PASS, seed=0)
Zf_id = subsample_rows(Z_fail, MAX_N_ID_FAIL, seed=1)

twonn_pass = id_twonn(Zp_id)
twonn_fail = id_twonn(Zf_id)

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

# ID figure
plt.figure(figsize=(6.8, 5.0))
plt.plot(k_values, pass_mean, marker="o", color="0.0", label="PASS")
plt.fill_between(k_values, pass_mean - pass_std, pass_mean + pass_std, color="0.0", alpha=0.12)
plt.plot(k_values, fail_mean, marker="o", color="0.35", label="FAIL")
plt.fill_between(k_values, fail_mean - fail_std, fail_mean + fail_std, color="0.35", alpha=0.12)
plt.xlabel("k (neighbourhood size)")
plt.ylabel("Intrinsic dimension estimate d (Levina–Bickel MLE)")
plt.title("Intrinsic dimension vs neighbourhood scale (3τ)")
plt.legend(frameon=False)
plt.tight_layout()

f3_png = OUTDIR / "fig_id_vs_k_three.png"
f3_pdf = OUTDIR / "fig_id_vs_k_three.pdf"
plt.savefig(f3_png, dpi=320, bbox_inches="tight")
plt.savefig(f3_pdf, bbox_inches="tight")
plt.show()
plt.close()
print("Saved:", f3_png.resolve())

delta_min = float(np.nanmin(fail_mean - pass_mean))
delta_max = float(np.nanmax(fail_mean - pass_mean))

# Summary CSV (one row)
summary = pd.DataFrame([{
    "SEED": SEED,
    "N": N,
    "N_DIAG": N_DIAG,
    "P": P,
    "PN": PN,
    "cL": cL,
    "eps1_lo": eps1_lo, "eps1_hi": eps1_hi,
    "eps2_lo": eps2_lo, "eps2_hi": eps2_hi,
    "N_PASS": int(PASS.sum()),
    "f_PASS": f_pass,
    "PCA4": PCA4,
    "PCA6": PCA6,
    "var_ratio_PC5": float(var_ratio[4]),
    "var_ratio_PC6": float(var_ratio[5]),
    "TwoNN_PASS": twonn_pass,
    "TwoNN_FAIL": twonn_fail,
    "delta_d_min": delta_min,
    "delta_d_max": delta_max,
}])

out_csv = OUTDIR / "three_tau_summary_v033.csv"
summary.to_csv(out_csv, index=False)
print("Saved:", out_csv.resolve())

print("\nDONE.")
print("Key numbers:")
print("N_PASS =", int(PASS.sum()), " | f_PASS =", f_pass)
print("PCA4 =", PCA4, "| PCA6 =", PCA6)
print("Var ratios PC5/PC6 (PASS/FAIL):", float(var_ratio[4]), float(var_ratio[5]))
print("Δd range (FAIL-PASS) over k:", delta_min, delta_max)