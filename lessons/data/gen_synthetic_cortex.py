import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(123)

# ── Grid ──────────────────────────────────────────────────────────────
NX, NY = 160, 160
xs = np.linspace(0, 1, NX)
ys = np.linspace(0, 1, NY)
xg, yg = np.meshgrid(xs, ys)

# ── Semi-ellipse geometry ─────────────────────────────────────────────
a, b = 0.5, 1.0
Xn = (xg - 0.5) / a
Yn = yg / b
r      = np.sqrt(Xn**2 + Yn**2)
inside = (r <= 1.0) & (yg >= 0)

x  = xg[inside].ravel()
y  = yg[inside].ravel()
Xn = Xn[inside].ravel()
Yn = Yn[inside].ravel()
r  = r[inside].ravel()

N = x.size
print("Number of cells:", N)

depth = 1.0 - r   # 0 = pia, 1 = WM

theta = np.arctan2(Yn, Xn)
theta = np.where(theta < 0, theta + 2*np.pi, theta)
theta = np.clip(theta, 0, np.pi)

# ── Layer proportions ─────────────────────────────────────────────────
props = {
    "L1":   0.0328173668,
    "L2":   0.0637398297,
    "L3_4": 0.1987458105,
    "L5":   0.2190508900,
    "L6":   0.2104409204,
    "WM":   0.2752051826,
}

base_bounds = {}
cum = 0.0
for layer, p in props.items():
    base_bounds[layer] = (cum, cum + p)
    cum += p

layers_order = ["L1", "L2", "L3_4", "L5", "L6", "WM"]
n_borders    = len(layers_order) - 1
base_borders = np.array([base_bounds[l][1] for l in layers_order[:-1]])

# ── Border jitter + depth noise ───────────────────────────────────────
amp, freq = 0.02, 3.0
phases = rng.uniform(0, 2*np.pi, size=n_borders)

depth_noisy = depth + rng.normal(0, 0.0075, size=N)
depth_noisy = np.clip(depth_noisy, 0.0, 1.0)

def depth_borders_at_theta(th):
    return base_borders + amp * np.sin(freq * th + phases)

def assign_layer_core(d_val, th):
    b = depth_borders_at_theta(th)
    if   d_val < b[0]: return 0
    elif d_val < b[1]: return 1
    elif d_val < b[2]: return 2
    elif d_val < b[3]: return 3
    elif d_val < b[4]: return 4
    else:              return 5

layer_idx = np.array([assign_layer_core(d, t)
                      for d, t in zip(depth_noisy, theta)])

# ── Probabilistic border fuzz ─────────────────────────────────────────
border_width = 0.015
for k in range(n_borders):
    center = base_borders[k]
    dist   = np.abs(depth_noisy - center)
    near   = dist < border_width
    p_flip = np.exp(-(dist[near] / border_width)**2) * 0.6
    rand   = rng.uniform(0, 1, size=p_flip.size)
    below  = depth_noisy[near] < center
    flip_up   = below    & (rand < p_flip)
    flip_down = (~below) & (rand < p_flip)
    idx_near  = np.where(near)[0]
    layer_idx[idx_near[flip_up]]   = np.minimum(layer_idx[idx_near[flip_up]]   + 1, 5)
    layer_idx[idx_near[flip_down]] = np.maximum(layer_idx[idx_near[flip_down]] - 1, 0)

idx_to_layer = {0:"L1", 1:"L2", 2:"L3_4", 3:"L5", 4:"L6", 5:"WM"}
layers = np.array([idx_to_layer[i] for i in layer_idx])

# ── Helper functions ──────────────────────────────────────────────────
def gaussian(v, center, width):
    return np.exp(-0.5 * ((v - center) / width) ** 2)

def sigmoid_rise(v, center, width):
    """Smooth 0→1 transition as v increases past centre."""
    return 1.0 / (1.0 + np.exp(-(v - center) / width))

d = depth   # shorthand

max_levels = {
    "AQP4":    5.0,
    "HPCAL1":  5.0,
    "FREM3":   2.5,
    "TRABD2A": 2.5,
    "KRT17":   3.0,
    "MOBP":    6.0,
}

# Precompute layer centres in depth units
L1_c  = (base_bounds["L1"][0]   + base_bounds["L1"][1])   / 2   # ≈ 0.016
L2_c  = (base_bounds["L2"][0]   + base_bounds["L2"][1])   / 2   # ≈ 0.065
L3_c  = (base_bounds["L3_4"][0] + base_bounds["L3_4"][1]) / 2   # ≈ 0.196
L5_c  = (base_bounds["L5"][0]   + base_bounds["L5"][1])   / 2   # ≈ 0.405
L6_c  = (base_bounds["L6"][0]   + base_bounds["L6"][1])   / 2   # ≈ 0.620
WM_start = base_bounds["WM"][0]                                   # ≈ 0.725

# ─────────────────────────────────────────────────────────────────────
# AQP4 (L1 marker)
#   • Primary: narrow Gaussian at L1 (width matches thin L1 layer)
#   • Secondary: weaker ghost band at L6
# ─────────────────────────────────────────────────────────────────────
AQP4_L1   = gaussian(d, L1_c, 0.018)    # thin — L1 occupies only ~3 % of depth
AQP4_L6   = gaussian(d, L6_c, 0.07)     # broader ghost band
AQP4_base = max_levels["AQP4"] * (AQP4_L1 + 0.35 * AQP4_L6)

# ─────────────────────────────────────────────────────────────────────
# HPCAL1 (L2 + L6 double band)
#   • Band 1 at L2, Band 2 at L6
#   • Hard gap mask suppresses L3_4 and L5 by 95 %
#     using two sigmoid transitions (one at L2 end, one at L6 start)
# ─────────────────────────────────────────────────────────────────────
HPCAL1_L2  = gaussian(d, L2_c, 0.030)
HPCAL1_L6  = gaussian(d, L6_c, 0.070)
HPCAL1_raw = HPCAL1_L2 + 0.70 * HPCAL1_L6

# Gap mask: ~1 inside [L2_end … L6_start], ~0 outside
gap_mask = (sigmoid_rise(d, base_bounds["L2"][1] + 0.010, 0.015) *
            (1.0 - sigmoid_rise(d, base_bounds["L6"][0] - 0.010, 0.015)))

HPCAL1_base = max_levels["HPCAL1"] * HPCAL1_raw * (1.0 - 0.95 * gap_mask)

# ─────────────────────────────────────────────────────────────────────
# FREM3 (L3/4 marker): broad gradient
# ─────────────────────────────────────────────────────────────────────
FREM3_base = max_levels["FREM3"] * gaussian(d, L3_c, 0.10)

# ─────────────────────────────────────────────────────────────────────
# TRABD2A (L5 marker): broad gradient
# ─────────────────────────────────────────────────────────────────────
TRABD2A_base = max_levels["TRABD2A"] * gaussian(d, L5_c, 0.11)

# ─────────────────────────────────────────────────────────────────────
# KRT17 (L6 marker): broad gradient
# ─────────────────────────────────────────────────────────────────────
KRT17_base = max_levels["KRT17"] * gaussian(d, L6_c, 0.11)

# ─────────────────────────────────────────────────────────────────────
# MOBP (WM marker)
#   • Sigmoid centred at depth ≈ 0.42 (L4/L5 boundary), width = 0.15
#   • Gives ~10 % at L3 start (depth 0.097)  → faint low gradient
#   •        ~50 % at depth 0.42             → mid-cortex gradient
#   •        ~88 % at WM start (depth 0.725) → near-full in WM
#   •        ~95 % at WM centre (depth 0.86) → max in WM
# ─────────────────────────────────────────────────────────────────────
MOBP_base = max_levels["MOBP"] * sigmoid_rise(d, 0.42, 0.15)

# ── Noise & dropouts ─────────────────────────────────────────────────
def add_imperfections(base, max_level,
                      dropout_rate=0.05, noise_frac=0.1, lognorm_sigma=0.2):
    scale = rng.lognormal(mean=0, sigma=lognorm_sigma, size=base.size)
    expr  = base * scale
    noise = rng.normal(0, max_level * noise_frac, size=base.size)
    expr  = np.clip(expr + noise, 0, None)
    expr[rng.uniform(0, 1, base.size) < dropout_rate] = 0.0
    return expr

AQP4    = add_imperfections(AQP4_base,    max_levels["AQP4"],    dropout_rate=0.08, noise_frac=0.08)
HPCAL1  = add_imperfections(HPCAL1_base,  max_levels["HPCAL1"],  dropout_rate=0.08, noise_frac=0.08)
FREM3   = add_imperfections(FREM3_base,   max_levels["FREM3"],   dropout_rate=0.06, noise_frac=0.08)
TRABD2A = add_imperfections(TRABD2A_base, max_levels["TRABD2A"], dropout_rate=0.06, noise_frac=0.08)
KRT17   = add_imperfections(KRT17_base,   max_levels["KRT17"],   dropout_rate=0.06, noise_frac=0.08)
MOBP    = add_imperfections(MOBP_base,    max_levels["MOBP"],    dropout_rate=0.06, noise_frac=0.08)

# ── Assemble & save ───────────────────────────────────────────────────

df = pd.DataFrame({
    "cell_barcode":   [f"cell_{i:06d}" for i in range(N)],
    "x":              x,
    "y":              y,
    "depth":          depth,
    "cortical_layer": layers,
    "AQP4":           AQP4,
    "HPCAL1":         HPCAL1,
    "FREM3":          FREM3,
    "TRABD2A":        TRABD2A,
    "KRT17":          KRT17,
    "MOBP":           MOBP,
})

df.to_csv("synthetic_cortex_data.csv", index=False)
print("Saved synthetic_cortex_data.csv with", len(df), "cells")