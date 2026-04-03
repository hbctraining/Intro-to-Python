import numpy as np
import pandas as pd

# =====================================================================
# Semi-oval "onion" cortex with moderate noise:
#   - concentric semi-elliptical layers
#   - softer, but still irregular borders
#   - gene expression with:
#       - MOBP: strong WM + weaker extension into L6
#       - AQP4: strong L1 + weaker band in L6
#       - HPCAL1 double band, etc.
#
# Output: synthetic_cortex_oval_moderate_noise.csv
# =====================================================================

rng = np.random.default_rng(123)

# -----------------------------
# Grid in [0,1] x [0,1]
# -----------------------------
NX = 160
NY = 160

xs = np.linspace(0, 1, NX)
ys = np.linspace(0, 1, NY)
xg, yg = np.meshgrid(xs, ys)   # (NY, NX)

# -----------------------------
# Semi-ellipse geometry
# -----------------------------
a = 0.5   # horizontal radius
b = 1.0   # vertical radius

Xn = (xg - 0.5) / a
Yn = yg / b
r = np.sqrt(Xn**2 + Yn**2)      # elliptical radius
inside = (r <= 1.0) & (yg >= 0)

x = xg[inside].ravel()
y = yg[inside].ravel()
Xn = Xn[inside].ravel()
Yn = Yn[inside].ravel()
r = r[inside].ravel()

N = x.size
print("Number of cells:", N)

# Depth: 0 = pia (outer), 1 = WM (center)
depth = 1.0 - r

# Angle in [0, pi]
theta = np.arctan2(Yn, Xn)
theta = np.where(theta < 0, theta + 2*np.pi, theta)
theta = np.clip(theta, 0, np.pi)

# -----------------------------
# Layer proportions (in depth)
# -----------------------------
props = {
    "L1":   0.0328173668,
    "L2":   0.0637398297,
    "L3_4": 0.1987458105,
    "L5":   0.2190508900,
    "L6":   0.2104409204,
    "WM":   0.2752051826
}

base_bounds = {}
cum = 0.0
for layer, p in props.items():
    base_bounds[layer] = (cum, cum + p)
    cum += p

layers_order = ["L1", "L2", "L3_4", "L5", "L6", "WM"]
n_borders = len(layers_order) - 1
base_borders = np.array([base_bounds[l][1] for l in layers_order[:-1]])

# -----------------------------
# Border jitter + mild depth noise
# (roughly half of previous iteration)
# -----------------------------
amp = 0.02      # previously 0.04
freq = 3.0
phases = rng.uniform(0, 2*np.pi, size=n_borders)

# per-cell depth noise
depth_noisy = depth + rng.normal(0, 0.0075, size=N)  # previously 0.015
depth_noisy = np.clip(depth_noisy, 0.0, 1.0)

def depth_borders_at_theta(th):
    return base_borders + amp * np.sin(freq * th + phases)

def assign_layer_core(d_val, th):
    b = depth_borders_at_theta(th)
    if d_val < b[0]:
        return 0
    elif d_val < b[1]:
        return 1
    elif d_val < b[2]:
        return 2
    elif d_val < b[3]:
        return 3
    elif d_val < b[4]:
        return 4
    else:
        return 5

layer_idx = np.array([assign_layer_core(d, t) for d, t in zip(depth_noisy, theta)])

# -----------------------------
# Softer border fuzz (probabilistic flipping)
# -----------------------------
border_width = 0.015   # previously 0.03

for k in range(n_borders):
    center = base_borders[k]
    dist = np.abs(depth_noisy - center)
    near = dist < border_width
    p_flip = np.exp(- (dist[near] / border_width)**2) * 0.6  # overall cap on flip prob
    rand = rng.uniform(0, 1, size=p_flip.size)
    below = depth_noisy[near] < center
    flip_up   = below & (rand < p_flip)
    flip_down = (~below) & (rand < p_flip)
    idx_near = np.where(near)[0]
    layer_idx[idx_near[flip_up]]   = np.minimum(layer_idx[idx_near[flip_up]] + 1, 5)
    layer_idx[idx_near[flip_down]] = np.maximum(layer_idx[idx_near[flip_down]] - 1, 0)

idx_to_layer = {0:"L1",1:"L2",2:"L3_4",3:"L5",4:"L6",5:"WM"}
layers = np.array([idx_to_layer[i] for i in layer_idx])

# -----------------------------
# Gene expression bands (functions of depth)
# -----------------------------
def gaussian(v, center, width):
    return np.exp(-0.5 * ((v - center) / width) ** 2)

centers = {
    "AQP4_L1":       (base_bounds["L1"][0]   + base_bounds["L1"][1])   / 2,
    "AQP4_L6":       (base_bounds["L6"][0]   + base_bounds["L6"][1])   / 2,  # weaker deep band
    "HPCAL1_top":    (base_bounds["L2"][0]   + base_bounds["L2"][1])   / 2,
    "HPCAL1_bottom": (base_bounds["L5"][0]   + base_bounds["L5"][1])   / 2,
    "FREM3":         (base_bounds["L3_4"][0] + base_bounds["L3_4"][1]) / 2,
    "TRABD2A":       (base_bounds["L5"][0]   + base_bounds["L5"][1])   / 2,
    "KRT17":         (base_bounds["L6"][0]   + base_bounds["L6"][1])   / 2,
    "MOBP_WM":       (base_bounds["WM"][0]   + base_bounds["WM"][1])   / 2,
    "MOBP_L6":       (base_bounds["L6"][0]   + base_bounds["L6"][1])   / 2,  # weaker extension
}

widths = {
    "AQP4_L1":       0.03,
    "AQP4_L6":       0.04,
    "HPCAL1_top":    0.035,
    "HPCAL1_bottom": 0.045,
    "FREM3":         0.05,
    "TRABD2A":       0.05,
    "KRT17":         0.05,
    "MOBP_WM":       0.06,
    "MOBP_L6":       0.05,
}

max_levels = {
    "AQP4":    5.0,
    "HPCAL1":  5.0,
    "FREM3":   2.5,
    "TRABD2A": 2.5,
    "KRT17":   3.0,
    "MOBP":    6.0,
}

d = depth

# AQP4: strong in L1, weaker in L6
AQP4_L1 = gaussian(d, centers["AQP4_L1"], widths["AQP4_L1"])
AQP4_L6 = gaussian(d, centers["AQP4_L6"], widths["AQP4_L6"])
AQP4_base = max_levels["AQP4"] * (AQP4_L1 + 0.25 * AQP4_L6)

# HPCAL1: double band
HPCAL1_band1 = gaussian(d, centers["HPCAL1_top"],    widths["HPCAL1_top"])
HPCAL1_band2 = gaussian(d, centers["HPCAL1_bottom"], widths["HPCAL1_bottom"])
HPCAL1_base = max_levels["HPCAL1"] * (HPCAL1_band1 + HPCAL1_band2)
gap_center = (centers["HPCAL1_top"] + centers["HPCAL1_bottom"]) / 2
gap = gaussian(d, gap_center, 0.03)
HPCAL1_base = HPCAL1_base * (1 - 0.9 * gap)

# FREM3, TRABD2A, KRT17: single bands
FREM3_base   = max_levels["FREM3"]   * gaussian(d, centers["FREM3"],   widths["FREM3"])
TRABD2A_base = max_levels["TRABD2A"] * gaussian(d, centers["TRABD2A"], widths["TRABD2A"])
KRT17_base   = max_levels["KRT17"]   * gaussian(d, centers["KRT17"],   widths["KRT17"])

# MOBP: strong in WM, weaker extension into L6
MOBP_WM = gaussian(d, centers["MOBP_WM"], widths["MOBP_WM"])
MOBP_L6 = gaussian(d, centers["MOBP_L6"], widths["MOBP_L6"])
MOBP_base = max_levels["MOBP"] * (MOBP_WM + 0.3 * MOBP_L6)

# -----------------------------
# Noise & dropouts (reduced)
# -----------------------------
def add_imperfections(base, max_level, dropout_rate=0.05, noise_frac=0.1, lognorm_sigma=0.2):
    scale = rng.lognormal(mean=0, sigma=lognorm_sigma, size=base.size)
    expr = base * scale
    noise = rng.normal(0, max_level * noise_frac, size=base.size)
    expr = np.clip(expr + noise, 0, None)
    mask = rng.uniform(0, 1, base.size) < dropout_rate
    expr[mask] = 0.0
    return expr

AQP4    = add_imperfections(AQP4_base,    max_levels["AQP4"],    dropout_rate=0.08, noise_frac=0.08)
HPCAL1  = add_imperfections(HPCAL1_base,  max_levels["HPCAL1"],  dropout_rate=0.08, noise_frac=0.08)
FREM3   = add_imperfections(FREM3_base,   max_levels["FREM3"],   dropout_rate=0.06, noise_frac=0.08)
TRABD2A = add_imperfections(TRABD2A_base, max_levels["TRABD2A"], dropout_rate=0.06, noise_frac=0.08)
KRT17   = add_imperfections(KRT17_base,   max_levels["KRT17"],   dropout_rate=0.06, noise_frac=0.08)
MOBP    = add_imperfections(MOBP_base,    max_levels["MOBP"],    dropout_rate=0.06, noise_frac=0.08)

# -----------------------------
# Assemble DataFrame and save
# -----------------------------
df = pd.DataFrame({
    "cell_barcode": [f"cell_{i:06d}" for i in range(N)],
    "x": x,
    "y": y,
    "depth": depth,
    "cortical_layer": layers,
    "AQP4": AQP4,
    "HPCAL1": HPCAL1,
    "FREM3": FREM3,
    "TRABD2A": TRABD2A,
    "KRT17": KRT17,
    "MOBP": MOBP,
})

df.to_csv("synthetic_cortex_data.csv", index=False)
print("Saved synthetic_cortex_oval_moderate_noise.csv with", len(df), "cells")
