#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data_csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_ks = list(range(2, 21))

datasets = [
    "BPI12_train.csv",
    "bpi_12_w_train.csv",
    "helpdesk_train.csv",
    "helpdesk17_train.csv",
    "BPI13_train.csv",
    "BPI17_train.csv",
    "BPI20DD_train.csv",
    "BPI20ID_train.csv",
    "BPI20TC_train.csv",
    "BPI20RP_train.csv",
    "BPI20PD_train.csv",
]

all_losses = {}

for name in datasets:
    path = DATA_DIR / name
    if not path.exists():
        print(f"Skipping missing: {name}")
        continue
    df = pd.read_csv(path, low_memory=False)
    if "next_time" not in df.columns:
        print(f"Skipping (no next_time): {name}")
        continue
    next_time = pd.to_numeric(df["next_time"], errors="coerce").dropna().clip(lower=0.0)
    if next_time.empty:
        print(f"Skipping (no values): {name}")
        continue
    values = np.log1p(next_time.to_numpy(dtype=float))

    losses = []
    for k in candidate_ks:
        try:
            bins = pd.qcut(values, q=k, duplicates="drop")
            actual_k = len(bins.categories)
            if actual_k < k:
                losses.append(np.nan)
                continue
            codes = bins.codes
            reconstructed = np.zeros_like(values)
            for bucket in range(len(bins.categories)):
                idx = codes == bucket
                med = np.median(values[idx])
                reconstructed[idx] = med
            loss = np.mean((values - reconstructed) ** 2)
            losses.append(float(loss))
        except Exception:
            losses.append(np.nan)
    all_losses[name] = np.array(losses)

# Plot combined
fig, ax = plt.subplots(figsize=(7, 5))
for label, loss_arr in all_losses.items():
    ax.plot(candidate_ks, loss_arr, marker='o', label=label)
ax.axvline(4, linestyle=':', color='black', linewidth=1.5, label='Temporal regimes K=4')
ax.set_xlabel('Number of temporal regimes (K)')
ax.set_ylabel('Reconstruction Loss L(K)')
ax.grid(True, alpha=0.25)
ax.legend(loc='best', fontsize=12)
out = OUT_DIR / 'quantile_analysis_reconstruction_multi.pdf'
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches='tight')
print('Saved combined reconstruction plot:', out)
