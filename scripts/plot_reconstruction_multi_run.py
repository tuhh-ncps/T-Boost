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

# Optional mapping from CSV filename or stem -> display title used in plot legends.
# Edit this mapping to customize how datasets are named on the combined figure.
FILE_TITLE_MAP: dict[str, str] = {
    "BPI12_train.csv": "BPI12",
    "bpi_12_w_train.csv": "BPI12 (W)",
    "helpdesk_train.csv": "Helpdesk16",
    "helpdesk17_train.csv": "Helpdesk17",
    "BPI13_train.csv": "BPI13",
    "BPI17_train.csv": "BPI17",
    "BPI20DD_train.csv": "BPI20 DD",
    "BPI20ID_train.csv": "BPI20 ID",
    "BPI20TC_train.csv": "BPI20 TC",
    "BPI20RP_train.csv": "BPI20 RP",
    "BPI20PD_train.csv": "BPI20 PD",
}

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
    display_name = FILE_TITLE_MAP.get(name, FILE_TITLE_MAP.get(Path(name).stem, name))
    all_losses[display_name] = np.array(losses)

# Plot combined
fig, ax = plt.subplots(figsize=(7, 5))
for label, loss_arr in all_losses.items():
    ax.plot(candidate_ks, loss_arr, marker='o', label=label)
ax.axvline(4, linestyle=':', color='black', linewidth=1.5, label='Temporal scales R=4')
ax.set_xlabel('Number of temporal scales (R)')
ax.set_ylabel('Reconstruction Loss L(R)')
ax.grid(True, alpha=0.25)
ax.legend(loc='best', fontsize=11)
out = OUT_DIR / 'quantile_analysis_reconstruction_multi.pdf'
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches='tight')
print('Saved combined reconstruction plot:', out)
