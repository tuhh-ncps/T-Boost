import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ["Tax + CS", "Transformer", "TRAM"]

# Raw values
params = np.array([209_000, 36_000, 22_700])       # parameters
flops = np.array([9_250_000, 250_000, 45_400])     # FLOPs
epochs = np.array([89, 100, 50])                   # epochs

metrics = {
    "Parameters": params,
    "FLOPs": flops,
    "Epochs": epochs,
}

# Normalize each metric by the maximum value so they can be compared visually
normalized = {
    name: values / values.max()
    for name, values in metrics.items()
}

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - width, normalized["Parameters"], width, label="Parameters")
bars2 = ax.bar(x, normalized["FLOPs"], width, label="FLOPs")
bars3 = ax.bar(x + width, normalized["Epochs"], width, label="Epochs")

ax.set_ylabel("Normalized computational cost")
ax.set_xlabel("Model")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Add raw value labels above each bar
raw_labels = {
    "Parameters": ["209K", "36K", "22.7K"],
    "FLOPs": ["9.25M", "250K", "45.4K"],
    "Epochs": ["~89", "~100", "~50"],
}

for bars, labels in zip(
    [bars1, bars2, bars3],
    [raw_labels["Parameters"], raw_labels["FLOPs"], raw_labels["Epochs"]],
):
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("computational_comparison_bar_chart.pdf", bbox_inches="tight")
plt.savefig("computational_comparison_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()