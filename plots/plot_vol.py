import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

with Path("vol_run_times.json").open("r") as f:
    plot_data = json.load(f)


df = pd.DataFrame(plot_data)

plt.figure(figsize=(19.20, 10.80))
sns.set_context("talk")

ax = sns.lineplot(x="n", y="res", hue="type", data=df, palette="muted")
# ax = sns.barplot(x="n", y="res", hue="type", data=df, palette="muted")
ax.axhline(1000/30., color='black', ls=':')
ax.set_ylim(1, 70)

plt.text(x=-0.0, y=1000/30. -2, s="30fps", color="black",  ha="center")

# Labels and title
plt.title("Runtime for volumetric integration (i5-12400F / NVIDIA GeForce RTX 3070)")
plt.xlabel("Grid size")
plt.ylabel("Runtime [ms]")

# plt.yscale('log')

# Show plot
plt.legend(title='Type', loc='upper left')
plt.tight_layout()
plt.savefig("vol_int.png")
plt.show()