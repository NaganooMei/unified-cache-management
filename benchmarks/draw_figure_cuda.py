import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Raw Data
# ============================================================
raw_data = [
    # input, conc, full,
    # ucm100, lm100, ucm50, lm50, ucm30, lm30, ucm0, lm0
    (4000,1,1018.34,77.36,205.85,594.23,676.10,766.80,847.04,1036.05,1043.81),
    (8000,1,2071.69,118.14,292.86,1152.15,1333.84,1562.78,1658.79,2104.93,2118.06),
    (16000,1,4440.74,170.65,542.49,2485.93,2776.63,3389.05,3604.00,4508.03,4535.78),
    (32000,1,10088.07,293.75,1020.86,5846.70,6500.16,7782.51,8208.73,10226.55,10281.63),

    (4000,2,1521.47,94.56,355.09,896.24,996.44,1160.65,1243.41,1550.40,1558.80),
    (8000,2,3128.48,195.14,416.39,1743.95,1947.81,2371.55,2454.85,3152.78,3168.43),
    (16000,2,6712.17,220.88,798.05,3769.76,4056.89,5844.64,6108.55,6819.86,6852.33),
    (32000,2,15238.17,392.31,1428.96,8920.83,9742.54,11982.43,12657.77,15437.94,15522.46),

    (4000,4,2788.26,171.36,553.30,2183.67,2074.11,2398.89,2561.19,2831.33,2851.39),
    (8000,4,5199.94,319.94,849.24,3164.74,3838.78,4462.72,4608.70,5293.15,5363.50),
    (16000,4,11236.31,458.93,1659.52,6323.41,6812.54,9560.07,9881.96,11335.18,11485.71),
    (32000,4,25485.75,772.31,2981.72,14949.62,16073.07,22264.47,21144.45,25644.76,25907.49),

    (4000,8,4936.89,297.75,1143.09,3365.86,3794.06,4237.71,4406.81,4996.38,5043.16),
    (8000,8,9419.03,587.07,1791.54,5673.78,6948.09,8058.17,8191.08,9549.99,9640.46),
    (16000,8,20248.53,804.39,3812.96,11508.60,12441.67,16360.17,16712.87,20441.36,20654.01),
    (32000,8,45813.20,1633.21,8358.88,26978.08,29023.65,36209.00,37755.02,46295.78,46593.21),
]

df = pd.DataFrame(
    raw_data,
    columns=[
        "input_len","conc","full",
        "ucm_100","lm_100",
        "ucm_50","lm_50",
        "ucm_30","lm_30",
        "ucm_0","lm_0",
    ]
)

# ============================================================
# Helper: build averaged DF for concurrency 1~4
# ============================================================
df_avg = (
    df[df["conc"].isin([1,2,4])]
    .groupby("input_len", as_index=False)
    .mean(numeric_only=True)
)

df_c8 = df[df["conc"] == 8].copy()

# ============================================================
# Plot
# ============================================================
hit_rates = ["100","50","30","0"]

fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey="row")

COLOR_FULL = "#4D4D4D"
COLOR_UCM  = "#d62728"
COLOR_LM   = "#1f77b4"
bar_width = 1200

def draw_block(ax, data, hr, title):
    full = data["full"]
    ucm = data[f"ucm_{hr}"]
    lm  = data[f"lm_{hr}"]

    ax.plot(data["input_len"], full, marker="o", color=COLOR_FULL, label="Full Recompute")
    ax.plot(data["input_len"], lm,  marker="o", color=COLOR_LM, label="LMCache")
    ax.plot(data["input_len"], ucm, marker="o", color=COLOR_UCM, label="UCM")

    ax.set_title(title)
    ax.set_xlabel("Input Length")
    ax.set_ylabel("TTFT (ms)")
    ax.grid(True)

    ax2 = ax.twinx()
    imp_ucm = (full / ucm - 1) * 100
    imp_lm  = (full / lm  - 1) * 100

    ax2.bar(data["input_len"] - bar_width/2, imp_lm,
            width=bar_width, alpha=0.35, color=COLOR_LM, label="LMCache Improvement")
    ax2.bar(data["input_len"] + bar_width/2, imp_ucm,
            width=bar_width, alpha=0.35, color=COLOR_UCM, label="UCM Improvement")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Improvement over Full (%)")
    return ax, ax2

# -------- Top row: avg concurrency 1~4
for i, hr in enumerate(hit_rates):
    if i == 0:
        ax_main, ax_bar = draw_block(
            axes[0, i],
            df_avg,
            hr,
            f"Avg Concurrency 1~4 | Hit Rate = {hr}%"
        )
    else:
        draw_block(
            axes[0, i],
            df_avg,
            hr,
            f"Avg Concurrency 1~4 | Hit Rate = {hr}%"
        )


# -------- Bottom row: concurrency = 8
for i, hr in enumerate(hit_rates):
    draw_block(
        axes[1, i],
        df_c8,
        hr,
        f"Concurrency = 8 | Hit Rate = {hr}%"
    )

# Legends
ax_main.legend(loc="upper left", frameon=False)
ax_bar.legend(loc="upper left", bbox_to_anchor=(0, 0.82), frameon=False)

fig.suptitle(
    "UCM / LMCache TTFT and Improvement over Full Recompute (CUDA)",
    fontsize=18
)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("ucm_lmcache_vs_full_all_concurrency.png", dpi=300)
plt.close()
