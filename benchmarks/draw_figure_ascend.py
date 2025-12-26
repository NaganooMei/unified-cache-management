import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Raw Data (Ascend)
# ============================================================
raw_data = [
    # input, conc, full,
    # ucm100, moon100, ucm50, moon50, ucm30, moon30, ucm0, moon0
    (4000,1,943.65,164.23,215.38,669.53,631.52,772.03,741.81,981.95,949.02),
    (8000,1,1926.93,219.57,264.50,1154.69,1136.69,1558.70,1477.41,1999.62,1941.91),
    (16000,1,4156.75,352.50,596.44,2513.22,2452.73,3330.32,3256.01,4294.11,4179.55),
    (32000,1,9511.53,617.16,760.26,5745.56,5790.63,7574.74,7454.83,9837.76,9566.61),

    (4000,2,1630.98,231.61,314.33,1162.77,1117.15,1356.01,1305.39,1695.87,1649.19),
    (8000,2,3375.99,396.92,491.86,1965.97,1990.34,2714.11,2640.47,3563.57,3412.61),
    (16000,2,7296.22,559.51,722.93,4324.69,4312.60,5752.46,5695.48,7577.17,7341.59),
    (32000,2,16579.22,998.61,1870.41,10103.39,10213.51,13170.88,13045.78,17126.61,16700.31),

    (4000,4,3033.09,397.57,480.18,1972.05,1923.82,2447.44,2346.15,3172.97,3078.86),
    (8000,4,6287.14,588.64,856.30,3668.71,3629.33,4926.82,4796.95,6561.25,6342.23),
    (16000,4,13437.58,957.18,1686.73,8085.99,8062.14,10710.57,10430.09,13981.23,13569.78),
    (32000,4,30814.34,1977.04,2385.94,18860.10,19066.64,24767.57,24425.07,31909.83,31090.59),

    (4000,8,5863.99,577.77,797.86,3634.16,3538.76,4682.52,4446.13,6092.80,5913.18),
    (8000,8,12000.44,1262.85,1536.23,7084.26,6971.85,9515.93,9431.86,12496.01,12154.17),
    (16000,8,25820.22,2182.75,2897.94,15535.87,15752.73,20336.08,20066.99,26847.16,26058.16),
    (32000,8,59150.62,4618.86,6217.44,36190.11,35948.49,47041.80,46803.93,61279.44,59678.08),
]

df = pd.DataFrame(
    raw_data,
    columns=[
        "input_len","conc","full",
        "ucm_100","moon_100",
        "ucm_50","moon_50",
        "ucm_30","moon_30",
        "ucm_0","moon_0",
    ]
)

# ============================================================
# Avg concurrency 1~4
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
COLOR_MOON = "#1f77b4"
bar_width = 1200

def draw_block(ax, data, hr, title):
    full = data["full"]
    ucm  = data[f"ucm_{hr}"]
    moon = data[f"moon_{hr}"]

    ax.plot(data["input_len"], full, marker="o",
            color=COLOR_FULL, label="Full Recompute")
    ax.plot(data["input_len"], moon, marker="o",
            color=COLOR_MOON, label="Mooncake")
    ax.plot(data["input_len"], ucm, marker="o",
            color=COLOR_UCM, label="UCM")

    ax.set_title(title)
    ax.set_xlabel("Input Length")
    ax.set_ylabel("TTFT (ms)")
    ax.grid(True)

    ax2 = ax.twinx()
    imp_ucm  = (full / ucm  - 1) * 100
    imp_moon = (full / moon - 1) * 100

    ax2.bar(data["input_len"] - bar_width/2, imp_moon,
            width=bar_width, alpha=0.35,
            color=COLOR_MOON, label="Mooncake Improvement")
    ax2.bar(data["input_len"] + bar_width/2, imp_ucm,
            width=bar_width, alpha=0.35,
            color=COLOR_UCM, label="UCM Improvement")

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

# Legends (bind to real axes)
ax_main.legend(loc="upper left", frameon=False)
ax_bar.legend(loc="upper left", bbox_to_anchor=(0, 0.82), frameon=False)

fig.suptitle(
    "UCM / Mooncake TTFT and Improvement over Full Recompute (Ascend)",
    fontsize=18
)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("ucm_mooncake_vs_full_ascend.png", dpi=300)
plt.close()
