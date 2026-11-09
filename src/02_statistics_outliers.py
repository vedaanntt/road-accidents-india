# Step 2: Summary statistics and outliers
#
# 1. Builds one "profile" per state: the average of each ratio over the
#    years that state has data for.
# 2. Summary statistics of every feature across states.
# 3. Finds outlier states with the IQR rule and draws box plots.
# 4. Tests our under-reporting (MNAR) idea: if some states skip minor
#    accidents, they should show a HIGH fatality ratio together with a
#    LOW injury ratio.
#
# Run from the Project folder:
#     .venv\Scripts\python src\02_statistics_outliers.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # draw straight to files, no window
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CLEAN = "data/clean/"
OUTPUTS = "outputs/"

# chart colours (from a colour-blind-safe palette)
BOX_BLUE = "#2a78d6"
OUTLIER_ORANGE = "#eb6834"
TEXT_DARK = "#0b0b0b"
TEXT_GREY = "#52514e"
GRID_GREY = "#e4e3df"

# the ratio and share columns we analyse (all in %)
ratio_columns = [
    "fatality_ratio", "injury_ratio",
    "share_expressway", "share_national_hw", "share_state_hw", "share_other_road",
    "fr_national_hw", "fr_state_hw", "fr_other_road",
    "night_share", "evening_share", "late_night_share",
    "monsoon_share", "winter_share",
    "rural_death_share",
    "two_wheeler_death_share", "pedestrian_death_share", "car_death_share",
    "truck_death_share", "bus_death_share",
]


def short_name(state):
    # Long names make charts messy, so "Himachal Pradesh" becomes "HP".
    if len(state) <= 13:
        return state
    # both would become "AP", so give Arunachal its own label
    if state == "Arunachal Pradesh":
        return "ArP"
    letters = ""
    for word in state.split(" "):
        if word != "And":
            letters = letters + word[0]
    return letters


# 1. State profiles
features = pd.read_csv(CLEAN + "state_year_features.csv")

# mean() skips blanks, so a state with a missing year or a 2023 blank
# is simply averaged over the years it has.
profile = features.drop(columns=["year"]).groupby("state").mean()
profile["years_of_data"] = features.groupby("state")["year"].count()
profile = profile.rename(columns={"accidents": "accidents_per_year", "deaths": "deaths_per_year"})
profile = profile.round(2)
profile.to_csv(CLEAN + "state_profiles.csv")

print("Smallest regions by accidents per year:")
smallest = profile.sort_values("accidents_per_year").head(10)
for state in smallest.index:
    print("  ", state.ljust(42), int(smallest.loc[state, "accidents_per_year"]), " fatality ratio:", smallest.loc[state, "fatality_ratio"])


# 2. Summary statistics
summary = profile[ratio_columns].describe().T
summary["skew"] = profile[ratio_columns].skew()
summary = summary.round(2)
summary.to_csv(OUTPUTS + "summary_statistics.csv")
print("\nSummary statistics across states:")
print(summary[["mean", "50%", "std", "min", "max", "skew"]].to_string())


# 3. Outliers with the IQR rule.
# A value is an outlier if it is more than 1.5 x IQR below Q1 or above Q3.
# We use IQR rather than z-scores because several features are skewed,
# and a z-score assumes a roughly bell-shaped distribution.
outlier_rows = []
for column in ratio_columns:
    values = profile[column].dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    high_limit = q3 + 1.5 * iqr
    for state in values.index:
        value = values[state]
        if value < low_limit:
            outlier_rows.append([column, state, value, "low", round(low_limit, 2)])
        if value > high_limit:
            outlier_rows.append([column, state, value, "high", round(high_limit, 2)])

outliers = pd.DataFrame(outlier_rows, columns=["feature", "state", "value", "side", "limit"])
outliers.to_csv(OUTPUTS + "outlier_states.csv", index=False)

print("\nNumber of features in which each state is an outlier:")
counts = outliers["state"].value_counts()
for state in counts.index:
    which = outliers[outliers["state"] == state]["feature"]
    print("  ", state.ljust(42), counts[state], " ", ", ".join(which))


# Box plots: one small box plot per feature, outlier states labelled.
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.patch.set_facecolor("white")
for i in range(len(ratio_columns)):
    column = ratio_columns[i]
    ax = axes[i // 5][i % 5]
    values = profile[column].dropna()
    ax.boxplot(values, widths=0.5, showfliers=False,
               boxprops={"color": BOX_BLUE, "linewidth": 1.5},
               whiskerprops={"color": BOX_BLUE, "linewidth": 1.5},
               capprops={"color": BOX_BLUE, "linewidth": 1.5},
               medianprops={"color": TEXT_DARK, "linewidth": 2})

    # draw every state as a small dot, outliers bigger and orange
    these_outliers = outliers[outliers["feature"] == column]
    for state in values.index:
        x = 1 + np.random.uniform(-0.12, 0.12)
        if state in list(these_outliers["state"]):
            ax.scatter(x, values[state], s=50, color=OUTLIER_ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(short_name(state), (x, values[state]), xytext=(8, 0), textcoords="offset points",
                        fontsize=9, color=TEXT_DARK, va="center")
        else:
            ax.scatter(x, values[state], s=14, color=TEXT_GREY, alpha=0.5, zorder=2)

    ax.set_title(column.replace("_", " "), fontsize=12, color=TEXT_DARK)
    ax.set_xticks([])
    ax.tick_params(colors=TEXT_GREY, labelsize=9)
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID_GREY)

fig.suptitle("State profiles (average over years, all values in %). Orange = outlier by the 1.5 x IQR rule",
             fontsize=15, color=TEXT_DARK)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTPUTS + "boxplots_state_profiles.png", dpi=120)
plt.close(fig)


# 4. Under-reporting (MNAR) test.
# If a state records only serious accidents, each recorded accident will
# have many deaths and few injuries. So we expect a NEGATIVE relationship
# between fatality ratio and injury ratio across states.
# Spearman correlation uses ranks, so the extreme states cannot dominate it.
both = profile[["fatality_ratio", "injury_ratio", "accidents_per_year"]].dropna()
rho, p_value = spearmanr(both["fatality_ratio"], both["injury_ratio"])
print("\nSpearman correlation, fatality ratio vs injury ratio: rho =", round(rho, 2), " p =", round(p_value, 5))

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(both["injury_ratio"], both["fatality_ratio"], s=60, color=BOX_BLUE, edgecolor="white", linewidth=1.5, zorder=3)
for state in both.index:
    ax.annotate(short_name(state), (both.loc[state, "injury_ratio"], both.loc[state, "fatality_ratio"]),
                xytext=(6, 3), textcoords="offset points", fontsize=8, color=TEXT_GREY)
ax.set_xlabel("Injured people per 100 recorded accidents", color=TEXT_DARK)
ax.set_ylabel("Deaths per 100 recorded accidents", color=TEXT_DARK)
ax.set_title("States that record few injuries record many deaths per accident\n(Spearman rho = "
             + str(round(rho, 2)) + ")", color=TEXT_DARK)
ax.grid(color=GRID_GREY, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUTS + "fatality_vs_injury.png", dpi=120)
plt.close(fig)

# 5. Outlier treatment for PCA and clustering.
# (a) DROP Lakshadweep. It has about 2 accidents a year, so its shares and
#     ratios are noise (one accident moves a share by 50 points). It is an
#     outlier in 7 of 20 features for that reason, not because it is different.
# (b) WINSORIZE the rest: values beyond the 1.5 x IQR limits are pulled back
#     to the limit. These outliers are real (Delhi is a city, J&K has snow),
#     so we keep the states, but PCA and k-means work with distances and one
#     extreme value could pull a whole cluster towards it.
# The dashboard still shows the true, uncapped values (state_profiles.csv).
modelling = profile.drop(index="Lakshadweep")

capped_rows = []
for column in ratio_columns:
    values = modelling[column]
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    # rounded to 2 decimals like the data, so a value equal to the limit is not "capped"
    low_limit = round(q1 - 1.5 * iqr, 2)
    high_limit = round(q3 + 1.5 * iqr, 2)
    for state in values.index:
        value = values[state]
        if value < low_limit:
            capped_rows.append([column, state, value, low_limit])
            modelling.loc[state, column] = low_limit
        if value > high_limit:
            capped_rows.append([column, state, value, high_limit])
            modelling.loc[state, column] = high_limit

capped = pd.DataFrame(capped_rows, columns=["feature", "state", "true_value", "capped_to"])
capped.to_csv(OUTPUTS + "winsorized_values.csv", index=False)
modelling.to_csv(CLEAN + "state_profiles_for_modelling.csv")

print("\nWinsorized", len(capped), "values (out of", len(modelling) * len(ratio_columns), "):")
for i in range(len(capped)):
    row = capped.iloc[i]
    print("  ", row["feature"].ljust(24), row["state"].ljust(42), row["true_value"], "->", row["capped_to"])

print("\nSaved state_profiles.csv, state_profiles_for_modelling.csv, summary_statistics.csv,")
print("outlier_states.csv, winsorized_values.csv and two charts.")
