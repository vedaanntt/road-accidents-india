# Step 3: Dimensionality reduction with PCA
#
# Each state is described by 13 "condition" features: road mix, time of day,
# season, rural share and vehicle mix. 13 dimensions cannot be drawn, so PCA
# finds a few new axes (principal components) that keep most of the
# differences between states.
#
# The fatality ratio is NOT an input. It is the outcome: we colour the PCA
# map by it to see which kind of profile goes with deadly accidents.
#
# Run from the Project folder:
#     .venv\Scripts\python src\03_pca.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CLEAN = "data/clean/"
OUTPUTS = "outputs/"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_DARK = "#0b0b0b"
TEXT_GREY = "#52514e"
GRID_GREY = "#e4e3df"

# 13 condition features. Left out on purpose:
#   fatality and injury ratios -> they are the outcome, not conditions
#   share_expressway           -> almost always 0
#   share_other_road           -> always 100 minus the other road shares
condition_columns = [
    "share_national_hw", "share_state_hw",
    "night_share", "evening_share", "late_night_share",
    "monsoon_share", "winter_share",
    "rural_death_share",
    "two_wheeler_death_share", "pedestrian_death_share", "car_death_share",
    "truck_death_share", "bus_death_share",
]


def short_name(state):
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


def style_axes(ax):
    ax.grid(color=GRID_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT_GREY)


# 1. Load the winsorized profiles (Lakshadweep already dropped in step 2)
profiles = pd.read_csv(CLEAN + "state_profiles_for_modelling.csv", index_col=0)
X = profiles[condition_columns]
print("States:", len(X), " Features:", len(condition_columns), " Blanks:", int(X.isna().sum().sum()))

# 2. Standardise: every feature gets mean 0 and standard deviation 1.
# Without this, features with a big spread (rural share, std 17) would
# dominate features with a small spread (winter share, std 2) only because
# of their units, not because they matter more.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA with all components first, to see how much each one explains.
pca = PCA()
scores = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100
eigenvalues = pca.explained_variance_

print("\nComponent  eigenvalue  % variance  cumulative %")
cumulative = 0
for i in range(len(explained)):
    cumulative = cumulative + explained[i]
    print("  PC" + str(i + 1), "    ", round(eigenvalues[i], 2), "      ", round(explained[i], 1), "      ", round(cumulative, 1))

# How many components to keep? Two common rules:
#   Kaiser rule: keep components with eigenvalue above 1 (they explain
#                more than one original feature would on its own).
#   80% rule:    keep enough components to explain 80% of the variance.
kaiser_count = 0
for value in eigenvalues:
    if value > 1:
        kaiser_count = kaiser_count + 1
count_for_80 = 0
running = 0
for value in explained:
    running = running + value
    count_for_80 = count_for_80 + 1
    if running >= 80:
        break
print("\nKaiser rule keeps", kaiser_count, "components; the 80% rule keeps", count_for_80)

# 4. Loadings: how much each original feature contributes to each component.
# Big positive or negative loadings tell us what a component "means".
loadings = pd.DataFrame(pca.components_.T, index=condition_columns)
column_names = []
for i in range(len(explained)):
    column_names.append("PC" + str(i + 1))
loadings.columns = column_names
loadings.round(3).to_csv(OUTPUTS + "pca_loadings.csv")

for pc in ["PC1", "PC2", "PC3"]:
    ordered = loadings[pc].sort_values()
    print("\n" + pc, "most negative:", ", ".join(ordered.index[:3]))
    print(pc, "most positive:", ", ".join(ordered.index[-3:]))

# 5. Save the scores (each state's position on the components)
score_table = pd.DataFrame(scores[:, :4], index=X.index, columns=["PC1", "PC2", "PC3", "PC4"])
score_table["fatality_ratio"] = profiles["fatality_ratio"]
score_table = score_table.round(3)
score_table.to_csv(CLEAN + "pca_scores.csv")

# 6. Does position on a component go with a higher fatality ratio?
print("\nSpearman correlation of each component with fatality ratio:")
for pc in ["PC1", "PC2", "PC3", "PC4"]:
    rho, p_value = spearmanr(score_table[pc], score_table["fatality_ratio"])
    print("  ", pc, " rho =", round(rho, 2), " p =", round(p_value, 4))


# Chart 1: scree plot (bars) with the Kaiser line
fig, ax = plt.subplots(figsize=(9, 5))
positions = np.arange(1, len(explained) + 1)
ax.bar(positions, explained, color=BLUE, width=0.6)
for i in range(len(explained)):
    ax.text(positions[i], explained[i] + 0.6, str(round(explained[i], 1)), ha="center", fontsize=9, color=TEXT_GREY)
ax.set_xticks(positions)
ax.set_xlabel("Principal component", color=TEXT_DARK)
ax.set_ylabel("% of variance explained", color=TEXT_DARK)
ax.set_title("Scree plot: the first " + str(kaiser_count) + " components have eigenvalue above 1", color=TEXT_DARK)
style_axes(ax)
fig.tight_layout()
fig.savefig(OUTPUTS + "pca_scree.png", dpi=120)
plt.close(fig)


# Chart 2: states on PC1 and PC2, coloured by fatality ratio, with the
# strongest feature arrows (a "biplot").
# One-hue colour scale: light blue = low fatality ratio, dark blue = high.
blue_scale = LinearSegmentedColormap.from_list("blues", ["#cfe0f6", "#2a78d6", "#0b2f5c"])

fig, ax = plt.subplots(figsize=(12, 9))
dots = ax.scatter(score_table["PC1"], score_table["PC2"], c=score_table["fatality_ratio"], cmap=blue_scale,
                  s=90, edgecolor="white", linewidth=1.5, zorder=3)
for state in score_table.index:
    ax.annotate(short_name(state), (score_table.loc[state, "PC1"], score_table.loc[state, "PC2"]),
                xytext=(7, 4), textcoords="offset points", fontsize=8, color=TEXT_GREY)

# arrows for the 6 features with the longest loading in the PC1-PC2 plane
arrow_length = []
for feature in condition_columns:
    length = np.sqrt(loadings.loc[feature, "PC1"] ** 2 + loadings.loc[feature, "PC2"] ** 2)
    arrow_length.append([length, feature])
arrow_length.sort(reverse=True)
stretch = 4.5   # loadings are small numbers, so stretch the arrows to be visible
for i in range(6):
    feature = arrow_length[i][1]
    dx = loadings.loc[feature, "PC1"] * stretch
    dy = loadings.loc[feature, "PC2"] * stretch
    ax.arrow(0, 0, dx, dy, color=ORANGE, width=0.012, head_width=0.1, length_includes_head=True, zorder=2)
    ax.text(dx * 1.1, dy * 1.1, feature.replace("_", " "), color=TEXT_DARK, fontsize=9, ha="center")

colour_bar = fig.colorbar(dots, ax=ax)
colour_bar.set_label("Deaths per 100 accidents", color=TEXT_DARK)
ax.set_xlabel("PC1 (" + str(round(explained[0], 1)) + "% of variance)", color=TEXT_DARK)
ax.set_ylabel("PC2 (" + str(round(explained[1], 1)) + "% of variance)", color=TEXT_DARK)
ax.set_title("States by accident conditions (PCA), coloured by how deadly their accidents are", color=TEXT_DARK)
ax.axhline(0, color=GRID_GREY, linewidth=1)
ax.axvline(0, color=GRID_GREY, linewidth=1)
style_axes(ax)
fig.tight_layout()
fig.savefig(OUTPUTS + "pca_biplot.png", dpi=120)
plt.close(fig)

print("\nSaved pca_loadings.csv, pca_scores.csv, pca_scree.png and pca_biplot.png")
