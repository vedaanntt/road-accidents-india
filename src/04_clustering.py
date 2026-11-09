# Step 4: Clustering states with similar accident conditions
#
# Input: each state's scores on the 4 PCA components kept by the Kaiser
# rule (step 3). They hold 74% of the differences between states and
# leave out the noisy small components.
#
# Method: k-means. We choose k with the elbow method and the silhouette
# score, and cross-check k-means against a second method (Ward hierarchical
# clustering). If two different methods find the same groups, the groups
# are more likely to be real.
#
# The fatality ratio is NOT used to build the clusters. We compare it
# across clusters afterwards.
#
# Run from the Project folder:
#     .venv\Scripts\python src\04_clustering.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

CLEAN = "data/clean/"
OUTPUTS = "outputs/"
CHOSEN_K = 4

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
TEXT_DARK = "#0b0b0b"
TEXT_GREY = "#52514e"
GRID_GREY = "#e4e3df"

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
    if state == "Arunachal Pradesh":
        return "ArP"
    letters = ""
    for word in state.split(" "):
        if word != "And":
            letters = letters + word[0]
    return letters


def name_cluster(members):
    # k-means numbers its clusters randomly (0, 1, 2, 3), so we name each
    # cluster after a state we know is in it. These names describe where the
    # members are; location was never an input.
    if "Mizoram" in members:
        return "North-East hills"
    if "Kerala" in members:
        return "South and West"
    if "Himachal Pradesh" in members:
        return "Himalayan"
    if "Uttar Pradesh" in members:
        return "North and Gangetic plain"
    return "Other"


def style_axes(ax):
    ax.grid(color=GRID_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT_GREY)


# 1. Load data
scores = pd.read_csv(CLEAN + "pca_scores.csv", index_col=0)
X = scores[["PC1", "PC2", "PC3", "PC4"]].values
true_profiles = pd.read_csv(CLEAN + "state_profiles.csv", index_col=0)
true_profiles = true_profiles.loc[scores.index]   # same 34 regions, true (uncapped) values


# 2. Try k = 2 to 8 and record three measures for each:
#    inertia    - total squared distance of states to their cluster centre
#                 (always falls as k grows; we look for the "elbow")
#    silhouette - how much closer each state is to its own cluster than to
#                 the next one (-1 to 1, higher is better)
#    agreement  - adjusted Rand index between k-means and Ward clustering
#                 (1 = both methods give exactly the same groups)
# n_init=50 runs k-means 50 times from different random starts and keeps the
# best, because with only 34 points one bad start can give odd clusters.
k_values = []
inertia_values = []
silhouette_values = []
agreement_values = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    k_values.append(k)
    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(X, kmeans.labels_))
    agreement_values.append(adjusted_rand_score(kmeans.labels_, ward.labels_))

choice_table = pd.DataFrame({"k": k_values, "inertia": inertia_values,
                             "silhouette": silhouette_values, "agreement_with_ward": agreement_values})
choice_table = choice_table.round(3)
choice_table.to_csv(OUTPUTS + "choosing_k.csv", index=False)
print("Choosing k:")
print(choice_table.to_string(index=False))


# 3. Final clustering with the chosen k
kmeans = KMeans(n_clusters=CHOSEN_K, n_init=50, random_state=0).fit(X)
labels = kmeans.labels_

cluster_names = []
for i in range(len(labels)):
    members = list(scores.index[labels == labels[i]])
    cluster_names.append(name_cluster(members))

result = pd.DataFrame(index=scores.index)
result["cluster"] = cluster_names
result["PC1"] = scores["PC1"]
result["PC2"] = scores["PC2"]
result["fatality_ratio"] = true_profiles["fatality_ratio"]
result.to_csv(CLEAN + "state_clusters.csv")

order = ["North-East hills", "North and Gangetic plain", "South and West", "Himalayan"]

print("\nClusters (k = " + str(CHOSEN_K) + "):")
for name in order:
    members = result[result["cluster"] == name]
    print("  ", name, "| n =", len(members), "| median fatality ratio:", round(members["fatality_ratio"].median(), 1))
    print("      ", ", ".join(members.index))


# 4. Describe each cluster with the original (true) features.
# For each feature we measure how far the cluster average is from the
# all-India average, in standard deviations, and list the biggest.
cluster_profile = true_profiles[condition_columns + ["fatality_ratio", "injury_ratio"]].copy()
cluster_profile["cluster"] = result["cluster"]
cluster_means = cluster_profile.groupby("cluster").mean().loc[order]
cluster_means.insert(0, "states", result["cluster"].value_counts())
cluster_means.round(1).to_csv(OUTPUTS + "cluster_profiles.csv")

overall_mean = true_profiles[condition_columns].mean()
overall_std = true_profiles[condition_columns].std()
print("\nWhat makes each cluster different (cluster mean vs national mean, in std units):")
for name in order:
    difference = (cluster_means.loc[name, condition_columns] - overall_mean) / overall_std
    difference = difference.sort_values()
    print("  ", name)
    print("      lower than average: ", end="")
    for feature in difference.index[:3]:
        print(feature, "(" + str(round(difference[feature], 1)) + ")", end="  ")
    print("\n      higher than average:", end=" ")
    for feature in difference.index[-3:]:
        print(feature, "(+" + str(round(difference[feature], 1)) + ")", end="  ")
    print()


# 5. Is the fatality ratio really different between clusters?
# Kruskal-Wallis compares groups using ranks. We use it instead of ANOVA
# because the groups are small (2 to 15 states) and the ratio is skewed,
# so we cannot assume normal distributions.
groups = []
for name in order:
    groups.append(result[result["cluster"] == name]["fatality_ratio"].values)
statistic, p_value = kruskal(groups[0], groups[1], groups[2], groups[3])
print("\nKruskal-Wallis test, fatality ratio across clusters: H =", round(statistic, 2), " p =", round(p_value, 4))


# Chart 1: elbow and silhouette, side by side (two panels, not two y-axes)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(k_values, inertia_values, color=BLUE, linewidth=2, marker="o", markersize=8)
axes[0].set_title("Elbow method: inertia", color=TEXT_DARK)
axes[0].set_xlabel("Number of clusters k", color=TEXT_DARK)
axes[0].set_ylabel("Inertia (lower = tighter)", color=TEXT_DARK)
axes[1].plot(k_values, silhouette_values, color=BLUE, linewidth=2, marker="o", markersize=8)
axes[1].set_title("Silhouette score", color=TEXT_DARK)
axes[1].set_xlabel("Number of clusters k", color=TEXT_DARK)
axes[1].set_ylabel("Silhouette (higher = better)", color=TEXT_DARK)
for ax in axes:
    ax.axvline(CHOSEN_K, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.text(CHOSEN_K + 0.1, ax.get_ylim()[1] * 0.98, "chosen k = " + str(CHOSEN_K), color=TEXT_DARK, fontsize=9, va="top")
    ax.set_xticks(k_values)
    style_axes(ax)
fig.tight_layout()
fig.savefig(OUTPUTS + "choosing_k.png", dpi=120)
plt.close(fig)


# Chart 2: clusters on the PCA map. Each cluster has its own colour AND its
# own marker shape, so the groups can be told apart without colour too.
colours = [ORANGE, BLUE, AQUA, YELLOW]
markers = ["o", "s", "^", "D"]
fig, ax = plt.subplots(figsize=(12, 8.5))
for i in range(len(order)):
    name = order[i]
    members = result[result["cluster"] == name]
    label = name + " (median " + str(round(members["fatality_ratio"].median())) + " deaths per 100)"
    ax.scatter(members["PC1"], members["PC2"], s=110, color=colours[i], marker=markers[i],
               edgecolor="white", linewidth=1.5, label=label, zorder=3)
for state in result.index:
    ax.annotate(short_name(state), (result.loc[state, "PC1"], result.loc[state, "PC2"]),
                xytext=(7, 4), textcoords="offset points", fontsize=8, color=TEXT_GREY)
ax.set_xlabel("PC1: two-wheelers and pedestrians  <-->  cars, trucks and national highways", color=TEXT_DARK)
ax.set_ylabel("PC2: evening and rural  <-->  night-time", color=TEXT_DARK)
ax.set_title("Four groups of states with similar accident conditions (k-means on 4 PCA components)", color=TEXT_DARK)
ax.legend(loc="upper right", frameon=False, fontsize=10)
style_axes(ax)
fig.tight_layout()
fig.savefig(OUTPUTS + "clusters_pca_map.png", dpi=120)
plt.close(fig)


# Chart 3: fatality ratio of every state, grouped by cluster
fig, ax = plt.subplots(figsize=(10, 5.5))
for i in range(len(order)):
    name = order[i]
    values = result[result["cluster"] == name]["fatality_ratio"]
    for state in values.index:
        x = i + np.random.uniform(-0.12, 0.12)
        ax.scatter(x, values[state], s=60, color=colours[i], marker=markers[i], edgecolor="white", linewidth=1.5, zorder=3)
    ax.hlines(values.median(), i - 0.3, i + 0.3, color=TEXT_DARK, linewidth=2.5, zorder=4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, color=TEXT_DARK)
ax.set_ylabel("Deaths per 100 accidents", color=TEXT_DARK)
ax.set_title("How deadly are accidents in each cluster? (black line = median; Kruskal-Wallis p = "
             + str(round(p_value, 3)) + ")", color=TEXT_DARK)
style_axes(ax)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUTPUTS + "clusters_fatality_ratio.png", dpi=120)
plt.close(fig)

print("\nSaved state_clusters.csv, choosing_k.csv, cluster_profiles.csv and three charts.")
