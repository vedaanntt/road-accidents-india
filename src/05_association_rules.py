# Step 5: Association rules - which combinations of conditions go with deadly accidents?
#
# Apriori needs "baskets" of items, like shopping baskets. Our data has no
# single-accident records, so each STATE-YEAR is one basket (237 baskets).
# Each condition becomes an item by splitting it into thirds across all
# state-years:
#     top third    -> "night share: high"
#     bottom third -> "night share: low"
#     middle third -> no item (it is neither clearly high nor low)
# The fatality ratio is split the same way. "Deadly" = top third.
#
# Splitting into thirds also deals with outliers: an extreme value just
# lands in the top third, so no winsorizing is needed here.
#
# Safeguard: each state appears up to 7 times (once per year) and its years
# look alike, so one state could make a rule look strong on its own. We
# therefore keep only rules whose conditions occur in at least 5 different states.
#
# Run from the Project folder:
#     .venv\Scripts\python src\05_association_rules.py

import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore", category=DeprecationWarning)

CLEAN = "data/clean/"
OUTPUTS = "outputs/"

MIN_SUPPORT = 0.05      # the item set must appear in at least 5% of baskets (12 of 237)
MIN_CONFIDENCE = 0.7    # the rule must be right at least 70% of the time
MIN_STATES = 5          # the conditions must occur in at least 5 different states
MAX_CONDITIONS = 3      # at most 3 conditions on the left side, to keep rules readable

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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

DEADLY = "fatality ratio: high"
LESS_DEADLY = "fatality ratio: low"


def item_name(column, level):
    # "two_wheeler_death_share", "high" -> "two wheeler death share: high"
    return column.replace("_", " ") + ": " + level


def set_to_text(item_set):
    names = sorted(item_set)
    return " + ".join(names)


# 1. Load state-years. Lakshadweep is dropped, as in step 2.
features = pd.read_csv(CLEAN + "state_year_features.csv")
features = features[features["state"] != "Lakshadweep"]
features = features.dropna(subset=condition_columns + ["fatality_ratio"])
features = features.reset_index(drop=True)
print("Baskets (state-years):", len(features), " States:", features["state"].nunique())


# 2. Turn numbers into high / low items using thirds.
baskets = pd.DataFrame(index=features.index)
cut_points = []
for column in condition_columns + ["fatality_ratio"]:
    low_cut = features[column].quantile(1 / 3)
    high_cut = features[column].quantile(2 / 3)
    cut_points.append([column, round(low_cut, 2), round(high_cut, 2)])
    baskets[item_name(column, "high")] = features[column] > high_cut
    baskets[item_name(column, "low")] = features[column] < low_cut

cut_table = pd.DataFrame(cut_points, columns=["feature", "low_below", "high_above"])
cut_table.to_csv(OUTPUTS + "rules_cut_points.csv", index=False)
print("'Deadly' means a fatality ratio above", cut_table.iloc[-1]["high_above"], "deaths per 100 accidents")


# 3. Apriori: find item sets that appear often enough.
frequent = apriori(baskets, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_CONDITIONS + 1)
all_rules = association_rules(frequent, metric="confidence", min_threshold=MIN_CONFIDENCE)
print("Frequent item sets:", len(frequent), " Rules above", MIN_CONFIDENCE, "confidence:", len(all_rules))


def rules_for_target(target):
    # Keep rules of the form  conditions -> target,  where the conditions
    # do not themselves mention the fatality ratio.
    kept = []
    for i in range(len(all_rules)):
        row = all_rules.iloc[i]
        if row["consequents"] != frozenset([target]):
            continue
        mentions_fatality = False
        for item in row["antecedents"]:
            if item.startswith("fatality ratio"):
                mentions_fatality = True
        if mentions_fatality:
            continue

        # count the different states where all the conditions are true together
        has_all = baskets[list(row["antecedents"])].all(axis=1)
        states = sorted(features.loc[has_all, "state"].unique())
        if len(states) < MIN_STATES:
            continue

        kept.append({
            "conditions": row["antecedents"],
            "rule": set_to_text(row["antecedents"]),
            "number_of_conditions": len(row["antecedents"]),
            "support": row["support"],
            "confidence": row["confidence"],
            "lift": row["lift"],
            "number_of_states": len(states),
            "states": ", ".join(states),
        })
    return kept


def remove_redundant(rules):
    # A longer rule is only worth showing if it is MORE reliable than every
    # shorter rule built from some of its conditions. Otherwise the extra
    # condition adds nothing. Example: if "night: high -> deadly" already has
    # 80% confidence, "night: high + monsoon: low -> deadly" with 78% is dropped.
    useful = []
    for rule in rules:
        is_useful = True
        for other in rules:
            smaller = other["conditions"] < rule["conditions"]   # strict subset
            if smaller and other["confidence"] >= rule["confidence"]:
                is_useful = False
                break
        if is_useful:
            useful.append(rule)
    return useful


for target, file_name in [(DEADLY, "rules_deadly.csv"), (LESS_DEADLY, "rules_less_deadly.csv")]:
    found = rules_for_target(target)
    useful = remove_redundant(found)
    table = pd.DataFrame(useful)
    table = table.drop(columns=["conditions"])
    table = table.sort_values(["lift", "number_of_states"], ascending=False)
    table = table.round(3)
    table.to_csv(OUTPUTS + file_name, index=False)
    print("\nRules ->", target, ":", len(found), "passed the filters,", len(useful), "left after removing redundant ones")
    for i in range(min(10, len(table))):
        row = table.iloc[i]
        print("  ", row["rule"])
        print("        confidence", row["confidence"], " lift", row["lift"], " support", row["support"], " states", row["number_of_states"])


# 4. Chart: the top 10 BROAD rules for deadly accidents, by lift.
# Lift = how many times more likely a deadly year is when the conditions hold,
# compared with an average state-year. Lift 1 = no effect.
# Ranking by lift alone puts narrow rules first (the top one comes from just
# the 5 North-East hill states, which the clustering already showed). So the
# chart shows only rules that hold across at least 10 states; the CSV keeps all.
BROAD_STATES = 10
deadly = pd.read_csv(OUTPUTS + "rules_deadly.csv")
deadly = deadly[deadly["number_of_states"] >= BROAD_STATES].head(10)
print("\nBroad rules (" + str(BROAD_STATES) + "+ states) -> deadly:")
for i in range(len(deadly)):
    row = deadly.iloc[i]
    print("  ", row["rule"], "| confidence", row["confidence"], " lift", row["lift"], " states", row["number_of_states"])
deadly = deadly.iloc[::-1]   # reverse so the strongest rule is at the top of the chart

fig, ax = plt.subplots(figsize=(12, 6.5))
positions = range(len(deadly))
ax.barh(positions, deadly["lift"], color=ORANGE, height=0.6)
ax.axvline(1, color=TEXT_GREY, linewidth=1, linestyle="--")
ax.text(1.02, len(deadly) - 0.4, "lift 1 = no effect", color=TEXT_GREY, fontsize=8)
for i in range(len(deadly)):
    row = deadly.iloc[i]
    note = str(round(row["confidence"] * 100)) + "% right, " + str(row["number_of_states"]) + " states"
    ax.text(row["lift"] + 0.03, i, note, va="center", fontsize=9, color=TEXT_DARK)
ax.set_yticks(list(positions))
ax.set_yticklabels(deadly["rule"], fontsize=9, color=TEXT_DARK)
ax.set_xlabel("Lift (times more likely to be a deadly state-year than average)", color=TEXT_DARK)
ax.set_title("Condition combinations that go with deadly accidents\n(top 10 rules that hold in "
             + str(BROAD_STATES) + " or more states)", color=TEXT_DARK)
ax.set_xlim(0, deadly["lift"].max() + 0.9)
ax.grid(axis="x", color=GRID_GREY, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUTS + "rules_deadly_top10.png", dpi=120)
plt.close(fig)

print("\nSaved rules_deadly.csv, rules_less_deadly.csv, rules_cut_points.csv and rules_deadly_top10.png")
