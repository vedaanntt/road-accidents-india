# Road Accidents in India: When and Where Do They Turn Deadly?
# Interactive dashboard (step 6).
#
# Run from the Project folder:
#     .venv\Scripts\streamlit run app.py
#
# The app reads the clean files made by src/01 and src/02. It then reruns
# PCA, k-means and Apriori itself, so the viewer can change the settings.
# With the default settings it gives the same results as src/03 to src/05.

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="Road Accidents in India", layout="wide")

CLEAN = "data/clean/"
OUTPUTS = "outputs/"
BAD_YEAR = 2023          # NCRB counts for 2023 are about 3 times too large
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# colours (colour-blind-safe palette, same as the saved charts)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
TEXT_DARK = "#0b0b0b"
TEXT_GREY = "#52514e"
MUTED = "#898781"
GRID_GREY = "#e1e0d9"
AXIS_GREY = "#c3c2b7"
SURFACE = "#fcfcfb"
NO_DATA = "#e4e3df"

# groups always take colours in this order (deadliest group first, so it is
# orange as in the saved charts), and each also gets its own marker shape
GROUP_COLOURS = [ORANGE, BLUE, AQUA, YELLOW, MAGENTA, GREEN]
GROUP_SHAPES = ["circle", "square", "triangle-up", "diamond", "cross", "x"]

# one-hue scale for amounts: light blue = low, dark blue = high
BLUE_SCALE = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
# two-hue scale for correlations: blue = negative, grey = 0, red = positive
DIVERGING_SCALE = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a3a2"], [1, "#c93a3a"]]

# the 13 "condition" features used for PCA, clustering and rules
condition_columns = [
    "share_national_hw", "share_state_hw",
    "night_share", "evening_share", "late_night_share",
    "monsoon_share", "winter_share",
    "rural_death_share",
    "two_wheeler_death_share", "pedestrian_death_share", "car_death_share",
    "truck_death_share", "bus_death_share",
]

# every ratio and share column (all in %)
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

TIME_ORDER = ["12 AM to 3 AM", "3 AM to 6 AM", "6 AM to 9 AM", "9 AM to 12 PM",
              "12 PM to 3 PM", "3 PM to 6 PM", "6 PM to 9 PM", "9 PM to 12 AM"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

DEADLY = "fatality ratio: high"
LESS_DEADLY = "fatality ratio: low"


# ---------------------------------------------------------------- helpers

def nice_name(column):
    # "share_national_hw" -> "National highway share"
    name = column.replace("_", " ")
    name = name.replace(" hw", " highway")
    if name.startswith("fr "):
        name = "fatality ratio on " + name[3:]
    if name.startswith("share "):
        name = name[6:] + " share"
    return name.capitalize()


def unit_of(column):
    if column == "fatality_ratio" or column.startswith("fr_"):
        return "deaths per 100 accidents"
    if column == "injury_ratio":
        return "injured per 100 accidents"
    if column == "accidents":
        return "accidents"
    if column == "deaths":
        return "deaths"
    return "%"


def format_value(value, column):
    if column == "accidents" or column == "deaths":
        return format(int(round(value)), ",")
    return str(round(value, 1))


def in_lakh(number):
    # 452000 -> "4.52 lakh"
    return str(round(number / 100000, 2)) + " lakh"


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


def item_name(column, level):
    return column.replace("_", " ") + ": " + level


def iqr_limits(values):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return q1, values.median(), q3, round(q1 - 1.5 * iqr, 2), round(q3 + 1.5 * iqr, 2)


def style_chart(fig, x_title, y_title, height):
    fig.update_layout(height=height, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                      font_color=TEXT_DARK, xaxis_title=x_title, yaxis_title=y_title,
                      margin_l=10, margin_r=10, margin_t=30, margin_b=10,
                      hoverlabel_bgcolor="white", legend_orientation="h", legend_y=1.08)
    # automargin makes room for long labels such as state names
    fig.update_xaxes(gridcolor=GRID_GREY, linecolor=AXIS_GREY, zeroline=False, color=TEXT_GREY,
                     automargin=True, title_font_color=TEXT_DARK)
    fig.update_yaxes(gridcolor=GRID_GREY, linecolor=AXIS_GREY, zeroline=False, color=TEXT_GREY,
                     automargin=True, title_font_color=TEXT_DARK)
    return fig


def show(fig, title=None):
    # titles are written by Streamlit above the chart, so they never overlap
    # the legend or the chart's zoom buttons
    if title is not None:
        st.markdown("**" + title + "**")
    st.plotly_chart(fig, width="stretch", theme=None)


# ---------------------------------------------------------------- data

@st.cache_data
def load_data():
    features = pd.read_csv(CLEAN + "state_year_features.csv")
    profiles = pd.read_csv(CLEAN + "state_profiles.csv", index_col=0)
    modelling = pd.read_csv(CLEAN + "state_profiles_for_modelling.csv", index_col=0)
    grid = pd.read_csv("data/tile_grid.csv")
    time_of_day = pd.read_csv(CLEAN + "time_of_day.csv")
    month = pd.read_csv(CLEAN + "month.csv")
    return features, profiles, modelling, grid, time_of_day, month


features, profiles, modelling, grid, time_of_day, month = load_data()


# ---------------------------------------------------------------- tile map
# Each state is one equal square, placed roughly where it is in India
# (positions in data/tile_grid.csv). Small states are as easy to see as big
# ones, and there are no border lines to get wrong.

def tile_figure(values, value_text, colour_scale, low, high):
    value_grid = np.full((9, 7), np.nan)
    empty_grid = np.full((9, 7), np.nan)
    value_labels = np.full((9, 7), "", dtype=object)
    empty_labels = np.full((9, 7), "", dtype=object)
    hover_grid = np.full((9, 7), "", dtype=object)
    for i in range(len(grid)):
        state = grid.iloc[i]["state"]
        row = grid.iloc[i]["row"]
        col = grid.iloc[i]["col"]
        has_value = state in values.index and not pd.isna(values[state])
        if has_value:
            value_grid[row, col] = values[state]
            value_labels[row, col] = grid.iloc[i]["short"]
            hover_grid[row, col] = state + "<br>" + value_text[state]
        else:
            empty_grid[row, col] = 0
            empty_labels[row, col] = grid.iloc[i]["short"]
            hover_grid[row, col] = state + "<br>no data"

    fig = go.Figure()
    # grey tiles for states with no value
    fig.add_trace(go.Heatmap(z=empty_grid, text=empty_labels, customdata=hover_grid,
                             texttemplate="%{text}", textfont_size=13, textfont_color=TEXT_GREY,
                             colorscale=[[0, NO_DATA], [1, NO_DATA]], showscale=False,
                             xgap=4, ygap=4, hoverongaps=False,
                             hovertemplate="%{customdata}<extra></extra>"))
    # coloured tiles
    fig.add_trace(go.Heatmap(z=value_grid, text=value_labels, customdata=hover_grid,
                             texttemplate="%{text}", textfont_size=13,
                             colorscale=colour_scale, zmin=low, zmax=high,
                             xgap=4, ygap=4, hoverongaps=False,
                             hovertemplate="%{customdata}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
    fig.update_layout(height=540, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                      margin_l=0, margin_r=0, margin_t=10, margin_b=0, hoverlabel_bgcolor="white")
    return fig


def value_tile_map(values, column):
    value_text = pd.Series(dtype=object)
    for state in values.index:
        if not pd.isna(values[state]):
            value_text[state] = format_value(values[state], column) + " " + unit_of(column)
    fig = tile_figure(values, value_text, BLUE_SCALE, values.min(), values.max())
    fig.data[1].colorbar.title.text = unit_of(column)
    return fig


def group_tile_map(groups, names):
    # groups: state -> group number (0, 1, 2 ...). One flat colour per group.
    k = len(names)
    scale = []
    for i in range(k):
        scale.append([i / k, GROUP_COLOURS[i]])
        scale.append([(i + 1) / k, GROUP_COLOURS[i]])
    value_text = pd.Series(dtype=object)
    for state in groups.index:
        value_text[state] = names[groups[state]]
    fig = tile_figure(groups, value_text, scale, -0.5, k - 0.5)
    fig.data[1].colorbar.tickvals = list(range(k))
    fig.data[1].colorbar.ticktext = names
    return fig


# ---------------------------------------------------------------- models (cached)

@st.cache_data
def run_pca(columns):
    X = modelling[list(columns)]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    names = []
    for i in range(scores.shape[1]):
        names.append("PC" + str(i + 1))
    score_table = pd.DataFrame(scores, index=X.index, columns=names)
    loadings = pd.DataFrame(pca.components_.T, index=list(columns), columns=names)
    return score_table, loadings, pca.explained_variance_, pca.explained_variance_ratio_ * 100


@st.cache_data
def run_kmeans(X, k):
    # 50 random starts, keep the best (with 34 points one bad start can mislead)
    model = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
    return model.labels_, model.inertia_


def name_cluster(members):
    # names for the 4 groups found with the default settings (see src/04)
    if "Mizoram" in members:
        return "North-East hills"
    if "Kerala" in members:
        return "South and West"
    if "Himachal Pradesh" in members:
        return "Himalayan"
    if "Uttar Pradesh" in members:
        return "North and Gangetic plain"
    return "Other"


def order_groups(labels, states, fatality, k):
    # k-means numbers its groups randomly, so we renumber them from the
    # deadliest (group 0) to the least deadly, by median fatality ratio.
    medians = []
    for g in range(k):
        medians.append([np.median(fatality[labels == g]), g])
    medians.sort(reverse=True)
    new_labels = np.zeros(len(labels), dtype=int)
    names = []
    for position in range(k):
        old = medians[position][1]
        new_labels[labels == old] = position
        names.append(name_cluster(list(states[labels == old])))
    # the place names only fit the default 4 groups; otherwise just number them
    if k != 4 or "Other" in names or len(set(names)) < k:
        names = []
        for position in range(k):
            names.append("Group " + str(position + 1))
    return new_labels, names


@st.cache_data
def make_baskets():
    # one basket per state-year; each feature split into thirds (see src/05)
    data = features[features["state"] != "Lakshadweep"]
    data = data.dropna(subset=condition_columns + ["fatality_ratio"]).reset_index(drop=True)
    baskets = pd.DataFrame(index=data.index)
    cut_rows = []
    for column in condition_columns + ["fatality_ratio"]:
        low_cut = data[column].quantile(1 / 3)
        high_cut = data[column].quantile(2 / 3)
        cut_rows.append([nice_name(column), round(low_cut, 2), round(high_cut, 2)])
        baskets[item_name(column, "high")] = data[column] > high_cut
        baskets[item_name(column, "low")] = data[column] < low_cut
    cuts = pd.DataFrame(cut_rows, columns=["feature", "low = below", "high = above"])
    return data, baskets, cuts


@st.cache_data
def mine_rules(min_support, max_conditions):
    data, baskets, cuts = make_baskets()
    frequent = apriori(baskets, min_support=min_support, use_colnames=True, max_len=max_conditions + 1)
    return association_rules(frequent, metric="confidence", min_threshold=0.5)


def rules_for_target(all_rules, target, min_confidence, min_states):
    data, baskets, cuts = make_baskets()
    kept = []
    for i in range(len(all_rules)):
        row = all_rules.iloc[i]
        if row["consequents"] != frozenset([target]):
            continue
        if row["confidence"] < min_confidence:
            continue
        mentions_fatality = False
        for item in row["antecedents"]:
            if item.startswith("fatality ratio"):
                mentions_fatality = True
        if mentions_fatality:
            continue
        # how many different states have all the conditions together?
        has_all = baskets[list(row["antecedents"])].all(axis=1)
        states = sorted(data.loc[has_all, "state"].unique())
        if len(states) < min_states:
            continue
        kept.append({
            "conditions": row["antecedents"],
            "rule": " + ".join(sorted(row["antecedents"])),
            "confidence": round(row["confidence"], 3),
            "lift": round(row["lift"], 2),
            "support": round(row["support"], 3),
            "states": len(states),
            "which states": ", ".join(states),
        })
    return kept


def remove_redundant(rules):
    # a longer rule is kept only if it is more reliable than every shorter
    # rule made from some of its conditions
    useful = []
    for rule in rules:
        is_useful = True
        for other in rules:
            if other["conditions"] < rule["conditions"] and other["confidence"] >= rule["confidence"]:
                is_useful = False
                break
        if is_useful:
            useful.append(rule)
    return useful


# ---------------------------------------------------------------- page

st.title("Road Accidents in India: When and Where Do They Turn Deadly?")
st.caption("NCRB Accidental Deaths and Suicides in India (2018-2024) for every state and union territory. "
           "MoRTH Road Accidents in India 2024 used only to check the numbers. "
           "Hover over any chart for exact values.")

(tab_overview, tab_quality, tab_stats, tab_pca, tab_clusters,
 tab_rules, tab_state) = st.tabs(["Overview", "Data quality", "Statistics", "PCA",
                                  "Clusters", "Association rules", "State explorer"])


# ================================================================ 1. Overview
with tab_overview:
    year_options = ["All years (average)"]
    for year in YEARS:
        year_options.append(str(year))
    map_columns = ["fatality_ratio", "injury_ratio", "accidents", "deaths"] + ratio_columns[2:]

    left, right = st.columns(2)
    year_choice = left.selectbox("Year", year_options, index=0)
    measure = right.selectbox("Colour the map by", map_columns, format_func=nice_name)

    # national totals for the headline numbers
    usable = features[features["year"] != BAD_YEAR]
    if year_choice == "All years (average)":
        number_of_years = usable["year"].nunique()
        accidents_total = usable["accidents"].sum() / number_of_years
        deaths_total = usable["deaths"].sum() / number_of_years
        period = "per year, 2018-2024 without 2023"
    else:
        this_year = features[features["year"] == int(year_choice)]
        accidents_total = this_year["accidents"].sum()
        deaths_total = this_year["deaths"].sum()
        period = "in " + year_choice

    tile1, tile2, tile3 = st.columns(3)
    if year_choice == str(BAD_YEAR):
        tile1.metric("Recorded road accidents", "not reliable")
        tile2.metric("Deaths", "not reliable")
        tile3.metric("Deaths per 100 accidents", "not reliable")
        st.warning("NCRB counts for 2023 are about 3 times too large (MoRTH reports 4.8 lakh accidents, "
                   "NCRB 14.1 lakh). Counts are hidden; state ratios and shares are still shown on the map.")
    else:
        tile1.metric("Recorded road accidents " + period, in_lakh(accidents_total))
        tile2.metric("Deaths " + period, in_lakh(deaths_total))
        tile3.metric("Deaths per 100 accidents", str(round(deaths_total / accidents_total * 100, 1)))

    if year_choice == "All years (average)":
        if measure == "accidents":
            map_values = profiles["accidents_per_year"]
        elif measure == "deaths":
            map_values = profiles["deaths_per_year"]
        else:
            map_values = profiles[measure]
    else:
        map_values = features[features["year"] == int(year_choice)].set_index("state")[measure]

    map_side, ranking_side = st.columns([3, 2])
    with map_side:
        st.subheader(nice_name(measure) + " by state")
        show(value_tile_map(map_values, measure))
        st.caption("Each square is one state or UT, placed roughly where it is in India "
                   "(not to scale). Grey = no data for this choice.")
    with ranking_side:
        st.subheader("Ranking")
        ranked = map_values.dropna().sort_values(ascending=True).tail(15)
        fig = go.Figure(go.Bar(x=ranked.values, y=ranked.index, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
        style_chart(fig, unit_of(measure), "", 540)
        show(fig, "Top 15")

    st.subheader("All-India trend")
    trend_rows = []
    for year in YEARS:
        rows = features[features["year"] == year]
        if year == BAD_YEAR:
            trend_rows.append([year, np.nan, np.nan])
        else:
            trend_rows.append([year, rows["accidents"].sum(), rows["deaths"].sum() / rows["accidents"].sum() * 100])
    trend = pd.DataFrame(trend_rows, columns=["year", "accidents", "fatality_ratio"])

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=trend["year"], y=trend["accidents"] / 100000, marker_color=BLUE,
                               hovertemplate="%{x}: %{y:.2f} lakh<extra></extra>"))
        fig.add_annotation(x=BAD_YEAR, y=1, text="2023 left out<br>(counts 3x too large)",
                           showarrow=False, font_color=TEXT_GREY, font_size=11)
        style_chart(fig, "", "Recorded accidents (lakh)", 320)
        fig.update_xaxes(dtick=1)
        show(fig)
    with right:
        fig = go.Figure(go.Scatter(x=trend["year"], y=trend["fatality_ratio"], mode="lines+markers",
                                   line_color=BLUE, line_width=2, marker_size=9,
                                   hovertemplate="%{x}: %{y:.1f} deaths per 100<extra></extra>"))
        style_chart(fig, "", "Deaths per 100 accidents", 320)
        fig.update_xaxes(dtick=1)
        show(fig)
    st.caption("2020 is low because of the COVID-19 lockdown. Deaths per 100 accidents rose from about "
               "34 in 2018 to about 38 in 2021 and have stayed there since.")


# ================================================================ 2. Data quality
with tab_quality:
    st.header("How we handled problems in the data")
    st.write("Every problem we found is listed at the bottom of this page. "
             "Below are the three that matter most for the results.")

    # 2a. coverage of every state-year
    st.subheader("1. Which state-years do we have?")
    states_sorted = sorted(profiles.index, reverse=True)
    status_grid = np.zeros((len(states_sorted), len(YEARS)))
    status_text = np.full((len(states_sorted), len(YEARS)), "", dtype=object)
    for i in range(len(states_sorted)):
        state = states_sorted[i]
        for j in range(len(YEARS)):
            year = YEARS[j]
            found = features[(features["state"] == state) & (features["year"] == year)]
            if len(found) == 0:
                status_grid[i, j] = 2
                status_text[i, j] = state + ", " + str(year) + ": missing (not imputed)"
            elif year == BAD_YEAR:
                status_grid[i, j] = 1
                status_text[i, j] = state + ", " + str(year) + ": ratios kept, counts blanked"
            else:
                status_text[i, j] = state + ", " + str(year) + ": complete"
    scale = [[0, "#cde2fb"], [0.333, "#cde2fb"], [0.333, YELLOW], [0.667, YELLOW], [0.667, ORANGE], [1, ORANGE]]
    fig = go.Figure(go.Heatmap(z=status_grid, x=YEARS, y=states_sorted, customdata=status_text,
                               colorscale=scale, zmin=-0.5, zmax=2.5, xgap=2, ygap=2,
                               hovertemplate="%{customdata}<extra></extra>",
                               colorbar_tickvals=[0, 1, 2],
                               colorbar_ticktext=["complete", "2023: ratios only", "missing"]))
    style_chart(fig, "", "", 760)
    fig.update_xaxes(dtick=1, side="top", showgrid=False)
    fig.update_yaxes(showgrid=False)
    show(fig)
    st.markdown(
        "- **Ladakh 2018-19** did not exist as a UT yet (structurally missing), so Ladakh is merged into "
        "*Jammu And Kashmir And Ladakh* in every year.\n"
        "- **Andaman and Nicobar 2019** is a reporting gap unrelated to the number of accidents "
        "(close to MCAR). It is left as a gap: its average uses 6 years instead of 7.\n"
        "- **2023**: NCRB counts are about 3 times too large in every state, but top and bottom of a "
        "ratio are inflated by the same factor, so ratios and shares are kept.")
    with st.expander("Check: is the 2023 fatality ratio in line with 2022 and 2024?"):
        st.dataframe(pd.read_csv(OUTPUTS + "check_2023_inflation.csv"), width="stretch", hide_index=True)

    # 2b. MNAR
    st.subheader("2. Under-reporting of minor accidents (MNAR)")
    both = profiles[["fatality_ratio", "injury_ratio"]].dropna()
    rho, p_value = spearmanr(both["fatality_ratio"], both["injury_ratio"])
    labels = []
    for state in both.index:
        labels.append(short_name(state))
    fig = go.Figure(go.Scatter(x=both["injury_ratio"], y=both["fatality_ratio"], mode="markers+text",
                               text=labels, textposition="top right", textfont_size=10,
                               textfont_color=TEXT_GREY, customdata=both.index,
                               marker_color=BLUE, marker_size=10, marker_line_color="white",
                               marker_line_width=2,
                               hovertemplate="%{customdata}<br>%{x:.0f} injured, %{y:.0f} deaths per 100 accidents<extra></extra>"))
    style_chart(fig, "Injured people per 100 recorded accidents", "Deaths per 100 recorded accidents", 480)
    show(fig)
    st.markdown(
        "If a state's police record only serious accidents, each recorded accident has many deaths and "
        "few injuries. So under-reporting predicts a **negative** relationship, and we find one: "
        "Spearman rho = **" + str(round(rho, 2)) + "** (p = " + str(round(p_value, 3)) + "). "
        "The missing accidents are missing *because* they are minor: this is MNAR, and it cannot be "
        "fixed by imputation. We keep the fatality ratio but read it partly as a measure of reporting.")

    # 2c. outliers
    st.subheader("3. Outliers: drop or winsorize?")
    left, right = st.columns([1, 2])
    outlier_feature = left.selectbox("Feature", ratio_columns, format_func=nice_name, index=0)
    view = right.radio("Show", ["True values (all 35 regions)",
                                "Used for PCA and clustering (Lakshadweep dropped, winsorized)"], horizontal=True)
    if view.startswith("True"):
        values = profiles[outlier_feature].dropna()
    else:
        values = modelling[outlier_feature].dropna()
    q1, median, q3, low_limit, high_limit = iqr_limits(values)

    random_numbers = np.random.default_rng(0).uniform(-0.3, 0.3, len(values))
    colours = []
    labels = []
    for state in values.index:
        if values[state] < low_limit or values[state] > high_limit:
            colours.append(ORANGE)
            labels.append(short_name(state))
        else:
            colours.append(BLUE)
            labels.append("")
    fig = go.Figure(go.Scatter(x=random_numbers, y=values.values, mode="markers+text", text=labels,
                               textposition="middle right", textfont_color=TEXT_DARK,
                               customdata=values.index, marker_color=colours, marker_size=11,
                               marker_line_color="white", marker_line_width=2,
                               hovertemplate="%{customdata}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=high_limit, line_dash="dash", line_color=MUTED,
                  annotation_text="upper limit Q3 + 1.5 IQR = " + str(high_limit), annotation_font_color=TEXT_GREY)
    fig.add_hline(y=low_limit, line_dash="dash", line_color=MUTED,
                  annotation_text="lower limit Q1 - 1.5 IQR = " + str(low_limit), annotation_font_color=TEXT_GREY)
    fig.add_hline(y=median, line_color=TEXT_DARK, line_width=1,
                  annotation_text="median", annotation_position="bottom left", annotation_font_color=TEXT_GREY)
    style_chart(fig, "", nice_name(outlier_feature) + " (" + unit_of(outlier_feature) + ")", 460)
    fig.update_xaxes(visible=False, range=[-0.5, 1])
    show(fig)
    st.markdown(
        "Orange = outside the 1.5 x IQR limits. **Lakshadweep is dropped** from the models: it has about "
        "2 accidents a year, so one accident moves a share by 50 points. **Other outliers are real** "
        "(Delhi is a city, Sikkim records many injuries), so they are kept but **winsorized**: pulled "
        "back to the limit in the copy used for PCA and clustering, because those methods work with "
        "distances and one extreme value can pull a whole cluster. Switch the view above to compare.")
    capped = pd.read_csv(OUTPUTS + "winsorized_values.csv")
    capped = capped[capped["feature"] == outlier_feature]
    if len(capped) > 0:
        st.write("Values capped for this feature:")
        st.dataframe(capped, width="stretch", hide_index=True)

    with st.expander("Full data problems log (22 problems)"):
        with open("DATA_PROBLEMS.md", encoding="utf-8") as log_file:
            st.markdown(log_file.read())


# ================================================================ 3. Statistics
with tab_stats:
    st.header("Summary statistics")
    st.caption("One profile per state: the average of each ratio over the years it has data. All values in %, "
               "except ratios, which are per 100 accidents.")
    summary = profiles[ratio_columns].describe().T
    summary["skew"] = profiles[ratio_columns].skew()
    summary = summary.drop(columns=["count"]).round(2)
    new_index = []
    for column in summary.index:
        new_index.append(nice_name(column))
    summary.index = new_index
    st.dataframe(summary, width="stretch")
    st.caption("Several features are strongly skewed (skew beyond +-1: expressway, bus, monsoon and car "
               "shares, and rural death share to the left). That is why we use medians, IQR and Spearman "
               "correlation instead of means, z-scores and Pearson correlation.")

    st.subheader("Compare two features")
    left, right = st.columns(2)
    x_feature = left.selectbox("X axis", ratio_columns, index=ratio_columns.index("late_night_share"),
                               format_func=nice_name)
    y_feature = right.selectbox("Y axis", ratio_columns, index=0, format_func=nice_name)
    pair = modelling[[x_feature, y_feature]].dropna()
    if x_feature == y_feature:
        st.info("Pick two different features.")
    else:
        rho, p_value = spearmanr(pair[x_feature], pair[y_feature])
        labels = []
        for state in pair.index:
            labels.append(short_name(state))
        fig = go.Figure(go.Scatter(x=pair[x_feature], y=pair[y_feature], mode="markers+text", text=labels,
                                   textposition="top right", textfont_size=10, textfont_color=TEXT_GREY,
                                   customdata=pair.index, marker_color=BLUE, marker_size=10,
                                   marker_line_color="white", marker_line_width=2,
                                   hovertemplate="%{customdata}<br>x = %{x:.1f}, y = %{y:.1f}<extra></extra>"))
        style_chart(fig, nice_name(x_feature), nice_name(y_feature), 460)
        show(fig)
        strength = "weak"
        if abs(rho) >= 0.3:
            strength = "moderate"
        if abs(rho) >= 0.6:
            strength = "strong"
        st.write("Spearman rho = **" + str(round(rho, 2)) + "**, p = " + str(round(p_value, 4))
                 + " (" + strength + "; 34 regions, winsorized, Lakshadweep dropped).")

    st.subheader("Spearman correlation between all condition features and the fatality ratio")
    corr_columns = condition_columns + ["fatality_ratio"]
    corr = modelling[corr_columns].corr(method="spearman")
    names = []
    for column in corr_columns:
        names.append(nice_name(column))
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, colorscale=DIVERGING_SCALE, zmin=-1, zmax=1,
                               xgap=2, ygap=2, texttemplate="%{z:.1f}", textfont_size=10,
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                               colorbar_title_text="rho"))
    style_chart(fig, "", "", 620)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-40)
    show(fig)
    st.caption("Look at the bottom row: it shows which conditions go with deadlier accidents "
               "(red) or less deadly ones (blue).")


# ================================================================ 4. PCA
with tab_pca:
    st.header("PCA: summarising 13 conditions in a few axes")
    pca_columns = st.multiselect("Features used for PCA (and for clustering in the next tab)",
                                 condition_columns, default=condition_columns, format_func=nice_name)
    if len(pca_columns) < 3:
        st.error("Pick at least 3 features.")
        st.stop()

    scores, loadings, eigenvalues, explained = run_pca(tuple(pca_columns))
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

    tile1, tile2, tile3 = st.columns(3)
    tile1.metric("Kaiser rule (eigenvalue above 1) keeps", str(kaiser_count) + " components")
    tile2.metric("80% rule keeps", str(count_for_80) + " components")
    tile3.metric("PC1 + PC2 explain", str(round(explained[0] + explained[1], 1)) + "%")

    left, right = st.columns([2, 3])
    with left:
        positions = []
        for i in range(len(eigenvalues)):
            positions.append(i + 1)
        fig = go.Figure(go.Bar(x=positions, y=eigenvalues, marker_color=BLUE, customdata=explained,
                               hovertemplate="PC%{x}: eigenvalue %{y:.2f}, %{customdata:.1f}% of variance<extra></extra>"))
        fig.add_hline(y=1, line_dash="dash", line_color=ORANGE, annotation_text="Kaiser line (eigenvalue 1)",
                      annotation_font_color=TEXT_DARK)
        style_chart(fig, "Principal component", "Eigenvalue", 420)
        fig.update_xaxes(dtick=1)
        show(fig, "Scree plot")
    with right:
        pc_names = list(scores.columns)
        a, b = st.columns(2)
        x_pc = a.selectbox("Horizontal axis", pc_names, index=0)
        y_pc = b.selectbox("Vertical axis", pc_names, index=1)
        labels = []
        for state in scores.index:
            labels.append(short_name(state))
        fatality = modelling.loc[scores.index, "fatality_ratio"]
        fig = go.Figure(go.Scatter(x=scores[x_pc], y=scores[y_pc], mode="markers+text", text=labels,
                                   textposition="top right", textfont_size=10, textfont_color=TEXT_GREY,
                                   customdata=np.stack([scores.index, fatality.round(1)], axis=1),
                                   marker_color=fatality, marker_colorscale=BLUE_SCALE, marker_size=11,
                                   marker_line_color="white", marker_line_width=2, marker_showscale=True,
                                   marker_colorbar_title_text="deaths per<br>100 accidents",
                                   hovertemplate="%{customdata[0]}<br>fatality ratio %{customdata[1]}<extra></extra>"))
        # arrows for the 6 features with the longest loadings on these two axes
        arrow_length = []
        for feature in pca_columns:
            length = np.sqrt(loadings.loc[feature, x_pc] ** 2 + loadings.loc[feature, y_pc] ** 2)
            arrow_length.append([length, feature])
        arrow_length.sort(reverse=True)
        stretch = 4.5
        for i in range(min(6, len(arrow_length))):
            feature = arrow_length[i][1]
            dx = loadings.loc[feature, x_pc] * stretch
            dy = loadings.loc[feature, y_pc] * stretch
            # the arrow itself (plotly draws an arrow from (ax, ay) to (x, y))
            fig.add_annotation(x=dx, y=dy, ax=0, ay=0, axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowcolor=ORANGE, arrowwidth=2, text="")
            # its label, just past the arrow head
            fig.add_annotation(x=dx * 1.2, y=dy * 1.2, showarrow=False, text=nice_name(feature),
                               font_color=TEXT_DARK, font_size=11, bgcolor="rgba(252,252,251,0.8)")
        style_chart(fig, x_pc + " (" + str(round(explained[pc_names.index(x_pc)], 1)) + "%)",
                    y_pc + " (" + str(round(explained[pc_names.index(y_pc)], 1)) + "%)", 520)
        show(fig, "Biplot: states coloured by fatality ratio (not a PCA input)")

    st.subheader("What does each component mean?")
    st.caption("Loadings: how much each feature pushes a component up (positive) or down (negative).")
    first_pcs = pc_names[:max(kaiser_count, 2)]
    loading_table = loadings[first_pcs].round(2)
    new_index = []
    for column in loading_table.index:
        new_index.append(nice_name(column))
    loading_table.index = new_index
    fig = go.Figure(go.Heatmap(z=loading_table.values, x=first_pcs, y=list(loading_table.index),
                               colorscale=DIVERGING_SCALE, zmin=-0.7, zmax=0.7, xgap=2, ygap=2,
                               texttemplate="%{z:.2f}", hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>"))
    style_chart(fig, "", "", 460)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    show(fig)

    rho_rows = []
    for pc in first_pcs:
        rho, p_value = spearmanr(scores[pc], fatality)
        rho_rows.append([pc, round(rho, 2), round(p_value, 4)])
    st.write("Does a state's position on each component go with a higher fatality ratio?")
    st.dataframe(pd.DataFrame(rho_rows, columns=["component", "Spearman rho with fatality ratio", "p value"]),
                 hide_index=True)


# ================================================================ 5. Clusters
with tab_clusters:
    st.header("Clusters: which states have similar accident conditions?")
    max_pcs = min(6, len(pca_columns))
    left, right = st.columns(2)
    k = left.slider("Number of clusters k", 2, 6, 4)
    number_of_pcs = right.slider("Number of PCA components used", 2, max_pcs, min(4, max_pcs))
    st.caption("Uses the features chosen in the PCA tab. The fatality ratio is not used to form the "
               "clusters; it is compared across them afterwards.")

    X = scores.iloc[:, :number_of_pcs].values
    raw_labels, inertia = run_kmeans(X, k)
    fatality = modelling.loc[scores.index, "fatality_ratio"].values
    labels, group_names = order_groups(raw_labels, scores.index, fatality, k)
    silhouette = silhouette_score(X, labels)
    groups = []
    for g in range(k):
        groups.append(fatality[labels == g])
    statistic, p_value = kruskal(*groups)

    tile1, tile2, tile3 = st.columns(3)
    tile1.metric("Silhouette score", str(round(silhouette, 2)), help="-1 to 1; higher = better separated groups")
    tile2.metric("Kruskal-Wallis p (fatality ratio)", str(round(p_value, 4)),
                 help="Below 0.05: the groups really differ in how deadly accidents are")
    tile3.metric("Smallest group", str(int(np.bincount(labels).min())) + " states")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for g in range(k):
            members = scores.index[labels == g]
            member_labels = []
            for state in members:
                member_labels.append(short_name(state))
            fig.add_trace(go.Scatter(x=scores.loc[members, "PC1"], y=scores.loc[members, "PC2"],
                                     mode="markers+text", name=group_names[g], text=member_labels,
                                     textposition="top right", textfont_size=10, textfont_color=TEXT_GREY,
                                     customdata=members, marker_color=GROUP_COLOURS[g],
                                     marker_symbol=GROUP_SHAPES[g], marker_size=12,
                                     marker_line_color="white", marker_line_width=2,
                                     hovertemplate="%{customdata}<extra>" + group_names[g] + "</extra>"))
        style_chart(fig, "PC1", "PC2", 520)
        show(fig, "Clusters on the first two components")
    with right:
        group_of = pd.Series(labels, index=scores.index)
        st.markdown("**Clusters on the map**")
        show(group_tile_map(group_of, group_names))
        st.caption("Location was never an input, yet the groups mostly follow geography.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for g in range(k):
            members = scores.index[labels == g]
            jitter = np.random.default_rng(g).uniform(-0.15, 0.15, len(members))
            fig.add_trace(go.Scatter(x=g + jitter, y=fatality[labels == g], mode="markers", name=group_names[g],
                                     customdata=members, marker_color=GROUP_COLOURS[g],
                                     marker_symbol=GROUP_SHAPES[g], marker_size=11,
                                     marker_line_color="white", marker_line_width=2, showlegend=False,
                                     hovertemplate="%{customdata}: %{y:.1f}<extra></extra>"))
            median = np.median(fatality[labels == g])
            fig.add_shape(type="line", x0=g - 0.3, x1=g + 0.3, y0=median, y1=median,
                          line_color=TEXT_DARK, line_width=3)
        style_chart(fig, "", "Deaths per 100 accidents", 420)
        fig.update_xaxes(tickvals=list(range(k)), ticktext=group_names, showgrid=False)
        show(fig, "How deadly is each cluster? (black line = median)")
    with right:
        k_values = []
        inertias = []
        silhouettes = []
        for k_try in range(2, 9):
            try_labels, try_inertia = run_kmeans(X, k_try)
            k_values.append(k_try)
            inertias.append(try_inertia)
            silhouettes.append(silhouette_score(X, try_labels))
        choice = st.radio("Choosing k", ["Elbow (inertia)", "Silhouette"], horizontal=True)
        if choice.startswith("Elbow"):
            y_values = inertias
            y_title = "Inertia (lower = tighter)"
        else:
            y_values = silhouettes
            y_title = "Silhouette (higher = better)"
        fig = go.Figure(go.Scatter(x=k_values, y=y_values, mode="lines+markers", line_color=BLUE,
                                   line_width=2, marker_size=9, hovertemplate="k = %{x}: %{y:.2f}<extra></extra>"))
        fig.add_vline(x=k, line_dash="dash", line_color=ORANGE, annotation_text="chosen k",
                      annotation_font_color=TEXT_DARK)
        style_chart(fig, "Number of clusters k", y_title, 360)
        show(fig)

    st.subheader("Members and profile of each cluster")
    profile_rows = []
    for g in range(k):
        members = scores.index[labels == g]
        row = [group_names[g], len(members), round(np.median(fatality[labels == g]), 1)]
        for column in condition_columns:
            row.append(round(profiles.loc[members, column].mean(), 1))
        row.append(", ".join(members))
        profile_rows.append(row)
    header = ["cluster", "states", "median fatality ratio"]
    for column in condition_columns:
        header.append(nice_name(column))
    header.append("members")
    st.dataframe(pd.DataFrame(profile_rows, columns=header), width="stretch", hide_index=True)


# ================================================================ 6. Association rules
with tab_rules:
    st.header("Association rules: which combinations of conditions go with deadly accidents?")
    st.caption("Each state-year is one 'basket'. Each feature is split into thirds: top third = high, "
               "bottom third = low, middle third left out. 'Deadly' = fatality ratio in the top third.")

    data, baskets, cuts = make_baskets()
    c1, c2, c3 = st.columns(3)
    target_choice = c1.radio("Outcome", ["Deadly (fatality ratio high)", "Less deadly (fatality ratio low)"])
    min_support = c2.slider("Minimum support", 0.03, 0.15, 0.05, 0.01,
                            help="Share of the 237 state-years where the conditions and the outcome occur together")
    min_confidence = c2.slider("Minimum confidence", 0.5, 0.95, 0.7, 0.05,
                               help="How often the outcome follows when the conditions hold")
    min_states = c3.slider("Minimum number of different states", 1, 15, 5,
                           help="Guards against one state's 7 similar years faking a rule")
    max_conditions = c3.slider("Maximum conditions per rule", 1, 3, 3)
    drop_redundant = c3.checkbox("Remove redundant rules", value=True,
                                 help="Drop a longer rule if a shorter rule inside it is at least as reliable")

    target = DEADLY
    bar_colour = ORANGE
    if target_choice.startswith("Less"):
        target = LESS_DEADLY
        bar_colour = BLUE

    all_rules = mine_rules(min_support, max_conditions)
    found = rules_for_target(all_rules, target, min_confidence, min_states)
    if drop_redundant:
        found = remove_redundant(found)

    if len(found) == 0:
        st.warning("No rules pass these settings. Try a lower confidence or fewer states.")
    else:
        table = pd.DataFrame(found).drop(columns=["conditions"])
        table = table.sort_values(["lift", "states"], ascending=False).reset_index(drop=True)
        st.metric("Rules found", len(table))

        top = table.head(10).iloc[::-1]
        notes = []
        for i in range(len(top)):
            notes.append(str(round(top.iloc[i]["confidence"] * 100)) + "% right, " + str(top.iloc[i]["states"]) + " states")
        fig = go.Figure(go.Bar(x=top["lift"], y=top["rule"], orientation="h", marker_color=bar_colour,
                               text=notes, textposition="outside", textfont_color=TEXT_DARK,
                               customdata=top["which states"],
                               hovertemplate="lift %{x}<br>%{customdata}<extra></extra>"))
        fig.add_vline(x=1, line_dash="dash", line_color=MUTED)
        style_chart(fig, "Lift (times more likely than an average state-year)", "", 520)
        fig.update_xaxes(range=[0, table["lift"].max() + 0.9])
        show(fig, "Top 10 rules by lift")
        st.caption("Dashed line: lift 1 = no effect. Ranking by lift favours narrow rules. Raise the minimum number of states to 10 to see "
                   "only rules that hold across much of India.")
        st.dataframe(table, width="stretch", hide_index=True)

        st.subheader("Where does a rule hold?")
        chosen_rule = st.selectbox("Rule", list(table["rule"]))
        conditions = chosen_rule.split(" + ")
        has_all = baskets[conditions].all(axis=1)
        hits = baskets[target]
        years_count = pd.Series(dtype=float)
        hover = pd.Series(dtype=object)
        for state in sorted(data["state"].unique()):
            rows = data["state"] == state
            held = int((has_all & rows).sum())
            right_years = int((has_all & rows & hits).sum())
            if held > 0:
                years_count[state] = held
                hover[state] = ("conditions held in " + str(held) + " year(s), outcome followed in "
                                + str(right_years))
        if len(years_count) > 0:
            fig = tile_figure(years_count, hover, BLUE_SCALE, 0, 7)
            fig.data[1].colorbar.title.text = "years the<br>conditions held"
            show(fig)
            st.caption("Grey = the conditions never held together in that state.")

    with st.expander("Cut points: what counts as high and low"):
        st.dataframe(cuts, hide_index=True)


# ================================================================ 7. State explorer
with tab_state:
    st.header("State explorer")
    all_states = sorted(profiles.index)
    state = st.selectbox("State or UT", all_states, index=all_states.index("Uttar Pradesh"))
    profile = profiles.loc[state]

    rank = int(profiles["fatality_ratio"].rank(ascending=False)[state])
    cluster_text = "not clustered (dropped)"
    if state in scores.index:
        cluster_text = group_names[group_of[state]]

    tile1, tile2, tile3, tile4 = st.columns(4)
    tile1.metric("Accidents per year", format(int(profile["accidents_per_year"]), ","))
    tile2.metric("Deaths per year", format(int(profile["deaths_per_year"]), ","))
    tile3.metric("Deaths per 100 accidents", str(round(profile["fatality_ratio"], 1)),
                 help="Rank " + str(rank) + " of " + str(len(profiles)) + " (1 = deadliest)")
    tile4.metric("Cluster (Clusters tab settings)", cluster_text)
    st.caption("Fatality ratio rank: " + str(rank) + " of " + str(len(profiles)) + " (1 = deadliest). "
               "Averages over " + str(int(profile["years_of_data"])) + " years; 2023 counts excluded.")

    left, right = st.columns(2)
    with left:
        state_years = features[features["state"] == state]
        india_ratio = []
        for year in YEARS:
            rows = features[features["year"] == year]
            if year == BAD_YEAR:
                india_ratio.append(np.nan)
            else:
                india_ratio.append(rows["deaths"].sum() / rows["accidents"].sum() * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=state_years["year"], y=state_years["fatality_ratio"], mode="lines+markers",
                                 name=state, line_color=BLUE, line_width=2, marker_size=9,
                                 hovertemplate="%{x}: %{y:.1f}<extra>" + state + "</extra>"))
        fig.add_trace(go.Scatter(x=YEARS, y=india_ratio, mode="lines+markers", name="All India",
                                 line_color=MUTED, line_width=2, line_dash="dot", marker_size=8,
                                 marker_symbol="diamond", hovertemplate="%{x}: %{y:.1f}<extra>All India</extra>"))
        style_chart(fig, "", "Deaths per 100 accidents", 380)
        fig.update_xaxes(dtick=1)
        show(fig, "Fatality ratio by year")
        st.caption("All India is missing for 2023 because NCRB counts for that year are not reliable.")
    with right:
        # state value vs the median state, for every condition feature
        names = []
        state_values = []
        median_values = []
        for column in condition_columns:
            names.append(nice_name(column))
            state_values.append(profile[column])
            median_values.append(profiles[column].median())
        fig = go.Figure()
        for i in range(len(names)):
            fig.add_trace(go.Scatter(x=[median_values[i], state_values[i]], y=[names[i], names[i]], mode="lines",
                                     line_color=AXIS_GREY, line_width=2, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=median_values, y=names, mode="markers", name="Median state",
                                 marker_color=MUTED, marker_symbol="diamond", marker_size=10,
                                 hovertemplate="%{y}: %{x:.1f}%<extra>Median state</extra>"))
        fig.add_trace(go.Scatter(x=state_values, y=names, mode="markers", name=state,
                                 marker_color=BLUE, marker_size=11, marker_line_color="white", marker_line_width=2,
                                 hovertemplate="%{y}: %{x:.1f}%<extra>" + state + "</extra>"))
        style_chart(fig, "%", "", 460)
        fig.update_yaxes(autorange="reversed")
        show(fig, "Conditions compared with the median state")

    # time of day and month: share of accidents, state vs India (2023 left out of the totals)
    left, right = st.columns(2)
    for table, column, order, side, title in [(time_of_day, "time", TIME_ORDER, left, "Time of day"),
                                              (month, "month", MONTH_ORDER, right, "Month")]:
        good_years = table[table["year"] != BAD_YEAR]
        state_counts = good_years[good_years["state"] == state].groupby(column)["number_of_accidents"].sum()
        india_counts = good_years.groupby(column)["number_of_accidents"].sum()
        state_share = state_counts.reindex(order) / state_counts.sum() * 100
        india_share = india_counts.reindex(order) / india_counts.sum() * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=order, y=state_share, name=state, marker_color=BLUE,
                             hovertemplate="%{x}: %{y:.1f}%<extra>" + state + "</extra>"))
        fig.add_trace(go.Bar(x=order, y=india_share, name="All India", marker_color=AXIS_GREY,
                             hovertemplate="%{x}: %{y:.1f}%<extra>All India</extra>"))
        style_chart(fig, "", "% of accidents", 360)
        fig.update_layout(bargap=0.25, bargroupgap=0.08)
        with side:
            show(fig, title)
