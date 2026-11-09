# Step 1: Data processing
#
# Loads the raw NCRB tables, cleans them, and builds one table with
# one row per state per year (data/clean/state_year_features.csv).
#
# Almost every column in that table is a ratio or a share. The 2023 NCRB
# counts are about 3 times too big (DATA_PROBLEMS.md #2), but a ratio stays
# the same when top and bottom are both multiplied by 3, so 2023 is kept
# for ratios. The raw counts for 2023 are blanked out.
#
# Run from the Project folder:
#     .venv\Scripts\python src\01_clean_data.py

import numpy as np
import pandas as pd

RAW = "data/raw/ncrb/"
CLEAN = "data/clean/"
OUTPUTS = "outputs/"
BAD_YEAR = 2023


def clean_state_name(name):
    # NCRB and MoRTH spell some states differently, for example
    # "Andaman & Nicobar Islands" and "Andaman And Nicobar Islands".
    # This turns every spelling into one standard form.
    name = str(name)
    name = name.replace("#", "")
    name = name.replace("*", "")
    name = name.replace("&", "and")
    name = name.strip()
    name = name.title()
    if name.startswith("The "):
        name = name[4:]
    if name == "J And K":
        name = "Jammu And Kashmir"
    # Ladakh only became a separate UT in October 2019. Before that its
    # accidents are counted inside Jammu and Kashmir. To compare the same
    # region in every year, we treat the two as one region throughout.
    # This happens before any adding up, so counts are summed first and
    # ratios are worked out after.
    if name == "Jammu And Kashmir" or name == "Ladakh":
        name = "Jammu And Kashmir And Ladakh"
    return name


def clean_vehicle_name(name):
    # Vehicle categories changed in 2021: Car, Jeep and SUV were merged
    # into one group. We merge them for the older years too so that
    # all years use the same groups.
    if name == "Car" or name == "Jeep" or name.startswith("Suv"):
        return "Car/Jeep/SUV"
    if name.startswith("Truck"):
        return "Truck/Lorry"
    return name


def to_number(text):
    # MoRTH writes numbers like "3,72,181". Remove the commas.
    text = str(text).replace(",", "").strip()
    if text == "" or text == "nan":
        return np.nan
    return float(text)


def safe_divide(top, bottom):
    # Returns blank (NaN) instead of infinity when the bottom is 0.
    # Example: Lakshadweep had 0 accidents in 2024, so its ratio is undefined.
    bottom = bottom.replace(0, np.nan)
    return top / bottom


def load_ncrb(file_name):
    table = pd.read_csv(RAW + file_name)
    table["state"] = table["state_name"].apply(clean_state_name)
    table = table.drop(columns=["id", "state_name", "state_code"])
    return table


# 1. Road type table: accidents, injured and deaths on each road type.
# This is our main table for counts.
road = load_ncrb("road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv")
road.to_csv(CLEAN + "road_type.csv", index=False)

totals = road.groupby(["state", "year"])[["cases", "injured", "died"]].sum()

features = pd.DataFrame(index=totals.index)
features["accidents"] = totals["cases"]
features["injured"] = totals["injured"]
features["deaths"] = totals["died"]
features["fatality_ratio"] = safe_divide(totals["died"], totals["cases"]) * 100
features["injury_ratio"] = safe_divide(totals["injured"], totals["cases"]) * 100

cases_wide = road.pivot_table(index=["state", "year"], columns="road_type", values="cases", aggfunc="sum")
died_wide = road.pivot_table(index=["state", "year"], columns="road_type", values="died", aggfunc="sum")

# share of accidents on each road type (in %)
features["share_expressway"] = safe_divide(cases_wide["Expressways"], totals["cases"]) * 100
features["share_national_hw"] = safe_divide(cases_wide["National Highways"], totals["cases"]) * 100
features["share_state_hw"] = safe_divide(cases_wide["State Highways"], totals["cases"]) * 100
features["share_other_road"] = safe_divide(cases_wide["Other Roads"], totals["cases"]) * 100

# deaths per 100 accidents on each road type.
# Not done for expressways: most states have none, so it would be mostly blank.
features["fr_national_hw"] = safe_divide(died_wide["National Highways"], cases_wide["National Highways"]) * 100
features["fr_state_hw"] = safe_divide(died_wide["State Highways"], cases_wide["State Highways"]) * 100
features["fr_other_road"] = safe_divide(died_wide["Other Roads"], cases_wide["Other Roads"]) * 100


# 2. Time of day table. Keep only road accidents (railway rows are mixed in).
time = load_ncrb("time-of-occurrence-wise-number-of-traffic-accidents.csv")
time = time[time["accident_type"] == "Road Accidents"]
time = time.drop(columns=["accident_type"])
time.to_csv(CLEAN + "time_of_day.csv", index=False)

time_wide = time.pivot_table(index=["state", "year"], columns="time", values="number_of_accidents", aggfunc="sum")
time_total = time_wide.sum(axis=1)

# In this data "Night" means 6 PM to 6 AM.
night = time[time["day_night"] == "Night"].groupby(["state", "year"])["number_of_accidents"].sum()
late_night = time_wide["12 AM to 3 AM"] + time_wide["3 AM to 6 AM"]

features["night_share"] = safe_divide(night, time_total) * 100
features["evening_share"] = safe_divide(time_wide["6 PM to 9 PM"], time_total) * 100
features["late_night_share"] = safe_divide(late_night, time_total) * 100


# 3. Month table: how much of the year's accidents happen in the monsoon
# (June to September) and in winter (December to February, fog season in the north).
month = load_ncrb("month-of-occurrence-wise-number-of-traffic-accidents.csv")
month = month[month["accident_type"] == "Road Accidents"]
month = month.drop(columns=["accident_type"])
month.to_csv(CLEAN + "month.csv", index=False)

month_total = month.groupby(["state", "year"])["number_of_accidents"].sum()
is_monsoon = month["month"].isin(["June", "July", "August", "September"])
is_winter = month["month"].isin(["December", "January", "February"])
monsoon = month[is_monsoon].groupby(["state", "year"])["number_of_accidents"].sum()
winter = month[is_winter].groupby(["state", "year"])["number_of_accidents"].sum()

features["monsoon_share"] = safe_divide(monsoon, month_total) * 100
features["winter_share"] = safe_divide(winter, month_total) * 100


# 4. Place table: deaths in rural vs urban areas.
# The file has both detailed places and "Sub Total" rows. Using both would
# count every death twice, so we keep only the two sub totals.
place = load_ncrb("place-of-occurrence-wise-road-accident-deaths.csv")
place = place[place["place"].str.contains("Sub Total")]
place.to_csv(CLEAN + "place.csv", index=False)

# pivot_table adds up male, female and transgender for us
place_wide = place.pivot_table(index=["state", "year"], columns="place", values="number_of_deaths", aggfunc="sum")
rural = place_wide["Rural Area (Sub Total)"]
urban = place_wide["Urban Area (Sub Total)"]
features["rural_death_share"] = safe_divide(rural, rural + urban) * 100


# 5. Vehicle (mode of transport) tables: who died, by the vehicle they were in.
# 2018-2020 splits deaths into "victims" and "offenders". We checked that the
# two add up to the total deaths in the road type table, so died = victims + offenders.
old_mode = load_ncrb("mode-of-transport-wise-number-of-persons-died-in-road-accidents-2018-2020.csv")
old_mode["died"] = old_mode["victims"] + old_mode["offenders"]
old_mode = old_mode[["state", "year", "mode_of_transport", "died"]]

new_mode = load_ncrb("mode-of-transport-wise-number-of-persons-died-in-road-accidents-2021-onwards.csv")
new_mode = new_mode[["state", "year", "mode_of_transport", "died"]]

mode = pd.concat([old_mode, new_mode])
mode["vehicle"] = mode["mode_of_transport"].apply(clean_vehicle_name)
mode = mode.groupby(["state", "year", "vehicle"])["died"].sum().reset_index()
mode.to_csv(CLEAN + "vehicle_deaths.csv", index=False)

mode_wide = mode.pivot_table(index=["state", "year"], columns="vehicle", values="died", aggfunc="sum")
mode_total = mode_wide.sum(axis=1)

features["two_wheeler_death_share"] = safe_divide(mode_wide["Two Wheeler"], mode_total) * 100
features["pedestrian_death_share"] = safe_divide(mode_wide["Pedestrian"], mode_total) * 100
features["car_death_share"] = safe_divide(mode_wide["Car/Jeep/SUV"], mode_total) * 100
features["truck_death_share"] = safe_divide(mode_wide["Truck/Lorry"], mode_total) * 100
features["bus_death_share"] = safe_divide(mode_wide["Bus"], mode_total) * 100


# 6. Check the 2023 problem state by state, and save it as evidence.
# If ratios are safe to keep, the 2023 fatality ratio should be close to 2022 and 2024.
check = pd.DataFrame()
check["accidents_2022"] = features["accidents"].xs(2022, level="year")
check["accidents_2023"] = features["accidents"].xs(2023, level="year")
check["accidents_2024"] = features["accidents"].xs(2024, level="year")
check["times_bigger_2023"] = check["accidents_2023"] / check["accidents_2022"]
check["fatality_ratio_2022"] = features["fatality_ratio"].xs(2022, level="year")
check["fatality_ratio_2023"] = features["fatality_ratio"].xs(2023, level="year")
check["fatality_ratio_2024"] = features["fatality_ratio"].xs(2024, level="year")
check = check.round(2).sort_values("times_bigger_2023")
check.to_csv(OUTPUTS + "check_2023_inflation.csv")

# Compare with MoRTH's national totals, which come from a different ministry.
morth = pd.read_csv("data/raw/morth/road-accidents-annual-2020-2024.csv")
print("National accidents, NCRB vs MoRTH:")
for i in range(len(morth)):
    year = int(morth.loc[i, "Year"])
    morth_count = to_number(morth.loc[i, "Accidents"])
    ncrb_count = features["accidents"].xs(year, level="year").sum()
    print("  ", year, " NCRB:", int(ncrb_count), " MoRTH:", int(morth_count))


# 7. Blank out the 2023 counts. Ratios and shares for 2023 are kept.
features = features.reset_index()
bad_rows = features["year"] == BAD_YEAR
features.loc[bad_rows, "accidents"] = np.nan
features.loc[bad_rows, "injured"] = np.nan
features.loc[bad_rows, "deaths"] = np.nan

features = features.round(2)
features.to_csv(CLEAN + "state_year_features.csv", index=False)


# 8. Missing values report.
# (a) state-years with no row at all
all_states = sorted(features["state"].unique())
all_years = sorted(features["year"].unique())
print("\nState-years with no data at all:")
for state in all_states:
    for year in all_years:
        found = features[(features["state"] == state) & (features["year"] == year)]
        if len(found) == 0:
            print("  ", state, year)

# (b) blank cells inside the table
print("\nBlank cells per column:")
blank_counts = features.isna().sum()
for column in features.columns:
    if blank_counts[column] > 0:
        print("  ", column, blank_counts[column])

print("\nSaved", len(features), "rows and", len(features.columns), "columns to", CLEAN + "state_year_features.csv")
