# Road Accidents in India: When and Where Do They Turn Deadly?

DS3030 Data Analytics course project, IIT Palakkad.

**Live dashboard:** https://road-accidents-india.streamlit.app

**Question.** India records over 4.5 lakh road accidents and about 1.7 lakh deaths a year.
Not every accident is equally deadly. We ask: which states, road types, times of day and
vehicle types go with a high **fatality ratio** (deaths per 100 accidents), and which
states do worse than their accident count alone would suggest?

---

## 1. Data

We use only official Indian government data. All files are in `data/raw/`.

| Source | What it has | Years | Granularity |
|--------|-------------|-------|-------------|
| NCRB, *Accidental Deaths and Suicides in India* (via India Data Portal) | Cases, injured and died; time of day; month; road type; vehicle mode; place (urban or rural) | 2018-2024 | State x year x category |
| MoRTH, *Road Accidents in India 2024* (via OpenCity) | State accidents and deaths; collision type; violation type; licence; safety devices; 50 large cities | 2020-2024 | State x year, and national tables |

**Why not the Kaggle "Road Accident Severity in India" dataset?** On inspection it turned out to
be Ethiopian data mislabelled as Indian (see problem 1 in `DATA_PROBLEMS.md`). Checking where
the data really comes from was our first data-quality step. (The rejected file is not
included in this repository, as it is not ours to redistribute.)

**Main limitation.** Official data is aggregated (counts per state and year), not one row per
accident. So our "records" are state-years (36 states x up to 7 years), and we explain how
each method was adapted to this.

## 2. Project structure

```
Project/
  data/raw/ncrb/      original NCRB CSVs (never edited)
  data/raw/morth/     original MoRTH CSVs (never edited)
  data/clean/         cleaned tables made by our scripts
  src/                one Python script per analysis step
  outputs/            charts and result tables
  data/india_states.geojson  simplified state borders for the dashboard map (from DataMeet, CC BY 4.0)
  app.py              Streamlit dashboard (step 6)
  .streamlit/         dashboard colour theme
  DATA_PROBLEMS.md    running log of data problems and fixes
```

## 3. Analysis steps

Each step below gets filled in as we finish it: what we did, why, and which challenge we handled.

1. **Data processing** (`src/01_clean_data.py`, done).
   - Kept only road accident rows, gave every state one standard name, and merged the
     2018-20 and 2021+ vehicle tables after making their categories match.
   - Built `data/clean/state_year_features.csv`: 244 state-years (35 regions x 7 years, minus
     one gap) and 25 columns. Every analysis
     column is a **ratio or a share** (fatality ratio, share of accidents at night, share of deaths
     among two-wheeler riders, and so on). Using ratios means big states do not dominate just
     because they are big.
   - **The 2023 problem.** NCRB's 2023 counts are about 3 times too large in every state
     (14.1 lakh accidents, while MoRTH reports 4.8 lakh). Because the top and bottom of a ratio are
     inflated by the same factor, ratios survive. We checked this state by state
     (`outputs/check_2023_inflation.csv`), then kept 2023 ratios and blanked the 2023 counts.
   - **Validation.** NCRB and MoRTH national totals agree within 1 to 4% in 2020-22 and 2024.
   - **Missing values.** Three kinds, each handled differently:
     (a) *structurally missing*: Ladakh 2018-19, before it was a separate UT. We merged Ladakh
     into J&K for all years, so the region is the same every year;
     (b) *close to MCAR*: Andaman and Nicobar 2019, a reporting gap. Left as a gap, not imputed,
     because state profiles are averages over the years each state has;
     (c) *not applicable*: ratios with a zero bottom, such as Lakshadweep, which has no national highway accidents.
   - **MNAR in the measure itself.** Non-fatal accidents are under-reported in some states
     *because* they are non-fatal, which pushes the fatality ratio up (Mizoram above 100,
     Bihar about 78, Kerala about 10).
2. **Statistics and outliers** (`src/02_statistics_outliers.py`, done).
   - **State profiles.** Each state's ratios are averaged over the years it has
     (`data/clean/state_profiles.csv`). Summary statistics with skewness are in
     `outputs/summary_statistics.csv`.
   - **Spread is huge.** The fatality ratio ranges from 9 (Goa) and 10 (Kerala) to 111 (Mizoram) deaths
     per 100 accidents; the median state is 45. Night (6 PM to 6 AM) accounts for 42% of accidents
     in the median state. Two-wheeler riders are the largest group of deaths (median 39%).
   - **Outlier detection.** We used the 1.5 x IQR rule, not z-scores, because several features are
     clearly skewed (skew up to 3.8), and a z-score assumes a roughly bell-shaped distribution.
     Box plots: `outputs/boxplots_state_profiles.png`; list: `outputs/outlier_states.csv`.
   - **Drop or winsorize? We did both, each where it fits.**
     - *Dropped Lakshadweep.* It averages 2 accidents a year, so a single accident moves a share by
       50 points. It was an outlier in 7 of 20 features because of noise, not because it is different.
     - *Winsorized the rest (29 of 680 values, 4%).* The other outliers are real differences
       (Delhi and Chandigarh are cities with almost no rural deaths, J&K + Ladakh has very few
       winter accidents, UP has most of India's expressways), so dropping them would throw away real information. But PCA
       and k-means work with distances, so one extreme value can pull a whole cluster. Values
       beyond the IQR limits are pulled back to the limit (`outputs/winsorized_values.csv`).
       This capped copy is used **only** for PCA and clustering
       (`data/clean/state_profiles_for_modelling.csv`); the dashboard shows true values.
   - **Testing the under-reporting (MNAR) idea.** If a state records only serious accidents,
     each recorded accident should have many deaths and few injuries. Across states, the fatality
     ratio and the injury ratio are indeed negatively related (Spearman rho = -0.37, p = 0.03;
     `outputs/fatality_vs_injury.png`). Punjab, Bihar, UP and Jharkhand have high death ratios and
     low injury ratios. This is moderate support, not proof: Goa is low on both, which suggests it
     records many minor crashes with no injury at all.
3. **Dimensionality reduction** (`src/03_pca.py`, done).
   - **Inputs: 13 "condition" features** (road mix, time of day, season, rural share, vehicle mix)
     for 34 regions, from the winsorized table.
   - **The fatality ratio is the outcome, not an input.** We found that each state's fatality ratio
     on national highways, state highways and other roads is 0.92 to 0.99 correlated with its
     overall fatality ratio. So the state matters far more than the road type, which again
     points to state-level reporting habits. Keeping the ratio out of PCA stops this reporting
     effect from shaping the map, and lets us ask cleanly which *conditions* go with deadly accidents.
   - **Dropped two weak features**: the expressway share (almost always 0) and the other-roads
     share (always 100 minus the other road shares).
   - **Standardised first**, so a feature with a big spread (rural share, std 17) does not outweigh
     one with a small spread (winter share, std 2) just because of its units.
   - **How many components?** The Kaiser rule (eigenvalue > 1) keeps 4 components (73.8% of the
     variance); the 80% rule keeps 5. We plot PC1 and PC2 (51.5%). Charts: `outputs/pca_scree.png`
     and `outputs/pca_biplot.png`.
   - **What the components mean** (from `outputs/pca_loadings.csv`):
     - **PC1 (28.8%), heavy vehicles vs two-wheelers.** Positive: national highway share, car and
       truck deaths. Negative: two-wheeler and pedestrian deaths, monsoon share.
     - **PC2 (22.7%), night-time vs evening and rural.** Positive: night and late-night share.
       Negative: rural death share, evening and winter share.
   - **Finding.** PC1 is clearly linked to the fatality ratio (Spearman rho = 0.49, p = 0.003);
     PC2, PC3 and PC4 are not (p > 0.1). States whose deaths involve cars and trucks on national
     highways (Mizoram, Nagaland, Sikkim, Arunachal) have the deadliest accidents. States dominated
     by two-wheeler and pedestrian deaths (Kerala, Goa, Tamil Nadu) have the least deadly ones,
     though these are also the states that record the most minor accidents.
4. **Clustering** (`src/04_clustering.py`, done).
   - **Input:** each state's scores on the 4 PCA components kept by the Kaiser rule (74% of the
     variance; the noisy small components are left out). The fatality ratio was not an input.
   - **Choosing k** (`outputs/choosing_k.png`, `outputs/choosing_k.csv`).
     - k = 2 has the best silhouette (0.34), but it only splits off the North-East, which is too
       coarse to be useful.
     - k = 4 has the next best silhouette (0.29) and sits at the bend of the elbow curve.
     - **Cross-check with a second method:** Ward hierarchical clustering finds largely the same
       4 groups (adjusted Rand index 0.71).
     - With only 34 points, k-means can land on odd groups from one unlucky random start, so it
       is run 50 times (`n_init=50`) and the best result kept.
   - **Honest limitation.** Silhouette scores of 0.2 to 0.34 mean the groups overlap; Indian
     states form a continuum rather than neat islands. We report the clusters as broad profiles.
   - **Result: the clusters rediscover geography even though no location data was used**
     (`outputs/clusters_pca_map.png`, `outputs/cluster_profiles.csv`):

     | Cluster | States | Median deaths per 100 accidents | What sets it apart |
     |---------|--------|---------------------------------|--------------------|
     | North-East hills | Arunachal, Meghalaya, Mizoram, Nagaland, Sikkim | **72** | Deaths in cars and trucks, more national highway accidents, few two-wheeler or pedestrian deaths |
     | North and Gangetic plain | UP, Bihar, Punjab, Haryana, WB, Delhi, Jharkhand, Odisha, Assam, Uttarakhand, Chandigarh, Puducherry | **49** | Night-time and especially late-night (12 to 6 AM) accidents |
     | South and West | Kerala, TN, Karnataka, AP, Telangana, Maharashtra, Gujarat, MP, Rajasthan, Goa and others | **39** | Two-wheeler and pedestrian deaths, evening accidents |
     | Himalayan | Himachal Pradesh, J&K + Ladakh | **27** | Far *fewer* accidents in winter and more in the monsoon and summer months (likely less travel in snow, and the summer tourist season); more bus and car deaths |

   - **The difference in deadliness is statistically real:** Kruskal-Wallis H = 12.8, p = 0.005
     (`outputs/clusters_fatality_ratio.png`). We used Kruskal-Wallis instead of ANOVA because the
     groups are small (2 to 15 states) and the ratio is skewed.
   - **Exceptions worth noting:** Manipur, Tripura and Assam are in the North-East but do *not*
     join the North-East hills cluster; their conditions look more like the plains.
5. **Association rules** (`src/05_association_rules.py`, done).
   - **Adapting Apriori to aggregate data.** We have no single-accident records, so each
     **state-year is one basket** (237 baskets). Each feature is split into thirds across all
     state-years: the top third becomes "high", the bottom third "low", and the middle third gets
     no item, so that "high" really means high. "Deadly" means a fatality ratio in the top third,
     above 50 deaths per 100 accidents (`outputs/rules_cut_points.csv`). Splitting into thirds also
     absorbs outliers, so no winsorizing was needed here.
   - **Settings:** support of at least 5% (12 baskets), confidence of at least 70%, and at most
     3 conditions per rule.
   - **Safeguard against repeated years.** Each state appears up to 7 times and its years look
     alike, so one state alone could make a rule look strong. We keep only rules whose conditions
     occur in **at least 5 different states**, and show the state count next to each rule.
   - **Removing redundant rules.** A longer rule is kept only if it is more reliable than every
     shorter rule inside it (104 deadly rules and 21 less-deadly rules remain;
     `outputs/rules_deadly.csv`, `outputs/rules_less_deadly.csv`).
   - **Ranking.** By lift alone, the top rule (lift 2.85) comes from only the 5 North-East hill
     states and repeats the clustering result. The chart (`outputs/rules_deadly_top10.png`)
     therefore shows the strongest rules that hold in **10 or more states**.
   - **Key rules for deadly accidents** (lift = how many times more likely than average):
     - *Late-night share high + national highway share high*: 79% confidence, lift 2.4, 10 states.
       The "night + highway" pattern suspected at the start, now confirmed across states.
     - *Late-night share high + state highway share high*: 78%, lift 2.3, 11 states.
     - *Night share high + truck deaths high + pedestrian deaths low*: 83%, lift 2.5, 10 states.
     - *Pedestrian deaths low* appears in most top rules. One possible reason is that such states have
       more of their accidents on highways between towns than on city streets; another is that cities
       record more minor accidents. State-level data cannot tell these apart.
   - **Key rules for less deadly accidents:** *bus death share high + late-night share low*
     (83%, lift 2.5, 10 states), and *pedestrian deaths high + state highway share low*: urban,
     daytime, public-transport-heavy patterns.
6. **Dashboard** (`app.py`, done). One Streamlit file with seven tabs:

   | Tab | What the viewer can do | Lab functionality |
   |-----|------------------------|-------------------|
   | Overview | Pick a year and a measure; see the tile map, a top-15 ranking, national totals and trends | Data processing |
   | Data quality | See which state-years are missing or blanked, the MNAR test, and outliers before and after winsorizing | Missing data, outliers |
   | Statistics | Summary table with skew, a scatter of any two features with Spearman rho, a correlation heatmap | Statistics |
   | PCA | Choose the input features; scree plot, biplot on any two components, loadings | Dimensionality reduction |
   | Clusters | Choose k (2-6) and the number of components; clusters on the PCA map and the tile map, silhouette, Kruskal-Wallis, elbow | Clustering |
   | Association rules | Choose the outcome, support, confidence, minimum states and rule length; top rules, full table, and a map of where a rule holds | Association rules |
   | State explorer | Pick a state; its trend, its conditions against the median state, its time-of-day and month pattern | All of the above, per state |

   - **Live, not pre-computed.** PCA, k-means and Apriori rerun when a setting changes (the data is
     small, so this is instant; results are cached). With the default settings the app gives
     exactly the results of `src/03` to `src/05` (4 components by Kaiser, Kruskal-Wallis p = 0.005,
     104 deadly rules), which we checked with an automated test.
   - **A real map of India.** State borders come from DataMeet's open boundaries (CC BY 4.0,
     github.com/datameet/maps). We checked the outline against the official map of India (J&K and
     Ladakh including PoK and Aksai Chin, all of Arunachal Pradesh) before using it.
     `src/06_india_map.py` joins J&K and Ladakh into one shape (as in our data) and simplifies the
     17 MB file to 126 KB. Small UTs are hard to see on a real map, so a ranking chart sits beside it.
   - **The algorithm is named on every tab**, in a blue box at the top (for example "k-means on the PCA
     scores, best of 50 random starts"), with a summary table on the Overview tab.
   - **Cluster outlines.** On the PCA plot each cluster is wrapped in its convex hull (the smallest
     shape around its points); on the map each cluster is one colour, so cluster borders are visible.
   - **Honest defaults.** 2023 counts are shown as "not reliable", missing values are grey tiles,
     and cluster place names are shown only for the default k = 4 (other values of k give
     numbered groups, ordered from deadliest to least deadly).
   - Colours: one colour-blind-safe palette throughout; clusters also differ by marker shape, so
     colour is never the only cue.

## 4. Challenges handled

See `DATA_PROBLEMS.md` for the full list of 22. The main ones:

- A mislabelled Kaggle dataset (rejected after checking).
- 2023 NCRB values about 3 times too large in every state (systematic portal error).
- States that did not exist in some years (Ladakh before 2019).
- Vehicle categories that change definition in 2021.
- Under-reporting of minor accidents (MNAR), which inflates the fatality ratio in some states.
- Real but extreme outliers (winsorized for modelling, kept as they are for display).
- Too few records for the methods (34 states, 237 state-years), handled with repeated k-means
  starts, a second clustering method, and a minimum-states rule for association rules.

## 5. How to run

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/01_clean_data.py          # only needed to rebuild data/clean
python src/02_statistics_outliers.py
python src/03_pca.py
python src/04_clustering.py
python src/05_association_rules.py
python src/06_india_map.py           # only to rebuild the map; needs pip install pyshp shapely
                                     # and DataMeet's States/Admin2 files in data/raw/datameet/

streamlit run app.py                 # opens the dashboard in the browser
```

The dashboard needs only `data/clean/`, `data/india_states.geojson` and `outputs/`, which are already
in the folder, so `streamlit run app.py` works on its own.

## 6. Key insights

**In one line:** a recorded road accident in Bihar is about 8 times as deadly as one in Kerala.
Part of that gap is real (highways, night-time, heavy vehicles) and part is how accidents get
recorded, and our four methods let us separate the two.

1. **The fatality ratio measures road danger *and* record-keeping.**
   It ranges from 9 (Goa) and 10 (Kerala) to 111 (Mizoram) deaths per 100 recorded accidents. States
   that record few injuries record many deaths per accident (Spearman rho = -0.37, p = 0.03), which
   is what we expect if minor accidents go unrecorded (MNAR). And a state's fatality ratio on
   national highways, state highways and other roads is 0.92 to 0.99 correlated with its overall
   ratio: *which state* matters more than *which road*. So a ranking of states by fatality ratio
   alone is misleading; it should be read together with how completely each state records accidents.

2. **States and years with more highway, night-time and heavy-vehicle accidents have deadlier
   accidents on average, and four methods agree.**
   - *Descriptive:* across India, national highways carry 30% of accidents but 34% of deaths
     (42 deaths per 100 accidents against 32 on other roads; 2018-2024 without 2023).
   - *PCA:* the only component linked to deadliness is PC1, which runs from two-wheeler and
     pedestrian deaths to car, truck and national-highway accidents (rho = 0.49, p = 0.003).
   - *Clustering:* the deadliest cluster (North-East hills, median 72) is the one with the most
     national-highway accidents and car and truck deaths.
   - *Association rules:* state-years with a high late-night share **and** a high national-highway
     share were deadly 79% of the time, **2.4 times** the average rate, across 10 states. A high
     night share with many truck deaths and few pedestrian deaths: 83%, lift 2.5, 10 states.

3. **The busiest hours are not the deadliest pattern.** The most accidents happen from 6 to 9 PM
   (20% of all accidents), while only 11% happen between midnight and 6 AM. Yet it is the states
   with a larger *late-night* share (the North and Gangetic plain cluster, median 49) whose
   accidents are deadlier. (NCRB gives no deaths by time of day, so this is a state-level link,
   not a per-accident death rate; see the limitations below.)

4. **The data rediscovers India's geography.** Without any location input, the states fall into
   four groups that match real regions: North-East hills (72 deaths per 100 accidents), North and
   Gangetic plain (49), South and West (39) and Himalayan (27). The difference is statistically
   real (Kruskal-Wallis p = 0.005). Manipur, Tripura and Assam are North-Eastern but have
   plains-like conditions, and fall outside the hill cluster.

5. **Where most people die is not where accidents are deadliest.** Two-wheeler riders are the
   largest group of road deaths (43%), followed by car occupants (17%), pedestrians (12%) and truck
   occupants (11%). But the states where two-wheeler and pedestrian deaths dominate (Kerala, Goa,
   Tamil Nadu) have the *lowest* fatality ratios, while the states with the highest ratios have more
   highway, late-night and truck accidents. So the **number** of deaths and the **fatality ratio**
   point to different states, which may need different responses. These are state-level patterns:
   they do not show that any single two-wheeler crash is less deadly than a truck crash.

6. **Fewer but deadlier crashes.** In the 2020 lockdown year, recorded accidents fell 21% (4.52 to
   3.59 lakh) but deaths fell only 13%. Deaths per 100 accidents rose from 34 in 2018 to 38 in
   2021 and have stayed near 38 since.

7. **Checking the data changed the results.** A dataset sold as "Indian" turned out to be
   Ethiopian, and the official 2023 figures were about 3 times too large in every state.
   Without these checks, our 2023 trend and any model trained on the Kaggle file would have been wrong.

**What this suggests.** These are hypotheses to test, not proven causes. If the state-level
pattern also holds for individual accidents (which needs accident-level data to check), night-time
patrols and lighting on national and state highways and limits on truck drivers' hours would target
the states with the deadliest accidents, while helmet and pedestrian measures would target the states
with the most deaths. States with very high fatality ratios and very low injury ratios
(Bihar, Punjab, Jharkhand, Mizoram, Nagaland) should also be checked for under-recording of
minor accidents before their roads are judged the most dangerous.

**Limitations.**
- *State-level data only.* All links are between states and years, not between individual
  accidents (the ecological fallacy): "late-night states are deadlier" does not prove that each
  late-night accident is deadlier.
- *MNAR cannot be fixed by imputation.* We can show under-reporting exists, but not correct it.
- *Small samples.* 34 regions and 237 state-years; the clusters overlap (silhouette 0.29).
- *2023* is used for ratios only, and NCRB has no deaths by time of day.

**Future work.** The real fix for the state-level limit is accident-level data, where each crash
has its own time, road type, vehicle and outcome. MoRTH's iRAD (Integrated Road Accident Database)
collects exactly this, but it is not openly downloadable; with access, the same association-rule
method could test combinations like "night + highway + two-wheeler" for single accidents.
