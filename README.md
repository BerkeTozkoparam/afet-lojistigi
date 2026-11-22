# Disaster Response: Resource Allocation and Logistics Optimization

A scenario-based decision support model for allocating emergency tents and
scheduling deliveries after an earthquake. It is built on field observations
from the 6 February 2023 Kahramanmaraş earthquakes and uses mixed-integer
linear programming (MILP) for allocation and a discrete-event simulation for
72-hour vehicle scheduling.

> **Important:** demand, depot locations, stock levels and travel parameters
> are modeling assumptions. The results do not measure the performance of the
> actual 2023 response.

## Workflow

The script is organized into numbered sections (also marked with `# %%` cells,
so it can be run cell by cell in Jupyter, Colab or VS Code):

| # | Section | What it does |
|---|---------|--------------|
| 0 | Configuration | Scenario parameters, depot table, capacities, equity threshold |
| 1 | Data loading and cleaning | Loads the observation data, simplifies column names, maps damage and shelter categories, creates flags |
| 2 | Exploratory analysis | Missing-data rates, damage distributions, district-level indicators |
| 3 | Synthetic demand and network | Random district demand (100–500 people), depot–district distances (haversine) |
| 4 | Static allocation | Minimum-unmet tent allocation for several stock levels (MILP) |
| 5 | Depot–district dispatch | Two-stage MILP: efficiency (min. unmet demand, then min. tent-km) vs. equity (≥ 70% coverage per district) |
| 6 | 72-hour scheduling | Trip splitting and dispatch under LPT, Equity First and MaxMin rules |
| 7 | Service metrics | Coverage by hour (24 / 48 / 72), time to reach 70% (T70) |
| 8 | Local search and depot swap | Swap / insertion local search on trip order; inter-depot allocation swaps |
| 9 | Validation | Stock limits, vehicle overlap, allocation conservation, fleet limit, equity check |
| 10 | Speed sensitivity | Re-runs the fixed schedule at 15–40 km/h, and re-optimizes at 20 km/h |

## Model assumptions

- **Depots:** three hypothetical depots (Adiyaman, Hatay, Maras), 500 tents each.
- **Demand:** synthetic, drawn uniformly from 100–500 people per district (seeded, `seed=42`).
- **Capacity:** 4 people per tent, 50 tents per truck trip, 3 trucks per depot.
- **Equity target:** at least 70% of each district's demand covered.
- **Travel time:** `distance × road_factor / speed`, with the damaged-road scenario at 25 km/h and a 1.8 road factor. The baseline scenario uses 40 km/h and 1.4.
- **Handling time:** 0.5 h loading and 0.5 h unloading per trip in the 72-hour simulation.

## Requirements

- Python 3
- `pandas`, `numpy`, `scipy`
- `datasets` (only if downloading the data from Hugging Face)

```bash
pip install pandas numpy scipy datasets
```

## Usage

Download the dataset from Hugging Face (it is also saved locally as
`deprem_gozlem_train.csv`):

```bash
python afet_lojistigi.py
```

Or use a local copy of the training CSV:

```bash
python afet_lojistigi.py --csv deprem_gozlem_train.csv
```

The script prints the results of each section and returns a dictionary of
DataFrames (`main()` in the script) for further analysis.

## Results (synthetic damaged-road scenario)

These figures come from an accompanying analysis report (in Turkish, not
included in this repository). They are based on synthetic demand (seed 42),
hypothetical depots, and the travel assumptions above. They do not measure the
actual 2023 response.

- **Static allocation:** across the 23 districts (7,026 people), 1,500 tents
  (4 people each) cover 85.40% of demand. Full coverage needs 1,764 tents.
- **Equity constraint cost:** requiring at least 70% coverage in every district
  raises total tent-km by 1,438.32 (+1.40%). Without the constraint, the lowest
  district receives 0%.
- **72-hour final plan:** all 21 districts reach at least 70% coverage.
  5,916 of 6,510 people (90.88%) are served. The lowest district is Besni at
  70.24%. The plan combines an Equity First schedule, local search, and one
  inter-depot swap (18 tents for Besni moved from Hatay to Adıyaman, and 18 for
  Tut moved the other way).
- **Scheduling rules:** within 72 hours, LPT serves 4,728 people, leaving five
  districts with no delivery. Equity First serves 5,772 people. Equity First
  with local search serves 5,972, but one district stays below 70%.
- **Speed sensitivity:** with the fixed task order, 25 km/h is the lowest tested
  speed at which every district reaches 70%. At 15 km/h and 20 km/h, four and
  three districts fall below the threshold.

## Data

The observations come from the public dataset
[`electricsheepasia/asia-turkiye-earthquake-response-humanitiarian-observation-form`](https://huggingface.co/datasets/electricsheepasia/asia-turkiye-earthquake-response-humanitiarian-observation-form)
on Hugging Face. The raw dataset is not included in this repository.

## Notes

- Column names in the raw data are long survey field names. They are mapped
  to short English names in `RENAME_MAP`, and categories are mapped in
  `HOUSING_MAP`, `INFRASTRUCTURE_MAP` and `PRIORITY_MAP`. These keys must match
  the raw dataset exactly.
- The equity MILP and the local search use `scipy.optimize.milp`, which
  requires SciPy 1.9 or newer.
