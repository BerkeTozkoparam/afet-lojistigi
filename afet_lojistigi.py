# -*- coding: utf-8 -*-
# %% [markdown]
"""
Afet Yönetiminde Kaynak Tahsisi ve Lojistik Optimizasyonu
=========================================================

6 Şubat 2023 Kahramanmaraş depremleri saha gözlemlerinden senaryo tabanlı
karar destek modeli.

İş akışı
--------
1. Veri yükleme ve sütun adlarının sadeleştirilmesi
2. Keşifsel analiz (EDA) ve ilçe bazlı hasar / barınma göstergeleri
3. Sentetik talep, varsayımsal depolar ve depo–ilçe mesafeleri
4. Statik çadır tahsisi ve stok duyarlılığı (MILP)
5. İki aşamalı depo–ilçe sevkiyat modeli: verimlilik ve %70 adalet (MILP)
6. 72 saatlik araç çizelgeleme simülasyonu: LPT, Equity First, MaxMin
7. Yerel arama (swap + insertion) ve depolar arası tahsis takası
8. Nihai planın doğrulanması
9. Hız duyarlılığı

Önemli: Talep, depo, stok ve seyahat parametreleri modelleme varsayımıdır.
Sonuçlar 2023'teki gerçek müdahalenin performansını ölçmez.

Kullanım
--------
    pip install pandas numpy scipy datasets
    python afet_lojistigi.py                     # Hugging Face'ten indirir
    python afet_lojistigi.py --csv deprem_gozlem_train.csv

Colab / VS Code'da `# %%` hücreleri ayrı ayrı çalıştırılabilir.
"""

# %% Kütüphaneler
from __future__ import annotations

import argparse
import heapq
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


# %% 0. Yapılandırma
@dataclass(frozen=True)
class Scenario:
    """Ulaşım senaryosu.

    Süre modeli (tüm simülasyonlarda tek ve aynı):
        τ        = mesafe_km × road_factor / speed_kmh           (tek yön)
        teslim   = başlangıç + load_hours + τ + unload_hours
        dönüş    = teslim + τ
    Rapordaki 72 saatlik sonuçlar load = unload = 0,5 sa (toplam 1 sa sabit
    operasyon) ile, Bölüm 4.3'teki temel tur süreleri 1 + 1 sa ile üretilmiştir.
    """
    name: str = "Damaged_Roads"
    speed_kmh: float = 25.0
    road_factor: float = 1.8
    trucks_per_depot: int = 3
    load_hours: float = 0.5
    unload_hours: float = 0.5


@dataclass(frozen=True)
class Config:
    dataset_id: str = (
        "electricsheepasia/asia-turkiye-earthquake-response-"
        "humanitiarian-observation-form"
    )
    seed: int = 42
    demand_low: int = 100
    demand_high: int = 500            # dahil
    tent_capacity: int = 4            # kişi / çadır
    truck_capacity: int = 50          # çadır / sefer
    min_coverage: float = 0.70        # adalet eşiği
    horizons: tuple = (24, 48, 72)
    stock_levels: tuple = (500, 750, 1000, 1250, 1500, 2000)
    local_search_iters: int = 3000
    depots: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "depot": ["Adiyaman_Depot", "Hatay_Depot", "Maras_Depot"],
        "latitude": [37.76, 36.20, 37.58],
        "longitude": [38.28, 36.16, 36.93],
        "stock": [500, 500, 500],
    }))
    baseline: Scenario = Scenario("Baseline", 40, 1.4, 5, 1.0, 1.0)
    damaged: Scenario = Scenario()


CFG = Config()
CRITICAL_DAMAGE = ["Destroyed", "Severe_Uninhabitable"]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# %% 1. Veri yükleme ve temizleme
RENAME_MAP = {
    "start": "survey_start",
    "end": "survey_end",
    "record_your_current_location_latitude": "latitude",
    "record_your_current_location_longitude": "longitude",
    "group_kv3uy25_province": "province",
    "group_kv3uy25_district": "district",
    "group_kv3uy25_village_neigborhood_koy_mahalle_name": "neighborhood",
    "group_kv3uy25_urban_vs_rural": "settlement_type",
    "group_ck2if28_what_is_the_primary_gestion_debris_etc": "primary_observed_issue",
    "group_rg7cc33_level_of_destruction_nd_sewage_system_etc": "infrastructure_damage",
    "group_of79u43_level_of_destruction_to_dwellings": "housing_damage",
    "group_sj1rb16_were_there_idps_living_in_your": "idps_present",
    "group_sj1rb16_1_if_yes_can_you_estimate_how_many_idps": "idps_estimated",
    "group_sj1rb16_how_many_people_are_districts_provinces": "population_estimated",
    "group_sj1rb16_population_complexity": "population_type",
    "group_yb8oz38_type_of_shelter_visible_on_site": "shelter_type",
    "group_yb8oz38_shelter_related_challenges_con": "shelter_challenges",
    "group_yb8oz38_type_of_shelter_visible_on_site_afad_tents": "afad_tent",
    "group_yb8oz38_shelter_related_challenges_con_not_received_shelter_assistanc":
        "no_shelter_assistance",
    "group_jg54y99_which_items_are_avai_ers_and_supermarkets": "market_items",
    "group_jg54y99_did_you_notice_a_change_in_prices": "price_changes",
    "group_jg54y99_are_atms_functioning_n_this_neighbourhood": "atm_functioning",
    "group_yf2rz25_facilities_observed_on_site": "facilities",
    "group_el8ar77_10_population_includes": "vulnerable_population",
    "group_nm6wg34_are_there_observation_and_or_s": "assistance_observed",
    "group_nm6wg34_if_yes_what_are_the_tance_being_provided": "assistance_types",
    "group_da4tz78_top_priority_1": "priority_1",
    "group_da4tz78_top_priority_2": "priority_2",
    "group_da4tz78_top_priority_3": "priority_3",
    "group_kj3dz68_shelter": "need_shelter",
    "group_kj3dz68_temporary_shelter_and_nfis": "need_temporary_shelter",
    "group_kj3dz68_wash": "need_wash",
    "group_kj3dz68_health_and_nutrition": "need_health",
    "group_kj3dz68_education": "need_education",
    "group_kj3dz68_protection": "need_protection",
    "group_oe1ew37_please_add_any_other_observation_you_have":
        "additional_observations",
}

PREFIX_MAP = {
    "group_sj1rb16_": "population_",
    "group_yb8oz38_": "shelter_",
    "group_jg54y99_": "market_",
    "group_yf2rz25_": "facility_",
    "group_el8ar77_": "vulnerable_",
    "group_nm6wg34_": "assistance_",
    "group_kj3dz68_": "need_",
}

HOUSING_MAP = {
    "totally_destroyed": "Destroyed",
    "severely_damaged_and_not_habitable": "Severe_Uninhabitable",
    "significantly_damaged_but_habitable": "Significant_Habitable",
    "moderate_damage": "Moderate",
    "minor_damage_only": "Minor",
    "no_visible_damage": "No_Damage",
}

INFRASTRUCTURE_MAP = {
    "mostly_destroyed": "Mostly_Destroyed",
    "severely_damaged": "Severe",
    "significant_damage": "Significant",
    "moderately_damage": "Moderate",
    "minor_damage": "Minor",
    "no_visible_damage": "No_Damage",
}

PRIORITY_MAP = {
    "water__sanitation_and_hygiene__wash": "Water_Sanitation",
    "temporary_settlement_support": "Temporary_Shelter",
    "multipurpose_cash___social_protection": "Cash_Assistance",
    "emergency_shelter___non_food_items": "Emergency_Shelter",
    "food_security_and_livelihoods__fsl": "Food_Security",
    "health___nutrition": "Health",
    "early_recovery___debris_removal": "Debris_Removal",
    "education": "Education",
}

ANALYSIS_COLUMNS = [
    "province", "district", "latitude", "longitude", "settlement_type",
    "housing_damage", "infrastructure_damage", "population_type",
    "priority_1", "shelter_type", "afad_tent", "no_shelter_assistance",
    "idps_estimated",
]


def load_raw(csv_path: str | None = None, cfg: Config = CFG) -> pd.DataFrame:
    """Train bölümünü yükler. CSV verilirse onu, yoksa Hugging Face'i kullanır."""
    if csv_path:
        return pd.read_csv(csv_path)
    from datasets import load_dataset  # yalnızca gerektiğinde içe aktar
    ds = load_dataset(cfg.dataset_id, verification_mode="no_checks")
    for split in ds:                     # yerel kopya (tekrarlanabilirlik)
        ds[split].to_pandas().to_csv(f"deprem_gozlem_{split}.csv", index=False)
    return ds["train"].to_pandas()


def _clean_column_name(col: str) -> str:
    for old, new in PREFIX_MAP.items():
        if col.startswith(old):
            return new + col[len(old):]
    return re.sub(r"^group_[a-z0-9]+_", "", col)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Sütunları okunur hale getirir, kategorileri kısaltır, bayrak üretir."""
    df = raw.rename(columns=RENAME_MAP)
    df = df.rename(columns={c: _clean_column_name(c) for c in df.columns})

    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"Aynı ada dönüşen sütunlar: {dup}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["housing_damage"] = df["housing_damage"].replace(HOUSING_MAP)
    df["infrastructure_damage"] = df["infrastructure_damage"].replace(
        INFRASTRUCTURE_MAP)
    df["priority_1"] = df["priority_1"].replace(PRIORITY_MAP)

    shelter = df["shelter_type"].fillna("")
    for flag, token in {
        "has_makeshift": "makeshift_shelters",
        "has_emergency_tent": "emergency_relief_tents",
        "has_private_tent": "private_camping_tents",
        "has_container": "containers",
    }.items():
        df[flag] = shelter.str.contains(rf"\b{token}\b", regex=True)
    df["has_afad_tent"] = df["afad_tent"] == 1

    # Konut hasarı eksikse NaN kalır (oranlarda paydaya girmez)
    df["critical_damage"] = np.where(
        df["housing_damage"].isna(), np.nan,
        df["housing_damage"].isin(CRITICAL_DAMAGE).astype(float))
    return df[ANALYSIS_COLUMNS + [
        "has_makeshift", "has_emergency_tent", "has_private_tent",
        "has_container", "has_afad_tent", "critical_damage"]].copy()


# %% 2. Keşifsel analiz
def run_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    section("2. KEŞİFSEL ANALİZ")
    print(f"Kayıt: {len(df)} · İl: {df['province'].nunique()} · "
          f"İlçe: {df['district'].nunique()}")

    print("\nEksik veri oranı (%):")
    print(df[ANALYSIS_COLUMNS].isna().mean().mul(100).round(1)
          .sort_values(ascending=False).to_string())

    for col in ["housing_damage", "infrastructure_damage", "priority_1",
                "settlement_type", "population_type"]:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False).to_string())

    print("\nBarınma türü gözlem sayıları:")
    print(df[["has_makeshift", "has_emergency_tent", "has_private_tent",
              "has_container", "has_afad_tent"]].sum().to_string())

    print("\nAğır hasar × derme çatma barınak × yardım almama:")
    print(pd.crosstab([df["critical_damage"], df["has_makeshift"]],
                      df["no_shelter_assistance"], dropna=False))

    critical_without_tent = df[(df["critical_damage"] == 1) & (df["afad_tent"] == 0)]
    print(f"\nAğır hasar + AFAD çadırı gözlenmeyen kayıt: {len(critical_without_tent)}")
    print(critical_without_tent[["province", "district", "housing_damage",
                                 "shelter_type", "no_shelter_assistance",
                                 "idps_estimated"]].to_string(index=False))

    indicators = district_indicators(df)
    print("\nİlçe bazlı göstergeler (veri yeterliliği ile):")
    print(indicators.sort_values(["observations", "critical_rate"],
                                 ascending=False).round(2).to_string(index=False))
    return {"indicators": indicators,
            "critical_without_tent": critical_without_tent}


def district_indicators(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.dropna(subset=["province", "district"])
           .groupby(["province", "district"])
           .agg(observations=("district", "size"),
                critical_count=("critical_damage", "sum"),
                valid_damage=("critical_damage", "count"),
                makeshift=("has_makeshift", "sum"),
                afad_tent=("has_afad_tent", "sum"),
                assistance_gap=("no_shelter_assistance", lambda s: (s == 1).sum()),
                valid_assistance=("no_shelter_assistance", "count"))
           .reset_index())
    g["critical_rate"] = g["critical_count"] / g["valid_damage"].replace(0, np.nan)
    g["makeshift_rate"] = g["makeshift"] / g["observations"]
    g["assistance_gap_rate"] = (g["assistance_gap"]
                                / g["valid_assistance"].replace(0, np.nan))
    g["data_quality"] = np.where(g["observations"] < 5, "Limited", "Preliminary")
    return g


# %% 3. Sentetik talep, depolar ve mesafeler
def haversine(lat1, lon1, lat2, lon2):
    """Kuş uçuşu mesafe (km)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_network(df: pd.DataFrame, indicators: pd.DataFrame, cfg: Config = CFG):
    """İlçe talebi (sentetik) ve depo–ilçe rota tablosunu üretir.

    Talep, 23 ilçelik tabloda üretilir (orijinal çalışmayla aynı RNG sırası);
    koordinatı olmayan ilçeler ardından elenir.
    """
    section("3. SENTETİK SENARYO VE AĞ")
    rng = np.random.default_rng(cfg.seed)
    all_districts = indicators[["province", "district"]].copy()
    all_districts["demand"] = rng.integers(
        cfg.demand_low, cfg.demand_high + 1, size=len(all_districts))

    locations = (df.groupby(["province", "district"])
                   .agg(latitude=("latitude", "median"),
                        longitude=("longitude", "median"))
                   .reset_index()
                   .dropna(subset=["latitude", "longitude"]))
    missing = set(all_districts["district"]) - set(locations["district"])
    print(f"Koordinatı olmayan ilçeler (modelden çıkarıldı): {sorted(missing)}")

    districts = (all_districts
                 .merge(locations[["province", "district"]],
                        on=["province", "district"], how="inner",
                        validate="one_to_one")
                 .reset_index(drop=True))

    routes = cfg.depots[["depot", "latitude", "longitude"]].merge(
        locations, how="cross", suffixes=("_d", "_j"))
    routes["distance_km"] = haversine(routes["latitude_d"], routes["longitude_d"],
                                      routes["latitude_j"], routes["longitude_j"]).round(2)
    routes = (routes[["depot", "province", "district", "distance_km"]]
              .merge(districts, on=["province", "district"], how="inner",
                     validate="many_to_one")
              .reset_index(drop=True))

    print(f"İlçe: {len(districts)} · Depo: {len(cfg.depots)} · "
          f"Rota: {len(routes)} · Talep: {districts['demand'].sum():,} kişi · "
          f"Stok: {cfg.depots['stock'].sum():,} çadır")
    return all_districts, districts, routes


# %% 4. Statik tahsis ve stok duyarlılığı
def solve_static_allocation(demand: np.ndarray, total_tents: int,
                            cap: int = CFG.tent_capacity) -> dict:
    """min Σu  s.t. cap·x + u ≥ d,  Σx ≤ S,  0 ≤ x ≤ ⌈d/cap⌉ (tam sayı)."""
    n = len(demand)
    c = np.concatenate([np.zeros(n), np.ones(n)])
    a = np.vstack([np.hstack([cap * np.eye(n), np.eye(n)]),
                   np.concatenate([np.ones(n), np.zeros(n)])])
    lb = np.concatenate([demand, [-np.inf]])
    ub = np.concatenate([np.full(n, np.inf), [total_tents]])
    bounds = Bounds(np.zeros(2 * n),
                    np.concatenate([np.ceil(demand / cap), demand]))
    res = milp(c, integrality=np.r_[np.ones(n), np.zeros(n)],
               bounds=bounds, constraints=LinearConstraint(a, lb, ub))
    if not res.success:
        raise RuntimeError(res.message)
    x = np.rint(res.x[:n]).astype(int)
    unmet = np.maximum(0, demand - cap * x)
    return {"stock": total_tents, "used": int(x.sum()),
            "served": int(demand.sum() - unmet.sum()), "unmet": int(unmet.sum()),
            "coverage_pct": round(100 * (1 - unmet.sum() / demand.sum()), 2)}


def stock_sensitivity(all_districts: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    section("4. STATİK TAHSİS — STOK DUYARLILIĞI")
    demand = all_districts["demand"].to_numpy()
    out = pd.DataFrame([solve_static_allocation(demand, s) for s in cfg.stock_levels])
    print(out.to_string(index=False))
    print(f"Kapasiteye dayalı alt sınır: {int(np.ceil(demand.sum() / cfg.tent_capacity))} çadır · "
          f"İlçe bazında tam sayı gereksinim: "
          f"{int(np.ceil(demand / cfg.tent_capacity).sum())} çadır")
    return out


# %% 5. İki aşamalı depo–ilçe MILP (verimlilik ve adalet)
def solve_depot_allocation(districts: pd.DataFrame, routes: pd.DataFrame,
                           min_coverage: float | None = None,
                           cfg: Config = CFG) -> pd.DataFrame:
    """Aşama 1: Σu en aza; Aşama 2: Σu ≤ U* tutularak Σ c·x (çadır-km) en aza.

    min_coverage verilirse her ilçe için u_j ≤ (1-α)·d_j eklenir.
    Dönüş: routes + allocated_tents sütunu.
    """
    nr, nd = len(routes), len(districts)
    d = districts["demand"].to_numpy()
    key_r = list(zip(routes["province"], routes["district"]))
    key_d = list(zip(districts["province"], districts["district"]))
    member = np.array([[kr == kd for kr in key_r] for kd in key_d], dtype=float)

    a_demand = np.hstack([cfg.tent_capacity * member, np.eye(nd)])
    a_stock = np.zeros((len(cfg.depots), nr + nd))
    for i, dep in enumerate(cfg.depots["depot"]):
        a_stock[i, :nr] = (routes["depot"] == dep).to_numpy()
    a_limit = np.hstack([member, np.zeros((nd, nd))])

    cons = [LinearConstraint(
        np.vstack([a_demand, a_stock, a_limit]),
        np.concatenate([d, np.zeros(len(cfg.depots) + nd)]),
        np.concatenate([np.full(nd, np.inf), cfg.depots["stock"].to_numpy(),
                        np.ceil(d / cfg.tent_capacity)]))]
    if min_coverage is not None:
        a_eq = np.hstack([np.zeros((nd, nr)), np.eye(nd)])
        cons.append(LinearConstraint(a_eq, np.zeros(nd), (1 - min_coverage) * d))

    bounds = Bounds(np.zeros(nr + nd), np.concatenate([np.full(nr, np.inf), d]))
    integrality = np.r_[np.ones(nr), np.zeros(nd)]

    stage1 = milp(np.r_[np.zeros(nr), np.ones(nd)], integrality=integrality,
                  bounds=bounds, constraints=cons)
    if not stage1.success:
        raise RuntimeError(f"Aşama 1: {stage1.message}")

    a_unmet = np.r_[np.zeros(nr), np.ones(nd)].reshape(1, -1)
    stage2 = milp(np.r_[routes["distance_km"].to_numpy(), np.zeros(nd)],
                  integrality=integrality, bounds=bounds,
                  constraints=cons + [LinearConstraint(a_unmet, -np.inf,
                                                       stage1.fun + 1e-6)])
    if not stage2.success:
        raise RuntimeError(f"Aşama 2: {stage2.message}")

    out = routes.copy()
    out["allocated_tents"] = np.rint(stage2.x[:nr]).astype(int)
    return out


def allocation_summary(alloc: pd.DataFrame, districts: pd.DataFrame,
                       cfg: Config = CFG) -> dict:
    per = (alloc.groupby(["province", "district"])["allocated_tents"].sum()
                .reset_index().merge(districts, on=["province", "district"]))
    served = np.minimum(per["demand"], cfg.tent_capacity * per["allocated_tents"])
    cov = 100 * served / per["demand"]
    return {"tents": int(alloc["allocated_tents"].sum()),
            "served": int(served.sum()),
            "unmet": int((per["demand"] - served).sum()),
            "min_coverage_pct": round(float(cov.min()), 2),
            "zero_service_districts": int((served == 0).sum()),
            "tent_km": round(float((alloc["allocated_tents"]
                                    * alloc["distance_km"]).sum()), 2)}


def compare_efficiency_equity(districts, routes, cfg: Config = CFG):
    section("5. DEPO–İLÇE SEVKİYATI: VERİMLİLİK vs ADALET")
    eff = solve_depot_allocation(districts, routes, None, cfg)
    eq = solve_depot_allocation(districts, routes, cfg.min_coverage, cfg)
    table = pd.DataFrame({"Verimlilik": allocation_summary(eff, districts, cfg),
                          "Adalet (≥%70)": allocation_summary(eq, districts, cfg)})
    print(table.to_string())
    d_km = table.loc["tent_km"]
    print(f"Adaletin ek taşıma yükü: {d_km.iloc[1] - d_km.iloc[0]:,.2f} çadır-km "
          f"(%{100 * (d_km.iloc[1] / d_km.iloc[0] - 1):.2f})")
    return eff, eq, table


# %% 6. Süre modeli ve çizelgeleme
def with_travel_times(alloc: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    out = alloc.copy()
    out["estimated_road_km"] = out["distance_km"] * sc.road_factor
    out["travel_hours"] = out["estimated_road_km"] / sc.speed_kmh
    out["round_trip_hours"] = (sc.load_hours + 2 * out["travel_hours"]
                               + sc.unload_hours)
    return out


def split_into_trips(alloc: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Her rotanın tahsisini ≤ truck_capacity çadırlık seferlere böler."""
    rows = []
    for _, r in alloc.iterrows():
        remaining, trip_no = int(r["allocated_tents"]), 0
        while remaining > 0:
            trip_no += 1
            load = min(remaining, cfg.truck_capacity)
            rows.append({"depot": r["depot"], "province": r["province"],
                         "district": r["district"], "tents": load,
                         "distance_km": r["distance_km"],
                         "travel_hours": r["travel_hours"], "trip_no": trip_no})
            remaining -= load
    return pd.DataFrame(rows)


def order_trips(trips: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "LPT":
        return trips.sort_values("travel_hours", ascending=False)
    if strategy == "Equity_First":
        return trips.sort_values(["trip_no", "travel_hours"], ascending=[True, True])
    raise ValueError(f"Bilinmeyen strateji: {strategy}")


def dispatch(trips: pd.DataFrame, sc: Scenario, sort_depots: bool = True) -> pd.DataFrame:
    """Sıralı sefer listesini, her depoda ilk boşalan araca atar.

    trips sırası korunur; her satırda travel_hours bulunmalıdır.
    """
    results = []
    for depot, group in trips.groupby("depot", sort=sort_depots):
        fleet = [(0.0, t) for t in range(sc.trucks_per_depot)]
        heapq.heapify(fleet)
        for _, trip in group.iterrows():
            start, truck = heapq.heappop(fleet)
            tau = float(trip["travel_hours"])
            delivery = start + sc.load_hours + tau + sc.unload_hours
            finish = delivery + tau
            results.append({"depot": depot, "truck_id": truck,
                            "province": trip["province"], "district": trip["district"],
                            "tents": trip["tents"], "distance_km": trip["distance_km"],
                            "travel_hours": tau, "start_hour": start,
                            "delivery_hour": delivery, "return_hour": finish})
            heapq.heappush(fleet, (finish, truck))
    return pd.DataFrame(results)


def simulate(alloc: pd.DataFrame, sc: Scenario, strategy: str,
             cfg: Config = CFG) -> pd.DataFrame:
    """LPT veya Equity_First kuralıyla 72 saatlik teslimat çizelgesi."""
    trips = split_into_trips(with_travel_times(alloc, sc), cfg)
    return dispatch(order_trips(trips, strategy), sc)


def simulate_maxmin(alloc: pd.DataFrame, districts: pd.DataFrame, sc: Scenario,
                    cfg: Config = CFG) -> pd.DataFrame:
    """Boşalan araç, kendi deposundan o an planlanan hizmeti en düşük ilçeye gider."""
    df = with_travel_times(alloc, sc)
    demand = {(r.province, r.district): r.demand for r in districts.itertuples()}
    remaining = {i: int(r["allocated_tents"]) for i, r in df.iterrows()}
    planned = {k: 0 for k in demand}
    fleet = [(0.0, dep, t) for dep in df["depot"].unique()
             for t in range(sc.trucks_per_depot)]
    heapq.heapify(fleet)

    results = []
    while any(v > 0 for v in remaining.values()):
        start, depot, truck = heapq.heappop(fleet)
        candidates = []
        for i, r in df[df["depot"] == depot].iterrows():
            if remaining[i] <= 0:
                continue
            key = (r["province"], r["district"])
            cov = min(1.0, cfg.tent_capacity * planned[key] / demand[key])
            candidates.append((cov, r["travel_hours"], i,
                               min(cfg.truck_capacity, remaining[i]), key))
        if not candidates:          # bu deponun işi bitti; araç havuza dönmez
            continue
        _, tau, i, load, key = min(candidates)
        delivery = start + sc.load_hours + tau + sc.unload_hours
        r = df.loc[i]
        results.append({"depot": depot, "truck_id": truck,
                        "province": r["province"], "district": r["district"],
                        "tents": load, "distance_km": r["distance_km"],
                        "travel_hours": tau, "start_hour": start,
                        "delivery_hour": delivery, "return_hour": delivery + tau})
        remaining[i] -= load
        planned[key] += load
        heapq.heappush(fleet, (delivery + tau, depot, truck))
    return pd.DataFrame(results)


# %% 7. Hizmet ölçüleri
def district_coverage(schedule: pd.DataFrame, districts: pd.DataFrame,
                      hour: float, cfg: Config = CFG) -> pd.DataFrame:
    delivered = (schedule.loc[schedule["delivery_hour"] <= hour]
                 .groupby(["province", "district"])["tents"].sum()
                 .rename("delivered_tents").reset_index())
    out = districts.merge(delivered, on=["province", "district"], how="left")
    out["delivered_tents"] = out["delivered_tents"].fillna(0).astype(int)
    out["served"] = np.minimum(out["demand"], cfg.tent_capacity * out["delivered_tents"])
    out["unmet"] = out["demand"] - out["served"]
    out["coverage_pct"] = 100 * out["served"] / out["demand"]
    return out


def score(schedule: pd.DataFrame, districts: pd.DataFrame,
          horizon: float = 72, cfg: Config = CFG) -> tuple:
    """Leksikografik skor (küçük iyi):
    (eşik altı ilçe, eşik için eksik çadır, −min hizmet %, −toplam hizmet)."""
    cov = district_coverage(schedule, districts, horizon, cfg)
    required = np.ceil(cfg.min_coverage * cov["demand"] / cfg.tent_capacity)
    shortage = np.maximum(0, required - cov["delivered_tents"]).sum()
    return (int((cov["coverage_pct"] < 100 * cfg.min_coverage).sum()),
            int(shortage), -float(cov["coverage_pct"].min()),
            -int(cov["served"].sum()))


def horizon_table(schedules: dict, districts: pd.DataFrame,
                  cfg: Config = CFG) -> pd.DataFrame:
    rows = []
    for name, sch in schedules.items():
        for h in cfg.horizons:
            cov = district_coverage(sch, districts, h, cfg)
            rows.append({"model": name, "hour": h,
                         "delivered_tents": int(sch.loc[sch["delivery_hour"] <= h, "tents"].sum()),
                         "served": int(cov["served"].sum()),
                         "coverage_pct": round(100 * cov["served"].sum() / cov["demand"].sum(), 2),
                         "below_target": int((cov["coverage_pct"] < 100 * cfg.min_coverage).sum()),
                         "zero_service": int((cov["served"] == 0).sum()),
                         "min_coverage": round(float(cov["coverage_pct"].min()), 2)})
    return pd.DataFrame(rows)


def time_to_target(schedule: pd.DataFrame, districts: pd.DataFrame,
                   cfg: Config = CFG) -> pd.DataFrame:
    """İlk teslimat saati ve %70 hedefine ulaşma saati (T70)."""
    rows = []
    for r in districts.itertuples():
        s = (schedule[(schedule["province"] == r.province)
                      & (schedule["district"] == r.district)]
             .sort_values("delivery_hour"))
        served = np.minimum(r.demand, cfg.tent_capacity * s["tents"].cumsum())
        hit = s.loc[served >= cfg.min_coverage * r.demand, "delivery_hour"]
        rows.append({"district": r.district, "demand": r.demand,
                     "first_delivery_hour": s["delivery_hour"].min() if len(s) else np.nan,
                     "T70": hit.iloc[0] if len(hit) else np.nan})
    out = pd.DataFrame(rows)
    out["T70_within_72h"] = out["T70"] <= 72
    return out


# %% 8. Yerel arama ve depo takası
def local_search(trips: pd.DataFrame, districts: pd.DataFrame, sc: Scenario,
                 iterations: int, seed: int, cfg: Config = CFG):
    """Aynı depodaki iki seferin yerini değiştir (swap) ya da birini başka
    sıraya taşı (insertion); yalnızca skoru iyileştiren hamleyi kabul et."""
    trips = trips.reset_index(drop=True)
    best_sched = dispatch(trips, sc, sort_depots=False)
    best = score(best_sched, districts, cfg=cfg)
    rng = np.random.default_rng(seed)

    for _ in range(iterations):
        cand = trips.copy()
        depot = rng.choice(cand["depot"].unique())
        idx = cand.index[cand["depot"] == depot].to_numpy()
        if len(idx) < 2:
            continue
        a, b = rng.choice(idx, size=2, replace=False)
        if rng.random() < 0.5:                                   # swap
            row_a, row_b = cand.loc[a].copy(), cand.loc[b].copy()
            cand.loc[a], cand.loc[b] = row_b, row_a
        else:                                                    # insertion
            row = cand.loc[a].copy()
            cand = cand.drop(a).reset_index(drop=True)
            cand = pd.concat([cand.iloc[:b], pd.DataFrame([row]), cand.iloc[b:]],
                             ignore_index=True)
        sched = dispatch(cand, sc, sort_depots=False)
        s = score(sched, districts, cfg=cfg)
        if s < best:
            trips, best_sched, best = cand, sched, s
    return trips, best_sched, best


def optimize_schedule(alloc: pd.DataFrame, districts: pd.DataFrame, sc: Scenario,
                      seed: int = CFG.seed, iterations: int = CFG.local_search_iters,
                      cfg: Config = CFG) -> dict:
    """Equity First başlangıcından yerel arama."""
    initial = simulate(alloc, sc, "Equity_First", cfg)
    trips = (initial.sort_values(["depot", "start_hour", "truck_id"])
             .reset_index(drop=True)[["depot", "province", "district", "tents",
                                      "distance_km", "travel_hours"]])
    trips, sched, s = local_search(trips, districts, sc, iterations, seed, cfg)
    return {"trips": trips, "schedule": sched, "score": s,
            "initial_score": score(initial, districts, cfg=cfg)}


def swap_allocation(alloc: pd.DataFrame, receiver: str = "Besni",
                    partner: str | None = None, transfer: int = 18,
                    near_depot: str = "Adiyaman_Depot",
                    far_depot: str = "Hatay_Depot") -> pd.DataFrame:
    """receiver ilçesinin `transfer` çadırını uzak depodan yakın depoya alır;
    karşılığında partner ilçesinin aynı miktarını yakın depodan uzak depoya verir.
    Depo stokları ve ilçe toplamları değişmez."""
    out = alloc.copy()
    if partner is None:
        return out

    def idx(depot, district):
        m = out.index[(out["depot"] == depot) & (out["district"] == district)]
        assert len(m) == 1, f"Tekil rota yok: {depot} → {district}"
        return m[0]

    for depot, district, delta in [(near_depot, receiver, +transfer),
                                   (far_depot, receiver, -transfer),
                                   (near_depot, partner, -transfer),
                                   (far_depot, partner, +transfer)]:
        out.loc[idx(depot, district), "allocated_tents"] += delta

    assert (out["allocated_tents"] >= 0).all(), "Negatif tahsis oluştu"
    assert out.groupby("depot")["allocated_tents"].sum().equals(
        alloc.groupby("depot")["allocated_tents"].sum())
    assert out.groupby(["province", "district"])["allocated_tents"].sum().equals(
        alloc.groupby(["province", "district"])["allocated_tents"].sum())
    return out


def swap_candidates(alloc: pd.DataFrame, receiver="Besni", transfer=18,
                    near_depot="Adiyaman_Depot", far_depot="Hatay_Depot") -> list:
    """Yakın depodan en az `transfer` çadır alan ve uzak depodan da rotası olan ilçeler."""
    near = alloc[(alloc["depot"] == near_depot) & (alloc["district"] != receiver)
                 & (alloc["allocated_tents"] >= transfer)]
    far = set(alloc.loc[alloc["depot"] == far_depot, "district"])
    return [d for d in near["district"] if d in far]


# %% 9. Doğrulama
def validate_plan(alloc: pd.DataFrame, schedule: pd.DataFrame,
                  districts: pd.DataFrame, sc: Scenario, cfg: Config = CFG) -> pd.DataFrame:
    checks = []
    stock = alloc.groupby("depot")["allocated_tents"].sum()
    limits = cfg.depots.set_index("depot")["stock"]
    checks.append(("Depo stoku", bool((stock <= limits.reindex(stock.index)).all()),
                   " · ".join(f"{k.split('_')[0]} {v}" for k, v in stock.items())))

    overlaps = 0
    for _, g in schedule.groupby(["depot", "truck_id"]):
        g = g.sort_values("start_hour")
        overlaps += int((g["start_hour"].to_numpy()[1:]
                         < g["return_hour"].to_numpy()[:-1] - 1e-8).sum())
    checks.append(("Araç çakışması", overlaps == 0, f"{overlaps} çakışma"))

    planned = alloc.groupby(["depot", "province", "district"])["allocated_tents"].sum()
    sched = schedule.groupby(["depot", "province", "district"])["tents"].sum()
    diff = pd.concat([planned, sched], axis=1).fillna(0).diff(axis=1).iloc[:, 1].abs().max()
    checks.append(("Tahsis korunumu", diff == 0, f"maks. fark {diff:.0f}"))

    trucks = schedule.groupby("depot")["truck_id"].nunique()
    checks.append(("Filo sınırı", bool((trucks <= sc.trucks_per_depot).all()),
                   f"maks. {trucks.max()} araç/depo"))
    checks.append(("Sefer yükü", bool(schedule["tents"].between(1, cfg.truck_capacity).all()),
                   f"{schedule['tents'].min()}–{schedule['tents'].max()} çadır"))

    cov = district_coverage(schedule, districts, 72, cfg)
    ok = int((cov["coverage_pct"] >= 100 * cfg.min_coverage - 1e-8).sum())
    checks.append(("72 saatlik adalet", ok == len(cov), f"{ok}/{len(cov)} ilçe ≥ %70"))
    total = 100 * cov["served"].sum() / cov["demand"].sum()
    worst = cov.loc[cov["coverage_pct"].idxmin()]
    checks.append(("Toplam hizmet", True,
                   f"{cov['served'].sum():,} / {cov['demand'].sum():,} kişi (%{total:.2f}); "
                   f"en düşük {worst['district']} %{worst['coverage_pct']:.2f}"))
    return pd.DataFrame(checks, columns=["kontrol", "geçti", "sonuç"])


# %% 10. Hız duyarlılığı
def speed_sensitivity(trips: pd.DataFrame, districts: pd.DataFrame, base: Scenario,
                      speeds=(15, 20, 25, 30, 35, 40), cfg: Config = CFG) -> pd.DataFrame:
    """Görev sırası sabit, seyahat süreleri yeni hızla yeniden hesaplanır."""
    rows = []
    for v in speeds:
        t = trips.copy()
        t["travel_hours"] = t["distance_km"] * base.road_factor / v
        cov = district_coverage(dispatch(t, base, sort_depots=False), districts, 72, cfg)
        rows.append({"speed_kmh": v,
                     "below_target": int((cov["coverage_pct"] < 100 * cfg.min_coverage).sum()),
                     "min_coverage": round(float(cov["coverage_pct"].min()), 2),
                     "served": int(cov["served"].sum()),
                     "service_pct": round(100 * cov["served"].sum() / cov["demand"].sum(), 2)})
    return pd.DataFrame(rows)


# %% 11. Ana akış
def main(csv_path: str | None = None, cfg: Config = CFG) -> dict:
    section("1. VERİ")
    df = clean_data(load_raw(csv_path, cfg))
    eda = run_eda(df)
    all_districts, districts, routes = build_network(df, eda["indicators"], cfg)
    stock_table = stock_sensitivity(all_districts, cfg)
    eff, eq, eff_eq_table = compare_efficiency_equity(districts, routes, cfg)

    # Temel senaryo tur süreleri (Bölüm 4.3)
    base_times = with_travel_times(eq, cfg.baseline)
    base_times["required_trips"] = np.ceil(base_times["allocated_tents"]
                                           / cfg.truck_capacity).astype(int)
    print("\nTemel senaryo — en uzun turlar:")
    print(base_times.loc[base_times["allocated_tents"] > 0,
                         ["depot", "district", "allocated_tents", "estimated_road_km",
                          "round_trip_hours", "required_trips"]]
          .sort_values("round_trip_hours", ascending=False).head(10)
          .round(2).to_string(index=False))

    section("6. 72 SAATLİK ÇİZELGELEME — AĞIR YOL HASARI")
    sc = cfg.damaged
    schedules = {"LPT": simulate(eq, sc, "LPT", cfg),
                 "Equity_First": simulate(eq, sc, "Equity_First", cfg),
                 "MaxMin": simulate_maxmin(eq, districts, sc, cfg)}
    print(horizon_table(schedules, districts, cfg).to_string(index=False))
    for name, sch in schedules.items():
        t70 = time_to_target(sch, districts, cfg)
        print(f"{name:13s} 72 saatte %70'e ulaşan: {int(t70['T70_within_72h'].sum())}/{len(t70)}"
              f" · ilk teslimat medyanı: {t70['first_delivery_hour'].median():.2f} sa")

    section("7. YEREL ARAMA VE DEPO TAKASI")
    variants = {"Original": None}
    variants.update({d: d for d in swap_candidates(eq)})
    results, models = [], {}
    for name, partner in variants.items():
        try:
            alloc = swap_allocation(eq, partner=partner)
        except AssertionError as err:        # uygulanamaz takas → atla
            print(f"{name}: takas atlandı ({err})")
            continue
        res = optimize_schedule(alloc, districts, sc, cfg=cfg)
        res["allocation"] = alloc
        models[name] = res
        s = res["score"]
        results.append({"model": name, "below_target": s[0], "shortage_tents": s[1],
                        "min_coverage": round(-s[2], 2), "served": -s[3]})
    comparison = pd.DataFrame(results).sort_values(
        ["below_target", "shortage_tents", "min_coverage", "served"],
        ascending=[True, True, False, False])
    print(comparison.to_string(index=False))
    best_name = comparison.iloc[0]["model"]
    best = models[best_name]
    print(f"\nSeçilen plan: {best_name}")

    late = best["schedule"][best["schedule"]["delivery_hour"] > 72]
    if len(late):
        print("\n72 saatten sonra ulaşan seferler:")
        print(late[["depot", "district", "tents", "delivery_hour"]]
              .round(2).to_string(index=False))

    section("8. NİHAİ PLANIN DOĞRULANMASI")
    validation = validate_plan(best["allocation"], best["schedule"], districts, sc, cfg)
    print(validation.to_string(index=False))
    final_cov = district_coverage(best["schedule"], districts, 72, cfg)
    print("\nİlçe bazında 72 saatlik hizmet:")
    print(final_cov[["district", "demand", "delivered_tents", "coverage_pct"]]
          .sort_values("coverage_pct").round(2).to_string(index=False))

    section("9. HIZ DUYARLILIĞI")
    speed = speed_sensitivity(best["trips"], districts, sc, cfg=cfg)
    print("Sabit görev sırası:")
    print(speed.to_string(index=False))
    slow = replace(sc, speed_kmh=20)
    reopt = optimize_schedule(best["allocation"], districts, slow,
                              seed=2026, iterations=5000, cfg=cfg)
    print(f"\n20 km/sa yeniden çizelgeleme: başlangıç {reopt['initial_score']} "
          f"→ sonuç {reopt['score']}  (eşik altı, eksik çadır, −min %, −hizmet)")

    return {"data": df, "districts": districts, "routes": routes,
            "stock_table": stock_table, "efficiency_vs_equity": eff_eq_table,
            "efficiency_alloc": eff, "equity_alloc": eq, "schedules": schedules,
            "swap_comparison": comparison, "best": best, "validation": validation,
            "final_coverage": final_cov, "speed_sensitivity": speed,
            "reopt_20kmh": reopt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--csv", help="Yerel train CSV yolu (deprem_gozlem_train.csv)")
    args, _ = parser.parse_known_args()   # Colab/Jupyter argümanlarını yok say
    results = main(args.csv)
