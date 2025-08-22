
# nycsbus_streamlit_app.py
# ---------------------------------------------------------
# Streamlit app for NYCSBUS-style spatiotemporal hotspot analysis
# + a two-stage alarm detector.
#
# Stage 0: Ingest crash-like data (date, lat, lon).
# Stage 1 (EDA/Features): H3 hex binning, weekly/monthly aggregation, local Gi* (G_Local),
#                         neighbor features (k-NN), lags, and trend (Mann-Kendall).
# Stage 2 (Model): KNN classifier to predict Alarm vs No Alarm for the latest period;
#                  then a rule-based second stage to label alarm type:
#                  New / Intensifying / Persistent / Diminishing / No Alarm.
#
# Usage:
#   streamlit run nycsbus_streamlit_app.py
#
# Required columns in your CSV (auto-detected case-insensitively):
#   - crash date/time: one of ["CRASH DATE", "CRASH_DATE", "date", "datetime", "timestamp"]
#   - latitude:        one of ["LATITUDE", "latitude", "lat", "y"]
#   - longitude:       one of ["LONGITUDE", "longitude", "lon", "lng", "x"]
#
# ---------------------------------------------------------

import io
import sys
import math
import json
import time
import folium
import numpy as np
import pandas as pd
import streamlit as st

# Spatial + stats
import h3
import geopandas as gpd
from shapely.geometry import Polygon, Point

from libpysal.weights import KNN
from esda.getisord import G_Local
import pymannkendall as mk

# ML
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Folium in Streamlit
from streamlit_folium import st_folium


# ------------------------------
# Helpers
# ------------------------------

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def to_period_str(dt_series, freq="W"):
    """Return a label for each datetime according to freq ('W' or 'M').
    For weekly, we use the week start (Monday) as label; for monthly, 'YYYY-MM'.
    """
    if freq.upper().startswith("W"):
        # Normalize to week start (Monday)
        week_start = (dt_series.dt.to_period('W-MON').start_time.dt.date.astype(str))
        return week_start
    elif freq.upper().startswith("M"):
        return dt_series.dt.to_period('M').astype(str)
    else:
        raise ValueError("freq must be 'W' or 'M'")


def h3_to_polygon(h3_cell):
    boundary = h3.h3_to_geo_boundary(h3_cell, geo_json=True)
    return Polygon(boundary)


def make_hex_gdf(cell_stats):
    """Given a DataFrame with columns ['h3_cell','centroid_lat','centroid_lon', ...],
    return a GeoDataFrame with polygon geometries and centroid points."""
    tmp = cell_stats.copy()
    tmp["geometry"] = tmp["h3_cell"].apply(h3_to_polygon)
    gdf = gpd.GeoDataFrame(tmp, geometry="geometry", crs="EPSG:4326")
    gdf["centroid"] = gdf.geometry.centroid
    gdf["centroid_lat"] = gdf["centroid"].y
    gdf["centroid_lon"] = gdf["centroid"].x
    return gdf.drop(columns=["centroid"])


def safe_g_local(y_values, centroids_xy, k=8, permutations=499):
    """Compute G* (Getis-Ord Local) safely. Returns (Zs, p_values).
    If fewer than 2 points, returns zeros/ones."""
    n = len(y_values)
    if n < 2:
        return np.zeros(n), np.ones(n)

    try:
        w = KNN.from_array(centroids_xy, k=min(k, n-1))
        g = G_Local(y_values.astype(float), w, transformation="r", permutations=permutations)
        # Some versions expose .Zs and .p_sim
        Zs = getattr(g, "Zs", np.zeros(n))
        p = getattr(g, "p_sim", np.ones(n))
        return np.array(Zs), np.array(p)
    except Exception as e:
        # Fail-safe
        return np.zeros(n), np.ones(n)


def mann_kendall_trend(x):
    """Return slope and categories from Mann-Kendall test for a 1D series.
    If too short or constant, returns zeros/no-trend."""
    x = pd.Series(x).dropna().values
    if len(x) < 4 or np.allclose(x, x[0]):
        return {"trend": "no trend", "slope": 0.0, "p": 1.0}
    try:
        res = mk.original_test(x)
        return {"trend": str(res.trend), "slope": float(getattr(res, "slope", 0.0)), "p": float(res.p)}
    except Exception:
        return {"trend": "no trend", "slope": 0.0, "p": 1.0}


def build_features(agg_df, latest_period, ngh_k=6):
    """Create lagged features, rolling stats, neighbor features (per period).

    Parameters
    ----------
    agg_df : DataFrame with columns
        ['h3_cell','time_period','crash_count','centroid_lat','centroid_lon','gi_star','gi_p']
    latest_period : str, the period label to predict
    ngh_k : int, k for neighbor features

    Returns
    -------
    features_all : DataFrame with engineered features for all periods
    train_df     : Subset (time_period != latest_period)
    pred_df      : Subset (time_period == latest_period)
    """
    df = agg_df.copy()
    # Ensure time ordering index
    t_order = (df[["time_period"]].drop_duplicates().sort_values("time_period")
               .reset_index(drop=True))
    t_order["t_idx"] = np.arange(len(t_order))
    df = df.merge(t_order, on="time_period", how="left").sort_values(["h3_cell","t_idx"]).reset_index(drop=True)

    # Binary significance flags
    df["is_hot_sig"] = ((df["gi_p"] < 0.05) & (df["gi_star"] > 0)).astype(int)

    # Lags/rollings per cell
    def add_lags(g, cols, lags=(1, 2)):
        g = g.copy()
        for col in cols:
            for L in lags:
                g[f"{col}_lag{L}"] = g[col].shift(L)
        return g

    def add_rolls(g):
        g = g.copy()
        g["crash_rolling_mean_2"] = g["crash_count"].rolling(2, min_periods=1).mean()
        g["crash_rolling_sum_3"]  = g["crash_count"].rolling(3, min_periods=1).sum()
        g["gi_star_rolling_mean_2"] = g["gi_star"].rolling(2, min_periods=1).mean()
        g["hot_sig_rolling_sum_3"]  = g["is_hot_sig"].rolling(3, min_periods=1).sum()
        return g

    df = df.groupby("h3_cell", group_keys=False).apply(add_lags, cols=["crash_count","gi_star","is_hot_sig"])
    df = df.groupby("h3_cell", group_keys=False).apply(add_rolls)

    # Neighbor features per period
    ngh_rows = []
    for tp, g in df.groupby("time_period", sort=False):
        if len(g) <= 1:
            out = g[["h3_cell"]].copy()
            out["ngh_crash_mean"] = np.nan
            out["ngh_gi_mean"] = np.nan
            out["ngh_hot_sig_sum"] = 0.0
        else:
            pts = g[["centroid_lat","centroid_lon"]].values
            tree = KDTree(pts)
            # Query k+1 to include self, then drop self
            k_eff = min(ngh_k + 1, len(pts))
            dists, inds = tree.query(pts, k=k_eff)
            mc, mg, sh = [], [], []
            for i, arr in enumerate(inds):
                nbrs = [j for j in arr if j != i]
                mc.append(np.nanmean(g.iloc[nbrs]["crash_count"].values) if nbrs else np.nan)
                mg.append(np.nanmean(g.iloc[nbrs]["gi_star"].values) if nbrs else np.nan)
                sh.append(np.nansum(g.iloc[nbrs]["is_hot_sig"].values) if nbrs else 0.0)
            out = g[["h3_cell"]].copy()
            out["ngh_crash_mean"] = mc
            out["ngh_gi_mean"] = mg
            out["ngh_hot_sig_sum"] = sh
        out["time_period"] = tp
        ngh_rows.append(out)

    ngh_all = pd.concat(ngh_rows, ignore_index=True)
    df = df.merge(ngh_all, on=["h3_cell","time_period"], how="left")

    # Simple neighbor lags
    def add_ngh_lags(g):
        g = g.copy()
        for col in ["ngh_crash_mean","ngh_gi_mean","ngh_hot_sig_sum"]:
            g[f"{col}_lag1"] = g[col].shift(1)
        return g
    df = df.groupby("h3_cell", group_keys=False).apply(add_ngh_lags)

    # Split train/predict
    train_df = df[df["time_period"] != latest_period].copy()
    pred_df  = df[df["time_period"] == latest_period].copy()
    return df, train_df, pred_df


def stage2_alarm_type(row, prev_is_hot: bool, trend_info: dict):
    """Derive alarm type using current/previous hotspot status and trend direction."""
    if row["stage1_alarm"] == 0:
        return "No Alarm"

    # Current is an alarm
    direction = trend_info.get("trend", "no trend")
    p = trend_info.get("p", 1.0)

    if not prev_is_hot:
        # "New" if last period wasn't hot
        return "New"
    # If it was hot before, see direction
    if direction == "increasing" and p < 0.05:
        return "Intensifying"
    if direction == "decreasing" and p < 0.05:
        return "Diminishing"
    return "Persistent"


def compute_gi_for_period(period_df, value_col="crash_count", ngh_k=8, permutations=299):
    """Compute Gi* Z and p for a single time period DataFrame that contains
    at least columns ['h3_cell','crash_count','centroid_lat','centroid_lon'].
    """
    period_df = period_df.copy()
    if len(period_df) < 2:
        period_df["gi_star"] = 0.0
        period_df["gi_p"] = 1.0
        return period_df
    coords = period_df[["centroid_lat","centroid_lon"]].values
    Zs, p = safe_g_local(period_df[value_col].values, coords, k=min(ngh_k, max(1, len(period_df)-1)), permutations=permutations)
    period_df["gi_star"] = Zs
    period_df["gi_p"] = p
    return period_df


def aggregate_by_h3(df, res=8, freq="W"):
    """Bin points to H3 cells and aggregate by period."""
    out = df.copy()
    out["h3_cell"] = out.apply(lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], res), axis=1)
    out["time_period"] = to_period_str(out["crash_dt"], freq=freq)
    agg = (out.groupby(["h3_cell","time_period"])
              .size().reset_index(name="crash_count"))
    # Centroids by cell from all points in that cell
    cell_centers = (out.groupby("h3_cell")[["latitude","longitude"]].mean()
                    .rename(columns={"latitude":"centroid_lat","longitude":"centroid_lon"})
                    .reset_index())
    agg = agg.merge(cell_centers, on="h3_cell", how="left")
    return agg


def derive_prev_hot_flags(agg_with_gi):
    """Return a dict mapping (h3_cell, time_period) -> previous_is_hot_sig flag for convenience."""
    agg_with_gi = agg_with_gi.sort_values(["h3_cell","time_period"]).reset_index(drop=True)
    prev_flag = {}
    for cell, g in agg_with_gi.groupby("h3_cell"):
        prev_hot = 0
        for _, row in g.iterrows():
            key = (row["h3_cell"], row["time_period"])
            prev_flag[key] = int(prev_hot)
            prev_hot = int((row["gi_p"] < 0.05) and (row["gi_star"] > 0))
    return prev_flag


def compute_trends_per_cell(agg_df):
    """Compute Mann-Kendall trend on crash_count time series per cell."""
    trend_rows = []
    for cell, g in agg_df.sort_values("time_period").groupby("h3_cell"):
        info = mann_kendall_trend(g["crash_count"].values)
        trend_rows.append({
            "h3_cell": cell,
            "trend_direction": info["trend"],
            "trend_slope": info["slope"],
            "trend_p_value": info["p"],
        })
    return pd.DataFrame(trend_rows)


def run_pipeline(raw_df, res=8, freq="W", ngh_k_gi=8, ngh_k_feats=6, permutations=299, n_neighbors=5):
    """Full pipeline: aggregate->Gi* per period->features->stage1 KNN->stage2 labels->map/table."""

    # 1) Aggregate to H3 + time period
    agg = aggregate_by_h3(raw_df, res=res, freq=freq)

    # 2) Compute Gi* per time period (historical + latest)
    periods = sorted(agg["time_period"].unique())
    if len(periods) < 2:
        raise ValueError("Need at least two time periods in the data to build features and compare.")
    latest_period = periods[-1]

    gi_rows = []
    for tp, g in agg.groupby("time_period", sort=False):
        gi_rows.append(compute_gi_for_period(g, ngh_k=ngh_k_gi, permutations=permutations))
    agg_gi = pd.concat(gi_rows, ignore_index=True)

    # 3) Trends
    trends = compute_trends_per_cell(agg_gi)

    # 4) Features
    features_all, train_df, pred_df = build_features(agg_gi, latest_period, ngh_k=ngh_k_feats)

    # 5) Stage 1: KNN Alarm vs No Alarm using historical labels (Gi* significance as ground truth)
    train_df = train_df.copy()
    train_df["is_alarm_label"] = ((train_df["gi_p"] < 0.05) & (train_df["gi_star"] > 0)).astype(int)

    numeric_cols = [
        "crash_count","gi_star","gi_p","is_hot_sig",
        "crash_count_lag1","crash_count_lag2",
        "gi_star_lag1","gi_star_lag2",
        "is_hot_sig_lag1","is_hot_sig_lag2",
        "crash_rolling_mean_2","crash_rolling_sum_3",
        "gi_star_rolling_mean_2","hot_sig_rolling_sum_3",
        "ngh_crash_mean","ngh_gi_mean","ngh_hot_sig_sum",
        "ngh_crash_mean_lag1","ngh_gi_mean_lag1","ngh_hot_sig_sum_lag1",
    ]
    # Fill missing
    for c in numeric_cols:
        if c not in train_df.columns:
            train_df[c] = np.nan
        if c not in pred_df.columns:
            pred_df[c] = np.nan

    X_train = train_df[numeric_cols].fillna(0.0).values
    y_train = train_df["is_alarm_label"].values

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-ish tolerable
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])
    if len(np.unique(y_train)) < 2:
        # Edge case: no positive labels historically => fallback to rule
        st.warning("Historical data has no hotspots by Gi*; Stage 1 falls back to a simple rule (gi_p<0.05 & gi_star>0).")
        pred_df["stage1_alarm"] = ((pred_df["gi_p"] < 0.05) & (pred_df["gi_star"] > 0)).astype(int)
    else:
        clf.fit(X_train, y_train)
        pred_df = pred_df.copy()
        pred_df["stage1_alarm"] = clf.predict(pred_df[numeric_cols].fillna(0.0).values)

    # 6) Stage 2 rule-based alarm typing
    prev_hot_flags = derive_prev_hot_flags(agg_gi)
    # Attach previous-flag for the latest period rows
    pred_df["prev_is_hot"] = pred_df.apply(lambda r: prev_hot_flags.get((r["h3_cell"], r["time_period"]), 0), axis=1)
    # Attach trend info
    pred_df = pred_df.merge(trends, on="h3_cell", how="left")

    pred_df["alarm_type"] = pred_df.apply(
        lambda r: stage2_alarm_type(
            row=r,
            prev_is_hot=bool(r.get("prev_is_hot", 0)),
            trend_info={"trend": r.get("trend_direction", "no trend"), "p": r.get("trend_p_value", 1.0)},
        ),
        axis=1
    )

    # 7) Create output tables
    latest_gdf = make_hex_gdf(pred_df.rename(columns={"h3_cell":"h3_cell"}))

    # Summary
    summary = (pred_df.groupby("alarm_type")
               .agg(num_cells=("h3_cell","nunique"),
                    total_crashes=("crash_count","sum"),
                    avg_gi=("gi_star","mean"))
               .reset_index())

    return {
        "latest_period": latest_period,
        "agg_all": agg_gi,
        "features_all": features_all,
        "pred_latest": pred_df,
        "trends": trends,
        "summary": summary,
        "latest_gdf": latest_gdf,
    }


def folium_map_from_predictions(gdf, color_by="alarm_type"):
    """Make a simple Folium map coloring hexagons by alarm type."""
    if gdf.empty:
        return folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    center = [gdf["centroid_lat"].mean(), gdf["centroid_lon"].mean()]
    m = folium.Map(location=center, zoom_start=11)
    palette = {
        "New": "#ff7f0e",
        "Intensifying": "#d62728",
        "Persistent": "#9467bd",
        "Diminishing": "#2ca02c",
        "No Alarm": "#1f77b4",
        None: "#999999",
    }
    for _, row in gdf.iterrows():
        poly = row["geometry"]
        color = palette.get(row.get(color_by), "#999999")
        gj = folium.GeoJson(
            {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]},
            style_function=lambda x, color=color: {"color": color, "weight": 1, "fillOpacity": 0.4}
        )
        popup = folium.Popup(html=(
            f"<b>Cell:</b> {row['h3_cell']}<br>"
            f"<b>Alarm:</b> {row.get('alarm_type', 'NA')}<br>"
            f"<b>Crash count:</b> {row.get('crash_count', 'NA')}<br>"
            f"<b>Gi* z:</b> {row.get('gi_star', 'NA'):.3f} (p={row.get('gi_p', 'NA'):.3f})"
        ), max_width=300)
        gj.add_child(popup)
        gj.add_to(m)
    return m


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="NYCSBUS Hotspots & Alarms", layout="wide")
st.title("NYCSBUS Hotspots & Two‑Stage Alarm Detector")

with st.expander("What this app does", expanded=True):
    st.markdown("""
This app combines two workflows you explored:
1. **EDA with H3 and Gi\\*** to discover spatial hotspots per time period.
2. **Two‑stage alarm logic** to flag whether a cell should raise an alarm now and to categorize that alarm as *New*, *Intensifying*, *Persistent*, or *Diminishing* based on trend.

**How it works (short):**
- We hex-bin points with **H3** at a resolution you pick and aggregate by **week** or **month**.
- For each period, we compute **Local Getis‑Ord G\\*** (Gi*) using **k‑NN** spatial weights to score hotspots.
- We engineer **lags**, **rollings**, and **neighbor** features.
- **Stage 1** trains a **KNN classifier** on historical periods (Gi* significance as proxy labels) and predicts alarm vs. no‑alarm for the **latest** period.
- **Stage 2** assigns one of the alarm types using the **Mann‑Kendall** trend test and whether the previous period was a hotspot.
    """)

st.sidebar.header("Configuration")
res = st.sidebar.slider("H3 Resolution (hex size)", min_value=6, max_value=9, value=8, step=1,
                        help="Higher = smaller hexes.")
freq = st.sidebar.selectbox("Time Aggregation", options=["W (weekly)", "M (monthly)"], index=0)
freq = "W" if freq.startswith("W") else "M"
ngh_k_gi = st.sidebar.slider("Neighbors (Gi* KNN)", 4, 12, 8, 1)
ngh_k_feats = st.sidebar.slider("Neighbors (feature KNN)", 3, 10, 6, 1)
permutations = st.sidebar.slider("Gi* permutations", 99, 999, 299, 100)
n_neighbors = st.sidebar.slider("Stage 1 KNN n_neighbors", 3, 15, 5, 1)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload CSV (NYC crash-like data)", type=["csv"])
run = st.sidebar.button("Run pipeline", type="primary", use_container_width=True)

sample_hint = st.sidebar.checkbox("Treat 'date' column as already parsed (skip auto-detect)", value=False)

if run:
    if uploaded is None:
        st.error("Please upload a CSV file first.")
        st.stop()

    df = pd.read_csv(uploaded)

    # Detect columns
    if sample_hint and "date" in df.columns:
        date_col = "date"
    else:
        date_col = find_col(df, ["CRASH DATE","CRASH_DATE","date","datetime","timestamp"])

    lat_col = find_col(df, ["LATITUDE","latitude","lat","y"])
    lon_col = find_col(df, ["LONGITUDE","longitude","lon","lng","x"])

    if not date_col or not lat_col or not lon_col:
        st.error(f"Could not find required columns. Detected: date={date_col}, lat={lat_col}, lon={lon_col}")
        st.stop()

    # Basic cleaning
    df = df.rename(columns={date_col:"crash_dt", lat_col:"latitude", lon_col:"longitude"})
    df["crash_dt"] = pd.to_datetime(df["crash_dt"], errors="coerce")
    df = df.dropna(subset=["crash_dt","latitude","longitude"]).copy()

    # Optional: bound to NYC-ish extent to discard bad geos
    nyc_bounds = (40.35, 41.10, -74.35, -73.60)  # minLat, maxLat, minLon, maxLon approx
    df = df[(df["latitude"].between(nyc_bounds[0], nyc_bounds[1])) &
            (df["longitude"].between(nyc_bounds[2], nyc_bounds[3]))].copy()

    # Run
    try:
        out = run_pipeline(
            raw_df=df,
            res=res,
            freq=freq,
            ngh_k_gi=ngh_k_gi,
            ngh_k_feats=ngh_k_feats,
            permutations=permutations,
            n_neighbors=n_neighbors
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    # Results
    st.subheader(f"Results for latest period: **{out['latest_period']}**")
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("#### Summary by Alarm Type")
        st.dataframe(out["summary"], use_container_width=True)

        st.markdown("#### Download predictions (latest period)")
        pred_csv = out["pred_latest"].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=pred_csv, file_name="nycsbus_predictions_latest.csv", mime="text/csv")

    with col2:
        st.markdown("#### Hotspot Map (latest period)")
        m = folium_map_from_predictions(out["latest_gdf"])
        st_folium(m, width=None, height=600)

    with st.expander("Detailed tables (debug)"):
        st.markdown("**Aggregated with Gi\\***")
        st.dataframe(out["agg_all"].head(200), use_container_width=True)
        st.markdown("**Engineered features (all periods)**")
        st.dataframe(out["features_all"].head(200), use_container_width=True)
        st.markdown("**Per-cell trend summary**")
        st.dataframe(out["trends"].head(200), use_container_width=True)

else:
    st.info("Upload a CSV and press **Run pipeline** from the sidebar to begin.")
