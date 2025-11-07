# app.py — Clean, user-friendly PatrolIQ Streamlit app (single-file)
# - Uses MiniBatchKMeans (scalable), sampled DBSCAN, sampled Hierarchical with KNN propagation
# - Clean sidebar labels, no tip/info clutter, hierarchical sampling guard
# - Matplotlib visuals (Geographic, PCA, Temporal heatmap)
# - Uses data/clustered_geodata_with_labels.parquet or data/sample_500k.csv automatically

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="PatrolIQ — Clustering Dashboard", layout="wide")

# ---------- Configuration ----------
PARQUET_PATH = "data/clustered_geodata_with_labels.parquet"
CSV_FALLBACK = "data/sample_500k.csv"
REQUIRED_COORDS = ("latitude", "longitude")

# ---------- Utility functions ----------
def load_dataset():
    if os.path.exists(PARQUET_PATH):
        return pd.read_parquet(PARQUET_PATH), PARQUET_PATH
    if os.path.exists(CSV_FALLBACK):
        return pd.read_csv(CSV_FALLBACK), CSV_FALLBACK
    return None, None

def ensure_datetime_features(df):
    # find first date-like column and create year/month/day/hour if present
    candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if candidates:
        date_col = candidates[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["hour"] = df[date_col].dt.hour
    else:
        for c in ["year", "month", "day", "hour"]:
            if c not in df.columns:
                df[c] = np.nan
    return df

def safe_numeric(df, cols):
    """
    Coerce selected columns to numeric safely.
    Returns DataFrame aligned to df.index with numeric columns (NaN when impossible).
    (Note: does NOT drop rows — caller may drop NaNs as needed.)
    """
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            out[c] = np.nan
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(int).astype(float)
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
            continue
        # fallback: coerce strings to numeric where possible
        out[c] = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    return out

def run_pca(df, features):
    """
    Run PCA on df[features].
    Returns (Xp, pca_model, scaler, valid_index)
    """
    X_df = safe_numeric(df, features)
    X_df = X_df.dropna()
    n_samples, n_features = X_df.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError(f"PCA cannot run: no valid numeric data. shape={X_df.shape}")
    n_components = min(2, n_samples, n_features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df.values)
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(Xs)
    return Xp, pca, scaler, X_df.index

def sample_for_algo(df, algo_name, requested_sample):
    """
    Return a sampled DataFrame for algorithms that cannot scale to full dataset.
    For KMeans, we allow larger requested_sample (MiniBatchKMeans is used).
    For DBSCAN/Hierarchical, we cap to safe limits.
    """
    n = len(df)
    if algo_name == "KMeans":
        return df.sample(n=requested_sample, random_state=42) if requested_sample < n else df
    caps = {"DBSCAN": 5000, "Hierarchical": 5000}
    cap = caps.get(algo_name, requested_sample)
    if n <= cap:
        return df
    else:
        return df.sample(n=cap, random_state=42)

def safe_hierarchical_labels(df, features, n_clusters=6, linkage="ward", sample_cap=5000, knn_k=5):
    """
    Hierarchical clustering safely:
    - If valid rows <= sample_cap: run AgglomerativeClustering and return labels.
    - Else: sample sample_cap rows, run AgglomerativeClustering on sample,
      train KNN on sampled features->labels, then predict labels for all valid rows.
    Returns pd.Series aligned to df.index with labels (NaN where invalid).
    """
    X = safe_numeric(df, features)
    valid_mask = X.notna().all(axis=1)
    if valid_mask.sum() == 0:
        return pd.Series(index=df.index, dtype=float)
    X_valid = X.loc[valid_mask]
    n_valid = len(X_valid)
    if n_valid <= sample_cap:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_valid)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels_valid = model.fit_predict(Xs)
        out = pd.Series(index=df.index, dtype=float)
        out.loc[X_valid.index] = labels_valid
        return out
    # sample and cluster
    sample_df = X_valid.sample(n=sample_cap, random_state=42)
    scaler_sample = StandardScaler()
    Xs_sample = scaler_sample.fit_transform(sample_df)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    sample_labels = model.fit_predict(Xs_sample)
    knn = KNeighborsClassifier(n_neighbors=min(knn_k, sample_cap - 1))
    knn.fit(Xs_sample, sample_labels)
    try:
        Xs_all = scaler_sample.transform(X_valid)
    except Exception:
        scaler_all = StandardScaler()
        Xs_all = scaler_all.fit_transform(X_valid)
    predicted = knn.predict(Xs_all)
    out = pd.Series(index=df.index, dtype=float)
    out.loc[X_valid.index] = predicted
    return out

# ---------- Load dataset ----------
df, source = load_dataset()
if df is None:
    st.error("No dataset found. Place data/clustered_geodata_with_labels.parquet or data/sample_500k.csv and rerun.")
    st.stop()

# Normalize column names to lowercase
df.columns = [c.lower() for c in df.columns]

# Try to standardize lat/lon names
lat_candidates = [c for c in df.columns if c in {"latitude", "lat", "y", "y_coord", "y_coordinate"}]
lon_candidates = [c for c in df.columns if c in {"longitude", "lon", "lng", "x", "x_coord", "x_coordinate"}]
if lat_candidates:
    df = df.rename(columns={lat_candidates[0]: "latitude"})
if lon_candidates:
    df = df.rename(columns={lon_candidates[0]: "longitude"})

# If combined location like "41.88, -87.62" exists, parse it
if "latitude" not in df.columns or "longitude" not in df.columns:
    loc_cols = [c for c in df.columns if "location" in c]
    if loc_cols:
        parsed = df[loc_cols[0]].astype(str).str.extract(r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)')
        if parsed.shape[1] == 2:
            df["latitude"] = pd.to_numeric(parsed[0], errors="coerce")
            df["longitude"] = pd.to_numeric(parsed[1], errors="coerce")

# Validate coords
if "latitude" not in df.columns or "longitude" not in df.columns:
    st.error("Dataset must contain latitude and longitude columns (case-insensitive).")
    st.stop()

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

# Ensure datetime-derived features (year/month/day/hour) exist where possible
df = ensure_datetime_features(df)

# ---------- Sidebar / Controls (clean & user-friendly) ----------
st.sidebar.title("Controls")

clustering_method = st.sidebar.selectbox("Choose clustering method", ("K-Means", "DBSCAN", "Hierarchical"))

n_clusters = st.sidebar.slider("Number of groups", min_value=2, max_value=20, value=6)

eps = st.sidebar.slider("Neighborhood radius (deg; ~0.001 ≈ 100m)", min_value=0.0005, max_value=0.01, value=0.0015, step=0.0001)
min_samples = st.sidebar.slider("Minimum points to form a group", min_value=3, max_value=30, value=8)

sample_size = st.sidebar.slider("Max records to use for computation (reduces memory)", min_value=2000, max_value=min(500000, len(df)), value=min(20000, len(df)), step=1000)

st.sidebar.write("Features for clustering")
possible_defaults = ["latitude", "longitude", "year", "month", "day", "hour"]
available_features = [c for c in df.columns if (df[c].dtype.kind in "biufc") or c in possible_defaults]
defaults = [f for f in possible_defaults if f in df.columns]
features_selected = st.sidebar.multiselect("Pick features (defaults shown)", options=available_features, default=defaults)

st.sidebar.divider()
run_button = st.sidebar.button("Run analysis")

# Safety guard: Hierarchical heavy; sample automatically if dataset > 10k
if clustering_method == "Hierarchical" and len(df) > 10000:
    st.sidebar.warning("Hierarchical will sample 10,000 rows automatically to avoid memory issues.")
    df_proc = df.sample(n=min(10000, len(df)), random_state=42)
else:
    df_proc = df.sample(n=min(sample_size, len(df)), random_state=42)

# Minimal dataset info (single line)
st.markdown(f"**Data source:** `{os.path.basename(source)}` — Rows loaded: {len(df):,}")

if not run_button:
    st.info("Adjust controls in sidebar and click 'Run analysis'.")
    st.stop()

# ---------- Validate features ----------
if len(features_selected) == 0:
    st.error("Select at least one feature for clustering.")
    st.stop()

# Build numeric matrix for df_proc and corresponding valid mask
X_proc = safe_numeric(df_proc, features_selected)
valid_mask = X_proc.notna().all(axis=1)
if valid_mask.sum() == 0:
    st.error("No valid numeric rows for chosen features in the selected dataset/sample.")
    st.stop()

df_valid = df_proc.loc[valid_mask].copy()
X_valid = X_proc.loc[valid_mask]

# ---------- PCA (safe) ----------
try:
    Xp, pca_model, pca_scaler, pca_index = run_pca(df_valid, features_selected)
    explained = pca_model.explained_variance_ratio_.sum() * 100.0
except Exception:
    Xp = None
    explained = None

# ---------- Clustering (safe paths) ----------
labels_series = pd.Series(index=df_proc.index, dtype=float)

if clustering_method == "K-Means":
    # Fit MiniBatchKMeans on df_valid (df_proc may be sampled via sample_size or hierarchical guard)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_valid)
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42)
    lab = mbk.fit_predict(Xs)
    labels_series.loc[X_valid.index] = lab

elif clustering_method == "DBSCAN":
    # DBSCAN runs on sampled df_proc (cap enforced in sample_for_algo)
    sample_df = sample_for_algo(df_valid, "DBSCAN", requested_sample=5000)
    X_sample = safe_numeric(sample_df, features_selected).dropna()
    if len(X_sample) == 0:
        st.error("No valid rows for DBSCAN after selecting features.")
        st.stop()
    scaler = StandardScaler()
    Xs_sample = scaler.fit_transform(X_sample)
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    lab = db.fit_predict(Xs_sample)
    labels_series.loc[X_sample.index] = lab

else:  # Hierarchical
    labels_series = safe_hierarchical_labels(df_proc, features_selected, n_clusters=n_clusters, linkage="ward", sample_cap=min(10000, len(df_proc)), knn_k=5)

# Attach cluster labels into df_proc (and keep original df untouched)
df_proc["cluster_label"] = labels_series

# ---------- Plots (matplotlib) ----------
# Use a plotting sample to avoid rendering excessive points (but preserve cluster labels)
plot_df = df_proc.dropna(subset=["cluster_label"]).copy()
if len(plot_df) == 0:
    st.error("No labeled rows to plot.")
    st.stop()
if len(plot_df) > sample_size:
    plot_df = plot_df.sample(n=sample_size, random_state=42)

# Geographic scatter
fig_geo, ax_geo = plt.subplots(figsize=(10, 5))
sc = ax_geo.scatter(plot_df["longitude"], plot_df["latitude"],
                    c=plot_df["cluster_label"].fillna(-1), cmap="tab20", s=8, alpha=0.8)
ax_geo.set_xlabel("Longitude")
ax_geo.set_ylabel("Latitude")
ax_geo.set_title("Geographic clusters")
plt.colorbar(sc, ax=ax_geo, fraction=0.03, pad=0.02, label="Cluster")
st.pyplot(fig_geo, width="stretch")

# PCA scatter (handles 1D or 2D PCA results)
if Xp is not None:
    pca_df = pd.DataFrame(Xp, index=pca_index, columns=[f"pca{i+1}" for i in range(Xp.shape[1])])
    pca_plot_df = pca_df.reindex(plot_df.index).dropna()
    if not pca_plot_df.empty:
        fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
        labels_pca = plot_df.loc[pca_plot_df.index, "cluster_label"].fillna(-1)
        if pca_plot_df.shape[1] > 1:
            sc2 = ax_pca.scatter(pca_plot_df.iloc[:, 0], pca_plot_df.iloc[:, 1], c=labels_pca, cmap="tab20", s=8, alpha=0.8)
            ax_pca.set_xlabel("PCA 1")
            ax_pca.set_ylabel("PCA 2")
            ax_pca.set_title("PCA projection (2D)")
        else:
            sc2 = ax_pca.scatter(pca_plot_df.iloc[:, 0], np.zeros(len(pca_plot_df)), c=labels_pca, cmap="tab20", s=8, alpha=0.8)
            ax_pca.set_xlabel("PCA 1")
            ax_pca.set_yticks([])
            ax_pca.set_title("PCA projection (1D)")
        plt.colorbar(sc2, ax=ax_pca, fraction=0.03, pad=0.02, label="Cluster")
        st.pyplot(fig_pca, width="stretch")
    else:
        st.write("PCA could not be aligned with plotted rows (not enough overlap).")
else:
    st.write("PCA could not be computed for the selected features (not enough numeric data).")

# Temporal heatmap (hour x month) if available
if "hour" in df_proc.columns and "month" in df_proc.columns:
    try:
        heat = pd.crosstab(df_proc["hour"].fillna(-1).astype(int), df_proc["month"].fillna(-1).astype(int))
        hours = [h for h in range(0, 24) if h in heat.index]
        months = [m for m in range(1, 13) if m in heat.columns]
        heat2 = heat.reindex(index=hours, columns=months, fill_value=0)
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        im = ax_h.imshow(heat2.values, aspect="auto", origin="lower", cmap="plasma")
        ax_h.set_xlabel("Month")
        ax_h.set_ylabel("Hour")
        ax_h.set_xticks(np.arange(len(months)))
        ax_h.set_xticklabels(months)
        ax_h.set_yticks(np.arange(len(hours)))
        ax_h.set_yticklabels(hours)
        ax_h.set_title("Crime frequency (Hour × Month)")
        plt.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02, label="Count")
        st.pyplot(fig_h, width="stretch")
    except Exception:
        st.write("Temporal heatmap could not be generated for selected data.")

# ---------- Minimal summary ----------
st.markdown("---")
st.write(f"Displayed rows: {len(plot_df):,}  •  Unique clusters shown: {int(df_proc['cluster_label'].nunique())}")
if explained is not None:
    st.write(f"PCA explained variance (2 components if available): {explained:.2f}%")
