# app/clustering_analysis.py
"""
Clustering analysis script.
Saves:
 - outputs/pca_components.npy  (full dataset PCA projection)
 - outputs/kmeans_labels.npy   (sample labels)
 - outputs/birch_labels.npy
 - outputs/dbscan_labels.npy
 - data/clustered_geodata_with_labels.parquet
"""

import os
import sys
import traceback
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, DBSCAN, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PARQUET_IN = "data/preprocessed.parquet"
PARQUET_OUT = "data/clustered_geodata_with_labels.parquet"
OUTPUT_DIR = "outputs"

PCA_N_COMPONENTS = 6           
PCA_SAMPLE_N = 50000            

KMEANS_K = 8
MBK_BATCH_SIZE = 4096
MBK_MAX_ITERS = 200

DO_DBSCAN = True               
DBSCAN_EPS = 0.01
DBSCAN_MIN_SAMPLES = 5

DO_BIRCH = True

USE_UMAP = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    print(">>> Loading parquet:", path)
    df = pd.read_parquet(path)
    print(">>> Loaded dataframe shape:", df.shape)
    return df

def ensure_numeric_columns(df, needed_numeric=None):
    """
    Ensure numeric columns exist. If needed_numeric is provided, try to coerce them.
    Returns df and list of numeric cols present.
    """
    if needed_numeric is None:
        needed_numeric = ['latitude', 'longitude', 'year', 'month', 'day', 'hour']

    cols_lower = {c.lower(): c for c in df.columns}
    coerced = []
    for cand in needed_numeric:
        if cand in cols_lower:
            col_name = cols_lower[cand]
            if not pd.api.types.is_numeric_dtype(df[col_name]):
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                coerced.append(col_name)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(">>> Numeric columns detected:", numeric_cols)
    return df, numeric_cols

def scale_and_pca(X, n_components=PCA_N_COMPONENTS, sample_n=PCA_SAMPLE_N):
    """
    Fit PCA on a sample and transform full dataset. Returns X_pca_full (n_rows x n_components)
    and explained variance ratios.
    """
    n_samples, n_features = X.shape
    max_comp = min(n_samples, n_features)
    n_components = max(1, min(n_components, max_comp))
    print(f">>> PCA: will use n_components={n_components} (clamped) on data shape {X.shape}")

    sample_n = min(sample_n, n_samples)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_samples, size=sample_n, replace=False) if sample_n < n_samples else np.arange(n_samples)

    X_sample = X[sample_idx, :]

    scaler = StandardScaler()
    print(">>> Scaling numeric features...")
    X_sample_scaled = scaler.fit_transform(X_sample)
    X_full_scaled = scaler.transform(X)  

    print(">>> Fitting PCA on sample...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_sample_scaled)
    print(">>> PCA explained variance ratio:", pca.explained_variance_ratio_)

    print(">>> Transforming full dataset with PCA...")
    X_pca_full = pca.transform(X_full_scaled)
    return X_pca_full, pca.explained_variance_ratio_

def run_kmeans(X_sample, k=KMEANS_K):
    print(f">>> Running MiniBatchKMeans (k={k}) on sample data shape {X_sample.shape}")
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=MBK_BATCH_SIZE, random_state=42, max_iter=MBK_MAX_ITERS)
    labels = mbk.fit_predict(X_sample)
    return labels, mbk

def run_birch(X_sample, threshold=0.5, branching_factor=50):
    print(">>> Running Birch on sample...")
    birch = Birch(threshold=threshold, branching_factor=branching_factor)
    labels = birch.fit_predict(X_sample)
    return labels, birch

def run_dbscan(X_sample, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    print(f">>> Running DBSCAN on sample with eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_sample)
    return labels, db

def evaluate_labels(X_sample, labels, metric_names=True):
    """
    Compute clustering metrics where applicable.
    Returns dict of metrics (silhouette, davies, calinski) when valid.
    """
    metrics = {}
    unique_labels = set(labels)
    num_clusters = len([l for l in unique_labels if l != -1]) 
    if num_clusters >= 2:
        try:
            metrics['silhouette'] = float(silhouette_score(X_sample, labels))
        except Exception as e:
            metrics['silhouette'] = None
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(X_sample, labels))
        except Exception:
            metrics['davies_bouldin'] = None
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_sample, labels))
        except Exception:
            metrics['calinski_harabasz'] = None
    else:
        metrics['silhouette'] = None
        metrics['davies_bouldin'] = None
        metrics['calinski_harabasz'] = None
    return metrics

def save_outputs(df, pca_components_full, kmeans_labels_sample, birch_labels_sample, dbscan_labels_sample, sample_idx):
    pca_path = os.path.join(OUTPUT_DIR, "pca_components.npy")
    np.save(pca_path, pca_components_full)
    print(">>> Saved outputs/pca_components.npy with shape", pca_components_full.shape)

    if kmeans_labels_sample is not None:
        np.save(os.path.join(OUTPUT_DIR, "kmeans_labels_sample.npy"), kmeans_labels_sample)
        print(">>> Saved outputs/kmeans_labels_sample.npy with shape", kmeans_labels_sample.shape)
    if birch_labels_sample is not None:
        np.save(os.path.join(OUTPUT_DIR, "birch_labels_sample.npy"), birch_labels_sample)
        print(">>> Saved outputs/birch_labels_sample.npy with shape", birch_labels_sample.shape)
    if dbscan_labels_sample is not None:
        np.save(os.path.join(OUTPUT_DIR, "dbscan_labels_sample.npy"), dbscan_labels_sample)
        print(">>> Saved outputs/dbscan_labels_sample.npy with shape", dbscan_labels_sample.shape)

    df_out = df.copy()
    df_out['kmeans_cluster'] = -999
    df_out['birch_cluster'] = -999
    df_out['dbscan_cluster'] = -999

    if kmeans_labels_sample is not None:
        df_out.loc[sample_idx, 'kmeans_cluster'] = kmeans_labels_sample
    if birch_labels_sample is not None:
        df_out.loc[sample_idx, 'birch_cluster'] = birch_labels_sample
    if dbscan_labels_sample is not None:
        df_out.loc[sample_idx, 'dbscan_cluster'] = dbscan_labels_sample

    out_path = PARQUET_OUT
    df_out.to_parquet(out_path, index=False)
    print(">>> Saved labeled dataset:", out_path)

def safe_mlflow_log(run_name, params, pca_var, clustering_scores, artifacts_to_log):
    try:
        import mlflow
        print(">>> MLflow detected -- logging experiment run")
        mlflow.set_experiment("clustering_experiments")
        with mlflow.start_run(run_name=run_name):
            for k,v in (params or {}).items():
                mlflow.log_param(k, v)
            for i, var in enumerate(pca_var):
                mlflow.log_metric(f"pca_var_{i+1}", float(var))
            for alg, metrics in clustering_scores.items():
                if isinstance(metrics, dict):
                    for mname, mval in metrics.items():
                        if mval is not None:
                            mlflow.log_metric(f"{alg}_{mname}", float(mval))
            for a in artifacts_to_log or []:
                if os.path.exists(a):
                    mlflow.log_artifact(a)
    except Exception as e:
        print(">>> MLflow logging failed or MLflow not installed; continuing. Error:", e)

def main():
    try:
        df = load_data(PARQUET_IN)

        df, numeric_cols = ensure_numeric_columns(df, needed_numeric=['latitude','longitude','year','month','day','hour'])

        if not numeric_cols:
            raise ValueError(f"No numeric columns found for dim red. Found: {numeric_cols}")

        use_cols = [c for c in ['latitude','longitude','year','month','day','hour'] if c in [col.lower() for col in df.columns]]
        col_map = {col.lower(): col for col in df.columns}
        use_cols_actual = [col_map[c] for c in use_cols]

        print(">>> Using numeric columns for DR/clustering:", use_cols_actual)
        X_all = df[use_cols_actual].to_numpy(dtype=float)

        X_pca_full, pca_var = scale_and_pca(X_all, n_components=PCA_N_COMPONENTS, sample_n=PCA_SAMPLE_N)

        n_rows = X_all.shape[0]
        sample_n = min(PCA_SAMPLE_N, n_rows)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_rows, size=sample_n, replace=False) if sample_n < n_rows else np.arange(n_rows)
        X_sample = X_all[sample_idx, :]

        scaler = StandardScaler()
        X_sample_scaled = scaler.fit_transform(X_sample)

        kmeans_labels, kmeans_model = run_kmeans(X_sample_scaled, k=KMEANS_K)
        kmeans_metrics = evaluate_labels(X_sample_scaled, kmeans_labels)
        print(">>> KMeans scores:", kmeans_metrics)

        birch_labels, birch_model = None, None
        if DO_BIRCH:
            birch_labels, birch_model = run_birch(X_sample_scaled)
            birch_metrics = evaluate_labels(X_sample_scaled, birch_labels)
            print(">>> Birch scores:", birch_metrics)
        else:
            birch_metrics = {}

        dbscan_labels, dbscan_model = None, None
        if DO_DBSCAN:
            dbscan_labels, dbscan_model = run_dbscan(X_sample_scaled)
            dbscan_metrics = evaluate_labels(X_sample_scaled, dbscan_labels)
            print(">>> DBSCAN scores:", dbscan_metrics)
        else:
            dbscan_metrics = {}

        save_outputs(df, X_pca_full, kmeans_labels, birch_labels, dbscan_labels, sample_idx)

        clustering_scores = {
            'kmeans': kmeans_metrics,
            'birch': birch_metrics,
            'dbscan': dbscan_metrics
        }
        artifacts = [
            os.path.join(OUTPUT_DIR, "pca_components.npy"),
            os.path.join(OUTPUT_DIR, "kmeans_labels_sample.npy"),
            os.path.join(OUTPUT_DIR, "birch_labels_sample.npy"),
            os.path.join(OUTPUT_DIR, "dbscan_labels_sample.npy"),
            PARQUET_OUT
        ]
        mlflow_params = {
            "kmeans_k": KMEANS_K,
            "pca_n_components": PCA_N_COMPONENTS,
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES
        }
        run_name = f"clustering_run_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        safe_mlflow_log(run_name, mlflow_params, pca_var, clustering_scores, artifacts)

        print(">>> COMPLETE: Clustering analysis finished successfully.")

    except Exception as e:
        print(">>> ERROR during clustering_analysis execution:")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
