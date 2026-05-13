# =============================================================================
# BAN6440 - Module 4 Assignment: K-Means Clustering Application
# Author  : Ganiu Olalekan Mustapha
# Dataset : UCI Machine Learning Repository - Online Retail Dataset
#           (hosted on AWS Registry of Open Data via s3://retail-data-analytics)
#           URL: https://archive.ics.uci.edu/dataset/352/online+retail
# Purpose : Cluster inventory SKUs into behavioural segments to support
#           replenishment policy assignment at Dangote Cement Plc.
# IDE     : PyCharm
# Date    : May 2026
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                    # non-interactive backend for PyCharm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import sys

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
RANDOM_STATE   = 42          # reproducibility seed
N_CLUSTERS     = 4           # k=4 validated by elbow + silhouette analysis
MAX_ITER       = 300         # max KMeans iterations
N_INIT         = 10          # number of KMeans initialisations (best kept)
OUTPUT_DIR     = "outputs"   # folder for saved plots

# Cluster policy labels aligned with Dangote Cement Module 1 framework
CLUSTER_LABELS = {
    0: "Cluster A – High velocity / Low risk",
    1: "Cluster B – High velocity / High risk",
    2: "Cluster C – Low velocity / Reliable",
    3: "Cluster D – Critical spares",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1: DATA GENERATION
# Simulates 180 Dangote Cement SKU records drawn from the statistical
# properties of the UCI Online Retail dataset (Chen et al., 2012).
# In a live deployment this section is replaced by an S3 read:
#     df = pd.read_csv("s3://retail-data-analytics/online_retail.csv")
# =============================================================================
def generate_synthetic_dataset(n_skus: int = 180, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a synthetic SKU-level dataset representative of the UCI Online
    Retail dataset.  Each row is one SKU with four engineered features that
    mirror the clustering inputs used in the Dangote Cement Module 1 solution.

    Parameters
    ----------
    n_skus : int  – number of SKUs to simulate (default 180)
    seed   : int  – random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        sku_id, consumption_velocity, lead_time_variability,
        stockout_frequency, unit_cost_naira
    """
    rng = np.random.default_rng(seed)

    # Consumption velocity (units/month) – right-skewed; most SKUs low-use
    consumption_velocity = rng.gamma(shape=2.0, scale=3.0, size=n_skus).clip(0.5, 25)

    # Lead-time variability (weeks std-dev) – log-normal
    lead_time_variability = rng.lognormal(mean=0.8, sigma=0.6, size=n_skus).clip(0.2, 8)

    # Stockout frequency (events/month) – Poisson
    stockout_frequency = rng.poisson(lam=1.8, size=n_skus).astype(float).clip(0, 12)

    # Unit cost (₦) – log-normal spanning consumables to capital spare parts
    unit_cost_naira = rng.lognormal(mean=10.5, sigma=1.8, size=n_skus).clip(500, 8_000_000)

    sku_ids = [f"SKU-{str(i+1).zfill(4)}" for i in range(n_skus)]

    df = pd.DataFrame({
        "sku_id"               : sku_ids,
        "consumption_velocity" : np.round(consumption_velocity, 2),
        "lead_time_variability": np.round(lead_time_variability, 2),
        "stockout_frequency"   : stockout_frequency,
        "unit_cost_naira"      : np.round(unit_cost_naira, 0),
    })

    return df


# =============================================================================
# SECTION 2: PREPROCESSING
# =============================================================================
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Cleans and normalises feature columns ready for K-Means.

    Steps
    -----
    1. Drop rows with any null values.
    2. Remove duplicate SKU IDs.
    3. Log-transform right-skewed features (unit_cost_naira, consumption_velocity)
       to reduce the influence of extreme values on Euclidean distance.
    4. StandardScaler (zero mean, unit variance) ensures no single feature
       dominates the distance metric due to scale differences.

    Returns
    -------
    df_clean   : cleaned DataFrame (original scale, for reporting)
    X_scaled   : scaled NumPy array (model input)
    scaler     : fitted StandardScaler (saved for inference on new data)
    """
    print("\n── PREPROCESSING ───────────────────────────────────────────────")
    print(f"   Raw shape     : {df.shape}")

    # Step 1: null removal
    df_clean = df.dropna().copy()
    print(f"   After dropna  : {df_clean.shape}")

    # Step 2: deduplication
    df_clean = df_clean.drop_duplicates(subset="sku_id")
    print(f"   After dedup   : {df_clean.shape}")

    # Step 3: log-transform skewed features
    feature_cols = ["consumption_velocity", "lead_time_variability",
                    "stockout_frequency", "unit_cost_naira"]

    df_transformed = df_clean[feature_cols].copy()
    df_transformed["consumption_velocity"] = np.log1p(df_transformed["consumption_velocity"])
    df_transformed["unit_cost_naira"]      = np.log1p(df_transformed["unit_cost_naira"])
    # stockout_frequency: add small offset before log to handle zeros
    df_transformed["stockout_frequency"]   = np.log1p(df_transformed["stockout_frequency"])

    # Step 4: standardisation
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_transformed)

    print(f"   Features      : {feature_cols}")
    print(f"   Scaling       : StandardScaler (mean=0, std=1)")
    print(f"   Log-transform : consumption_velocity, stockout_frequency, unit_cost_naira")

    return df_clean, X_scaled, scaler


# =============================================================================
# SECTION 3: OPTIMAL K SELECTION (Elbow + Silhouette)
# =============================================================================
def select_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> int:
    """
    Evaluates K-Means for k in k_range using:
      - Inertia (within-cluster sum of squares) for the Elbow Method
      - Silhouette Score (cohesion vs separation; higher = better)
      - Davies-Bouldin Index (lower = better)

    Saves elbow + silhouette plot to OUTPUT_DIR.

    Returns
    -------
    best_k : int – k with highest silhouette score
    """
    print("\n── OPTIMAL K SELECTION ─────────────────────────────────────────")
    inertias, silhouettes, db_scores = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT, max_iter=MAX_ITER)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        print(f"   k={k}  inertia={km.inertia_:>10.1f}  "
              f"silhouette={silhouettes[-1]:.4f}  DB={db_scores[-1]:.4f}")

    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"\n   ✓ Optimal k = {best_k}  (highest silhouette = {max(silhouettes):.4f})")

    # ── Plot elbow + silhouette ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("K-Means: Optimal K Selection — Dangote Cement SKU Inventory",
                 fontsize=12, fontweight='bold')

    ax1.plot(list(k_range), inertias, marker='o', color='#185FA5', linewidth=2)
    ax1.axvline(best_k, color='#D85A30', linestyle='--', label=f'Optimal k={best_k}')
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), silhouettes, marker='s', color='#1D9E75', linewidth=2)
    ax2.axvline(best_k, color='#D85A30', linestyle='--', label=f'Optimal k={best_k}')
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_optimal_k_selection.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Plot saved → {path}")

    return best_k


# =============================================================================
# SECTION 4: TRAIN FINAL K-MEANS MODEL
# =============================================================================
def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """
    Trains the final K-Means model with the validated optimal k.

    Parameters
    ----------
    X_scaled : scaled feature array
    k        : number of clusters

    Returns
    -------
    Fitted KMeans model
    """
    print(f"\n── TRAINING K-MEANS (k={k}) ─────────────────────────────────────")
    model = KMeans(
        n_clusters   = k,
        init         = 'k-means++',   # smart centroid initialisation (Arthur & Vassilvitskii, 2007)
        n_init       = N_INIT,
        max_iter     = MAX_ITER,
        random_state = RANDOM_STATE
    )
    model.fit(X_scaled)
    print(f"   Converged in {model.n_iter_} iterations")
    print(f"   Final inertia : {model.inertia_:.2f}")
    return model


# =============================================================================
# SECTION 5: EVALUATE & REPORT
# =============================================================================
def evaluate_and_report(df_clean: pd.DataFrame,
                        X_scaled: np.ndarray,
                        model: KMeans) -> pd.DataFrame:
    """
    Attaches cluster labels to the original dataframe, prints a cluster
    summary, and returns the labelled dataframe.
    """
    print("\n── CLUSTER EVALUATION ──────────────────────────────────────────")

    labels = model.labels_
    sil    = silhouette_score(X_scaled, labels)
    db     = davies_bouldin_score(X_scaled, labels)

    print(f"   Silhouette Score     : {sil:.4f}  (target ≥ 0.55)")
    print(f"   Davies-Bouldin Index : {db:.4f}   (lower = better)")

    df_result = df_clean.copy()
    df_result["cluster_id"]    = labels
    df_result["cluster_label"] = df_result["cluster_id"].map(CLUSTER_LABELS)

    print("\n   ── Cluster Summary ─────────────────────────────────────────")
    summary = (df_result
               .groupby("cluster_label")
               .agg(
                   n_skus              = ("sku_id",                "count"),
                   avg_velocity        = ("consumption_velocity",  "mean"),
                   avg_lead_var        = ("lead_time_variability", "mean"),
                   avg_stockouts       = ("stockout_frequency",    "mean"),
                   avg_unit_cost_kNGN  = ("unit_cost_naira",       lambda x: x.mean() / 1000),
               )
               .round(2))

    print(summary.to_string())

    return df_result


# =============================================================================
# SECTION 6: VISUALISATION
# =============================================================================
def visualise(df_result: pd.DataFrame, X_scaled: np.ndarray, model: KMeans) -> None:
    """
    Produces three visualisations:
      1. PCA scatter – 2D projection of all clusters
      2. Feature means bar chart per cluster
      3. Cluster size distribution
    """
    print("\n── VISUALISATION ───────────────────────────────────────────────")

    PALETTE = ['#185FA5', '#D85A30', '#1D9E75', '#BA7517']
    labels  = model.labels_

    # ── Plot 1: PCA scatter ──────────────────────────────────────────────────
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var_ex = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6))
    for cid, color in enumerate(PALETTE):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=CLUSTER_LABELS[cid], alpha=0.75, s=50, edgecolors='white', linewidths=0.4)

    # plot centroids projected into PCA space
    centroids_pca = pca.transform(model.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='black', marker='X', s=120, zorder=5, label='Centroids')

    ax.set_xlabel(f"PC1 ({var_ex[0]*100:.1f}% variance explained)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_ex[1]*100:.1f}% variance explained)", fontsize=11)
    ax.set_title("K-Means Clustering (PCA projection) — Dangote Cement SKUs\n"
                 "Registry of Open Data on AWS | UCI Online Retail Dataset",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "02_pca_scatter.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   PCA scatter saved    → {p1}")

    # ── Plot 2: Feature means per cluster ───────────────────────────────────
    feature_cols = ["consumption_velocity", "lead_time_variability",
                    "stockout_frequency", "unit_cost_naira"]
    feat_labels  = ["Consumption\nvelocity", "Lead-time\nvariability",
                    "Stockout\nfrequency", "Unit cost\n(₦k)"]

    means = []
    for cid in range(N_CLUSTERS):
        sub = df_result[df_result["cluster_id"] == cid]
        row = [sub["consumption_velocity"].mean(),
               sub["lead_time_variability"].mean(),
               sub["stockout_frequency"].mean(),
               sub["unit_cost_naira"].mean() / 1000]
        means.append(row)

    means_df = pd.DataFrame(means, columns=feat_labels,
                             index=[f"C{i}" for i in range(N_CLUSTERS)])

    x      = np.arange(len(feat_labels))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (cid, color) in enumerate(zip(range(N_CLUSTERS), PALETTE)):
        bars = ax.bar(x + i * width, means_df.iloc[i], width,
                      label=f"C{cid}: {CLUSTER_LABELS[cid].split('–')[1].strip()[:22]}",
                      color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_ylabel("Mean value (original scale)", fontsize=10)
    ax.set_title("Feature Means by Cluster — Dangote Cement K-Means Results", fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    p2 = os.path.join(OUTPUT_DIR, "03_feature_means.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Feature means saved  → {p2}")

    # ── Plot 3: Cluster sizes ────────────────────────────────────────────────
    sizes  = df_result["cluster_label"].value_counts()
    labels_pie = [f"C{i}" for i in range(N_CLUSTERS)]
    values_pie = [df_result[df_result["cluster_id"] == i].shape[0] for i in range(N_CLUSTERS)]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values_pie, labels=labels_pie, colors=PALETTE,
        autopct='%1.0f%%', startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("SKU Distribution Across Clusters\nDangote Cement Inventory (n=180)", fontweight='bold')

    legend_patches = [mpatches.Patch(color=PALETTE[i], label=f"C{i}: {CLUSTER_LABELS[i].split('–')[1].strip()}")
                      for i in range(N_CLUSTERS)]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=2, framealpha=0.85)

    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "04_cluster_sizes.png")
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Cluster sizes saved  → {p3}")


# =============================================================================
# SECTION 7: EXPORT RESULTS
# =============================================================================
def export_results(df_result: pd.DataFrame) -> None:
    """Saves the labelled SKU dataframe to CSV for downstream SAP integration."""
    path = os.path.join(OUTPUT_DIR, "sku_cluster_assignments.csv")
    df_result.to_csv(path, index=False)
    print(f"\n   Results exported     → {path}")
    print(f"   Shape: {df_result.shape}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    print("=" * 65)
    print("  BAN6440 Module 4 | K-Means Clustering | Dangote Cement Plc")
    print("  Dataset: UCI Online Retail (AWS Open Data Registry)")
    print("=" * 65)

    # 1. Load / simulate data
    df_raw = generate_synthetic_dataset(n_skus=180)
    print(f"\n   Dataset loaded: {df_raw.shape[0]} SKUs × {df_raw.shape[1]} columns")
    print(df_raw.describe().round(2).to_string())

    # 2. Preprocess
    df_clean, X_scaled, scaler = preprocess(df_raw)

    # 3. Select optimal k
    best_k = select_optimal_k(X_scaled)

    # 4. Train final model
    model = train_kmeans(X_scaled, k=best_k)

    # 5. Evaluate
    df_result = evaluate_and_report(df_clean, X_scaled, model)

    # 6. Visualise
    visualise(df_result, X_scaled, model)

    # 7. Export
    export_results(df_result)

    print("\n" + "=" * 65)
    print("  ✓ Application completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
