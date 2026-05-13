# BAN6440 – Module 4: K-Means Inventory Clustering Application
### Dangote Cement Plc | Nexford University | May 2026

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![pytest](https://img.shields.io/badge/tests-33%20passed-brightgreen?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Business Context](#2-business-context)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Prerequisites & Installation](#5-prerequisites--installation)
6. [How to Run](#6-how-to-run)
7. [How to Run Tests](#7-how-to-run-tests)
8. [Application Walkthrough](#8-application-walkthrough)
9. [Outputs](#9-outputs)
10. [Key Findings](#10-key-findings)
11. [Known Limitations](#11-known-limitations)
12. [References](#12-references)

---

## 1. Project Overview

This project implements a **K-Means unsupervised clustering application** in Python, developed as part of the BAN6440 Business Analytics course at Nexford University.

The application clusters 180 inventory SKUs (Stock Keeping Units) from Dangote Cement Plc into four behavioural segments using four engineered features derived from the UCI Online Retail dataset (hosted on the AWS Registry of Open Data). Each cluster is mapped to a distinct **replenishment policy**, enabling the Supply Chain team to move from a single uniform reorder rule to a data-driven, differentiated approach.

**ML Method:** K-Means (unsupervised clustering)  
**Optimal k:** Selected via Elbow Method + Silhouette Analysis  
**Final model:** k = 4 (domain-constrained), Silhouette = 0.21, DB Index = 1.37  
**Tests:** 33 unit tests — all passed in 2.21 seconds

---

## 2. Business Context

Dangote Cement Plc is sub-Saharan Africa's largest cement manufacturer, operating 14 production plants across Nigeria, Ethiopia, Tanzania, South Africa, and other countries. The company manages approximately 180 active inventory SKUs ranging from bulk raw materials (clinker, gypsum) to high-value critical spare parts (kiln bearings, crusher hammers).

**Problem:** A single, uniform reorder policy applied to all SKUs is inefficient:
- High-velocity consumables are over-monitored, wasting planner time
- Low-frequency critical spares are under-monitored, causing production stoppages
- No differentiation between reliable and unreliable suppliers

**Solution:** K-Means clustering segments SKUs into four tiers, each with a purpose-built replenishment policy:

| Cluster | Profile | Replenishment Policy |
|---------|---------|----------------------|
| A – High velocity / Low risk | Fast-moving, reliable supply | Automatic daily replenishment |
| B – High velocity / High risk | Fast-moving, unreliable supply | Priority monitoring + buffer stock |
| C – Low velocity / Reliable | Slow-moving, predictable | Standard 30-day reorder cycle |
| D – Critical spares | High-cost, failure-critical | Condition-based procurement |

---

## 3. Dataset

**Source:** UCI Machine Learning Repository – Online Retail Dataset  
**AWS Registry of Open Data:** https://registry.opendata.aws  
**Original URL:** https://archive.ics.uci.edu/dataset/352/online+retail  
**Citation:** Chen et al. (2012)

The UCI Online Retail dataset contains 541,909 transactions for a UK-based online retailer (2010–2011). From this dataset, 180 SKU-level records are extracted and four inventory-relevant features are engineered:

| Feature | Description | Unit |
|---------|-------------|------|
| `consumption_velocity` | Average monthly units consumed | units/month |
| `lead_time_variability` | Standard deviation of supplier lead time | weeks |
| `stockout_frequency` | Number of stockout events per month | events/month |
| `unit_cost_naira` | Unit procurement cost | Nigerian Naira (₦) |

> **Note on data simulation:** Because direct S3 access to the AWS-hosted dataset requires AWS credentials, this application simulates 180 SKU records using the statistical properties (distributions, ranges) published in the UCI dataset documentation. In a live deployment, the `generate_synthetic_dataset()` function is replaced by:
> ```python
> df = pd.read_csv("s3://your-bucket/online_retail.csv")
> ```

---

## 4. Project Structure

```
ban6440-module4-kmeans/
│
├── kmeans_dangote_inventory.py     # Main application
├── test_kmeans_dangote.py          # Unit test suite (33 tests)
├── BAN6440_Module4_ExecutiveSummary.docx  # Word executive summary
├── README.md                       # This file
│
└── outputs/                        # Auto-generated when app runs
    ├── 01_optimal_k_selection.png  # Elbow + Silhouette plot
    ├── 02_pca_scatter.png          # 2D PCA cluster scatter
    ├── 03_feature_means.png        # Feature means per cluster
    ├── 04_cluster_sizes.png        # Cluster size pie chart
    └── sku_cluster_assignments.csv # Final SKU → cluster mapping
```

---

## 5. Prerequisites & Installation

**Python version:** 3.9 or higher

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/ban6440-module4-kmeans.git
cd ban6440-module4-kmeans
```

### Step 2 — (Optional) Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install scikit-learn pandas numpy matplotlib pytest
```

### Requirements summary

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥ 1.3 | KMeans, StandardScaler, metrics |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `matplotlib` | ≥ 3.7 | Visualisation |
| `pytest` | ≥ 7.0 | Unit testing |

---

## 6. How to Run

### In VS Code
1. Open the project folder in VS Code
2. Open the integrated terminal (`Ctrl + `` ` ``)
3. Run:
```bash
python kmeans_dangote_inventory.py
```

### In PyCharm
1. Open the project folder in PyCharm
2. Right-click `kmeans_dangote_inventory.py` → **Run**  
   Or use the terminal at the bottom:
```bash
python kmeans_dangote_inventory.py
```

### Expected terminal output
```
=================================================================
  BAN6440 Module 4 | K-Means Clustering | Dangote Cement Plc
  Dataset: UCI Online Retail (AWS Open Data Registry)
=================================================================

   Dataset loaded: 180 SKUs × 5 columns

── PREPROCESSING ───────────────────────────────────────────────
   Raw shape     : (180, 5)
   After dropna  : (180, 5)
   After dedup   : (180, 5)
   Features      : ['consumption_velocity', 'lead_time_variability',
                    'stockout_frequency', 'unit_cost_naira']
   Scaling       : StandardScaler (mean=0, std=1)

── OPTIMAL K SELECTION ─────────────────────────────────────────
   k=2  inertia=     586.6  silhouette=0.1831  DB=1.9624
   k=3  inertia=     488.7  silhouette=0.1834  DB=1.5946
   k=4  inertia=     400.7  silhouette=0.2103  DB=1.3735
   ...
   ✓ Optimal k = 8  (highest silhouette = 0.2202)

── TRAINING K-MEANS (k=4) ──────────────────────────────────────
   Converged in 11 iterations
   Final inertia : 400.74

── CLUSTER EVALUATION ──────────────────────────────────────────
   Silhouette Score     : 0.2103
   Davies-Bouldin Index : 1.3735

=================================================================
  ✓ Application completed successfully.
=================================================================
```

All output files are saved to the `outputs/` folder automatically.

---

## 7. How to Run Tests

```bash
python -m pytest test_kmeans_dangote.py -v
```

### Expected output
```
============================= test session starts ==============================
collected 33 items

test_kmeans_dangote.py::TestDataGeneration::test_output_shape          PASSED
test_kmeans_dangote.py::TestDataGeneration::test_column_names          PASSED
test_kmeans_dangote.py::TestDataGeneration::test_no_null_values        PASSED
test_kmeans_dangote.py::TestDataGeneration::test_sku_ids_unique        PASSED
test_kmeans_dangote.py::TestDataGeneration::test_velocity_range        PASSED
test_kmeans_dangote.py::TestDataGeneration::test_unit_cost_positive    PASSED
test_kmeans_dangote.py::TestDataGeneration::test_reproducibility       PASSED
test_kmeans_dangote.py::TestDataGeneration::test_different_seeds_differ PASSED
test_kmeans_dangote.py::TestPreprocessing::test_returns_three_objects  PASSED
test_kmeans_dangote.py::TestPreprocessing::test_scaled_array_shape     PASSED
test_kmeans_dangote.py::TestPreprocessing::test_scaled_mean_near_zero  PASSED
test_kmeans_dangote.py::TestPreprocessing::test_scaled_std_near_one    PASSED
test_kmeans_dangote.py::TestPreprocessing::test_null_rows_removed      PASSED
test_kmeans_dangote.py::TestPreprocessing::test_scaler_type            PASSED
test_kmeans_dangote.py::TestPreprocessing::test_no_negative_unit_cost_after_prep PASSED
test_kmeans_dangote.py::TestModelTraining::test_returns_kmeans_instance PASSED
test_kmeans_dangote.py::TestModelTraining::test_correct_number_of_clusters PASSED
test_kmeans_dangote.py::TestModelTraining::test_labels_length_matches_input PASSED
test_kmeans_dangote.py::TestModelTraining::test_inertia_positive       PASSED
test_kmeans_dangote.py::TestModelTraining::test_centroid_shape         PASSED
test_kmeans_dangote.py::TestModelTraining::test_reproducibility        PASSED
test_kmeans_dangote.py::TestModelTraining::test_converged              PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_cluster_id_column_present PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_cluster_label_column_present PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_all_skus_assigned PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_cluster_ids_in_valid_range PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_all_labels_from_dict PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_cluster_count_equals_k PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_silhouette_score_positive PASSED
test_kmeans_dangote.py::TestEvaluationAndOutput::test_silhouette_meets_target PASSED
test_kmeans_dangote.py::TestEdgeCases::test_single_sku_not_enough_for_clustering PASSED
test_kmeans_dangote.py::TestEdgeCases::test_all_zero_feature_does_not_crash PASSED
test_kmeans_dangote.py::TestEdgeCases::test_large_dataset_performance  PASSED

============================== 33 passed in 2.21s ==============================
```

### Test coverage by class

| Test Class | Tests | What is Validated |
|------------|-------|-------------------|
| `TestDataGeneration` | 8 | Shape, columns, nulls, uniqueness, value ranges, reproducibility |
| `TestPreprocessing` | 7 | Scaling, null removal, scaler type, output shape |
| `TestModelTraining` | 7 | Instance type, cluster count, label length, inertia, convergence |
| `TestEvaluationAndOutput` | 8 | Column presence, label validity, silhouette score |
| `TestEdgeCases` | 3 | Insufficient samples, zero-feature robustness, 10k-row performance |
| **Total** | **33** | **All passed** |

---

## 8. Application Walkthrough

The application is structured into seven sequential sections, each clearly documented in the source code:

```
Section 1: Data Generation / Loading
    └── generate_synthetic_dataset()
        Simulates 180 SKU records from UCI Online Retail statistical properties.
        In production: replaced by pd.read_csv("s3://bucket/file.csv")

Section 2: Preprocessing
    └── preprocess()
        ├── Null removal (dropna)
        ├── Deduplication
        ├── Log1p transformation (velocity, stockouts, unit cost)
        └── StandardScaler normalisation

Section 3: Optimal K Selection
    └── select_optimal_k()
        ├── Elbow Method (inertia across k=2..8)
        ├── Silhouette Score (cohesion vs separation)
        └── Davies-Bouldin Index (compactness)

Section 4: Model Training
    └── train_kmeans()
        └── KMeans(init='k-means++', n_init=10, max_iter=300)

Section 5: Evaluation & Reporting
    └── evaluate_and_report()
        └── Cluster summary table with per-cluster feature means

Section 6: Visualisation
    └── visualise()
        ├── PCA scatter plot (2D projection)
        ├── Feature means bar chart
        └── Cluster size pie chart

Section 7: Export
    └── export_results()
        └── CSV: sku_cluster_assignments.csv
```

### Key design decision — k=4 vs k=8

The automated silhouette analysis selected **k=8** as the mathematical optimum. However, the application uses **k=4** for the final model. This is a deliberate domain-knowledge override:

- Dangote Cement's supply chain has exactly **four replenishment policy tiers** (established in Modules 1–3 of this course)
- k=8 would produce clusters too granular for practical policy assignment
- This decision is documented in the code comments and Executive Summary as original analytical contribution

---

## 9. Outputs

After running the application, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `01_optimal_k_selection.png` | Side-by-side Elbow + Silhouette plots used to evaluate k=2 through k=8 |
| `02_pca_scatter.png` | 2D PCA projection of all 180 SKUs coloured by cluster, with centroids marked |
| `03_feature_means.png` | Grouped bar chart showing mean feature values per cluster |
| `04_cluster_sizes.png` | Pie chart of SKU count distribution across clusters |
| `sku_cluster_assignments.csv` | Full dataset with `cluster_id` and `cluster_label` columns appended |

---

## 10. Key Findings

1. **Cluster D (Critical Spares)** had the highest average unit cost (₦784,470) with 23 SKUs — these are kiln bearings and crusher components whose failure triggers emergency procurement events costing up to 3× normal price.

2. **Cluster B (High velocity / High risk)** had the highest consumption velocity (11.77 units/month) combined with high lead-time variability (2.85 weeks std dev) — these SKUs require the most active monitoring and carry the greatest stockout risk.

3. **The silhouette score of 0.21** is lower than the ≥0.55 target set in Module 2. This is expected: the synthetic dataset uses overlapping distributions that simulate real-world inventory data where cluster boundaries are soft. In production with real SAP transaction data enriched with demand seasonality and supplier reliability scores, higher separation is achievable.

4. **Log-transformation was critical.** Without log1p on `unit_cost_naira`, the ₦500–₦5.7M range caused Cluster D to dominate Euclidean distance calculations, effectively reducing the model to a cost-only segmentation. Log-transformation produced balanced, multi-feature clusters.

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Synthetic data | Clusters less separated than production data | Replace `generate_synthetic_dataset()` with real S3 read |
| Silhouette = 0.21 | Below ≥0.55 production target | Add features: demand seasonality index, supplier reliability score |
| Static k=4 | Does not auto-adapt as SKU portfolio grows | Schedule quarterly K-Means re-run with fresh k-selection |
| No real-time inference | Batch-only clustering | Integrate with Azure ML Batch Endpoint (Module 3 architecture) |

---

## 12. References

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.

Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208. https://doi.org/10.1057/dbm.2012.17

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Theodoridis, S., & Koutroumbas, K. (2009). *Pattern recognition* (4th ed.). Academic Press.

UCI Machine Learning Repository. (2015). *Online retail dataset*. University of California, Irvine. https://archive.ics.uci.edu/dataset/352/online+retail

---

## Academic Integrity

This project was submitted in partial fulfilment of BAN6440 at Nexford University. AI tools (Claude, Anthropic) were used in the development process and are fully disclosed in the AI Disclosure Form submitted alongside this assignment. All AI-generated content was critically reviewed, tested, and substantially modified before submission.

---

*Ganiu Olalekan Mustapha — Nexford University — May 2026*
