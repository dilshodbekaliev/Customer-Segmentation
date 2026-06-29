# Customer Segmentation — UK Online Retail II

Unsupervised ML pipeline on 1M+ real transactions that segments ~5,900 UK e-commerce customers into four actionable groups, with full revenue attribution for marketing budget allocation.

---

## Business Context

A UK online gift retailer needs its marketing budget allocated by customer type, not spread evenly. Using the **UCI Online Retail II dataset** (1,067,371 transactions, Dec 2009 – Dec 2011), this project answers: *"Who are our customers, how much are they worth, and what should we do differently for each group?"*

---

## Dataset

**Online Retail II (UCI)** — loaded directly from HuggingFace (~95 MB)

| Column | Description |
|---|---|
| Invoice | Transaction ID (`C` prefix = cancellation) |
| StockCode | Product code |
| Quantity | Units (negative = return) |
| InvoiceDate | Timestamp |
| Price | Unit price (GBP) |
| Customer ID | 22.8% missing — dropped in cleaning |
| Country | Customer country |

After cleaning (drop nulls, cancellations, zero-price rows, duplicates): **~805,000 rows · 5,288 unique customers**

---

## Pipeline

```
Raw data (1,067,371 rows)
  → Part 0: Data quality audit
  → Part 1: Clean transactions — drop nulls, cancellations, returns, dupes
  → Part 2: 7-feature customer table (RFM + Tenure + DistinctProducts + AOV + ReturnRate)
  → Part 3: Log transform skewed features + StandardScaler
  → Part 4: K selection — Silhouette / Davies-Bouldin / Calinski-Harabasz over K=2–8
  → Part 5: Algorithm comparison — K-Means vs GMM vs Ward vs DBSCAN (k-distance plot for eps tuning)
  → Part 6: Stability — multi-seed ARI across 5 random seeds
  → Part 7: Anomaly detection — IsolationForest at 2% contamination
  → Part 8: PCA (cumulative variance) + t-SNE visualisation
  → Part 9: Segment profiling — behaviour + revenue share + names
  → Part 10: CMO-facing executive report
```

---

## Features Engineered

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of distinct invoices |
| Monetary | Total spend (£) |
| Tenure | Days between first and last purchase |
| DistinctProducts | Unique products bought |
| AOV | Average order value (Monetary / Frequency) |
| ReturnRate | Share of raw transaction lines that were returns |

---

## Key Results

### K = 4 chosen

| K | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| 2 | 0.29 | 1.31 | **2604** |
| 3 | 0.30 | 1.24 | 2146 |
| **4** | **0.31** | **1.10** | 2023 |
| 5 | 0.25 | 1.17 | 2036 |

Two of three metrics peak at K=4. Calinski-Harabasz favours K=2 but 4 segments are more actionable for marketing — a deliberate trade-off documented in the notebook.

### Algorithm Comparison

| Algorithm | Clusters | Silhouette | ARI vs K-Means |
|---|---|---|---|
| **K-Means** | 4 | **0.31** | 1.00 |
| Ward | 4 | ~0.30 | ~0.47 |
| GMM | 4 | ~0.28 | ~0.24 |
| DBSCAN | ~1 + noise | poor | ~0.00 |

DBSCAN was tuned using a **k-distance plot** (k=14, 2× n_features). The plot shows no sharp elbow — confirming there are no density gaps in this RFM space, which is why DBSCAN collapses. K-Means wins.

### Stability

Mean pairwise ARI across 5 random seeds **≥ 0.90** — the segments are reproducible, not a lucky run.

### Anomalies

~118 customers (2%) flagged by IsolationForest. Characterised as suspected wholesalers or data errors based on extreme Monetary and DistinctProducts values. Excluded from final segments.

---

## Customer Segments

| Cluster | Name | Customers | % Customers | Avg Spend | Key Trait |
|---|---|---|---|---|---|
| 0 | New / Promising | 724 | 13.7% | £639 | Low tenure (77 days), recent, low return rate (1%) |
| 1 | Loyal High-Value | 2,015 | 38.1% | £6,191 | Highest frequency (12.6×), longest tenure (587 days) |
| 2 | High-Spending At-Risk | 155 | 2.9% | £1,425 (AOV £520) | Highest AOV but 34% return rate, 328 days inactive |
| 3 | Dormant Low-Value | 2,394 | 45.3% | £549 | 355 days inactive, low frequency (2.2×) |

**Headline:** Cluster 1 is 38% of customers but drives a disproportionate share of total revenue. Losing this segment is the primary business risk — retention spend here has the highest ROI.

---

## Visualisations

- Elbow + 3-metric K selection plots
- k-Distance plot for DBSCAN eps tuning
- Algorithm comparison table
- PCA cumulative variance curve + PC1/PC2 scatter
- t-SNE scatter (2,000-point sample)
- Segment profile table with revenue attribution

---

## Tech Stack

```
pandas · numpy · scikit-learn · matplotlib · seaborn
```

Key sklearn modules: `KMeans`, `GaussianMixture`, `AgglomerativeClustering`, `DBSCAN`, `IsolationForest`, `PCA`, `TSNE`, `StandardScaler`, `NearestNeighbors`

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

jupyter notebook Module5_lab_fixed.ipynb
```

Dataset loads automatically from HuggingFace on first run (~1 min). Set `STUDENT_ID` in the personal variant cell before running.

---

## Repo Structure

```
├── Module5_lab_fixed.ipynb   # Full pipeline — runs top to bottom
└── README.md
```

---

## Author

Dilshodbek — B.Sc. Economics with Data Science, Westminster International University in Tashkent
