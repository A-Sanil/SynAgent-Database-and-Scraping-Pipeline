"""
SynAgent Phase 1 Task 8 -- Reaction-Type Clustering
=====================================================
No reaction_type labels exist in the DB, so we use KMeans on PCA-reduced
DRFP fingerprints as a proxy for reaction type.

Pipeline:
  1. Load DRFP blobs from ord_full.db (same split as train_xgboost.py)
  2. PCA: 2048 -> 100 dims  (speeds up KMeans, reduces curse of dimensionality)
  3. Elbow + silhouette -> pick optimal k
  4. KMeans with optimal k; assign cluster labels to all reactions
  5. For each cluster (min 2000 reactions): train XGBoost with global best_params
  6. Compare per-cluster MAE / RMSE / R2 vs global model
  7. Four-panel dark-theme chart:
       TL: PCA 2D scatter colored by cluster
       TR: MAE global vs cluster-specific (bar)
       BL: Cluster sizes + yield distribution
       BR: R2 improvement per cluster
  8. Save cluster labels, per-cluster models, and metrics to USB

Usage:
    python "Agent tools/reaction_clusters.py"
    python "Agent tools/reaction_clusters.py" --db D:\\SynAgent\\db\\ord_full.db --k 20
    python "Agent tools/reaction_clusters.py" --k 0   # auto-pick k via silhouette

Requires:
    pip install xgboost scikit-learn numpy matplotlib scipy joblib
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# -- Config -------------------------------------------------------------------
DEFAULT_DB   = r"D:\SynAgent\db\ord_full.db"
MODEL_DIR    = r"D:\SynAgent\models"
BEST_PARAMS  = r"D:\SynAgent\models\best_params.json"
GLOBAL_MODEL = r"D:\SynAgent\models\xgb_yield_v1.json"

FP_SIZE      = 2048
PCA_DIMS     = 100       # DRFP -> PCA dims for KMeans
TOP_SOLVENTS = 60
TOP_CATALYSTS= 60
MIN_YIELD    = 5.0
MAX_YIELD    = 100.0
MIN_CLUSTER  = 2000      # minimum reactions to train a per-cluster model
TEST_FRAC    = 0.10
VAL_FRAC     = 0.10
RANDOM_SEED  = 42
DEFAULT_K    = 20        # number of clusters; 0 = auto via silhouette

# Dark-theme palette
BG     = "#0d1117"; CARD   = "#161b27"; BORDER = "#21293d"
GREEN  = "#4ade80"; PURPLE = "#c084fc"; TEAL   = "#2dd4bf"
INDIGO = "#818cf8"; YELLOW = "#facc15"; TEXT   = "#e2e8f0"
MUTED  = "#64748b"; RED    = "#f87171"

CLUSTER_PALETTE = [
    "#818cf8","#4ade80","#f97316","#c084fc","#2dd4bf",
    "#facc15","#f87171","#38bdf8","#a3e635","#fb923c",
    "#e879f9","#34d399","#fbbf24","#60a5fa","#f472b6",
    "#84cc16","#a78bfa","#22d3ee","#fb7185","#86efac",
]


# -- Data loading -------------------------------------------------------------

def load_data(db_path: str) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """
    Returns (fps, yields, reaction_ids, conditions) for reactions with DRFP + yield.
    fps shape: (n, FP_SIZE) float32.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cols = {row[1] for row in cur.execute("PRAGMA table_info(reactions)")}
    has_reactants = "reactant_smiles_json" in cols

    cur.execute(f"""
        SELECT reaction_id, reaction_fp, yield_percent,
               temperature_celsius, time_hours, solvent, catalyst,
               {'reactant_smiles_json' if has_reactants else 'NULL'}
        FROM reactions
        WHERE reaction_fp IS NOT NULL
          AND yield_percent BETWEEN {MIN_YIELD} AND {MAX_YIELD}
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  Loaded {len(rows):,} reactions from {Path(db_path).name}")

    fps, yields, ids, conditions = [], [], [], []
    for rid, blob, yld, temp, time_h, solvent, catalyst, reactants_json in rows:
        try:
            fp = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:FP_SIZE].astype(np.float32)
        except Exception:
            continue
        n_reactants = 0
        if reactants_json:
            try:
                n_reactants = len(json.loads(reactants_json))
            except Exception:
                pass
        fps.append(fp)
        yields.append(float(yld))
        ids.append(rid)
        conditions.append({
            "temp":        float(temp)   if temp     is not None else None,
            "time_h":      float(time_h) if time_h   is not None else None,
            "solvent":     (solvent   or "").strip().lower() or None,
            "catalyst":    (catalyst  or "").strip().lower() or None,
            "n_reactants": n_reactants,
            "source":      1,   # ORD only in this script
        })

    return np.stack(fps), np.array(yields, dtype=np.float32), ids, conditions


# -- Feature engineering -- identical to train_xgboost.py -------------------

def build_features(fps: np.ndarray, conditions: list[dict],
                   top_solvents: list[str], top_catalysts: list[str]) -> np.ndarray:
    """
    Builds the SAME 2174-feature vectors as train_xgboost.py so the global
    model predictions are directly comparable.
    DRFP(2048) + scalars(4) + solvent_onehot(61) + catalyst_onehot(61) = 2174
    """
    n = len(fps)
    scalars = np.zeros((n, 4), dtype=np.float32)
    solv_mat = np.zeros((n, TOP_SOLVENTS + 1), dtype=np.float32)
    cat_mat  = np.zeros((n, TOP_CATALYSTS + 1), dtype=np.float32)

    for i, c in enumerate(conditions):
        scalars[i, 0] = (c["temp"] - 25.0) / 100.0 if c["temp"]     is not None else 0.0
        scalars[i, 1] = math.log1p(c["time_h"])     if c["time_h"]   is not None else 0.0
        scalars[i, 2] = float(c["n_reactants"])
        scalars[i, 3] = float(c["source"])

        s = c["solvent"]
        if s in top_solvents:
            solv_mat[i, top_solvents.index(s)] = 1.0
        else:
            solv_mat[i, -1] = 1.0

        cat = c["catalyst"]
        if cat in top_catalysts:
            cat_mat[i, top_catalysts.index(cat)] = 1.0
        else:
            cat_mat[i, -1] = 1.0

    return np.concatenate([fps, scalars, solv_mat, cat_mat], axis=1)


# -- KMeans clustering on PCA-reduced DRFPs ----------------------------------

def find_optimal_k(X_pca: np.ndarray, k_range: range) -> int:
    """Silhouette score over k_range on a 10k subsample. Returns best k."""
    from sklearn.metrics import silhouette_score
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X_pca), min(10_000, len(X_pca)), replace=False)
    X_s = X_pca[idx]

    scores = {}
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED,
                             batch_size=4096, n_init=3)
        labels = km.fit_predict(X_s)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X_s, labels, sample_size=5000,
                                     random_state=RANDOM_SEED)
        print(f"    k={k:>3}  silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"  Best k = {best_k}  (silhouette={scores[best_k]:.4f})")
    return best_k


def cluster_reactions(fps: np.ndarray, k: int) -> tuple[np.ndarray, PCA, KMeans]:
    """PCA reduce then KMeans. Returns (labels, pca, kmeans)."""
    print(f"  PCA: {FP_SIZE} -> {PCA_DIMS} dims ...")
    pca = PCA(n_components=PCA_DIMS, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(fps)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained by {PCA_DIMS} PCs: {var_explained:.1%}")

    print(f"  KMeans k={k} (MiniBatchKMeans for speed) ...")
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED,
                         batch_size=8192, n_init=5, max_iter=300)
    labels = km.fit_predict(X_pca)
    sizes  = np.bincount(labels)
    print(f"  Cluster sizes: min={sizes.min():,}  max={sizes.max():,}  "
          f"mean={sizes.mean():.0f}  median={np.median(sizes):.0f}")
    return labels, pca, km, X_pca


# -- Per-cluster XGBoost training --------------------------------------------

def load_best_params(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        print(f"  Loaded best params from {path}")
        return p
    print("  best_params.json not found -- using defaults")
    return {}


def train_cluster_model(X_tr, y_tr, X_val, y_val, best_params: dict,
                        cluster_id: int) -> xgb.XGBRegressor:
    params = {
        "n_estimators":          3000,
        "objective":             "reg:squarederror",
        "tree_method":           "hist",
        "device":                "cpu",      # cluster models on CPU (fast enough)
        "early_stopping_rounds": 30,
        "random_state":          RANDOM_SEED,
        "verbosity":             0,
        "learning_rate":         0.05,
        "max_depth":             6,
        "subsample":             0.8,
        "colsample_bytree":      0.7,
    }
    params.update(best_params)
    params["device"] = "cpu"   # force CPU for parallel cluster training
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


# -- Chart generation --------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def generate_cluster_report(X_pca2: np.ndarray, labels: np.ndarray,
                             yields: np.ndarray, cluster_results: list[dict],
                             global_mae: float, global_r2: float,
                             out_path: str) -> None:
    """
    Four-panel chart:
      TL: PCA 2D scatter colored by cluster (10k subsample)
      TR: MAE -- global vs per-cluster model (bar)
      BL: Cluster size + mean yield per cluster
      BR: R2 -- global vs per-cluster model (bar)
    """
    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                            left=0.06, right=0.97, top=0.90, bottom=0.07)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_mae     = fig.add_subplot(gs[0, 1])
    ax_sizes   = fig.add_subplot(gs[1, 0])
    ax_r2      = fig.add_subplot(gs[1, 1])

    for ax in [ax_scatter, ax_mae, ax_sizes, ax_r2]:
        _style_ax(ax)

    k = len(cluster_results)
    colors = [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(k)]

    # -- TL: PCA 2D scatter --------------------------------------------------
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X_pca2), min(12_000, len(X_pca2)), replace=False)
    for ci in range(k):
        mask = labels[idx] == ci
        if mask.sum() == 0:
            continue
        ax_scatter.scatter(X_pca2[idx][mask, 0], X_pca2[idx][mask, 1],
                           c=colors[ci], s=2, alpha=0.5, linewidths=0,
                           label=f"C{ci}")
    ax_scatter.set_xlabel("PC 1", color=TEXT, fontsize=10)
    ax_scatter.set_ylabel("PC 2", color=TEXT, fontsize=10)
    ax_scatter.set_title("DRFP Reaction Space (PCA 2D)", color=TEXT,
                         fontsize=12, fontweight="bold", pad=8)
    # small cluster legend only if k <= 20
    if k <= 20:
        leg = ax_scatter.legend(fontsize=6, ncol=4, framealpha=0.3,
                                facecolor=CARD, labelcolor=TEXT,
                                markerscale=4, loc="upper right")

    # -- TR: MAE comparison --------------------------------------------------
    cids  = [r["cluster"] for r in cluster_results]
    maes  = [r["mae"] for r in cluster_results]
    bar_colors = [GREEN if m < global_mae else RED for m in maes]

    x_pos = np.arange(len(cids))
    bars  = ax_mae.bar(x_pos, maes, color=bar_colors, edgecolor=BG, linewidth=0.5)
    ax_mae.axhline(global_mae, color=YELLOW, lw=1.6, ls="--",
                   label=f"Global MAE {global_mae:.1f}%")
    ax_mae.axhline(10, color=INDIGO, lw=1.0, ls=":", alpha=0.6,
                   label="Target 10%")
    ax_mae.set_xticks(x_pos)
    ax_mae.set_xticklabels([f"C{c}" for c in cids], color=MUTED, fontsize=8,
                            rotation=45 if len(cids) > 12 else 0)
    ax_mae.set_ylabel("MAE (%)", color=TEXT, fontsize=10)
    ax_mae.set_title("MAE: Per-Cluster vs Global", color=TEXT,
                     fontsize=12, fontweight="bold", pad=8)
    ax_mae.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)

    # annotate improvement/regression
    for bar, m in zip(bars, maes):
        delta = m - global_mae
        ax_mae.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{delta:+.1f}", ha="center", va="bottom",
                    color=GREEN if delta < 0 else RED, fontsize=7)

    # -- BL: Cluster sizes + mean yield --------------------------------------
    sizes      = [r["n_test"] * 10 for r in cluster_results]   # approx total
    mean_yields = [r["mean_yield"] for r in cluster_results]

    ax2 = ax_sizes.twinx()
    ax2.set_facecolor(CARD)
    ax_sizes.bar(x_pos, sizes, color=INDIGO, alpha=0.6, edgecolor=BG, label="Size (est.)")
    ax2.plot(x_pos, mean_yields, "o-", color=YELLOW, lw=1.5, ms=5, label="Mean yield")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.set_ylabel("Mean yield (%)", color=YELLOW, fontsize=9)
    ax_sizes.set_xticks(x_pos)
    ax_sizes.set_xticklabels([f"C{c}" for c in cids], color=MUTED, fontsize=8,
                              rotation=45 if len(cids) > 12 else 0)
    ax_sizes.set_ylabel("Approx. cluster size", color=TEXT, fontsize=10)
    ax_sizes.set_title("Cluster Sizes & Mean Yield", color=TEXT,
                       fontsize=12, fontweight="bold", pad=8)
    lines1, labels1 = ax_sizes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_sizes.legend(lines1+lines2, labels1+labels2, fontsize=8,
                    framealpha=0.3, facecolor=CARD, labelcolor=TEXT)

    # -- BR: R2 comparison ---------------------------------------------------
    r2s = [r["r2"] for r in cluster_results]
    bar_colors_r2 = [GREEN if v > global_r2 else RED for v in r2s]
    bars_r2 = ax_r2.bar(x_pos, r2s, color=bar_colors_r2, edgecolor=BG, linewidth=0.5)
    ax_r2.axhline(global_r2, color=YELLOW, lw=1.6, ls="--",
                  label=f"Global R2 {global_r2:.3f}")
    ax_r2.axhline(0.6, color=INDIGO, lw=1.0, ls=":", alpha=0.6,
                  label="Target R2 0.6")
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels([f"C{c}" for c in cids], color=MUTED, fontsize=8,
                           rotation=45 if len(cids) > 12 else 0)
    ax_r2.set_ylabel("R2", color=TEXT, fontsize=10)
    ax_r2.set_title("R2: Per-Cluster vs Global", color=TEXT,
                    fontsize=12, fontweight="bold", pad=8)
    ax_r2.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)

    for bar, v in zip(bars_r2, r2s):
        delta = v - global_r2
        ax_r2.text(bar.get_x() + bar.get_width()/2,
                   max(bar.get_height(), 0) + 0.01,
                   f"{delta:+.3f}", ha="center", va="bottom",
                   color=GREEN if delta > 0 else RED, fontsize=7)

    # clusters beating global on BOTH metrics
    n_better = sum(1 for r in cluster_results if r["mae"] < global_mae and r["r2"] > global_r2)
    fig.suptitle(
        f"SynAgent Phase 1 Task 8 -- Reaction Cluster Analysis  "
        f"(k={k}, {n_better}/{len(cluster_results)} clusters beat global on MAE + R2)",
        color=TEXT, fontsize=14, fontweight="bold", y=0.96
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart saved -> {out_path}")


# -- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SynAgent Task 8 -- Reaction clustering")
    parser.add_argument("--db",         default=DEFAULT_DB)
    parser.add_argument("--model_dir",  default=MODEL_DIR)
    parser.add_argument("--k",          type=int, default=DEFAULT_K,
                        help="Number of clusters (0 = auto via silhouette, range 5-40)")
    args = parser.parse_args()

    cluster_dir  = os.path.join(args.model_dir, "clusters")
    chart_out    = os.path.join(args.model_dir, "cluster_analysis.png")
    results_out  = os.path.join(args.model_dir, "cluster_results.json")
    os.makedirs(cluster_dir, exist_ok=True)

    print("=" * 64)
    print("  SynAgent Phase 1 Task 8 -- Reaction Clustering")
    print(f"  k = {'auto' if args.k == 0 else args.k}")
    print("=" * 64)

    # -- Load encoder lists from global model meta ----------------------------
    meta_path = os.path.join(args.model_dir, "xgb_yield_v1_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    top_solvents  = meta["top_solvents"]
    top_catalysts = meta["top_catalysts"]
    print(f"\nLoaded encoders: {len(top_solvents)} solvents, {len(top_catalysts)} catalysts")

    # -- Load data ------------------------------------------------------------
    print("\nLoading data...")
    fps, yields, ids, conditions = load_data(args.db)
    X_feat = build_features(fps, conditions, top_solvents, top_catalysts)
    print(f"  Feature matrix: {X_feat.shape}  (matches global model input)")

    # -- Same split as train_xgboost.py (same seed = same test set) ----------
    idx_all = np.arange(len(fps))
    idx_tmp, idx_test = train_test_split(idx_all, test_size=TEST_FRAC, random_state=RANDOM_SEED)
    val_frac_adj = VAL_FRAC / (1 - TEST_FRAC)
    idx_train, idx_val = train_test_split(idx_tmp, test_size=val_frac_adj, random_state=RANDOM_SEED)

    # -- Load global model for baseline --------------------------------------
    print("\nLoading global model for baseline comparison...")
    global_model = xgb.XGBRegressor()
    global_model.load_model(GLOBAL_MODEL)
    global_preds = np.clip(global_model.predict(X_feat[idx_test]), 0.0, 100.0)
    global_mae   = mean_absolute_error(yields[idx_test], global_preds)
    global_r2    = r2_score(yields[idx_test], global_preds)
    print(f"  Global model -> MAE: {global_mae:.2f}%  R2: {global_r2:.4f}")

    # -- Cluster on DRFP fingerprints ----------------------------------------
    print("\nClustering DRFP fingerprints...")
    k = args.k
    if k == 0:
        print("  Auto-selecting k via silhouette (range 5-40)...")
        pca_pre = PCA(n_components=PCA_DIMS, random_state=RANDOM_SEED)
        X_pre   = pca_pre.fit_transform(fps)
        k = find_optimal_k(X_pre, range(5, 41, 5))

    labels, pca, km, X_pca = cluster_reactions(fps, k)

    # PCA to 2D for scatter plot
    pca2  = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca2 = pca2.fit_transform(fps)

    # -- Load best hyperparams -----------------------------------------------
    best_params = load_best_params(BEST_PARAMS)

    # -- Per-cluster models --------------------------------------------------
    print(f"\nTraining per-cluster models (min cluster size: {MIN_CLUSTER:,})...")
    cluster_results = []

    for ci in range(k):
        # Get indices for this cluster within each split
        cl_train = idx_train[labels[idx_train] == ci]
        cl_val   = idx_val  [labels[idx_val]   == ci]
        cl_test  = idx_test [labels[idx_test]  == ci]

        n_train = len(cl_train)
        n_test  = len(cl_test)

        if n_train < MIN_CLUSTER:
            print(f"  Cluster {ci:>2}: {n_train:>5} train samples -- SKIP (< {MIN_CLUSTER})")
            continue

        X_tr = X_feat[cl_train];  y_tr = yields[cl_train]
        X_va = X_feat[cl_val];    y_va = yields[cl_val]
        X_te = X_feat[cl_test];   y_te = yields[cl_test]

        if len(cl_val) < 10 or len(cl_test) < 10:
            print(f"  Cluster {ci:>2}: val/test too small -- SKIP")
            continue

        model_c = train_cluster_model(X_tr, y_tr, X_va, y_va, best_params, ci)

        preds_c  = np.clip(model_c.predict(X_te), 0.0, 100.0)
        mae_c    = mean_absolute_error(y_te, preds_c)
        rmse_c   = math.sqrt(mean_squared_error(y_te, preds_c))
        r2_c     = r2_score(y_te, preds_c)

        # Global model on same test subset
        g_preds  = np.clip(global_model.predict(X_te), 0.0, 100.0)
        mae_g    = mean_absolute_error(y_te, g_preds)
        r2_g     = r2_score(y_te, g_preds)

        better   = mae_c < mae_g
        print(f"  Cluster {ci:>2}: n_train={n_train:>6,}  n_test={n_test:>5,}  "
              f"MAE {mae_c:.1f}% (global {mae_g:.1f}%)  "
              f"R2 {r2_c:.3f} (global {r2_g:.3f})  "
              f"{'BETTER' if better else 'worse'}")

        # Save cluster model
        model_c.save_model(os.path.join(cluster_dir, f"xgb_cluster_{ci:02d}.json"))

        cluster_results.append({
            "cluster":    ci,
            "n_train":    int(n_train),
            "n_test":     int(n_test),
            "mae":        round(mae_c, 3),
            "rmse":       round(rmse_c, 3),
            "r2":         round(r2_c, 4),
            "mae_global_on_cluster": round(mae_g, 3),
            "r2_global_on_cluster":  round(r2_g, 4),
            "delta_mae":  round(mae_c - mae_g, 3),
            "delta_r2":   round(r2_c - r2_g, 4),
            "mean_yield": round(float(yields[labels == ci].mean()), 2),
        })

    # -- Summary --------------------------------------------------------------
    if cluster_results:
        n_better_mae = sum(1 for r in cluster_results if r["delta_mae"] < 0)
        n_better_r2  = sum(1 for r in cluster_results if r["delta_r2"]  > 0)
        n_better_both= sum(1 for r in cluster_results if r["delta_mae"] < 0 and r["delta_r2"] > 0)
        avg_delta_mae = np.mean([r["delta_mae"] for r in cluster_results])
        avg_delta_r2  = np.mean([r["delta_r2"]  for r in cluster_results])

        print(f"\n{'='*64}")
        print(f"  CLUSTER SUMMARY  ({len(cluster_results)} models trained)")
        print(f"{'='*64}")
        print(f"  Global baseline:   MAE {global_mae:.2f}%   R2 {global_r2:.4f}")
        print(f"  Avg delta MAE:     {avg_delta_mae:+.2f}%  ({n_better_mae}/{len(cluster_results)} clusters improved)")
        print(f"  Avg delta R2:      {avg_delta_r2:+.4f}  ({n_better_r2}/{len(cluster_results)} clusters improved)")
        print(f"  Beat global on both MAE + R2: {n_better_both}/{len(cluster_results)}")

        # Best and worst clusters
        best_c = min(cluster_results, key=lambda r: r["mae"])
        worst_c= max(cluster_results, key=lambda r: r["mae"])
        print(f"\n  Best cluster:   C{best_c['cluster']}  MAE {best_c['mae']:.1f}%  R2 {best_c['r2']:.3f}  (n={best_c['n_train']:,})")
        print(f"  Worst cluster:  C{worst_c['cluster']}  MAE {worst_c['mae']:.1f}%  R2 {worst_c['r2']:.3f}  (n={worst_c['n_train']:,})")

    # -- Save results JSON ----------------------------------------------------
    summary = {
        "k": k,
        "global_mae":  round(global_mae, 3),
        "global_r2":   round(global_r2, 4),
        "n_clusters_trained": len(cluster_results),
        "clusters": cluster_results,
    }
    with open(results_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results -> {results_out}")

    # -- Chart ----------------------------------------------------------------
    print("\nGenerating cluster analysis chart...")
    generate_cluster_report(
        X_pca2, labels, yields, cluster_results,
        global_mae, global_r2, chart_out
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
