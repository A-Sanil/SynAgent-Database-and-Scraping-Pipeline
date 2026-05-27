"""
SynAgent Phase 1 -- XGBoost Yield Predictor
============================================
Features:
  - DRFP 2048-bit reaction fingerprint  (pre-computed blob in DB)
  - temperature_celsius  (normalised)
  - time_hours           (log1p)
  - solvent              (one-hot, top-N)
  - catalyst             (one-hot, top-N)
  - n_reactants          (count)
  - source               (0 = patent, 1 = ord)

Pipeline:
  1. Load reactions with reaction_fp + yield from SQLite
  2. Build feature matrix
  3. 80 / 10 / 10  train / val / test split
  4. Optuna: 30 trials x 5-fold CV on 50k subsample  (--optuna flag)
  5. Final retrain on full train set, early-stop on val
  6. SplitConformalRegressor (mapie 1.x) calibrated on val
     -> 80 / 90 / 95 % coverage evaluated on test
  7. 4-panel dark-theme results chart + metrics JSON saved to USB

Usage (Windows, system Python):
    python "Agent tools/train_xgboost.py" --gpu
    python "Agent tools/train_xgboost.py" --gpu --optuna
    python "Agent tools/train_xgboost.py" --db D:\\SynAgent\\db\\ord_full.db --gpu --optuna
    python "Agent tools/train_xgboost.py" --db D:\\SynAgent\\db\\ord_full.db \\
        --extra_db D:\\SynAgent\\db\\patent_pipeline.db --gpu --optuna

Savio cluster:
    python train_xgboost.py \\
        --db /scratch/.../ord_full.db \\
        --model_dir /scratch/.../models \\
        --n_jobs 24 --gpu --optuna

Requires:
    pip install xgboost scikit-learn numpy optuna mapie matplotlib joblib
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # no display / headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# -- Config --------------------------------------------------------------------
DEFAULT_DB     = r"D:\SynAgent\db\ord_full.db"
PATENT_DB      = r"D:\SynAgent\db\patent_pipeline.db"
MODEL_DIR      = r"D:\SynAgent\models"

MIN_YIELD      = 5.0
MAX_YIELD      = 100.0
FP_SIZE        = 2048
TOP_SOLVENTS   = 60
TOP_CATALYSTS  = 60
TEST_FRAC      = 0.10
VAL_FRAC       = 0.10
RANDOM_SEED    = 42

OPTUNA_TRIALS  = 30
OPTUNA_FOLDS   = 5
OPTUNA_SUBSAMP = 50_000   # reactions per Optuna CV search (speed <-> quality)

CI_LEVELS      = [0.80, 0.90, 0.95]   # coverage levels for conformal prediction

# Dark-theme palette (matches SynAgent UI)
BG      = "#0d1117"
CARD    = "#161b27"
BORDER  = "#21293d"
GREEN   = "#4ade80"
PURPLE  = "#c084fc"
TEAL    = "#2dd4bf"
INDIGO  = "#818cf8"
YELLOW  = "#facc15"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"


# -- Data loading --------------------------------------------------------------

def load_db(db_path: str, source_label: int) -> list[dict]:
    """Load reactions with reaction_fp blob + yield + conditions from SQLite."""
    if not os.path.exists(db_path):
        print(f"  [SKIP] DB not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cols = {row[1] for row in cur.execute("PRAGMA table_info(reactions)")}
    has_reactants = "reactant_smiles_json" in cols

    cur.execute(f"""
        SELECT reaction_id,
               reaction_fp,
               yield_percent,
               temperature_celsius,
               time_hours,
               solvent,
               catalyst,
               {'reactant_smiles_json' if has_reactants else 'NULL'}
        FROM reactions
        WHERE reaction_fp IS NOT NULL
          AND yield_percent BETWEEN {MIN_YIELD} AND {MAX_YIELD}
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  Loaded {len(rows):,} rows from {Path(db_path).name}")

    out = []
    for row in rows:
        rid, fp_blob, yld, temp, time_h, solvent, catalyst, reactants_json = row
        try:
            fp = np.unpackbits(np.frombuffer(fp_blob, dtype=np.uint8))[:FP_SIZE].astype(np.float32)
        except Exception:
            continue
        n_reactants = 0
        if reactants_json:
            try:
                n_reactants = len(json.loads(reactants_json))
            except Exception:
                pass
        out.append({
            "id":          rid,
            "fp":          fp,
            "yield":       float(yld),
            "temp":        float(temp)   if temp   is not None else None,
            "time_h":      float(time_h) if time_h is not None else None,
            "solvent":     (solvent  or "").strip().lower() or None,
            "catalyst":    (catalyst or "").strip().lower() or None,
            "n_reactants": n_reactants,
            "source":      source_label,
        })
    return out


# -- Feature engineering -------------------------------------------------------

def build_encoders(data: list[dict]) -> tuple[list[str], list[str]]:
    solvent_counts  = Counter(r["solvent"]  for r in data if r["solvent"])
    catalyst_counts = Counter(r["catalyst"] for r in data if r["catalyst"])
    return (
        [s for s, _ in solvent_counts.most_common(TOP_SOLVENTS)],
        [c for c, _ in catalyst_counts.most_common(TOP_CATALYSTS)],
    )


def to_feature_vector(row: dict,
                      top_solvents: list[str],
                      top_catalysts: list[str]) -> np.ndarray:
    fp        = row["fp"]
    temp_norm = (row["temp"] - 25.0) / 100.0 if row["temp"]   is not None else 0.0
    time_log  = math.log1p(row["time_h"])     if row["time_h"] is not None else 0.0
    n_react   = float(row["n_reactants"])
    source    = float(row["source"])

    solv_vec  = np.zeros(TOP_SOLVENTS  + 1, dtype=np.float32)
    if row["solvent"] in top_solvents:
        solv_vec[top_solvents.index(row["solvent"])] = 1.0
    else:
        solv_vec[-1] = 1.0

    cat_vec   = np.zeros(TOP_CATALYSTS + 1, dtype=np.float32)
    if row["catalyst"] in top_catalysts:
        cat_vec[top_catalysts.index(row["catalyst"])] = 1.0
    else:
        cat_vec[-1] = 1.0

    scalars = np.array([temp_norm, time_log, n_react, source], dtype=np.float32)
    return np.concatenate([fp, scalars, solv_vec, cat_vec])


def build_matrix(data: list[dict],
                 top_solvents: list[str],
                 top_catalysts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([to_feature_vector(r, top_solvents, top_catalysts) for r in data])
    y = np.array([r["yield"] for r in data], dtype=np.float32)
    return X, y


# -- Optuna K-fold hyperparameter search ---------------------------------------

def run_optuna_kfold(X_train: np.ndarray, y_train: np.ndarray,
                     device: str, n_jobs: int) -> dict:
    """
    30-trial Optuna search using 5-fold CV on a 50k subsample.

    Subsample keeps each trial fast (~15-30 s on GPU) while giving honest
    cross-validated RMSE rather than a single split, which reduces luck in
    the hyperparameter selection.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.RandomState(RANDOM_SEED)
    n   = min(OPTUNA_SUBSAMP, len(X_train))
    idx = rng.choice(len(X_train), n, replace=False)
    X_s, y_s = X_train[idx], y_train[idx]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":          3000,
            "learning_rate":         trial.suggest_float("learning_rate",    0.01, 0.3,  log=True),
            "max_depth":             trial.suggest_int("max_depth",          4,    10),
            "subsample":             trial.suggest_float("subsample",        0.5,  1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.4,  1.0),
            "min_child_weight":      trial.suggest_int("min_child_weight",   1,    20),
            "gamma":                 trial.suggest_float("gamma",            0.0,  2.0),
            "reg_alpha":             trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
            "reg_lambda":            trial.suggest_float("reg_lambda",       1e-4, 10.0, log=True),
            "objective":             "reg:squarederror",
            "tree_method":           "hist",
            "device":                device,
            "early_stopping_rounds": 30,
            "random_state":          RANDOM_SEED,
            "verbosity":             0,
        }
        if device == "cpu":
            params["nthread"] = n_jobs

        kf     = KFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, va_idx in kf.split(X_s):
            m = xgb.XGBRegressor(**params)
            m.fit(X_s[tr_idx], y_s[tr_idx],
                  eval_set=[(X_s[va_idx], y_s[va_idx])],
                  verbose=False)
            preds  = m.predict(X_s[va_idx])
            scores.append(math.sqrt(mean_squared_error(y_s[va_idx], preds)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print(f"\n  Optuna best CV-RMSE : {study.best_value:.3f}%")
    print(f"  Best params         : {study.best_params}\n")
    return study.best_params


# -- Final model training ------------------------------------------------------

def train_final(X_tr: np.ndarray, y_tr: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                best_params: dict,
                device: str, n_jobs: int) -> xgb.XGBRegressor:
    """Retrain with best hyperparams on full train set, early-stop on val."""
    params = {
        "n_estimators":          10_000,
        "objective":             "reg:squarederror",
        "tree_method":           "hist",
        "device":                device,
        "early_stopping_rounds": 50,
        "random_state":          RANDOM_SEED,
        "verbosity":             1,
        # Optuna best params override defaults below:
        "learning_rate":         0.05,
        "max_depth":             7,
        "subsample":             0.8,
        "colsample_bytree":      0.6,
        "min_child_weight":      5,
        "gamma":                 0.1,
        "reg_alpha":             0.1,
        "reg_lambda":            1.0,
    }
    params.update(best_params)
    if device == "cpu":
        params["nthread"] = n_jobs

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    return model


# -- Conformal prediction (manual split conformal -- no MAPIE dependency) -------
#
# Split conformal prediction is 3 steps:
#   1. Calibrate: compute residuals |y - ?| on the held-out val set
#   2. For each coverage level ?, find the (1-?) quantile of those residuals
#   3. At test time: CI = [? - q, ? + q]  (symmetric, absolute conformity score)
#
# This gives honest finite-sample coverage guarantees identical to MAPIE's
# "naive" / "split" method, with zero extra dependencies.

def fit_conformal(model: xgb.XGBRegressor,
                  X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Calibrate conformal intervals on the val set.
    Returns a dict  {coverage_level: threshold}  e.g. {0.90: 18.4}.
    """
    preds_val = np.clip(model.predict(X_val), 0.0, 100.0)
    residuals = np.abs(y_val - preds_val)
    n = len(residuals)

    thresholds = {}
    for level in CI_LEVELS:            # [0.80, 0.90, 0.95]
        alpha   = 1.0 - level
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        thresholds[level] = float(np.quantile(residuals, q_level))

    return thresholds


def apply_conformal(model: xgb.XGBRegressor,
                    X_test: np.ndarray,
                    thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply calibrated conformal intervals to the test set.
    Returns (point_preds, intervals) where intervals has shape (n, 2, n_levels).
    """
    preds  = np.clip(model.predict(X_test), 0.0, 100.0)
    levels = sorted(thresholds.keys())          # [0.80, 0.90, 0.95]
    n      = len(preds)
    ivs    = np.zeros((n, 2, len(levels)), dtype=np.float32)

    for i, level in enumerate(levels):
        q = thresholds[level]
        ivs[:, 0, i] = np.clip(preds - q, 0.0, 100.0)   # lower
        ivs[:, 1, i] = np.clip(preds + q, 0.0, 100.0)   # upper

    return preds, ivs


# -- Results chart -------------------------------------------------------------

def generate_report(y_test: np.ndarray,
                    preds: np.ndarray,
                    intervals: np.ndarray,   # shape (n, 2, n_levels)
                    mae: float, rmse: float, r2: float,
                    out_path: str) -> None:
    """
    4-panel dark-theme figure:
      TL - Predicted vs Actual hexbin
      TR - Residual histogram
      BL - MAE per yield decile
      BR - CI coverage at 80 / 90 / 95 %

    Plus a headline metrics band across the bottom.
    """
    fig = plt.figure(figsize=(16, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2,
                            height_ratios=[1, 1, 0.18],
                            hspace=0.40, wspace=0.35,
                            left=0.07, right=0.96,
                            top=0.90, bottom=0.04)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_resid   = fig.add_subplot(gs[0, 1])
    ax_bucket  = fig.add_subplot(gs[1, 0])
    ax_ci      = fig.add_subplot(gs[1, 1])
    ax_formula = fig.add_subplot(gs[2, :])

    _style_ax(ax_scatter)
    _style_ax(ax_resid)
    _style_ax(ax_bucket)
    _style_ax(ax_ci)
    ax_formula.axis("off")

    # -- TL: Predicted vs Actual -----------------------------------------------
    hb = ax_scatter.hexbin(y_test, preds, gridsize=60, cmap="YlOrRd",
                           mincnt=1, bins="log")
    lims = (0, 105)
    ax_scatter.plot(lims, lims, "--", color=INDIGO, lw=1.2, alpha=0.7,
                    label="Perfect prediction")
    # Linear regression line
    m_coef, b_coef = np.polyfit(y_test, preds, 1)
    xs = np.linspace(0, 100, 100)
    ax_scatter.plot(xs, m_coef * xs + b_coef, "-", color=GREEN, lw=1.5, alpha=0.8,
                    label=f"Fit  y={m_coef:.3f}x+{b_coef:.1f}")
    ax_scatter.set_xlim(lims); ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel("Actual yield (%)", color=TEXT, fontsize=10)
    ax_scatter.set_ylabel("Predicted yield (%)", color=TEXT, fontsize=10)
    ax_scatter.set_title("Predicted vs Actual", color=TEXT, fontsize=12, fontweight="bold", pad=8)
    ax_scatter.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)
    cb = fig.colorbar(hb, ax=ax_scatter, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED, labelsize=7)
    cb.set_label("log??(count)", color=MUTED, fontsize=8)
    ax_scatter.annotate(f"R2 = {r2:.3f}",
                        xy=(0.04, 0.94), xycoords="axes fraction",
                        color=TEAL, fontsize=11, fontweight="bold")

    # -- TR: Residual histogram ------------------------------------------------
    residuals = preds - y_test
    bins      = np.linspace(-60, 60, 61)
    ax_resid.hist(residuals, bins=bins, color=INDIGO, alpha=0.75, edgecolor="none")

    # Normal fit overlay
    mu, sigma = residuals.mean(), residuals.std()
    xs_r = np.linspace(bins[0], bins[-1], 300)
    from scipy.stats import norm as sp_norm
    pdf = sp_norm.pdf(xs_r, mu, sigma) * len(residuals) * (bins[1] - bins[0])
    ax_resid.plot(xs_r, pdf, color=YELLOW, lw=1.8, label=f"Normal fit  ?={sigma:.1f}%")
    ax_resid.axvline(0,  color=GREEN,  lw=1.2, ls="--", alpha=0.7)
    ax_resid.axvline(mu, color=PURPLE, lw=1.2, ls=":",  alpha=0.9, label=f"Mean={mu:+.1f}%")
    ax_resid.set_xlabel("Residual (pred ? actual, %)", color=TEXT, fontsize=10)
    ax_resid.set_ylabel("Count", color=TEXT, fontsize=10)
    ax_resid.set_title("Residual Distribution", color=TEXT, fontsize=12, fontweight="bold", pad=8)
    ax_resid.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)
    ax_resid.annotate(f"MAE = {mae:.2f}%\nRMSE = {rmse:.2f}%",
                      xy=(0.66, 0.88), xycoords="axes fraction",
                      color=TEAL, fontsize=10, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD,
                                edgecolor=BORDER, alpha=0.85))

    # -- BL: MAE per yield decile ----------------------------------------------
    edges    = np.arange(0, 110, 10)
    labels   = [f"{lo}-{lo+10}" for lo in edges[:-1]]
    mae_vals = []
    ns       = []
    for lo in edges[:-1]:
        hi   = lo + 10
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            mae_vals.append(mean_absolute_error(y_test[mask], preds[mask]))
            ns.append(mask.sum())
        else:
            mae_vals.append(0.0)
            ns.append(0)

    colors_b = [GREEN if v <= 10 else YELLOW if v <= 15 else "#f87171"
                for v in mae_vals]
    bars = ax_bucket.bar(labels, mae_vals, color=colors_b, edgecolor=BG, linewidth=0.5)
    ax_bucket.axhline(mae, color=INDIGO, lw=1.4, ls="--", label=f"Overall MAE {mae:.1f}%")
    ax_bucket.axhline(10,  color=GREEN,  lw=1.0, ls=":",  alpha=0.5, label="Target 10%")
    for bar, n in zip(bars, ns):
        if n > 0:
            ax_bucket.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + 0.3,
                           f"n={n:,}", ha="center", va="bottom",
                           color=MUTED, fontsize=6)
    ax_bucket.set_xlabel("Yield range (%)", color=TEXT, fontsize=10)
    ax_bucket.set_ylabel("MAE (%)", color=TEXT, fontsize=10)
    ax_bucket.set_title("MAE per Yield Decile", color=TEXT, fontsize=12,
                        fontweight="bold", pad=8)
    ax_bucket.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)
    ax_bucket.tick_params(axis="x", rotation=35, labelsize=8, colors=MUTED)

    # -- BR: CI coverage at 80 / 90 / 95 % ------------------------------------
    targets = [int(cl * 100) for cl in CI_LEVELS]   # [80, 90, 95]
    actuals = []
    widths  = []
    for i, cl in enumerate(CI_LEVELS):
        lo_i = intervals[:, 0, i]
        hi_i = intervals[:, 1, i]
        cov  = float(np.mean((y_test >= lo_i) & (y_test <= hi_i)))
        actuals.append(cov * 100)
        widths.append(float(np.mean(hi_i - lo_i)))

    x_ci  = np.arange(len(targets))
    width = 0.35
    bars_t = ax_ci.bar(x_ci - width/2, targets, width,
                       color=MUTED, alpha=0.55, label="Target", edgecolor=BG)
    bars_a = ax_ci.bar(x_ci + width/2, actuals, width,
                       color=[GREEN if abs(a-t) <= 3 else YELLOW if abs(a-t) <= 7 else "#f87171"
                               for a, t in zip(actuals, targets)],
                       label="Actual", edgecolor=BG)

    for bar, w in zip(bars_a, widths):
        ax_ci.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.6,
                   f"+/-{w/2:.1f}%", ha="center", va="bottom",
                   color=TEXT, fontsize=8)

    ax_ci.set_xticks(x_ci)
    ax_ci.set_xticklabels([f"{t}% CI" for t in targets], color=MUTED, fontsize=10)
    ax_ci.set_ylim(0, 110)
    ax_ci.set_ylabel("Coverage (%)", color=TEXT, fontsize=10)
    ax_ci.set_title("Conformal CI Coverage vs Target", color=TEXT, fontsize=12,
                    fontweight="bold", pad=8)
    ax_ci.legend(fontsize=9, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)
    ax_ci.axhline(100, color=BORDER, lw=0.8, ls="--")

    # -- Bottom formula band ---------------------------------------------------
    ci90_actual = actuals[1]   # index 1 = 90%
    ci90_width  = widths[1]
    formula = (
        f"? = XGBoost(DRFP???? ? conditions)     "
        f"MAE = {mae:.2f}%   RMSE = {rmse:.2f}%   R2 = {r2:.3f}   "
        f"90% CI coverage = {ci90_actual:.1f}%   Mean CI width = +/-{ci90_width/2:.1f}%"
    )
    ax_formula.text(0.5, 0.55, formula,
                    transform=ax_formula.transAxes,
                    ha="center", va="center",
                    color=TEAL, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                              edgecolor=INDIGO, linewidth=1.5))

    # -- Main title ------------------------------------------------------------
    fig.suptitle("SynAgent Phase 1 -- XGBoost Yield Prediction Results",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.96)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart saved to : {out_path}")


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SynAgent Phase 1 XGBoost yield predictor")
    parser.add_argument("--db",        default=DEFAULT_DB,
                        help="Primary DB (ord_full.db)")
    parser.add_argument("--extra_db",  default=None,
                        help="Optional second DB (patent_pipeline.db)")
    parser.add_argument("--model_dir", default=MODEL_DIR,
                        help="Output directory (USB or local)")
    parser.add_argument("--n_jobs",    type=int, default=-1,
                        help="CPU threads for XGBoost (ignored on GPU)")
    parser.add_argument("--gpu",       action="store_true",
                        help="Use CUDA GPU")
    parser.add_argument("--optuna",    action="store_true",
                        help="Run 30-trial x 5-fold Optuna search first")
    parser.add_argument("--best_params", default=None,
                        help="Path to JSON file of known best params -- skips Optuna entirely")
    parser.add_argument("--skip_train", action="store_true",
                        help="Load existing model from model_dir instead of retraining")
    args = parser.parse_args()

    device = "cuda" if args.gpu else "cpu"

    model_out  = os.path.join(args.model_dir, "xgb_yield_v1.json")
    mapie_out  = os.path.join(args.model_dir, "xgb_yield_v1_mapie.pkl")
    meta_out   = os.path.join(args.model_dir, "xgb_yield_v1_meta.json")
    chart_out  = os.path.join(args.model_dir, "xgb_yield_v1_results.png")

    print("=" * 64)
    print("  SynAgent Phase 1 -- XGBoost Yield Predictor")
    print(f"  Device : {device.upper()}   Optuna : {args.optuna}   Skip train : {args.skip_train}")
    print("=" * 64)

    # -- Load ------------------------------------------------------------------
    print("\nLoading data...")
    data = load_db(args.db, source_label=1)          # 1 = ORD
    if args.extra_db:
        data += load_db(args.extra_db, source_label=0)   # 0 = patent
    if not data:
        print("No data loaded -- check DB path and that reaction_fp is populated.")
        return
    print(f"  Total reactions with yield + DRFP : {len(data):,}")

    # -- Encoders + feature matrix ---------------------------------------------
    print("\nBuilding encoders + feature matrix...")
    top_solvents, top_catalysts = build_encoders(data)
    print(f"  Top solvents: {len(top_solvents)}   Top catalysts: {len(top_catalysts)}")
    X, y = build_matrix(data, top_solvents, top_catalysts)
    n_features = X.shape[1]
    print(f"  Feature matrix shape : {X.shape}"
          f"  (DRFP {FP_SIZE} + {n_features - FP_SIZE} condition features)")

    # -- Split -----------------------------------------------------------------
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_SEED)
    val_frac_adj = VAL_FRAC / (1 - TEST_FRAC)
    X_train, X_val, y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_adj, random_state=RANDOM_SEED)
    print(f"\n  Train: {len(X_train):,}   Val: {len(X_val):,}   Test: {len(X_test):,}")

    # -- Load or train model ---------------------------------------------------
    best_params: dict = {}
    if args.skip_train:
        print(f"\nLoading saved model from {model_out} ...")
        model = xgb.XGBRegressor()
        model.load_model(model_out)
        print("  Model loaded OK")
    else:
        if args.best_params:
            with open(args.best_params) as f:
                best_params = json.load(f)
            print(f"\nUsing provided best params (Optuna skipped): {best_params}")
        elif args.optuna:
            print(f"\nRunning Optuna ({OPTUNA_TRIALS} trials x {OPTUNA_FOLDS}-fold CV"
                  f" on {min(OPTUNA_SUBSAMP, len(X_train)):,} subsample)...")
            best_params = run_optuna_kfold(X_train, y_train, device, args.n_jobs)

        print(f"\nTraining final XGBoost on {device.upper()}...")
        model = train_final(X_train, y_train, X_val, y_val,
                            best_params, device, args.n_jobs)

    # -- Point metrics on test set ---------------------------------------------
    preds = np.clip(model.predict(X_test), 0.0, 100.0)
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\n{'='*64}")
    print(f"  TEST SET RESULTS  (n = {len(y_test):,})")
    print(f"{'='*64}")
    print(f"  MAE  : {mae:.2f}%")
    print(f"  RMSE : {rmse:.2f}%")
    print(f"  R2   : {r2:.4f}")

    for label, lo, hi in [
        ("High yield  (70-100%)", 70, 100),
        ("Mid  yield  (30-70%)",  30,  70),
        ("Low  yield   (5-30%)",   5,  30),
    ]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            r_ = math.sqrt(mean_squared_error(y_test[mask], preds[mask]))
            a_ = mean_absolute_error(y_test[mask], preds[mask])
            print(f"  {label} (n={mask.sum():,}): RMSE {r_:.1f}%  MAE {a_:.1f}%")

    # -- Save model (skip if we loaded it from disk) ---------------------------
    os.makedirs(args.model_dir, exist_ok=True)
    if not args.skip_train:
        model.save_model(model_out)
        print(f"\n  Model  -> {model_out}  (saved)")
    else:
        print(f"\n  Model  -> {model_out}  (loaded, not re-saved)")

    # -- Conformal prediction (manual split conformal) -------------------------
    print("\nCalibrating conformal intervals on val set...")
    thresholds     = fit_conformal(model, X_val, y_val)
    _, intervals_test = apply_conformal(model, X_test, thresholds)

    print(f"\n  {'Level':>8}  {'Target':>8}  {'Actual':>8}  {'Mean width':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
    ci_summary = {}
    for i, cl in enumerate(CI_LEVELS):
        lo_i = intervals_test[:, 0, i]
        hi_i = intervals_test[:, 1, i]
        cov  = float(np.mean((y_test >= lo_i) & (y_test <= hi_i)))
        wid  = float(np.mean(hi_i - lo_i))
        mark = "OK" if abs(cov - cl) <= 0.05 else "FAIL"
        print(f"  {int(cl*100):>7}%  {int(cl*100):>7}%  {cov*100:>7.1f}%  {wid:>10.1f}%  {mark}")
        ci_summary[f"ci_{int(cl*100)}"] = {"target": cl, "actual": round(cov, 4),
                                            "mean_width": round(wid, 2),
                                            "threshold":  round(thresholds[cl], 3)}

    # Save conformal thresholds as JSON (simpler than pkl, human-readable)
    with open(mapie_out.replace(".pkl", ".json"), "w") as f:
        json.dump({"thresholds": {str(k): v for k, v in thresholds.items()},
                   "ci_levels": CI_LEVELS}, f, indent=2)

    # -- Save metadata ---------------------------------------------------------
    meta = {
        "top_solvents":  top_solvents,
        "top_catalysts": top_catalysts,
        "fp_size":       FP_SIZE,
        "min_yield":     MIN_YIELD,
        "max_yield":     MAX_YIELD,
        "n_features":    int(n_features),
        "n_train":       int(len(X_train)),
        "n_val":         int(len(X_val)),
        "n_test":        int(len(X_test)),
        "mae_test":      round(mae,  3),
        "rmse_test":     round(rmse, 3),
        "r2_test":       round(r2,   4),
        "best_params":   best_params,
        "ci_levels":     ci_summary,
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  CIs    -> {mapie_out.replace('.pkl', '.json')}  (saved)")
    print(f"  Meta   -> {meta_out}  (saved)")

    # -- Chart -----------------------------------------------------------------
    print("\nGenerating results chart...")
    try:
        from scipy.stats import norm   # ensure scipy available for residual fit
        generate_report(y_test, preds, intervals_test,
                        mae, rmse, r2, chart_out)
    except ImportError:
        print("  [SKIP] scipy not installed -- chart skipped. pip install scipy")

    print("\n  Done.")


if __name__ == "__main__":
    main()
