"""
SynAgent Phase 1 -- XGBoost Yield Predictor (Extended Training v1 -> v2)
=========================================================================
Continues training from xgb_yield_v1.json for up to N additional trees
(early-stopping patience=100 on val RMSE).  Saves result as:
  xgb_yield_v2.json         -- updated model
  xgb_yield_v2_mapie.json   -- updated conformal thresholds
  xgb_yield_v2_meta.json    -- updated metrics
  xgb_yield_v2_results.png  -- updated chart

Encoders (top_solvents / top_catalysts) are loaded from xgb_yield_v1_meta.json
so feature vectors are byte-for-byte identical to the original training run.

Usage:
    python "Agent tools/extend_xgboost.py" --gpu
    python "Agent tools/extend_xgboost.py" --gpu --extra_rounds 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# -- Config (must match train_xgboost.py exactly) --------------------------------
DEFAULT_DB    = r"D:\SynAgent\db\ord_full.db"
MODEL_DIR     = r"D:\SynAgent\models"
FP_SIZE       = 2048
TOP_SOLVENTS  = 60
TOP_CATALYSTS = 60
TEST_FRAC     = 0.10
VAL_FRAC      = 0.10
RANDOM_SEED   = 42
MIN_YIELD     = 5.0
MAX_YIELD     = 100.0
CI_LEVELS     = [0.80, 0.90, 0.95]

# Dark-theme palette
BG     = "#0d1117"
CARD   = "#161b27"
BORDER = "#21293d"
GREEN  = "#4ade80"
PURPLE = "#c084fc"
TEAL   = "#2dd4bf"
INDIGO = "#818cf8"
YELLOW = "#facc15"
TEXT   = "#e2e8f0"
MUTED  = "#64748b"


# -- Data loading (identical to train_xgboost.py) --------------------------------

def load_db(db_path: str, source_label: int) -> list[dict]:
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
                import json as _json
                n_reactants = len(_json.loads(reactants_json))
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


def build_matrix(data, top_solvents, top_catalysts):
    X = np.stack([to_feature_vector(r, top_solvents, top_catalysts) for r in data])
    y = np.array([r["yield"] for r in data], dtype=np.float32)
    return X, y


# -- Conformal prediction -------------------------------------------------------

def fit_conformal(booster: xgb.Booster,
                  dval: xgb.DMatrix, y_val: np.ndarray) -> dict:
    preds_val = np.clip(booster.predict(dval), 0.0, 100.0)
    residuals = np.abs(y_val - preds_val)
    n = len(residuals)
    thresholds = {}
    for level in CI_LEVELS:
        alpha   = 1.0 - level
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        thresholds[level] = float(np.quantile(residuals, q_level))
    return thresholds


def apply_conformal(booster: xgb.Booster,
                    dtest: xgb.DMatrix,
                    thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    preds  = np.clip(booster.predict(dtest), 0.0, 100.0)
    levels = sorted(thresholds.keys())
    n      = len(preds)
    ivs    = np.zeros((n, 2, len(levels)), dtype=np.float32)
    for i, level in enumerate(levels):
        q = thresholds[level]
        ivs[:, 0, i] = np.clip(preds - q, 0.0, 100.0)
        ivs[:, 1, i] = np.clip(preds + q, 0.0, 100.0)
    return preds, ivs


# -- Chart (identical layout to train_xgboost.py) --------------------------------

def _style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def generate_report(y_test, preds, intervals, mae, rmse, r2, out_path,
                    v1_mae=None, v1_r2=None):
    from scipy.stats import norm as sp_norm

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

    for ax in [ax_scatter, ax_resid, ax_bucket, ax_ci]:
        _style_ax(ax)
    ax_formula.axis("off")

    # TL: Predicted vs Actual
    hb = ax_scatter.hexbin(y_test, preds, gridsize=60, cmap="YlOrRd",
                           mincnt=1, bins="log")
    lims = (0, 105)
    ax_scatter.plot(lims, lims, "--", color=INDIGO, lw=1.2, alpha=0.7,
                    label="Perfect prediction")
    m_coef, b_coef = np.polyfit(y_test, preds, 1)
    xs = np.linspace(0, 100, 100)
    ax_scatter.plot(xs, m_coef * xs + b_coef, "-", color=GREEN, lw=1.5, alpha=0.8,
                    label=f"Fit  y={m_coef:.3f}x+{b_coef:.1f}")
    ax_scatter.set_xlim(lims); ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel("Actual yield (%)", color=TEXT, fontsize=10)
    ax_scatter.set_ylabel("Predicted yield (%)", color=TEXT, fontsize=10)
    ax_scatter.set_title("Predicted vs Actual", color=TEXT, fontsize=12,
                         fontweight="bold", pad=8)
    ax_scatter.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)
    cb = fig.colorbar(hb, ax=ax_scatter, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED, labelsize=7)
    cb.set_label("log10(count)", color=MUTED, fontsize=8)
    r2_label = f"R2 = {r2:.3f}"
    if v1_r2 is not None:
        delta = r2 - v1_r2
        r2_label += f"  ({'+' if delta >= 0 else ''}{delta:.3f} vs v1)"
    ax_scatter.annotate(r2_label, xy=(0.04, 0.94), xycoords="axes fraction",
                        color=TEAL, fontsize=11, fontweight="bold")

    # TR: Residual histogram
    residuals = preds - y_test
    bins      = np.linspace(-60, 60, 61)
    ax_resid.hist(residuals, bins=bins, color=INDIGO, alpha=0.75, edgecolor="none")
    mu, sigma = residuals.mean(), residuals.std()
    xs_r = np.linspace(bins[0], bins[-1], 300)
    pdf  = sp_norm.pdf(xs_r, mu, sigma) * len(residuals) * (bins[1] - bins[0])
    ax_resid.plot(xs_r, pdf, color=YELLOW, lw=1.8, label=f"Normal fit  sigma={sigma:.1f}%")
    ax_resid.axvline(0,  color=GREEN,  lw=1.2, ls="--", alpha=0.7)
    ax_resid.axvline(mu, color=PURPLE, lw=1.2, ls=":",  alpha=0.9,
                     label=f"Mean={mu:+.1f}%")
    ax_resid.set_xlabel("Residual (pred - actual, %)", color=TEXT, fontsize=10)
    ax_resid.set_ylabel("Count", color=TEXT, fontsize=10)
    ax_resid.set_title("Residual Distribution", color=TEXT, fontsize=12,
                       fontweight="bold", pad=8)
    ax_resid.legend(fontsize=8, framealpha=0.3, facecolor=CARD, labelcolor=TEXT)
    mae_label = f"MAE = {mae:.2f}%\nRMSE = {rmse:.2f}%"
    if v1_mae is not None:
        mae_label += f"\ndelta MAE vs v1: {mae - v1_mae:+.2f}%"
    ax_resid.annotate(mae_label, xy=(0.63, 0.84), xycoords="axes fraction",
                      color=TEAL, fontsize=10, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD,
                                edgecolor=BORDER, alpha=0.85))

    # BL: MAE per yield decile
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
    ax_bucket.axhline(mae, color=INDIGO, lw=1.4, ls="--",
                      label=f"Overall MAE {mae:.1f}%")
    ax_bucket.axhline(10,  color=GREEN,  lw=1.0, ls=":",  alpha=0.5,
                      label="Target 10%")
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

    # BR: CI coverage
    targets = [int(cl * 100) for cl in CI_LEVELS]
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
    ax_ci.bar(x_ci - width/2, targets, width,
              color=MUTED, alpha=0.55, label="Target", edgecolor=BG)
    bars_a = ax_ci.bar(x_ci + width/2, actuals, width,
                       color=[GREEN if abs(a-t) <= 3 else YELLOW if abs(a-t) <= 7
                               else "#f87171" for a, t in zip(actuals, targets)],
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

    # Bottom formula band
    ci90_actual = actuals[1]
    ci90_width  = widths[1]
    formula = (
        f"y^ = XGBoost_v2(DRFP2048 + conditions)     "
        f"MAE = {mae:.2f}%   RMSE = {rmse:.2f}%   R2 = {r2:.3f}   "
        f"90% CI coverage = {ci90_actual:.1f}%   Mean CI width = +/-{ci90_width/2:.1f}%"
    )
    ax_formula.text(0.5, 0.55, formula,
                    transform=ax_formula.transAxes,
                    ha="center", va="center",
                    color=TEAL, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                              edgecolor=INDIGO, linewidth=1.5))

    fig.suptitle("SynAgent Phase 1 -- XGBoost v2 Yield Prediction Results",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.96)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart saved -> {out_path}")


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extend XGBoost training from v1 checkpoint")
    parser.add_argument("--db",           default=DEFAULT_DB)
    parser.add_argument("--model_dir",    default=MODEL_DIR)
    parser.add_argument("--extra_rounds", type=int, default=10_000,
                        help="Additional boosting rounds (default 10000)")
    parser.add_argument("--patience",     type=int, default=100,
                        help="Early-stopping patience (default 100)")
    parser.add_argument("--gpu",          action="store_true")
    args = parser.parse_args()

    device    = "cuda" if args.gpu else "cpu"
    model_in  = os.path.join(args.model_dir, "xgb_yield_v1.json")
    meta_in   = os.path.join(args.model_dir, "xgb_yield_v1_meta.json")
    model_out = os.path.join(args.model_dir, "xgb_yield_v2.json")
    mapie_out = os.path.join(args.model_dir, "xgb_yield_v2_mapie.json")
    meta_out  = os.path.join(args.model_dir, "xgb_yield_v2_meta.json")
    chart_out = os.path.join(args.model_dir, "xgb_yield_v2_results.png")

    print("=" * 64)
    print("  SynAgent Phase 1 -- XGBoost Extended Training (v1 -> v2)")
    print(f"  Device: {device.upper()}   Extra rounds: {args.extra_rounds}")
    print(f"  Early-stop patience: {args.patience}")
    print("=" * 64)

    # -- Load encoders from v1 meta (CRITICAL: must be identical to training run)
    print(f"\nLoading v1 encoders from {meta_in} ...")
    with open(meta_in) as f:
        v1_meta = json.load(f)
    top_solvents  = v1_meta["top_solvents"]
    top_catalysts = v1_meta["top_catalysts"]
    v1_mae = v1_meta.get("mae_test")
    v1_r2  = v1_meta.get("r2_test")
    v1_best_params = v1_meta.get("best_params", {})
    print(f"  Encoders: {len(top_solvents)} solvents, {len(top_catalysts)} catalysts")
    print(f"  v1 baseline: MAE {v1_mae}%   R2 {v1_r2}")

    # If meta didn't capture best_params (e.g. model loaded via --skip_train),
    # fall back to best_params.json on disk
    if not v1_best_params:
        bp_path = os.path.join(args.model_dir, "best_params.json")
        if os.path.exists(bp_path):
            with open(bp_path) as f:
                v1_best_params = json.load(f)
            print(f"  best_params loaded from {bp_path}")
        else:
            print("  WARNING: best_params.json not found, using built-in defaults")

    # -- Load data (identical pipeline to train_xgboost.py)
    print(f"\nLoading data from {args.db} ...")
    data = load_db(args.db, source_label=1)
    if not data:
        print("No data loaded.")
        return
    print(f"  Total reactions: {len(data):,}")

    print("\nBuilding feature matrix (using v1 encoders) ...")
    X, y = build_matrix(data, top_solvents, top_catalysts)
    print(f"  Feature matrix: {X.shape}  (must be 2174 features)")

    # -- Identical 80/10/10 split (random_state=42 -> same sets as v1)
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_SEED)
    val_frac_adj = VAL_FRAC / (1 - TEST_FRAC)
    X_train, X_val, y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_adj, random_state=RANDOM_SEED)
    print(f"  Train: {len(X_train):,}   Val: {len(X_val):,}   Test: {len(X_test):,}")

    # -- Convert to DMatrix
    print("\nBuilding DMatrix objects ...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # -- Load v1 model
    print(f"\nLoading v1 model from {model_in} ...")
    booster_v1 = xgb.Booster()
    booster_v1.load_model(model_in)
    # Get tree count from v1
    try:
        v1_ntrees = booster_v1.num_boosted_rounds()
        print(f"  v1 tree count: {v1_ntrees:,}")
    except Exception:
        v1_ntrees = "unknown"
        print("  v1 tree count: (could not determine)")

    # -- Build params for continued training (same as best_params.json)
    params = {
        "objective":        "reg:squarederror",
        "tree_method":      "hist",
        "device":           device,
        "eval_metric":      "rmse",
        "learning_rate":    v1_best_params.get("learning_rate", 0.01594),
        "max_depth":        int(v1_best_params.get("max_depth", 10)),
        "subsample":        v1_best_params.get("subsample", 0.676),
        "colsample_bytree": v1_best_params.get("colsample_bytree", 0.631),
        "min_child_weight": int(v1_best_params.get("min_child_weight", 4)),
        "gamma":            v1_best_params.get("gamma", 0.633),
        "reg_alpha":        v1_best_params.get("reg_alpha", 0.000914),
        "reg_lambda":       v1_best_params.get("reg_lambda", 0.531),
        "verbosity":        1,
        "seed":             RANDOM_SEED,
    }

    print(f"\nContinuing training: up to {args.extra_rounds:,} more rounds "
          f"(patience={args.patience}) on {device.upper()} ...")
    print("  (prints val RMSE every 100 rounds)\n")

    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=args.patience,
            metric_name="rmse",
            data_name="val",
            save_best=True,
        )
    ]

    booster_v2 = xgb.train(
        params,
        dtrain,
        num_boost_round=args.extra_rounds,
        xgb_model=booster_v1,           # <-- continue from v1
        evals=[(dval, "val")],
        callbacks=callbacks,
        verbose_eval=100,
    )

    try:
        v2_ntrees = booster_v2.num_boosted_rounds()
        print(f"\n  v2 tree count: {v2_ntrees:,}  (added {v2_ntrees - (v1_ntrees if isinstance(v1_ntrees, int) else 0):,} trees)")
    except Exception:
        print("\n  v2 tree count: (could not determine)")

    # -- Evaluate on test set
    preds = np.clip(booster_v2.predict(dtest), 0.0, 100.0)
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\n{'='*64}")
    print(f"  TEST SET RESULTS  (n = {len(y_test):,})")
    print(f"{'='*64}")
    print(f"  v1 -> v2  MAE:  {v1_mae:.2f}% -> {mae:.2f}%  "
          f"({'+' if mae - v1_mae >= 0 else ''}{mae - v1_mae:.2f}%)")
    print(f"  v1 -> v2  RMSE: (v1 prev)  -> {rmse:.2f}%")
    print(f"  v1 -> v2  R2:   {v1_r2:.4f} -> {r2:.4f}  "
          f"({'+' if r2 - v1_r2 >= 0 else ''}{r2 - v1_r2:.4f})")

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

    # -- Save model
    os.makedirs(args.model_dir, exist_ok=True)
    booster_v2.save_model(model_out)
    print(f"\n  Model  -> {model_out}  (saved)")

    # -- Conformal prediction
    print("\nCalibrating conformal intervals on val set ...")
    thresholds        = fit_conformal(booster_v2, dval, y_val)
    _, intervals_test = apply_conformal(booster_v2, dtest, thresholds)

    print(f"\n  {'Level':>8}  {'Target':>8}  {'Actual':>8}  {'Mean width':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
    ci_summary = {}
    for i, cl in enumerate(CI_LEVELS):
        lo_i = intervals_test[:, 0, i]
        hi_i = intervals_test[:, 1, i]
        cov  = float(np.mean((y_test >= lo_i) & (y_test <= hi_i)))
        wid  = float(np.mean(hi_i - lo_i))
        mark = "OK" if abs(cov - cl) <= 0.05 else "FAIL"
        print(f"  {int(cl*100):>7}%  {int(cl*100):>7}%  {cov*100:>7.1f}%  "
              f"{wid:>10.1f}%  {mark}")
        ci_summary[f"ci_{int(cl*100)}"] = {
            "target":    cl,
            "actual":    round(cov, 4),
            "mean_width": round(wid, 2),
            "threshold":  round(thresholds[cl], 3),
        }

    with open(mapie_out, "w") as f:
        json.dump({"thresholds": {str(k): v for k, v in thresholds.items()},
                   "ci_levels": CI_LEVELS}, f, indent=2)

    # -- Save metadata
    meta = {
        "version":        "v2",
        "base_model":     "xgb_yield_v1.json",
        "v1_mae":         v1_mae,
        "v1_r2":          v1_r2,
        "top_solvents":   top_solvents,
        "top_catalysts":  top_catalysts,
        "fp_size":        FP_SIZE,
        "min_yield":      MIN_YIELD,
        "max_yield":      MAX_YIELD,
        "n_features":     int(X.shape[1]),
        "n_train":        int(len(X_train)),
        "n_val":          int(len(X_val)),
        "n_test":         int(len(X_test)),
        "mae_test":       round(mae,  3),
        "rmse_test":      round(rmse, 3),
        "r2_test":        round(r2,   4),
        "delta_mae":      round(mae - v1_mae, 3),
        "delta_r2":       round(r2 - v1_r2,  4),
        "best_params":    v1_best_params,
        "extra_rounds":   args.extra_rounds,
        "early_stop_patience": args.patience,
        "ci_levels":      ci_summary,
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  CIs    -> {mapie_out}  (saved)")
    print(f"  Meta   -> {meta_out}  (saved)")

    # -- Chart
    print("\nGenerating results chart ...")
    try:
        generate_report(y_test, preds, intervals_test,
                        mae, rmse, r2, chart_out,
                        v1_mae=v1_mae, v1_r2=v1_r2)
    except ImportError:
        print("  [SKIP] scipy not installed -- pip install scipy")

    print("\n  Done.")


if __name__ == "__main__":
    main()
