from pathlib import Path
import json
import math
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as e:
    raise SystemExit(
        "\nMissing scikit-learn dependency.\n"
        "Run:\n"
        "  python -m pip install scikit-learn\n"
        "and then rerun this script.\n"
    ) from e

# =============================================================================
# Paths / fixed design
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "derived" / "dataset_A_phase1" / "dataset_A_phase1_modeling_table.csv"
DICT = ROOT / "derived" / "dataset_A_phase1" / "dataset_A_phase1_feature_dictionary.csv"
OUTDIR = ROOT / "results" / "dataset_A_phase1_nestedcv"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

FOLD_METRICS_OUT = OUTDIR / "dataset_A_nestedcv_fold_metrics.csv"
PRED_OUT = OUTDIR / "dataset_A_nestedcv_predictions.csv"
AVG_PRED_OUT = OUTDIR / "dataset_A_nestedcv_avg_oof_predictions.csv"
MODEL_SUMMARY_OUT = OUTDIR / "dataset_A_nestedcv_model_summary.csv"
INCREMENT_OUT = OUTDIR / "dataset_A_incremental_value_summary.csv"
HYPER_OUT = OUTDIR / "dataset_A_nestedcv_selected_hyperparameters.csv"
AUDIT_OUT = AUDITDIR / "dataset_A_nestedcv_audit.txt"

TARGET = "physical_fatigue_final"
GROUP = "participant_id"
ROW_ID = "source_file"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BASE_SEED = 20260831
BOOTSTRAP_B = 2000

FEATURE_SETS = {
    "M0a_core_context": {
        "roles": ["context_core"],
        "description": "Core low-cost context only",
    },
    "M0b_extended_context": {
        "roles": ["context_core", "context_extended"],
        "description": "Core + demographic context",
    },
    "M1_wearable_only": {
        "roles": ["wearable_feature"],
        "description": "Wearable handcrafted features only",
    },
    "M2a_core_plus_wearable": {
        "roles": ["context_core", "wearable_feature"],
        "description": "Core context + wearable",
    },
    "M2b_extended_plus_wearable": {
        "roles": ["context_core", "context_extended", "wearable_feature"],
        "description": "Extended context + wearable",
    },
}

COMPARISONS = [
    ("ElasticNet", "M0a_core_context", "M2a_core_plus_wearable", "Core-context increment"),
    ("ElasticNet", "M0b_extended_context", "M2b_extended_plus_wearable", "Extended-context increment"),
    ("HGB", "M0a_core_context", "M2a_core_plus_wearable", "Core-context increment"),
    ("HGB", "M0b_extended_context", "M2b_extended_plus_wearable", "Extended-context increment"),
]

ELASTIC_GRID = [
    {"alpha": a, "l1_ratio": l1}
    for a, l1 in product([0.01, 0.1, 1.0, 10.0], [0.1, 0.5, 0.9])
]

HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
    for lr, leaves, l2 in product([0.03, 0.08], [7, 15], [1.0, 10.0])
]

# =============================================================================
# Utilities
# =============================================================================
def participant_balanced_metrics(y_true, y_pred, groups):
    d = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_pred, dtype=float),
        "g": np.asarray(groups).astype(str),
    })
    d["ae"] = np.abs(d["y"] - d["p"])
    d["se"] = (d["y"] - d["p"]) ** 2

    by_g = d.groupby("g", sort=False).agg(
        mae=("ae", "mean"),
        mse=("se", "mean"),
    )

    mae_pb = float(by_g["mae"].mean())
    rmse_pb = float(np.sqrt(by_g["mse"].mean()))
    mae_rep = float(mean_absolute_error(d["y"], d["p"]))
    rmse_rep = float(np.sqrt(mean_squared_error(d["y"], d["p"])))
    r2_rep = float(r2_score(d["y"], d["p"])) if d["y"].nunique() > 1 else np.nan

    return {
        "mae_pb": mae_pb,
        "rmse_pb": rmse_pb,
        "mae_rep": mae_rep,
        "rmse_rep": rmse_rep,
        "r2_rep": r2_rep,
    }


def shuffled_group_folds(groups, n_splits, seed):
    """
    Deterministic group-only split.
    Groups are shuffled, then greedily assigned to the fold with the fewest rows.
    No outcome information is used in split construction.
    """
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique = groups.unique().tolist()

    if len(unique) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups; found {len(unique)}.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    sizes = groups.value_counts().to_dict()
    fold_groups = [set() for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    # randomized tie-breaking inherited from shuffled group order
    for g in unique:
        j = int(np.argmin(fold_sizes))
        fold_groups[j].add(g)
        fold_sizes[j] += int(sizes[g])

    all_idx = np.arange(len(groups))
    result = []
    for j, gs in enumerate(fold_groups):
        test_mask = groups.isin(gs).to_numpy()
        te = all_idx[test_mask]
        tr = all_idx[~test_mask]
        result.append((tr, te))
    return result


def make_pipeline(model_name, params, numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    if model_name == "ElasticNet":
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=30000,
            random_state=BASE_SEED,
            selection="cyclic",
        )
    elif model_name == "HGB":
        model = HistGradientBoostingRegressor(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            max_iter=250,
            min_samples_leaf=10,
            random_state=BASE_SEED,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([
        ("pre", pre),
        ("var", VarianceThreshold(threshold=0.0)),
        ("model", model),
    ])


def tune_model(model_name, X, y, groups, numeric_cols, categorical_cols, seed):
    grid = ELASTIC_GRID if model_name == "ElasticNet" else HGB_GRID
    inner_splits = shuffled_group_folds(groups, INNER_FOLDS, seed)

    best_params = None
    best_score = np.inf
    all_scores = []

    for params in grid:
        fold_scores = []
        for tr_idx, va_idx in inner_splits:
            pipe = make_pipeline(model_name, params, numeric_cols, categorical_cols)
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = pipe.predict(X.iloc[va_idx])

            m = participant_balanced_metrics(
                y.iloc[va_idx],
                pred,
                groups.iloc[va_idx],
            )
            fold_scores.append(m["mae_pb"])

        score = float(np.mean(fold_scores))
        all_scores.append((score, params))

        # deterministic tie-breaking by JSON parameter string
        if (score < best_score - 1e-12) or (
            abs(score - best_score) <= 1e-12
            and json.dumps(params, sort_keys=True) < json.dumps(best_params or {}, sort_keys=True)
        ):
            best_score = score
            best_params = params

    return best_params, best_score, all_scores


def bootstrap_increment(avg_pred_df, model_name, baseline_set, combined_set, b=2000, seed=0):
    a = avg_pred_df[
        (avg_pred_df["model"] == model_name) &
        (avg_pred_df["feature_set"] == baseline_set)
    ][[ROW_ID, GROUP, "task", "y_true", "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_base"}
    )

    c = avg_pred_df[
        (avg_pred_df["model"] == model_name) &
        (avg_pred_df["feature_set"] == combined_set)
    ][[ROW_ID, "y_pred_avg"]].rename(columns={"y_pred_avg": "pred_comb"})

    d = a.merge(c, on=ROW_ID, how="inner", validate="one_to_one")

    # Observed participant-balanced deltas.
    mb = participant_balanced_metrics(d["y_true"], d["pred_base"], d[GROUP])
    mc = participant_balanced_metrics(d["y_true"], d["pred_comb"], d[GROUP])

    observed = {
        "delta_mae_pb": mb["mae_pb"] - mc["mae_pb"],
        "delta_rmse_pb": mb["rmse_pb"] - mc["rmse_pb"],
        "delta_r2_rep": mc["r2_rep"] - mb["r2_rep"],
    }

    participants = sorted(d[GROUP].astype(str).unique())
    rng = np.random.default_rng(seed)

    boot = {
        "delta_mae_pb": [],
        "delta_rmse_pb": [],
        "delta_r2_rep": [],
    }

    by_pid = {pid: d[d[GROUP].astype(str) == pid].copy() for pid in participants}

    for _ in range(b):
        sampled = rng.choice(participants, size=len(participants), replace=True)

        parts = []
        for k, pid in enumerate(sampled):
            tmp = by_pid[pid].copy()
            # unique bootstrap cluster ID, preserving within-cluster observations
            tmp["__boot_group"] = f"{pid}__draw{k}"
            parts.append(tmp)

        z = pd.concat(parts, ignore_index=True)

        zb = participant_balanced_metrics(
            z["y_true"], z["pred_base"], z["__boot_group"]
        )
        zc = participant_balanced_metrics(
            z["y_true"], z["pred_comb"], z["__boot_group"]
        )

        boot["delta_mae_pb"].append(zb["mae_pb"] - zc["mae_pb"])
        boot["delta_rmse_pb"].append(zb["rmse_pb"] - zc["rmse_pb"])
        boot["delta_r2_rep"].append(zc["r2_rep"] - zb["r2_rep"])

    result = {}
    for key, vals in boot.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        result[key + "_ci_low"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
        result[key + "_ci_high"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
        result[key + "_bootstrap_positive_prob"] = float(np.mean(vals > 0)) if len(vals) else np.nan

    return observed, result, len(d), len(participants)


# =============================================================================
# Load and validate
# =============================================================================
if not TABLE.exists():
    raise SystemExit(f"Modeling table not found: {TABLE}")
if not DICT.exists():
    raise SystemExit(f"Feature dictionary not found: {DICT}")

df = pd.read_csv(TABLE)
fd = pd.read_csv(DICT)

required = {TARGET, GROUP, ROW_ID, "task"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

if df.columns.duplicated().any():
    raise SystemExit("Duplicate column names detected in modeling table.")

if df[ROW_ID].duplicated().any():
    raise SystemExit("source_file is not unique; row identity is ambiguous.")

if df[TARGET].isna().any():
    raise SystemExit("Primary outcome contains missing values.")

# Map role -> columns from dictionary.
role_map = {
    role: fd.loc[fd["role"] == role, "column"].tolist()
    for role in fd["role"].dropna().unique()
}

feature_columns = {}
for fs_name, fs_def in FEATURE_SETS.items():
    cols = []
    for role in fs_def["roles"]:
        cols.extend(role_map.get(role, []))
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]

    # Hard safety exclusions.
    forbidden = {
        "physical_fatigue_final",
        "physical_fatigue_change",
        "mental_fatigue_final",
        "performance_rating",
        "source_file",
        "participant_id",
        "participant_id_standard",
        "n_sensor_rows",
    }
    cols = [c for c in cols if c not in forbidden and not c.startswith("qc__")]

    if not cols:
        raise SystemExit(f"No columns found for feature set {fs_name}")

    feature_columns[fs_name] = cols

categorical_candidates = {"task", "gender"}

# =============================================================================
# Nested grouped CV
# =============================================================================
fold_metric_rows = []
prediction_rows = []
hyper_rows = []

start_time = time.time()
total_outer = OUTER_REPEATS * OUTER_FOLDS
done_outer = 0

for repeat in range(OUTER_REPEATS):
    outer_seed = BASE_SEED + repeat * 1000
    outer_splits = shuffled_group_folds(df[GROUP], OUTER_FOLDS, outer_seed)

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        done_outer += 1

        train_groups = set(df.iloc[tr_idx][GROUP].astype(str))
        test_groups = set(df.iloc[te_idx][GROUP].astype(str))
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Participant leakage in outer split: {sorted(overlap)}")

        y_train = df.iloc[tr_idx][TARGET].astype(float).reset_index(drop=True)
        y_test = df.iloc[te_idx][TARGET].astype(float).reset_index(drop=True)
        g_train = df.iloc[tr_idx][GROUP].astype(str).reset_index(drop=True)
        g_test = df.iloc[te_idx][GROUP].astype(str).reset_index(drop=True)

        for model_name in ["ElasticNet", "HGB"]:
            for fs_name, cols in feature_columns.items():
                cat_cols = [c for c in cols if c in categorical_candidates]
                num_cols = [c for c in cols if c not in categorical_candidates]

                X_train = df.iloc[tr_idx][cols].reset_index(drop=True)
                X_test = df.iloc[te_idx][cols].reset_index(drop=True)

                inner_seed = outer_seed + outer_fold * 100 + (
                    1 if model_name == "ElasticNet" else 2
                )

                best_params, inner_mae, _ = tune_model(
                    model_name=model_name,
                    X=X_train,
                    y=y_train,
                    groups=g_train,
                    numeric_cols=num_cols,
                    categorical_cols=cat_cols,
                    seed=inner_seed,
                )

                pipe = make_pipeline(
                    model_name,
                    best_params,
                    num_cols,
                    cat_cols,
                )
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)

                metrics = participant_balanced_metrics(
                    y_test,
                    pred,
                    g_test,
                )

                fold_metric_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "feature_set": fs_name,
                    "n_train_rows": len(tr_idx),
                    "n_test_rows": len(te_idx),
                    "n_train_participants": len(train_groups),
                    "n_test_participants": len(test_groups),
                    "inner_selected_mae_pb": inner_mae,
                    **metrics,
                })

                hyper_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "feature_set": fs_name,
                    "selected_params_json": json.dumps(best_params, sort_keys=True),
                    "inner_selected_mae_pb": inner_mae,
                })

                test_meta = df.iloc[te_idx][
                    [ROW_ID, GROUP, "task", "rep", TARGET]
                ].reset_index(drop=True)

                for j in range(len(test_meta)):
                    prediction_rows.append({
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "feature_set": fs_name,
                        ROW_ID: test_meta.loc[j, ROW_ID],
                        GROUP: str(test_meta.loc[j, GROUP]),
                        "task": test_meta.loc[j, "task"],
                        "rep": test_meta.loc[j, "rep"],
                        "y_true": float(test_meta.loc[j, TARGET]),
                        "y_pred": float(pred[j]),
                    })

        elapsed = time.time() - start_time
        print(
            f"Completed outer split {done_outer}/{total_outer} "
            f"(repeat {repeat+1}, fold {outer_fold}) | "
            f"elapsed {elapsed/60:.1f} min"
        )

fold_df = pd.DataFrame(fold_metric_rows)
pred_df = pd.DataFrame(prediction_rows)
hyper_df = pd.DataFrame(hyper_rows)

fold_df.to_csv(FOLD_METRICS_OUT, index=False)
pred_df.to_csv(PRED_OUT, index=False)
hyper_df.to_csv(HYPER_OUT, index=False)

# =============================================================================
# Average repeated OOF predictions by repetition
# =============================================================================
avg_pred = (
    pred_df
    .groupby(["model", "feature_set", ROW_ID, GROUP, "task", "rep", "y_true"], as_index=False)
    .agg(
        y_pred_avg=("y_pred", "mean"),
        y_pred_sd=("y_pred", "std"),
        n_oof_predictions=("y_pred", "count"),
    )
)
avg_pred.to_csv(AVG_PRED_OUT, index=False)

# =============================================================================
# Overall model summary from averaged OOF predictions
# =============================================================================
summary_rows = []

for (model_name, fs_name), g in avg_pred.groupby(["model", "feature_set"]):
    m = participant_balanced_metrics(
        g["y_true"], g["y_pred_avg"], g[GROUP]
    )
    summary_rows.append({
        "scope": "All",
        "model": model_name,
        "feature_set": fs_name,
        "n_repetitions": len(g),
        "n_participants": g[GROUP].nunique(),
        **m,
    })

    for task, gt in g.groupby("task"):
        mt = participant_balanced_metrics(
            gt["y_true"], gt["y_pred_avg"], gt[GROUP]
        )
        summary_rows.append({
            "scope": str(task),
            "model": model_name,
            "feature_set": fs_name,
            "n_repetitions": len(gt),
            "n_participants": gt[GROUP].nunique(),
            **mt,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(MODEL_SUMMARY_OUT, index=False)

# =============================================================================
# Incremental-value summary + participant-cluster bootstrap
# =============================================================================
increment_rows = []

for k, (model_name, baseline_set, combined_set, label) in enumerate(COMPARISONS):
    observed, boot, n_rows, n_pids = bootstrap_increment(
        avg_pred_df=avg_pred,
        model_name=model_name,
        baseline_set=baseline_set,
        combined_set=combined_set,
        b=BOOTSTRAP_B,
        seed=BASE_SEED + 9000 + k,
    )

    # Fold-level paired differences are descriptive only because repeated folds
    # are not independent; cluster bootstrap above is the inferential summary.
    f0 = fold_df[
        (fold_df["model"] == model_name) &
        (fold_df["feature_set"] == baseline_set)
    ].copy()
    f1 = fold_df[
        (fold_df["model"] == model_name) &
        (fold_df["feature_set"] == combined_set)
    ].copy()

    paired = f0.merge(
        f1,
        on=["repeat", "outer_fold", "model"],
        suffixes=("_base", "_comb"),
        validate="one_to_one",
    )

    d_mae_fold = paired["mae_pb_base"] - paired["mae_pb_comb"]
    d_rmse_fold = paired["rmse_pb_base"] - paired["rmse_pb_comb"]
    d_r2_fold = paired["r2_rep_comb"] - paired["r2_rep_base"]

    increment_rows.append({
        "model": model_name,
        "comparison": label,
        "baseline_feature_set": baseline_set,
        "combined_feature_set": combined_set,
        "n_repetitions": n_rows,
        "n_participants": n_pids,
        **observed,
        **boot,
        "fold_delta_mae_pb_mean": float(d_mae_fold.mean()),
        "fold_delta_mae_pb_sd": float(d_mae_fold.std(ddof=1)),
        "fold_delta_mae_pb_positive_fraction": float((d_mae_fold > 0).mean()),
        "fold_delta_rmse_pb_mean": float(d_rmse_fold.mean()),
        "fold_delta_rmse_pb_positive_fraction": float((d_rmse_fold > 0).mean()),
        "fold_delta_r2_rep_mean": float(d_r2_fold.mean()),
        "fold_delta_r2_rep_positive_fraction": float((d_r2_fold > 0).mean()),
    })

increment_df = pd.DataFrame(increment_rows)
increment_df.to_csv(INCREMENT_OUT, index=False)

# =============================================================================
# Audit report
# =============================================================================
elapsed = time.time() - start_time
lines = []

def add(s=""):
    lines.append(str(s))

add("DATASET A — REPEATED NESTED PARTICIPANT-GROUPED INCREMENTAL-VALUE BENCHMARK")
add("=" * 118)
add(f"Input rows: {len(df)}")
add(f"Unique participant labels: {df[GROUP].nunique()}")
add(f"Outcome: {TARGET}")
add(f"Outer validation: {OUTER_REPEATS} repeats x {OUTER_FOLDS} participant-grouped folds")
add(f"Inner validation: {INNER_FOLDS} participant-grouped folds")
add(f"Bootstrap: {BOOTSTRAP_B} participant-cluster resamples on averaged repeated-OOF predictions")
add(f"Elapsed minutes: {elapsed/60:.2f}")
add()

add("1. FEATURE SETS")
add("-" * 118)
for name, cols in feature_columns.items():
    add(f"{name}: {len(cols)} raw input columns")
    add(f"  {FEATURE_SETS[name]['description']}")
add()

add("2. LEAKAGE CONTROLS")
add("-" * 118)
add("PASS: outer train/test splits are participant-disjoint.")
add("PASS: inner hyperparameter selection uses only outer-training participants.")
add("PASS: imputation, scaling, one-hot encoding, and zero-variance filtering are fit inside each training fold.")
add("PASS: the same outer splits are reused for every feature set and learner, enabling paired comparisons.")
add("Excluded by design: outcome, change score, final mental fatigue, performance rating, QC columns, IDs, gyroscope.")
add()

add("3. PRIMARY MODEL SUMMARY — AVERAGED REPEATED OOF PREDICTIONS")
add("-" * 118)
show = summary_df[summary_df["scope"] == "All"][
    ["model","feature_set","n_repetitions","n_participants","mae_pb","rmse_pb","mae_rep","rmse_rep","r2_rep"]
].sort_values(["model","mae_pb"])
add(show.to_string(index=False))
add()

add("4. INCREMENTAL VALUE")
add("-" * 118)
show_inc_cols = [
    "model","comparison",
    "delta_mae_pb","delta_mae_pb_ci_low","delta_mae_pb_ci_high","delta_mae_pb_bootstrap_positive_prob",
    "delta_rmse_pb","delta_rmse_pb_ci_low","delta_rmse_pb_ci_high","delta_rmse_pb_bootstrap_positive_prob",
    "delta_r2_rep","delta_r2_rep_ci_low","delta_r2_rep_ci_high","delta_r2_rep_bootstrap_positive_prob",
    "fold_delta_mae_pb_positive_fraction",
]
add(increment_df[show_inc_cols].to_string(index=False))
add()
add("Sign convention:")
add("  delta_mae_pb > 0   => adding wearable features reduces participant-balanced MAE.")
add("  delta_rmse_pb > 0  => adding wearable features reduces participant-balanced RMSE.")
add("  delta_r2_rep > 0   => adding wearable features increases pooled R^2.")
add("Bootstrap positive probability is descriptive support, not a substitute for a prespecified hypothesis test.")
add()

add("5. INTERPRETATION GUARDRAILS")
add("-" * 118)
add("Do NOT claim wearable value from absolute wearable-only performance.")
add("Primary evidence is the paired increment over the corresponding context baseline.")
add("Repeated outer-fold metrics are dependent; use them descriptively.")
add("Participant-cluster bootstrap on averaged repeated-OOF predictions is the main uncertainty summary here.")
add("No claim about cross-dataset transportability can be made from Dataset A alone.")
add("No claim about foundation-model superiority can be made in Phase I.")
add()

add("6. OUTPUT FILES")
add("-" * 118)
for fp in [
    FOLD_METRICS_OUT,
    PRED_OUT,
    AVG_PRED_OUT,
    MODEL_SUMMARY_OUT,
    INCREMENT_OUT,
    HYPER_OUT,
]:
    add(f"  {fp}")
add()
add("=" * 118)
add("END OF BENCHMARK")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Nested grouped benchmark complete.")
print(f"Audit: {AUDIT_OUT}")
print(f"Increment summary: {INCREMENT_OUT}")
print("")
print("Upload dataset_A_nestedcv_audit.txt and dataset_A_incremental_value_summary.csv to ChatGPT.")
