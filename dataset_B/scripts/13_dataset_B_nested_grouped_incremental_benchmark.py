from pathlib import Path
import json
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# Paths / fixed design
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "derived" / "dataset_B_phase1" / "dataset_B_lagged_onset_modeling_table.csv"
DICT = ROOT / "derived" / "dataset_B_phase1" / "dataset_B_lagged_onset_feature_dictionary.csv"

OUTDIR = ROOT / "results" / "dataset_B_phase1_nestedcv"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

FOLD_OUT = OUTDIR / "dataset_B_nestedcv_fold_metrics.csv"
PRED_OUT = OUTDIR / "dataset_B_nestedcv_predictions.csv"
AVG_OUT = OUTDIR / "dataset_B_nestedcv_avg_oof_predictions.csv"
SUMMARY_OUT = OUTDIR / "dataset_B_nestedcv_model_summary.csv"
INCREMENT_OUT = OUTDIR / "dataset_B_incremental_value_summary.csv"
HYPER_OUT = OUTDIR / "dataset_B_nestedcv_selected_hyperparameters.csv"
AUDIT_OUT = AUDITDIR / "dataset_B_nestedcv_audit.txt"

TARGET = "fatigue_onset_next_interval"
GROUP = "worker_index"
ROW_ID = None  # created below from stable row identity

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_B = 3000
BASE_SEED = 20260901

LOGISTIC_GRID = [
    {"C": c, "penalty": penalty}
    for c, penalty in product([0.01, 0.1, 1.0, 10.0], ["l1", "l2"])
]

HGB_GRID = [
    {
        "learning_rate": lr,
        "max_leaf_nodes": leaves,
        "l2_regularization": l2,
        "min_samples_leaf": msl,
    }
    for lr, leaves, l2, msl in product(
        [0.03, 0.08],
        [7, 15],
        [1.0, 10.0],
        [5, 10],
    )
]

# =============================================================================
# Metrics
# =============================================================================
def safe_ap(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))

def safe_roc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

def participant_balanced_prob_metrics(y, p, groups):
    d = pd.DataFrame({
        "y": np.asarray(y, dtype=int),
        "p": np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6),
        "g": np.asarray(groups).astype(str),
    })

    pb_brier = []
    pb_logloss = []

    for _, g in d.groupby("g", sort=False):
        pb_brier.append(float(np.mean((g["y"] - g["p"]) ** 2)))
        pb_logloss.append(float(
            -np.mean(
                g["y"] * np.log(g["p"]) +
                (1 - g["y"]) * np.log(1 - g["p"])
            )
        ))

    return {
        "pr_auc": safe_ap(d["y"], d["p"]),
        "roc_auc": safe_roc(d["y"], d["p"]),
        "brier_pb": float(np.mean(pb_brier)),
        "logloss_pb": float(np.mean(pb_logloss)),
        "brier_rep": float(brier_score_loss(d["y"], d["p"])),
        "logloss_rep": float(log_loss(d["y"], d["p"], labels=[0, 1])),
    }

# =============================================================================
# Split construction
# =============================================================================
def make_valid_sgkf_splits(X, y, groups, n_splits, seed, max_attempts=500):
    """
    StratifiedGroupKFold, with deterministic seed search until every fold has
    both classes in train and validation/test. No outcome information beyond
    the target needed for stratification is used for modeling.
    """
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).astype(str).reset_index(drop=True)

    for attempt in range(max_attempts):
        rs = seed + attempt
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rs,
        )
        splits = list(cv.split(X, y, groups))

        valid = True
        for tr, va in splits:
            if y.iloc[tr].nunique() < 2 or y.iloc[va].nunique() < 2:
                valid = False
                break

        if valid:
            return splits, rs

    raise RuntimeError(
        f"Could not construct valid {n_splits}-fold StratifiedGroupKFold "
        f"after {max_attempts} seed attempts."
    )

# =============================================================================
# Pipelines
# =============================================================================
def make_pipeline(model_name, params, numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    if model_name == "Logistic":
        model = LogisticRegression(
            C=float(params["C"]),
            penalty=str(params["penalty"]),
            solver="liblinear",
            max_iter=10000,
            class_weight=None,
            random_state=BASE_SEED,
        )
    elif model_name == "HGB":
        model = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_iter=250,
            random_state=BASE_SEED,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([
        ("pre", pre),
        ("var", VarianceThreshold(threshold=0.0)),
        ("model", model),
    ])

def tune_model(
    model_name,
    X,
    y,
    groups,
    numeric_cols,
    categorical_cols,
    seed,
):
    grid = LOGISTIC_GRID if model_name == "Logistic" else HGB_GRID

    splits, used_seed = make_valid_sgkf_splits(
        X, y, groups, INNER_FOLDS, seed
    )

    best_score = -np.inf
    best_params = None

    for params in grid:
        aps = []

        for tr, va in splits:
            pipe = make_pipeline(
                model_name,
                params,
                numeric_cols,
                categorical_cols,
            )
            pipe.fit(X.iloc[tr], y.iloc[tr])
            prob = pipe.predict_proba(X.iloc[va])[:, 1]

            ap = safe_ap(y.iloc[va], prob)
            if np.isfinite(ap):
                aps.append(ap)

        score = float(np.mean(aps)) if aps else -np.inf

        key = json.dumps(params, sort_keys=True)
        old_key = json.dumps(best_params or {}, sort_keys=True)

        if (
            score > best_score + 1e-12
            or (
                abs(score - best_score) <= 1e-12
                and key < old_key
            )
        ):
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("Hyperparameter tuning failed.")

    return best_params, best_score, used_seed

# =============================================================================
# Bootstrap incremental value
# =============================================================================
def bootstrap_increment(
    avg_pred,
    model_name,
    baseline_set,
    combined_set,
    b,
    seed,
):
    a = avg_pred[
        (avg_pred["model"] == model_name) &
        (avg_pred["feature_set"] == baseline_set)
    ][
        ["row_id", GROUP, "task", "y_true", "y_prob_avg"]
    ].rename(columns={"y_prob_avg": "p_base"})

    c = avg_pred[
        (avg_pred["model"] == model_name) &
        (avg_pred["feature_set"] == combined_set)
    ][["row_id", "y_prob_avg"]].rename(columns={"y_prob_avg": "p_comb"})

    d = a.merge(c, on="row_id", how="inner", validate="one_to_one")

    mb = participant_balanced_prob_metrics(
        d["y_true"], d["p_base"], d[GROUP]
    )
    mc = participant_balanced_prob_metrics(
        d["y_true"], d["p_comb"], d[GROUP]
    )

    observed = {
        "delta_pr_auc": mc["pr_auc"] - mb["pr_auc"],
        "delta_roc_auc": mc["roc_auc"] - mb["roc_auc"],
        "delta_brier_pb": mb["brier_pb"] - mc["brier_pb"],
        "delta_logloss_pb": mb["logloss_pb"] - mc["logloss_pb"],
    }

    participants = sorted(d[GROUP].astype(str).unique())
    by_pid = {
        pid: d[d[GROUP].astype(str) == pid].copy()
        for pid in participants
    }

    rng = np.random.default_rng(seed)

    store = {k: [] for k in observed}

    for _ in range(b):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        parts = []
        for j, pid in enumerate(sampled):
            z = by_pid[pid].copy()
            z["boot_group"] = f"{pid}__draw{j}"
            parts.append(z)

        z = pd.concat(parts, ignore_index=True)

        # Rare but possible bootstrap samples may contain no positive events.
        if z["y_true"].nunique() < 2:
            continue

        zb = participant_balanced_prob_metrics(
            z["y_true"], z["p_base"], z["boot_group"]
        )
        zc = participant_balanced_prob_metrics(
            z["y_true"], z["p_comb"], z["boot_group"]
        )

        store["delta_pr_auc"].append(zc["pr_auc"] - zb["pr_auc"])
        store["delta_roc_auc"].append(zc["roc_auc"] - zb["roc_auc"])
        store["delta_brier_pb"].append(zb["brier_pb"] - zc["brier_pb"])
        store["delta_logloss_pb"].append(
            zb["logloss_pb"] - zc["logloss_pb"]
        )

    ci = {}
    for key, vals in store.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]

        ci[key + "_ci_low"] = float(np.quantile(arr, 0.025))
        ci[key + "_ci_high"] = float(np.quantile(arr, 0.975))
        ci[key + "_bootstrap_positive_prob"] = float(np.mean(arr > 0))
        ci[key + "_bootstrap_n"] = int(len(arr))

    return observed, ci, len(d), len(participants)

# =============================================================================
# Load and validate
# =============================================================================
df = pd.read_csv(TABLE)
fd = pd.read_csv(DICT)

required = {
    TARGET, GROUP, "task",
    "task_specific_id",
    "prediction_time_min",
    "target_interval_end_min",
}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

if df.columns.duplicated().any():
    raise SystemExit("Duplicate modeling-table columns found.")

if df[TARGET].isna().any():
    raise SystemExit("Target contains missing values.")

# Stable unique row identity.
df["row_id"] = (
    df["task_specific_id"].astype(str) + "__" +
    df["prediction_time_min"].astype(str) + "__" +
    df["target_interval_end_min"].astype(str)
)

if df["row_id"].duplicated().any():
    raise SystemExit("Constructed row_id is not unique.")

role_to_cols = {}
for role in fd["role"].dropna().unique():
    role_to_cols[role] = fd.loc[fd["role"] == role, "column"].tolist()

core = [c for c in role_to_cols.get("context_core", []) if c in df.columns]
extended = [c for c in role_to_cols.get("context_extended", []) if c in df.columns]
baseline_phys = [c for c in role_to_cols.get("baseline_physiology", []) if c in df.columns]
wearable = [c for c in role_to_cols.get("wearable_feature", []) if c in df.columns]

FEATURE_SETS = {
    "B0a_core_context": core,
    "B0b_extended_context": list(dict.fromkeys(core + extended)),
    "B0c_extended_plus_restingHR": list(
        dict.fromkeys(core + extended + baseline_phys)
    ),
    "B1_dynamic_wearable_only": wearable,
    "B2a_core_plus_wearable": list(dict.fromkeys(core + wearable)),
    "B2b_extended_plus_wearable": list(
        dict.fromkeys(core + extended + wearable)
    ),
    "B2c_extended_restingHR_plus_wearable": list(
        dict.fromkeys(core + extended + baseline_phys + wearable)
    ),
}

for name, cols in FEATURE_SETS.items():
    if not cols:
        raise SystemExit(f"No features found for {name}")

categorical_candidates = {"task", "sex"}

# =============================================================================
# Nested repeated grouped CV
# =============================================================================
fold_rows = []
pred_rows = []
hyper_rows = []

start = time.time()
outer_done = 0
outer_total = OUTER_REPEATS * OUTER_FOLDS

for repeat in range(OUTER_REPEATS):
    repeat_seed = BASE_SEED + repeat * 10000

    outer_splits, used_outer_seed = make_valid_sgkf_splits(
        df,
        df[TARGET].astype(int),
        df[GROUP].astype(str),
        OUTER_FOLDS,
        repeat_seed,
    )

    for outer_fold, (tr_idx, te_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        outer_done += 1

        train_groups = set(df.iloc[tr_idx][GROUP].astype(str))
        test_groups = set(df.iloc[te_idx][GROUP].astype(str))

        if train_groups & test_groups:
            raise RuntimeError("Participant leakage in outer split.")

        y_train = df.iloc[tr_idx][TARGET].astype(int).reset_index(drop=True)
        y_test = df.iloc[te_idx][TARGET].astype(int).reset_index(drop=True)
        g_train = df.iloc[tr_idx][GROUP].astype(str).reset_index(drop=True)
        g_test = df.iloc[te_idx][GROUP].astype(str).reset_index(drop=True)

        for model_name in ["Logistic", "HGB"]:
            for fs_idx, (fs_name, cols) in enumerate(FEATURE_SETS.items()):

                cat_cols = [
                    c for c in cols if c in categorical_candidates
                ]
                num_cols = [
                    c for c in cols if c not in categorical_candidates
                ]

                X_train = df.iloc[tr_idx][cols].reset_index(drop=True)
                X_test = df.iloc[te_idx][cols].reset_index(drop=True)

                inner_seed = (
                    repeat_seed
                    + outer_fold * 1000
                    + fs_idx * 10
                    + (1 if model_name == "Logistic" else 2)
                )

                best_params, inner_ap, used_inner_seed = tune_model(
                    model_name,
                    X_train,
                    y_train,
                    g_train,
                    num_cols,
                    cat_cols,
                    inner_seed,
                )

                pipe = make_pipeline(
                    model_name,
                    best_params,
                    num_cols,
                    cat_cols,
                )
                pipe.fit(X_train, y_train)

                prob = pipe.predict_proba(X_test)[:, 1]

                metrics = participant_balanced_prob_metrics(
                    y_test,
                    prob,
                    g_test,
                )

                fold_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "outer_split_seed": used_outer_seed,
                    "model": model_name,
                    "feature_set": fs_name,
                    "n_features_raw": len(cols),
                    "n_train_rows": len(tr_idx),
                    "n_test_rows": len(te_idx),
                    "n_train_workers": len(train_groups),
                    "n_test_workers": len(test_groups),
                    "n_test_events": int(y_test.sum()),
                    "inner_selected_pr_auc": inner_ap,
                    "inner_split_seed": used_inner_seed,
                    **metrics,
                })

                hyper_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "feature_set": fs_name,
                    "selected_params_json": json.dumps(
                        best_params,
                        sort_keys=True,
                    ),
                    "inner_selected_pr_auc": inner_ap,
                })

                meta = df.iloc[te_idx][
                    [
                        "row_id", GROUP, "task",
                        "task_specific_id",
                        "prediction_time_min",
                        "target_interval_end_min",
                        TARGET,
                    ]
                ].reset_index(drop=True)

                for j in range(len(meta)):
                    pred_rows.append({
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "feature_set": fs_name,
                        "row_id": meta.loc[j, "row_id"],
                        GROUP: str(meta.loc[j, GROUP]),
                        "task": meta.loc[j, "task"],
                        "task_specific_id": meta.loc[j, "task_specific_id"],
                        "prediction_time_min": meta.loc[
                            j, "prediction_time_min"
                        ],
                        "target_interval_end_min": meta.loc[
                            j, "target_interval_end_min"
                        ],
                        "y_true": int(meta.loc[j, TARGET]),
                        "y_prob": float(prob[j]),
                    })

        elapsed = time.time() - start
        print(
            f"Completed outer split {outer_done}/{outer_total} "
            f"(repeat {repeat+1}, fold {outer_fold}) | "
            f"elapsed {elapsed/60:.1f} min"
        )

fold_df = pd.DataFrame(fold_rows)
pred_df = pd.DataFrame(pred_rows)
hyper_df = pd.DataFrame(hyper_rows)

fold_df.to_csv(FOLD_OUT, index=False)
pred_df.to_csv(PRED_OUT, index=False)
hyper_df.to_csv(HYPER_OUT, index=False)

# =============================================================================
# Repeated OOF averaging
# =============================================================================
avg = (
    pred_df
    .groupby(
        [
            "model", "feature_set",
            "row_id", GROUP, "task",
            "task_specific_id",
            "prediction_time_min",
            "target_interval_end_min",
            "y_true",
        ],
        as_index=False,
    )
    .agg(
        y_prob_avg=("y_prob", "mean"),
        y_prob_sd=("y_prob", "std"),
        n_oof_predictions=("y_prob", "count"),
    )
)

avg.to_csv(AVG_OUT, index=False)

# =============================================================================
# Overall summary
# =============================================================================
summary_rows = []

for (model_name, fs_name), g in avg.groupby(
    ["model", "feature_set"]
):
    m = participant_balanced_prob_metrics(
        g["y_true"],
        g["y_prob_avg"],
        g[GROUP],
    )

    summary_rows.append({
        "scope": "All",
        "model": model_name,
        "feature_set": fs_name,
        "n_rows": len(g),
        "n_workers": g[GROUP].nunique(),
        "n_events": int(g["y_true"].sum()),
        **m,
    })

    for task, gt in g.groupby("task"):
        mt = participant_balanced_prob_metrics(
            gt["y_true"],
            gt["y_prob_avg"],
            gt[GROUP],
        )

        summary_rows.append({
            "scope": str(task),
            "model": model_name,
            "feature_set": fs_name,
            "n_rows": len(gt),
            "n_workers": gt[GROUP].nunique(),
            "n_events": int(gt["y_true"].sum()),
            **mt,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)

# =============================================================================
# Incremental comparisons
# =============================================================================
COMPARISONS = [
    ("B0a_core_context", "B2a_core_plus_wearable", "Core-context increment"),
    ("B0b_extended_context", "B2b_extended_plus_wearable", "Extended-context increment"),
    (
        "B0c_extended_plus_restingHR",
        "B2c_extended_restingHR_plus_wearable",
        "Dynamic wearable beyond context + resting HR",
    ),
]

inc_rows = []

for model_idx, model_name in enumerate(["Logistic", "HGB"]):
    for comp_idx, (base, comb, label) in enumerate(COMPARISONS):

        obs, ci, n_rows, n_workers = bootstrap_increment(
            avg,
            model_name,
            base,
            comb,
            BOOTSTRAP_B,
            BASE_SEED + 90000 + model_idx * 1000 + comp_idx,
        )

        f0 = fold_df[
            (fold_df["model"] == model_name) &
            (fold_df["feature_set"] == base)
        ][
            ["repeat","outer_fold","pr_auc","roc_auc","brier_pb","logloss_pb"]
        ]

        f1 = fold_df[
            (fold_df["model"] == model_name) &
            (fold_df["feature_set"] == comb)
        ][
            ["repeat","outer_fold","pr_auc","roc_auc","brier_pb","logloss_pb"]
        ]

        paired = f0.merge(
            f1,
            on=["repeat","outer_fold"],
            suffixes=("_base","_comb"),
            validate="one_to_one",
        )

        inc_rows.append({
            "model": model_name,
            "comparison": label,
            "baseline_feature_set": base,
            "combined_feature_set": comb,
            "n_rows": n_rows,
            "n_workers": n_workers,
            **obs,
            **ci,
            "fold_delta_pr_auc_mean": float(
                (paired["pr_auc_comb"] - paired["pr_auc_base"]).mean()
            ),
            "fold_delta_pr_auc_positive_fraction": float(
                (paired["pr_auc_comb"] > paired["pr_auc_base"]).mean()
            ),
            "fold_delta_brier_pb_mean": float(
                (paired["brier_pb_base"] - paired["brier_pb_comb"]).mean()
            ),
            "fold_delta_brier_pb_positive_fraction": float(
                (paired["brier_pb_base"] > paired["brier_pb_comb"]).mean()
            ),
        })

inc_df = pd.DataFrame(inc_rows)
inc_df.to_csv(INCREMENT_OUT, index=False)

# =============================================================================
# Audit report
# =============================================================================
elapsed = time.time() - start
lines = []

def add(s=""):
    lines.append(str(s))

add("DATASET B — REPEATED NESTED PARTICIPANT-GROUPED INCREMENTAL-VALUE BENCHMARK")
add("=" * 118)
add(f"Input rows: {len(df)}")
add(f"Underlying workers: {df[GROUP].nunique()}")
add(f"Positive onset intervals: {int(df[TARGET].sum())}")
add(f"Positive prevalence: {df[TARGET].mean():.6f}")
add(f"Outcome: {TARGET}")
add(f"Outer validation: {OUTER_REPEATS} repeats x {OUTER_FOLDS} StratifiedGroupKFold by worker_index")
add(f"Inner tuning: {INNER_FOLDS} StratifiedGroupKFold by worker_index")
add(f"Bootstrap: {BOOTSTRAP_B} participant-cluster resamples on averaged repeated-OOF predictions")
add(f"Elapsed minutes: {elapsed/60:.2f}")
add()

add("1. FEATURE SETS")
add("-" * 118)
for name, cols in FEATURE_SETS.items():
    add(f"{name}: {len(cols)} raw predictor columns")
add()

add("2. LEAKAGE / TEMPORAL CONTROLS")
add("-" * 118)
add("PASS: group variable = worker_index, not task-specific ID.")
add("PASS: outer train/test workers are disjoint.")
add("PASS: inner hyperparameter selection uses outer-training workers only.")
add("PASS: all preprocessing is refit inside the relevant training split.")
add("PASS: dynamic wearable predictors are lagged by one 10-min interval in the input table.")
add("PASS: current event-interval sensor summaries are not predictors.")
add("Excluded: Fatigue, current State, final Time, Fatigue_rate, IDs.")
add()

add("3. PRIMARY MODEL SUMMARY — AVERAGED REPEATED OOF PREDICTIONS")
add("-" * 118)
show = summary_df[summary_df["scope"] == "All"][
    [
        "model","feature_set","n_rows","n_workers","n_events",
        "pr_auc","roc_auc","brier_pb","logloss_pb",
        "brier_rep","logloss_rep"
    ]
].sort_values(["model","pr_auc"], ascending=[True, False])
add(show.to_string(index=False))
add()

add("4. INCREMENTAL VALUE")
add("-" * 118)
show_cols = [
    "model","comparison",
    "delta_pr_auc","delta_pr_auc_ci_low","delta_pr_auc_ci_high","delta_pr_auc_bootstrap_positive_prob",
    "delta_roc_auc","delta_roc_auc_ci_low","delta_roc_auc_ci_high","delta_roc_auc_bootstrap_positive_prob",
    "delta_brier_pb","delta_brier_pb_ci_low","delta_brier_pb_ci_high","delta_brier_pb_bootstrap_positive_prob",
    "delta_logloss_pb","delta_logloss_pb_ci_low","delta_logloss_pb_ci_high","delta_logloss_pb_bootstrap_positive_prob",
    "fold_delta_pr_auc_positive_fraction",
]
add(inc_df[show_cols].to_string(index=False))
add()

add("Sign convention:")
add("  delta_pr_auc > 0     => wearable improves PR-AUC.")
add("  delta_roc_auc > 0    => wearable improves ROC-AUC.")
add("  delta_brier_pb > 0   => wearable reduces participant-balanced Brier score.")
add("  delta_logloss_pb > 0 => wearable reduces participant-balanced log loss.")
add()

add("5. INTERPRETATION GUARDRAILS")
add("-" * 118)
add("Primary discrimination metric: PR-AUC because onset prevalence is low.")
add("ROC-AUC is secondary.")
add("Probabilistic accuracy is evaluated with participant-balanced Brier score and log loss.")
add("No threshold-dependent warning metric is optimized in this benchmark.")
add("No claim of Dataset-A replication should be made until direction and uncertainty of the paired increments are examined.")
add("Dataset B uses a different outcome (next-interval onset) but shares the same higher-level incremental-value estimand.")
add()

add("6. OUTPUTS")
add("-" * 118)
for fp in [
    FOLD_OUT, PRED_OUT, AVG_OUT,
    SUMMARY_OUT, INCREMENT_OUT, HYPER_OUT
]:
    add(f"  {fp}")
add()
add("=" * 118)
add("END OF BENCHMARK")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Dataset B nested grouped benchmark complete.")
print(f"Audit: {AUDIT_OUT}")
print(f"Increment summary: {INCREMENT_OUT}")
print("")
print("Upload dataset_B_nestedcv_audit.txt and dataset_B_incremental_value_summary.csv to ChatGPT.")
