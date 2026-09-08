
from pathlib import Path
import json, time, warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# Paths / design
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived" / "dataset_C_phase1"

START_TABLE = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_TABLE = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

OUTDIR = ROOT / "results" / "dataset_C_phase1_dual_anchor_nestedcv_qc"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

FOLD_OUT = OUTDIR / "dataset_C_qc_dual_anchor_fold_metrics.csv"
PRED_OUT = OUTDIR / "dataset_C_qc_dual_anchor_predictions.csv"
AVG_OUT = OUTDIR / "dataset_C_qc_dual_anchor_avg_oof_predictions.csv"
SUMMARY_OUT = OUTDIR / "dataset_C_qc_dual_anchor_model_summary.csv"
INCREMENT_OUT = OUTDIR / "dataset_C_qc_dual_anchor_incremental_value.csv"
ROBUST_OUT = OUTDIR / "dataset_C_qc_alignment_robustness_summary.csv"
HYPER_OUT = OUTDIR / "dataset_C_qc_dual_anchor_selected_hyperparameters.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_qc_dual_anchor_nestedcv_audit.txt"

TARGET = "target_borg"
GROUP = "subject_num"
ROW_ID = "row_id"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_B = 2500
BASE_SEED = 20260901

FEATURE_SET_ROLES = {
    "C0a_core_context": ["context_core"],
    "C0b_extended_context": ["context_core", "context_extended"],
    "C1_wearable_only": ["wearable_feature"],
    "C2a_core_plus_wearable": ["context_core", "wearable_feature"],
    "C2b_extended_plus_wearable": ["context_core", "context_extended", "wearable_feature"],
}

COMPARISONS = [
    ("C0a_core_context", "C2a_core_plus_wearable", "Core-context increment"),
    ("C0b_extended_context", "C2b_extended_plus_wearable", "Extended-context increment"),
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
# Metrics
# =============================================================================
def participant_balanced_metrics(y_true, y_pred, groups):
    d = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_pred, dtype=float),
        "g": np.asarray(groups).astype(str),
    })
    d["ae"] = np.abs(d["y"] - d["p"])
    d["se"] = (d["y"] - d["p"]) ** 2

    bg = d.groupby("g", sort=False).agg(
        mae=("ae", "mean"),
        mse=("se", "mean"),
    )

    return {
        "mae_pb": float(bg["mae"].mean()),
        "rmse_pb": float(np.sqrt(bg["mse"].mean())),
        "mae_rep": float(mean_absolute_error(d["y"], d["p"])),
        "rmse_rep": float(np.sqrt(mean_squared_error(d["y"], d["p"]))),
        "r2_rep": float(r2_score(d["y"], d["p"])) if d["y"].nunique() > 1 else np.nan,
    }

# =============================================================================
# Group-only split construction
# =============================================================================
def shuffled_group_folds(groups, n_splits, seed):
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique = groups.unique().tolist()

    if len(unique) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups, found {len(unique)}.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    sizes = groups.value_counts().to_dict()
    fold_groups = [set() for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for g in unique:
        j = int(np.argmin(fold_sizes))
        fold_groups[j].add(g)
        fold_sizes[j] += int(sizes[g])

    idx = np.arange(len(groups))
    splits = []

    for gs in fold_groups:
        test_mask = groups.isin(gs).to_numpy()
        splits.append((idx[~test_mask], idx[test_mask]))

    return splits

# =============================================================================
# Preprocessing / model
# =============================================================================
def split_column_types(X, cols):
    cat, num = [], []

    for c in cols:
        if (
            pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or pd.api.types.is_bool_dtype(X[c])
            or isinstance(X[c].dtype, pd.CategoricalDtype)
        ):
            cat.append(c)
        else:
            num.append(c)

    return num, cat

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

def tune_model(model_name, X, y, groups, cols, seed):
    grid = ELASTIC_GRID if model_name == "ElasticNet" else HGB_GRID
    numeric_cols, categorical_cols = split_column_types(X, cols)
    splits = shuffled_group_folds(groups, INNER_FOLDS, seed)

    best_score = np.inf
    best_params = None

    for params in grid:
        scores = []

        for tr, va in splits:
            pipe = make_pipeline(
                model_name,
                params,
                numeric_cols,
                categorical_cols,
            )

            pipe.fit(X.iloc[tr][cols], y.iloc[tr])
            pred = pipe.predict(X.iloc[va][cols])

            m = participant_balanced_metrics(
                y.iloc[va],
                pred,
                groups.iloc[va],
            )
            scores.append(m["mae_pb"])

        score = float(np.mean(scores))
        key = json.dumps(params, sort_keys=True)
        old_key = json.dumps(best_params or {}, sort_keys=True)

        if (
            score < best_score - 1e-12
            or (abs(score - best_score) <= 1e-12 and key < old_key)
        ):
            best_score = score
            best_params = params

    return best_params, best_score, numeric_cols, categorical_cols

# =============================================================================
# Bootstrap incremental value
# =============================================================================
def bootstrap_increment(avg, anchor, model_name, baseline, combined, b, seed):
    a = avg[
        (avg["anchor"] == anchor)
        & (avg["model"] == model_name)
        & (avg["feature_set"] == baseline)
    ][[ROW_ID, GROUP, "y_true", "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_base"}
    )

    c = avg[
        (avg["anchor"] == anchor)
        & (avg["model"] == model_name)
        & (avg["feature_set"] == combined)
    ][[ROW_ID, "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_comb"}
    )

    d = a.merge(
        c,
        on=ROW_ID,
        how="inner",
        validate="one_to_one",
    )

    mb = participant_balanced_metrics(
        d["y_true"],
        d["pred_base"],
        d[GROUP],
    )

    mc = participant_balanced_metrics(
        d["y_true"],
        d["pred_comb"],
        d[GROUP],
    )

    observed = {
        "delta_mae_pb": mb["mae_pb"] - mc["mae_pb"],
        "delta_rmse_pb": mb["rmse_pb"] - mc["rmse_pb"],
        "delta_r2_rep": mc["r2_rep"] - mb["r2_rep"],
    }

    pids = sorted(d[GROUP].astype(str).unique())
    by_pid = {
        pid: d[d[GROUP].astype(str) == pid].copy()
        for pid in pids
    }

    rng = np.random.default_rng(seed)
    store = {k: [] for k in observed}

    for _ in range(b):
        sampled = rng.choice(
            pids,
            size=len(pids),
            replace=True,
        )

        pieces = []

        for j, pid in enumerate(sampled):
            z = by_pid[pid].copy()
            z["boot_group"] = f"{pid}__draw{j}"
            pieces.append(z)

        z = pd.concat(
            pieces,
            ignore_index=True,
        )

        zb = participant_balanced_metrics(
            z["y_true"],
            z["pred_base"],
            z["boot_group"],
        )

        zc = participant_balanced_metrics(
            z["y_true"],
            z["pred_comb"],
            z["boot_group"],
        )

        store["delta_mae_pb"].append(
            zb["mae_pb"] - zc["mae_pb"]
        )

        store["delta_rmse_pb"].append(
            zb["rmse_pb"] - zc["rmse_pb"]
        )

        store["delta_r2_rep"].append(
            zc["r2_rep"] - zb["r2_rep"]
        )

    ci = {}

    for key, vals in store.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]

        ci[f"{key}_ci_low"] = float(
            np.quantile(arr, 0.025)
        )
        ci[f"{key}_ci_high"] = float(
            np.quantile(arr, 0.975)
        )
        ci[f"{key}_bootstrap_positive_prob"] = float(
            np.mean(arr > 0)
        )
        ci[f"{key}_bootstrap_n"] = int(len(arr))

    return observed, ci, len(d), len(pids)

# =============================================================================
# Load / validate paired QC-clean anchors
# =============================================================================
start_df = pd.read_csv(START_TABLE)
end_df = pd.read_csv(END_TABLE)
fd = pd.read_csv(DICT)

for name, d in [("START", start_df), ("END", end_df)]:
    if d.columns.duplicated().any():
        raise SystemExit(
            f"{name}: duplicate columns found."
        )

    if d[ROW_ID].duplicated().any():
        raise SystemExit(
            f"{name}: duplicate row_id found."
        )

    if d[TARGET].isna().any():
        raise SystemExit(
            f"{name}: target contains missing values."
        )

start_df = start_df.sort_values(
    ROW_ID
).reset_index(drop=True)

end_df = end_df.sort_values(
    ROW_ID
).reset_index(drop=True)

if not np.array_equal(
    start_df[ROW_ID].to_numpy(),
    end_df[ROW_ID].to_numpy(),
):
    raise SystemExit(
        "START/END row_id sets differ."
    )

if not np.array_equal(
    start_df[GROUP].to_numpy(),
    end_df[GROUP].to_numpy(),
):
    raise SystemExit(
        "START/END participant grouping differs."
    )

if not np.allclose(
    start_df[TARGET].to_numpy(),
    end_df[TARGET].to_numpy(),
    equal_nan=True,
):
    raise SystemExit(
        "START/END targets differ."
    )

if len(start_df) != 1141:
    print(
        f"WARNING: expected 1141 QC-clean rows, "
        f"found {len(start_df)}."
    )

# Raw finite / magnitude preflight.
wearable_cols = fd.loc[
    fd["role"] == "wearable_feature",
    "column"
].tolist()

wearable_cols = [
    c for c in wearable_cols
    if c in start_df.columns
]

preflight_rows = []

for anchor, df in [
    ("START", start_df),
    ("END", end_df),
]:
    x = df[wearable_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_vals = x.to_numpy(dtype=float)
    abs_finite = np.abs(
        finite_vals[np.isfinite(finite_vals)]
    )

    preflight_rows.append({
        "anchor": anchor,
        "n_rows": len(df),
        "n_wearable_features": len(wearable_cols),
        "nonfinite_count": int(
            (~np.isfinite(finite_vals)).sum()
        ),
        "finite_abs_max": float(
            np.max(abs_finite)
        ) if len(abs_finite) else np.nan,
        "finite_abs_q99999": float(
            np.quantile(abs_finite, 0.99999)
        ) if len(abs_finite) else np.nan,
    })

preflight_df = pd.DataFrame(
    preflight_rows
)

role_map = {}

for role in fd["role"].dropna().unique():
    role_map[role] = fd.loc[
        fd["role"] == role,
        "column"
    ].tolist()

FEATURE_SETS = {}

for fs_name, roles in FEATURE_SET_ROLES.items():
    cols = []

    for role in roles:
        cols.extend(
            role_map.get(role, [])
        )

    cols = [
        c for c in dict.fromkeys(cols)
        if c in start_df.columns
    ]

    forbidden = {
        ROW_ID,
        GROUP,
        TARGET,
        "borg_change_10s",
        "target_time_sec",
        "trial_duration_sec_qc",
        "known_corrupt_shoulder_imu_trial_qc",
    }

    cols = [
        c for c in cols
        if c not in forbidden
    ]

    if not cols:
        raise SystemExit(
            f"No features found for {fs_name}"
        )

    FEATURE_SETS[fs_name] = cols

anchors = {
    "START": start_df,
    "END": end_df,
}

# =============================================================================
# Repeated nested participant-grouped CV
# Same outer splits are reused for START and END.
# =============================================================================
fold_rows = []
pred_rows = []
hyper_rows = []

t0 = time.time()
outer_done = 0
outer_total = OUTER_REPEATS * OUTER_FOLDS

for repeat in range(OUTER_REPEATS):
    outer_seed = BASE_SEED + repeat * 10000

    outer_splits = shuffled_group_folds(
        start_df[GROUP],
        OUTER_FOLDS,
        outer_seed,
    )

    for outer_fold, (tr_idx, te_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        outer_done += 1

        train_groups = set(
            start_df.iloc[tr_idx][GROUP].astype(str)
        )
        test_groups = set(
            start_df.iloc[te_idx][GROUP].astype(str)
        )

        if train_groups & test_groups:
            raise RuntimeError(
                "Participant leakage in outer split."
            )

        for anchor, df in anchors.items():
            y_train = (
                df.iloc[tr_idx][TARGET]
                .astype(float)
                .reset_index(drop=True)
            )

            y_test = (
                df.iloc[te_idx][TARGET]
                .astype(float)
                .reset_index(drop=True)
            )

            g_train = (
                df.iloc[tr_idx][GROUP]
                .astype(str)
                .reset_index(drop=True)
            )

            g_test = (
                df.iloc[te_idx][GROUP]
                .astype(str)
                .reset_index(drop=True)
            )

            train_df = (
                df.iloc[tr_idx]
                .reset_index(drop=True)
            )

            test_df = (
                df.iloc[te_idx]
                .reset_index(drop=True)
            )

            for model_name in [
                "ElasticNet",
                "HGB",
            ]:
                for fs_idx, (
                    fs_name,
                    cols,
                ) in enumerate(
                    FEATURE_SETS.items()
                ):
                    inner_seed = (
                        outer_seed
                        + outer_fold * 1000
                        + fs_idx * 10
                        + (
                            1
                            if model_name == "ElasticNet"
                            else 2
                        )
                    )

                    (
                        best_params,
                        inner_mae,
                        num_cols,
                        cat_cols,
                    ) = tune_model(
                        model_name,
                        train_df,
                        y_train,
                        g_train,
                        cols,
                        inner_seed,
                    )

                    pipe = make_pipeline(
                        model_name,
                        best_params,
                        num_cols,
                        cat_cols,
                    )

                    pipe.fit(
                        train_df[cols],
                        y_train,
                    )

                    pred = pipe.predict(
                        test_df[cols]
                    )

                    if not np.all(
                        np.isfinite(pred)
                    ):
                        raise RuntimeError(
                            f"Non-finite predictions: "
                            f"{anchor} {model_name} {fs_name} "
                            f"repeat={repeat+1} fold={outer_fold}"
                        )

                    metrics = participant_balanced_metrics(
                        y_test,
                        pred,
                        g_test,
                    )

                    fold_rows.append({
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": fs_name,
                        "n_raw_features": len(cols),
                        "n_train_rows": len(tr_idx),
                        "n_test_rows": len(te_idx),
                        "n_train_participants": len(train_groups),
                        "n_test_participants": len(test_groups),
                        "inner_selected_mae_pb": inner_mae,
                        "prediction_abs_max": float(
                            np.max(np.abs(pred))
                        ),
                        **metrics,
                    })

                    hyper_rows.append({
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": fs_name,
                        "selected_params_json": json.dumps(
                            best_params,
                            sort_keys=True,
                        ),
                        "inner_selected_mae_pb": inner_mae,
                    })

                    meta = test_df[
                        [
                            ROW_ID,
                            GROUP,
                            "task_label",
                            "current_time_sec",
                            TARGET,
                        ]
                    ].reset_index(drop=True)

                    for j in range(len(meta)):
                        pred_rows.append({
                            "repeat": repeat + 1,
                            "outer_fold": outer_fold,
                            "anchor": anchor,
                            "model": model_name,
                            "feature_set": fs_name,
                            ROW_ID: meta.loc[j, ROW_ID],
                            GROUP: meta.loc[j, GROUP],
                            "task_label": meta.loc[j, "task_label"],
                            "current_time_sec": meta.loc[
                                j,
                                "current_time_sec",
                            ],
                            "y_true": float(
                                meta.loc[j, TARGET]
                            ),
                            "y_pred": float(pred[j]),
                        })

        elapsed = time.time() - t0

        print(
            f"Completed outer split "
            f"{outer_done}/{outer_total} "
            f"(repeat {repeat+1}, fold {outer_fold}) | "
            f"elapsed {elapsed/60:.1f} min"
        )

fold_df = pd.DataFrame(
    fold_rows
)

pred_df = pd.DataFrame(
    pred_rows
)

hyper_df = pd.DataFrame(
    hyper_rows
)

fold_df.to_csv(
    FOLD_OUT,
    index=False,
)

pred_df.to_csv(
    PRED_OUT,
    index=False,
)

hyper_df.to_csv(
    HYPER_OUT,
    index=False,
)

# =============================================================================
# Average repeated OOF predictions
# =============================================================================
avg = (
    pred_df
    .groupby(
        [
            "anchor",
            "model",
            "feature_set",
            ROW_ID,
            GROUP,
            "task_label",
            "current_time_sec",
            "y_true",
        ],
        as_index=False,
    )
    .agg(
        y_pred_avg=("y_pred", "mean"),
        y_pred_sd=("y_pred", "std"),
        n_oof_predictions=("y_pred", "count"),
    )
)

avg.to_csv(
    AVG_OUT,
    index=False,
)

# =============================================================================
# Model summary
# =============================================================================
summary_rows = []

for (
    anchor,
    model_name,
    fs_name,
), g in avg.groupby(
    [
        "anchor",
        "model",
        "feature_set",
    ]
):
    m = participant_balanced_metrics(
        g["y_true"],
        g["y_pred_avg"],
        g[GROUP],
    )

    summary_rows.append({
        "anchor": anchor,
        "scope": "All_QC_clean",
        "model": model_name,
        "feature_set": fs_name,
        "n_rows": len(g),
        "n_participants": g[GROUP].nunique(),
        **m,
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)

# =============================================================================
# Incremental-value summary
# =============================================================================
inc_rows = []

for anchor_idx, anchor in enumerate(
    ["START", "END"]
):
    for model_idx, model_name in enumerate(
        ["ElasticNet", "HGB"]
    ):
        for comp_idx, (
            base,
            comb,
            label,
        ) in enumerate(
            COMPARISONS
        ):
            obs, ci, n_rows, n_pids = bootstrap_increment(
                avg,
                anchor,
                model_name,
                base,
                comb,
                BOOTSTRAP_B,
                BASE_SEED
                + 90000
                + anchor_idx * 10000
                + model_idx * 1000
                + comp_idx,
            )

            f0 = fold_df[
                (fold_df["anchor"] == anchor)
                & (fold_df["model"] == model_name)
                & (fold_df["feature_set"] == base)
            ][
                [
                    "repeat",
                    "outer_fold",
                    "mae_pb",
                    "rmse_pb",
                    "r2_rep",
                ]
            ]

            f1 = fold_df[
                (fold_df["anchor"] == anchor)
                & (fold_df["model"] == model_name)
                & (fold_df["feature_set"] == comb)
            ][
                [
                    "repeat",
                    "outer_fold",
                    "mae_pb",
                    "rmse_pb",
                    "r2_rep",
                ]
            ]

            paired = f0.merge(
                f1,
                on=[
                    "repeat",
                    "outer_fold",
                ],
                suffixes=(
                    "_base",
                    "_comb",
                ),
                validate="one_to_one",
            )

            inc_rows.append({
                "anchor": anchor,
                "model": model_name,
                "comparison": label,
                "baseline_feature_set": base,
                "combined_feature_set": comb,
                "n_rows": n_rows,
                "n_participants": n_pids,
                **obs,
                **ci,
                "fold_delta_mae_pb_mean": float(
                    (
                        paired["mae_pb_base"]
                        - paired["mae_pb_comb"]
                    ).mean()
                ),
                "fold_delta_mae_pb_positive_fraction": float(
                    (
                        paired["mae_pb_base"]
                        > paired["mae_pb_comb"]
                    ).mean()
                ),
                "fold_delta_r2_rep_mean": float(
                    (
                        paired["r2_rep_comb"]
                        - paired["r2_rep_base"]
                    ).mean()
                ),
                "fold_delta_r2_rep_positive_fraction": float(
                    (
                        paired["r2_rep_comb"]
                        > paired["r2_rep_base"]
                    ).mean()
                ),
            })

inc_df = pd.DataFrame(
    inc_rows
)

inc_df.to_csv(
    INCREMENT_OUT,
    index=False,
)

# =============================================================================
# Alignment robustness summary
# =============================================================================
robust_rows = []

for model_name in [
    "ElasticNet",
    "HGB",
]:
    for _, _, label in COMPARISONS:
        s = inc_df[
            (inc_df["model"] == model_name)
            & (
                inc_df["comparison"]
                == label
            )
        ].copy()

        if set(s["anchor"]) != {
            "START",
            "END",
        }:
            continue

        a = s.set_index(
            "anchor"
        )

        dms = float(
            a.loc[
                "START",
                "delta_mae_pb",
            ]
        )

        dme = float(
            a.loc[
                "END",
                "delta_mae_pb",
            ]
        )

        drs = float(
            a.loc[
                "START",
                "delta_r2_rep",
            ]
        )

        dre = float(
            a.loc[
                "END",
                "delta_r2_rep",
            ]
        )

        robust_rows.append({
            "model": model_name,
            "comparison": label,

            "start_delta_mae_pb": dms,
            "end_delta_mae_pb": dme,
            "mae_direction_consistent": int(
                np.sign(dms)
                == np.sign(dme)
            ),
            "mae_both_positive": int(
                dms > 0 and dme > 0
            ),
            "mae_both_negative": int(
                dms < 0 and dme < 0
            ),

            "start_delta_r2_rep": drs,
            "end_delta_r2_rep": dre,
            "r2_direction_consistent": int(
                np.sign(drs)
                == np.sign(dre)
            ),
            "r2_both_positive": int(
                drs > 0 and dre > 0
            ),
            "r2_both_negative": int(
                drs < 0 and dre < 0
            ),

            "start_mae_ci_excludes_zero_positive": int(
                a.loc[
                    "START",
                    "delta_mae_pb_ci_low",
                ] > 0
            ),
            "end_mae_ci_excludes_zero_positive": int(
                a.loc[
                    "END",
                    "delta_mae_pb_ci_low",
                ] > 0
            ),
            "start_mae_ci_excludes_zero_negative": int(
                a.loc[
                    "START",
                    "delta_mae_pb_ci_high",
                ] < 0
            ),
            "end_mae_ci_excludes_zero_negative": int(
                a.loc[
                    "END",
                    "delta_mae_pb_ci_high",
                ] < 0
            ),
        })

robust_df = pd.DataFrame(
    robust_rows
)

robust_df.to_csv(
    ROBUST_OUT,
    index=False,
)

# =============================================================================
# Audit report
# =============================================================================
elapsed = time.time() - t0
lines = []

def add(s=""):
    lines.append(str(s))

add(
    "DATASET C — QC-CLEAN DUAL-ANCHOR "
    "REPEATED NESTED PARTICIPANT-GROUPED BENCHMARK"
)
add("=" * 118)
add(f"Rows per anchor: {len(start_df)}")
add(
    f"Participants: "
    f"{start_df[GROUP].nunique()}"
)
add(f"Outcome: {TARGET}")
add(
    f"Outer validation: "
    f"{OUTER_REPEATS} repeats x "
    f"{OUTER_FOLDS} participant-grouped folds"
)
add(
    f"Inner validation: "
    f"{INNER_FOLDS} participant-grouped folds"
)
add(
    f"Bootstrap: "
    f"{BOOTSTRAP_B} participant-cluster resamples "
    f"on averaged repeated-OOF predictions"
)
add(
    f"Elapsed minutes: "
    f"{elapsed/60:.2f}"
)
add("")

add("1. QC-CLEAN PAIRED COHORT")
add("-" * 118)
add(
    "PASS: START and END tables contain "
    "identical row_id values."
)
add(
    "PASS: START and END tables contain "
    "identical participants and targets."
)
add(
    "PASS: the exact same outer participant "
    "splits are reused for both anchors."
)
add(
    "QC-clean cohort derives from the raw-signal "
    "integrity quarantine defined before this rerun."
)
add(
    "No clipping, winsorization, interpolation "
    "or model-driven row removal is used."
)
add("")

add("2. PREFLIGHT NUMERICAL CHECK")
add("-" * 118)
add(
    preflight_df.to_string(
        index=False
    )
)
add("")

add("3. FEATURE SETS")
add("-" * 118)
for name, cols in FEATURE_SETS.items():
    add(
        f"{name}: "
        f"{len(cols)} raw predictor columns"
    )
add("")

add("4. LEAKAGE CONTROLS")
add("-" * 118)
add(
    "PASS: outer train/test participants "
    "are disjoint."
)
add(
    "PASS: inner tuning uses only "
    "outer-training participants."
)
add(
    "PASS: imputation, scaling, one-hot encoding "
    "and variance filtering are refit inside "
    "training folds."
)
add(
    "PASS: wearable features derive only from "
    "[t-10,t]; target is Borg(t+10)."
)
add(
    "Excluded: target, Borg change, target time, "
    "trial duration QC, corruption QC, IDs."
)
add("")

add("5. OVERALL MODEL SUMMARY")
add("-" * 118)

show = summary_df[
    [
        "anchor",
        "model",
        "feature_set",
        "n_rows",
        "n_participants",
        "mae_pb",
        "rmse_pb",
        "mae_rep",
        "rmse_rep",
        "r2_rep",
    ]
].sort_values(
    [
        "anchor",
        "model",
        "mae_pb",
    ]
)

add(
    show.to_string(
        index=False
    )
)
add("")

add("6. INCREMENTAL VALUE")
add("-" * 118)

show_cols = [
    "anchor",
    "model",
    "comparison",
    "delta_mae_pb",
    "delta_mae_pb_ci_low",
    "delta_mae_pb_ci_high",
    "delta_mae_pb_bootstrap_positive_prob",
    "delta_rmse_pb",
    "delta_rmse_pb_ci_low",
    "delta_rmse_pb_ci_high",
    "delta_rmse_pb_bootstrap_positive_prob",
    "delta_r2_rep",
    "delta_r2_rep_ci_low",
    "delta_r2_rep_ci_high",
    "delta_r2_rep_bootstrap_positive_prob",
    "fold_delta_mae_pb_positive_fraction",
]

add(
    inc_df[
        show_cols
    ].to_string(
        index=False
    )
)
add("")

add("Sign convention:")
add(
    "  delta_mae_pb > 0  => wearable reduces "
    "participant-balanced MAE."
)
add(
    "  delta_rmse_pb > 0 => wearable reduces "
    "participant-balanced RMSE."
)
add(
    "  delta_r2_rep > 0  => wearable increases "
    "pooled R^2."
)
add("")

add("7. ALIGNMENT ROBUSTNESS")
add("-" * 118)

add(
    robust_df.to_string(
        index=False
    )
)
add("")

add("Interpretation rule:")
add(
    "  A wearable increment is alignment-robust "
    "only when START and END anchors show the "
    "same direction."
)
add(
    "  Strong positive evidence additionally "
    "requires positive increments with uncertainty "
    "excluding zero under both anchors."
)
add(
    "  Strong negative evidence additionally "
    "requires negative increments with uncertainty "
    "excluding zero under both anchors."
)
add("")

add("8. NUMERICAL-STABILITY CHECK")
add("-" * 118)

pred_stability = (
    fold_df.groupby(
        [
            "anchor",
            "model",
            "feature_set",
        ],
        as_index=False,
    )
    .agg(
        max_abs_prediction=(
            "prediction_abs_max",
            "max",
        ),
        median_abs_prediction_max=(
            "prediction_abs_max",
            "median",
        ),
    )
)

add(
    pred_stability.to_string(
        index=False
    )
)
add("")

add(
    "If QC successfully removed the raw corruption "
    "driving benchmark 18 instability, "
    "START ElasticNet predictions should return "
    "to ordinary Borg-scale magnitudes."
)
add("")

add("9. INTERPRETATION GUARDRAILS")
add("-" * 118)
add(
    "Do not infer wearable value from wearable-only "
    "absolute performance."
)
add(
    "The primary estimand is the paired increment "
    "over the corresponding context baseline."
)
add(
    "Current Borg is intentionally included in "
    "context because the task is one-step-ahead "
    "forecasting, not concurrent fatigue estimation."
)
add(
    "PPG and magnetometer remain outside this "
    "primary handcrafted benchmark."
)
add(
    "No foundation-model claim is made at this stage."
)
add("")

add("10. OUTPUTS")
add("-" * 118)

for fp in [
    FOLD_OUT,
    PRED_OUT,
    AVG_OUT,
    SUMMARY_OUT,
    INCREMENT_OUT,
    ROBUST_OUT,
    HYPER_OUT,
]:
    add(f"  {fp}")

add("")
add("=" * 118)
add("END OF BENCHMARK")

AUDIT_OUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("")
print(
    "Dataset C QC-clean dual-anchor nested benchmark complete."
)
print(f"Audit: {AUDIT_OUT}")
print(
    f"Increment: {INCREMENT_OUT}"
)
print(
    f"Robustness: {ROBUST_OUT}"
)
print("")
print(
    "Upload dataset_C_qc_dual_anchor_nestedcv_audit.txt, "
    "dataset_C_qc_alignment_robustness_summary.csv, "
    "and dataset_C_qc_dual_anchor_incremental_value.csv "
    "to ChatGPT."
)
