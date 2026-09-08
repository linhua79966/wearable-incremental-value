from pathlib import Path
import json
import time
import warnings
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
# Fixed paths and design
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "derived" / "dataset_A_phase1" / "dataset_A_phase1_modeling_table.csv"
DICT = ROOT / "derived" / "dataset_A_phase1" / "dataset_A_phase1_feature_dictionary.csv"

OUTDIR = ROOT / "results" / "dataset_A_phase1_residualized"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

FOLD_OUT = OUTDIR / "dataset_A_residualized_fold_metrics.csv"
PRED_OUT = OUTDIR / "dataset_A_residualized_predictions.csv"
AVG_OUT = OUTDIR / "dataset_A_residualized_avg_oof_predictions.csv"
SUMMARY_OUT = OUTDIR / "dataset_A_residualized_summary.csv"
INCREMENT_OUT = OUTDIR / "dataset_A_residualized_increment_summary.csv"
HYPER_OUT = OUTDIR / "dataset_A_residualized_hyperparameters.csv"
AUDIT_OUT = AUDITDIR / "dataset_A_residualized_audit.txt"

TARGET = "physical_fatigue_final"
GROUP = "participant_id"
ROW_ID = "source_file"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_B = 2000
BASE_SEED = 20260901

ELASTIC_GRID = [
    {"alpha": a, "l1_ratio": l1}
    for a, l1 in product([0.01, 0.1, 1.0, 10.0], [0.1, 0.5, 0.9])
]

HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
    for lr, leaves, l2 in product([0.03, 0.08], [7, 15], [1.0, 10.0])
]

# =============================================================================
# Utility functions
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

    return {
        "mae_pb": float(by_g["mae"].mean()),
        "rmse_pb": float(np.sqrt(by_g["mse"].mean())),
        "mae_rep": float(mean_absolute_error(d["y"], d["p"])),
        "rmse_rep": float(np.sqrt(mean_squared_error(d["y"], d["p"]))),
        "r2_rep": float(r2_score(d["y"], d["p"])) if d["y"].nunique() > 1 else np.nan,
    }


def shuffled_group_folds(groups, n_splits, seed):
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique = groups.unique().tolist()

    if len(unique) < n_splits:
        raise ValueError(f"Need {n_splits} groups, found {len(unique)}")

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


def make_pipeline(model_name, params, numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        num = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num, numeric_cols))

    if categorical_cols:
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat, categorical_cols))

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


def tune(model_name, X, y, groups, numeric_cols, categorical_cols, seed):
    grid = ELASTIC_GRID if model_name == "ElasticNet" else HGB_GRID
    splits = shuffled_group_folds(groups, INNER_FOLDS, seed)

    best_score = np.inf
    best_params = None

    for params in grid:
        scores = []

        for tr, va in splits:
            pipe = make_pipeline(
                model_name, params, numeric_cols, categorical_cols
            )
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[va])

            m = participant_balanced_metrics(
                y.iloc[va], pred, groups.iloc[va]
            )
            scores.append(m["mae_pb"])

        score = float(np.mean(scores))
        key = json.dumps(params, sort_keys=True)

        if (
            score < best_score - 1e-12
            or (
                abs(score - best_score) <= 1e-12
                and key < json.dumps(best_params or {}, sort_keys=True)
            )
        ):
            best_score = score
            best_params = params

    return best_params, best_score


def crossfit_predictions(
    model_name,
    params,
    X,
    y,
    groups,
    numeric_cols,
    categorical_cols,
    seed,
):
    pred = np.full(len(X), np.nan, dtype=float)
    splits = shuffled_group_folds(groups, INNER_FOLDS, seed)

    for tr, va in splits:
        pipe = make_pipeline(
            model_name, params, numeric_cols, categorical_cols
        )
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred[va] = pipe.predict(X.iloc[va])

    if np.isnan(pred).any():
        raise RuntimeError("Cross-fitted context predictions contain NaN.")

    return pred


def bootstrap_increment(avg, model_name, variant_name, b=2000, seed=0):
    c = avg[
        (avg["model"] == model_name) &
        (avg["variant"] == "ContextOnly")
    ][[ROW_ID, GROUP, "y_true", "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_context"}
    )

    r = avg[
        (avg["model"] == model_name) &
        (avg["variant"] == variant_name)
    ][[ROW_ID, "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_resid"}
    )

    d = c.merge(r, on=ROW_ID, validate="one_to_one")

    mc = participant_balanced_metrics(
        d["y_true"], d["pred_context"], d[GROUP]
    )
    mr = participant_balanced_metrics(
        d["y_true"], d["pred_resid"], d[GROUP]
    )

    observed = {
        "delta_mae_pb": mc["mae_pb"] - mr["mae_pb"],
        "delta_rmse_pb": mc["rmse_pb"] - mr["rmse_pb"],
        "delta_r2_rep": mr["r2_rep"] - mc["r2_rep"],
    }

    pids = sorted(d[GROUP].astype(str).unique())
    by_pid = {pid: d[d[GROUP].astype(str) == pid] for pid in pids}
    rng = np.random.default_rng(seed)

    store = {k: [] for k in observed}

    for _ in range(b):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        pieces = []

        for k, pid in enumerate(sampled):
            z = by_pid[pid].copy()
            z["boot_group"] = f"{pid}__{k}"
            pieces.append(z)

        z = pd.concat(pieces, ignore_index=True)

        zc = participant_balanced_metrics(
            z["y_true"], z["pred_context"], z["boot_group"]
        )
        zr = participant_balanced_metrics(
            z["y_true"], z["pred_resid"], z["boot_group"]
        )

        store["delta_mae_pb"].append(zc["mae_pb"] - zr["mae_pb"])
        store["delta_rmse_pb"].append(zc["rmse_pb"] - zr["rmse_pb"])
        store["delta_r2_rep"].append(zr["r2_rep"] - zc["r2_rep"])

    ci = {}
    for key, values in store.items():
        a = np.asarray(values, dtype=float)
        a = a[np.isfinite(a)]
        ci[f"{key}_ci_low"] = float(np.quantile(a, 0.025))
        ci[f"{key}_ci_high"] = float(np.quantile(a, 0.975))
        ci[f"{key}_bootstrap_positive_prob"] = float(np.mean(a > 0))

    return observed, ci, len(d), len(pids)


# =============================================================================
# Load and define feature families
# =============================================================================
df = pd.read_csv(TABLE)
fd = pd.read_csv(DICT)

if df.columns.duplicated().any():
    raise SystemExit("Duplicate columns found.")

if df[ROW_ID].duplicated().any():
    raise SystemExit("source_file is not unique.")

context_cols = fd.loc[
    fd["role"] == "context_core", "column"
].tolist()
context_cols = [c for c in context_cols if c in df.columns]

wearable_cols = fd.loc[
    fd["role"] == "wearable_feature", "column"
].tolist()
wearable_cols = [c for c in wearable_cols if c in df.columns]

categorical_context = [c for c in ["task", "gender"] if c in context_cols]
numeric_context = [c for c in context_cols if c not in categorical_context]

def starts_any(c, prefixes):
    return any(c.startswith(p) for p in prefixes)

families = {
    "Physiology": [
        c for c in wearable_cols
        if starts_any(c, (
            "sensor__hr__", "sensor__hr_processed__",
            "sensor__hrv__", "sensor__rr__",
            "sensor__ecg__", "sensor__temperature__",
        ))
    ],
    "ChestAcceleration": [
        c for c in wearable_cols
        if starts_any(c, (
            "sensor__acc_x__", "sensor__acc_y__", "sensor__acc_z__",
            "sensor__chest_acc_",
        ))
    ],
    "IMUAcceleration": [
        c for c in wearable_cols
        if c.startswith("sensor__imu")
    ],
    "AllWearable": list(wearable_cols),
}

for name, cols in families.items():
    if not cols:
        raise SystemExit(f"No columns found for sensor family: {name}")

# =============================================================================
# Outer repeated grouped CV
# =============================================================================
fold_rows = []
pred_rows = []
hyper_rows = []

t0 = time.time()
outer_done = 0
outer_total = OUTER_REPEATS * OUTER_FOLDS

for repeat in range(OUTER_REPEATS):
    outer_seed = BASE_SEED + 1000 * repeat
    outer_splits = shuffled_group_folds(
        df[GROUP], OUTER_FOLDS, outer_seed
    )

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        outer_done += 1

        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)

        train_groups = set(train[GROUP].astype(str))
        test_groups = set(test[GROUP].astype(str))

        if train_groups & test_groups:
            raise RuntimeError("Outer participant leakage detected.")

        y_train = train[TARGET].astype(float).reset_index(drop=True)
        y_test = test[TARGET].astype(float).reset_index(drop=True)
        g_train = train[GROUP].astype(str).reset_index(drop=True)
        g_test = test[GROUP].astype(str).reset_index(drop=True)

        Xc_train = train[context_cols].reset_index(drop=True)
        Xc_test = test[context_cols].reset_index(drop=True)

        for model_name in ["ElasticNet", "HGB"]:
            # ---------------------------------------------------------
            # Step 1: tune context model using outer-training data only
            # ---------------------------------------------------------
            ctx_seed = outer_seed + outer_fold * 100 + (
                1 if model_name == "ElasticNet" else 2
            )

            ctx_params, ctx_inner_mae = tune(
                model_name,
                Xc_train,
                y_train,
                g_train,
                numeric_context,
                categorical_context,
                ctx_seed,
            )

            ctx_pipe = make_pipeline(
                model_name,
                ctx_params,
                numeric_context,
                categorical_context,
            )
            ctx_pipe.fit(Xc_train, y_train)
            ctx_test_pred = ctx_pipe.predict(Xc_test)

            ctx_metrics = participant_balanced_metrics(
                y_test, ctx_test_pred, g_test
            )

            fold_rows.append({
                "repeat": repeat + 1,
                "outer_fold": outer_fold,
                "model": model_name,
                "variant": "ContextOnly",
                "sensor_family": "None",
                "n_sensor_features": 0,
                "inner_context_mae_pb": ctx_inner_mae,
                **ctx_metrics,
            })

            hyper_rows.append({
                "repeat": repeat + 1,
                "outer_fold": outer_fold,
                "model": model_name,
                "stage": "context",
                "sensor_family": "None",
                "selected_params_json": json.dumps(ctx_params, sort_keys=True),
                "inner_mae_pb": ctx_inner_mae,
            })

            for j in range(len(test)):
                pred_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "variant": "ContextOnly",
                    "sensor_family": "None",
                    ROW_ID: test.loc[j, ROW_ID],
                    GROUP: str(test.loc[j, GROUP]),
                    "task": test.loc[j, "task"],
                    "rep": test.loc[j, "rep"],
                    "y_true": float(y_test.iloc[j]),
                    "y_pred": float(ctx_test_pred[j]),
                })

            # ---------------------------------------------------------
            # Step 2: cross-fit context prediction inside outer train
            #         and form residual target
            # ---------------------------------------------------------
            ctx_oof_pred = crossfit_predictions(
                model_name,
                ctx_params,
                Xc_train,
                y_train,
                g_train,
                numeric_context,
                categorical_context,
                ctx_seed + 50,
            )

            residual_target = pd.Series(
                y_train.to_numpy() - ctx_oof_pred,
                name="context_residual",
            )

            # ---------------------------------------------------------
            # Step 3: train sensor residual correctors by family
            # ---------------------------------------------------------
            for fam_idx, (family, sensor_cols) in enumerate(families.items()):
                Xw_train = train[sensor_cols].reset_index(drop=True)
                Xw_test = test[sensor_cols].reset_index(drop=True)

                resid_seed = (
                    outer_seed
                    + outer_fold * 1000
                    + fam_idx * 10
                    + (3 if model_name == "ElasticNet" else 4)
                )

                resid_params, resid_inner_mae = tune(
                    model_name,
                    Xw_train,
                    residual_target,
                    g_train,
                    numeric_cols=sensor_cols,
                    categorical_cols=[],
                    seed=resid_seed,
                )

                resid_pipe = make_pipeline(
                    model_name,
                    resid_params,
                    numeric_cols=sensor_cols,
                    categorical_cols=[],
                )
                resid_pipe.fit(Xw_train, residual_target)

                residual_test_pred = resid_pipe.predict(Xw_test)
                final_pred = ctx_test_pred + residual_test_pred

                metrics = participant_balanced_metrics(
                    y_test, final_pred, g_test
                )

                variant = f"Residualized_{family}"

                fold_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "variant": variant,
                    "sensor_family": family,
                    "n_sensor_features": len(sensor_cols),
                    "inner_context_mae_pb": ctx_inner_mae,
                    "inner_residual_mae_pb": resid_inner_mae,
                    **metrics,
                })

                hyper_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "stage": "residual",
                    "sensor_family": family,
                    "selected_params_json": json.dumps(resid_params, sort_keys=True),
                    "inner_mae_pb": resid_inner_mae,
                })

                for j in range(len(test)):
                    pred_rows.append({
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "variant": variant,
                        "sensor_family": family,
                        ROW_ID: test.loc[j, ROW_ID],
                        GROUP: str(test.loc[j, GROUP]),
                        "task": test.loc[j, "task"],
                        "rep": test.loc[j, "rep"],
                        "y_true": float(y_test.iloc[j]),
                        "y_pred": float(final_pred[j]),
                    })

        elapsed = time.time() - t0
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
# Averaged OOF predictions and summaries
# =============================================================================
avg = (
    pred_df
    .groupby(
        ["model", "variant", "sensor_family", ROW_ID, GROUP, "task", "rep", "y_true"],
        as_index=False,
    )
    .agg(
        y_pred_avg=("y_pred", "mean"),
        y_pred_sd=("y_pred", "std"),
        n_oof_predictions=("y_pred", "count"),
    )
)
avg.to_csv(AVG_OUT, index=False)

summary_rows = []

for (model_name, variant, family), g in avg.groupby(
    ["model", "variant", "sensor_family"]
):
    m = participant_balanced_metrics(
        g["y_true"], g["y_pred_avg"], g[GROUP]
    )
    summary_rows.append({
        "model": model_name,
        "variant": variant,
        "sensor_family": family,
        "n_repetitions": len(g),
        "n_participants": g[GROUP].nunique(),
        **m,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)

increment_rows = []

for model_idx, model_name in enumerate(["ElasticNet", "HGB"]):
    for fam_idx, family in enumerate(families.keys()):
        variant = f"Residualized_{family}"

        observed, ci, n_rows, n_pids = bootstrap_increment(
            avg,
            model_name=model_name,
            variant_name=variant,
            b=BOOTSTRAP_B,
            seed=BASE_SEED + 9000 + model_idx * 100 + fam_idx,
        )

        f0 = fold_df[
            (fold_df["model"] == model_name)
            & (fold_df["variant"] == "ContextOnly")
        ][["repeat","outer_fold","mae_pb","rmse_pb","r2_rep"]].copy()

        f1 = fold_df[
            (fold_df["model"] == model_name)
            & (fold_df["variant"] == variant)
        ][["repeat","outer_fold","mae_pb","rmse_pb","r2_rep"]].copy()

        paired = f0.merge(
            f1,
            on=["repeat","outer_fold"],
            suffixes=("_ctx","_resid"),
            validate="one_to_one",
        )

        increment_rows.append({
            "model": model_name,
            "sensor_family": family,
            "variant": variant,
            "n_repetitions": n_rows,
            "n_participants": n_pids,
            **observed,
            **ci,
            "fold_delta_mae_pb_mean": float(
                (paired["mae_pb_ctx"] - paired["mae_pb_resid"]).mean()
            ),
            "fold_delta_mae_pb_positive_fraction": float(
                (paired["mae_pb_ctx"] - paired["mae_pb_resid"] > 0).mean()
            ),
        })

inc_df = pd.DataFrame(increment_rows)
inc_df.to_csv(INCREMENT_OUT, index=False)

# =============================================================================
# Audit
# =============================================================================
elapsed = time.time() - t0
lines = []

def add(s=""):
    lines.append(str(s))

add("DATASET A — CONTEXT-RESIDUALIZED SENSOR-VALUE BENCHMARK")
add("=" * 118)
add(f"Input repetitions: {len(df)}")
add(f"Participants: {df[GROUP].nunique()}")
add(f"Outcome: {TARGET}")
add(f"Outer validation: {OUTER_REPEATS} x {OUTER_FOLDS} participant-grouped folds")
add(f"Inner tuning: {INNER_FOLDS} participant-grouped folds")
add(f"Bootstrap: {BOOTSTRAP_B} participant-cluster resamples")
add(f"Elapsed minutes: {elapsed/60:.2f}")
add()

add("1. CORE CONTEXT")
add("-" * 118)
for c in context_cols:
    add(f"  {c}")
add()

add("2. SENSOR FAMILIES")
add("-" * 118)
for family, cols in families.items():
    add(f"{family}: {len(cols)} features")
add()

add("3. METHOD")
add("-" * 118)
add("For each outer split:")
add("  1) Tune the context model using outer-training participants only.")
add("  2) Generate participant-grouped cross-fitted context predictions inside outer training.")
add("  3) Define training residuals = observed final fatigue - cross-fitted context prediction.")
add("  4) Tune a wearable residual model on those training residuals only.")
add("  5) Fit context and residual models on the complete outer-training set.")
add("  6) Outer-test prediction = context prediction + predicted wearable residual.")
add("Outer-test participants are untouched by all tuning and residual construction.")
add()

add("4. OVERALL PERFORMANCE")
add("-" * 118)
show = summary_df[
    ["model","variant","sensor_family","mae_pb","rmse_pb","mae_rep","rmse_rep","r2_rep"]
].sort_values(["model","mae_pb"])
add(show.to_string(index=False))
add()

add("5. INCREMENTAL VALUE OVER CORE CONTEXT")
add("-" * 118)
show_cols = [
    "model","sensor_family",
    "delta_mae_pb","delta_mae_pb_ci_low","delta_mae_pb_ci_high","delta_mae_pb_bootstrap_positive_prob",
    "delta_rmse_pb","delta_rmse_pb_ci_low","delta_rmse_pb_ci_high","delta_rmse_pb_bootstrap_positive_prob",
    "delta_r2_rep","delta_r2_rep_ci_low","delta_r2_rep_ci_high","delta_r2_rep_bootstrap_positive_prob",
    "fold_delta_mae_pb_positive_fraction",
]
add(inc_df[show_cols].to_string(index=False))
add()
add("Sign convention: positive delta = residualized wearable model improves over core context.")
add()

add("6. INTERPRETATION")
add("-" * 118)
add("This experiment tests whether wearable signals can explain context-unexplained fatigue variation.")
add("It is NOT a foundation-model experiment.")
add("A positive family-specific increment would justify carrying that modality into GPU representation learning.")
add("A null/negative result would support the view that handcrafted summaries do not yield transferable residual information.")
add("No cross-dataset claim is permitted until Dataset B/C are analyzed.")
add()
add("=" * 118)
add("END OF BENCHMARK")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Residualized benchmark complete.")
print(f"Audit: {AUDIT_OUT}")
print(f"Increment summary: {INCREMENT_OUT}")
print("")
print("Upload dataset_A_residualized_audit.txt and dataset_A_residualized_increment_summary.csv to ChatGPT.")
