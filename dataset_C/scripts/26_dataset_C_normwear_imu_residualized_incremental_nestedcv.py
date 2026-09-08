from pathlib import Path
import json
import time
import hashlib
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# 26_dataset_C_normwear_imu_residualized_incremental_nestedcv.py
#
# M5: CONTEXT-RESIDUALIZED FROZEN NORMWEAR REPRESENTATION
#
# Scientific question:
#   After removing the component of frozen NormWear mean768 that is predictable
#   from deployment-available context, does the residual representation provide
#   stable incremental predictive value beyond the SAME locked 18f context
#   baseline?
#
# Crucial leakage rule:
#   The context -> embedding residualizer is learned ONLY from the applicable
#   training participants:
#     - inner-train only during hyperparameter tuning;
#     - full outer-train only for the final outer-fold model.
#   Held-out participant embeddings are never used to fit the residualizer.
#
# Residualizer:
#   context preprocessing:
#       numeric median impute + StandardScaler
#       categorical most-frequent impute + OneHotEncoder
#   multi-output Ridge(alpha=1.0), FIXED before outcome inspection
#   residual = frozen_embedding - context_predicted_embedding
#
# Locked comparisons:
#   M5a = C0a core context + context-residualized NormWear mean768
#   M5b = C0b extended context + context-residualized NormWear mean768
#
# Context baselines:
#   literal 18f C0a/C0b held-out outputs, NOT re-fit here.
#
# Same evaluation family as 18f / Script 25:
#   - 5 x 5 participant-grouped outer validation
#   - 4-fold participant-grouped inner tuning
#   - ElasticNet and HGB
#   - identical hyperparameter grids
#   - participant-balanced MAE/RMSE
#   - pooled R2
#   - 2500 participant-cluster bootstrap
#
# NOT DONE:
#   - no change to frozen embedding or pooling;
#   - no CLS sensitivity;
#   - no NormWear adaptation/fine-tuning;
#   - no residualizer alpha tuning;
#   - no outcome-driven feature selection;
#   - no baseline re-fitting.
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DERIVED = ROOT / "derived" / "dataset_C_phase1"
START_TABLE = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_TABLE = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

NW_DIR = ROOT / "derived" / "dataset_C_normwear_imu"
START_EMB = NW_DIR / "dataset_C_normwear_imu_start_mean768.npz"
END_EMB = NW_DIR / "dataset_C_normwear_imu_end_mean768.npz"

LOCKED18 = ROOT / "results" / "dataset_C_phase1_dual_anchor_nestedcv_qc"
BASE_FOLD = LOCKED18 / "dataset_C_qc_dual_anchor_fold_metrics.csv"
BASE_PRED = LOCKED18 / "dataset_C_qc_dual_anchor_predictions.csv"
BASE_AVG = LOCKED18 / "dataset_C_qc_dual_anchor_avg_oof_predictions.csv"
BASE_SUMMARY = LOCKED18 / "dataset_C_qc_dual_anchor_model_summary.csv"

OUTDIR = ROOT / "results" / "dataset_C_normwear_imu_residualized_nestedcv"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

FOLD_OUT = OUTDIR / "dataset_C_normwear_residualized_fold_metrics.csv"
PRED_OUT = OUTDIR / "dataset_C_normwear_residualized_predictions.csv"
AVG_OUT = OUTDIR / "dataset_C_normwear_residualized_avg_oof_predictions.csv"
SUMMARY_OUT = OUTDIR / "dataset_C_normwear_residualized_model_summary_with_locked_baselines.csv"
INCREMENT_OUT = OUTDIR / "dataset_C_normwear_residualized_incremental_value.csv"
ROBUST_OUT = OUTDIR / "dataset_C_normwear_residualized_alignment_robustness.csv"
HYPER_OUT = OUTDIR / "dataset_C_normwear_residualized_selected_hyperparameters.csv"
RESID_AUDIT_OUT = OUTDIR / "dataset_C_normwear_residualizer_fold_audit.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_normwear_imu_residualized_nestedcv_audit.txt"

PARTIAL_FOLD = OUTDIR / "_partial_residualized_fold_metrics.csv"
PARTIAL_PRED = OUTDIR / "_partial_residualized_predictions.csv"
PARTIAL_HYPER = OUTDIR / "_partial_residualized_selected_hyperparameters.csv"
PARTIAL_RESID = OUTDIR / "_partial_residualizer_fold_audit.csv"

TARGET = "target_borg"
GROUP = "subject_num"
ROW_ID = "row_id"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_B = 2500
BASE_SEED = 20260901

RESID_RIDGE_ALPHA = 1.0

EXPECTED_START_EMB_SHA256 = "56115F2118773EC0C288816CB461DDD8F6D3803E182B35C0BB94B88A8B4156E9"
EXPECTED_END_EMB_SHA256 = "E736A6008BDCACF342920CA850A5ADC5A8AC6DC5BE52B583DF659B80E2216401"

ELASTIC_GRID = [
    {"alpha": a, "l1_ratio": l1}
    for a, l1 in product([0.01, 0.1, 1.0, 10.0], [0.1, 0.5, 0.9])
]

HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
    for lr, leaves, l2 in product([0.03, 0.08], [7, 15], [1.0, 10.0])
]

COMBINED_TO_BASELINE = {
    "M5a_core_plus_normwear_residual_mean768": "C0a_core_context",
    "M5b_extended_plus_normwear_residual_mean768": "C0b_extended_context",
}

COMPARISON_LABEL = {
    "M5a_core_plus_normwear_residual_mean768":
        "Core-context -> residualized NormWear mean768 increment",
    "M5b_extended_plus_normwear_residual_mean768":
        "Extended-context -> residualized NormWear mean768 increment",
}


def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


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


def make_context_preprocessor(X, context_cols):
    num, cat = split_column_types(X, context_cols)
    transformers = []

    if num:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num,
        ))

    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_residualizer(train_df, context_cols, emb_cols):
    # Outcome-free, fixed residualizer.
    pre = make_context_preprocessor(train_df, context_cols)
    Xc = pre.fit_transform(train_df[context_cols])
    E = train_df[emb_cols].to_numpy(dtype=np.float64)

    ridge = Ridge(alpha=RESID_RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(Xc, E)

    pred_train = ridge.predict(Xc)
    resid_train = E - pred_train

    return pre, ridge, resid_train


def apply_residualizer(pre, ridge, df, context_cols, emb_cols):
    Xc = pre.transform(df[context_cols])
    E = df[emb_cols].to_numpy(dtype=np.float64)
    pred = ridge.predict(Xc)
    resid = E - pred
    return resid, pred


def make_downstream_pipeline(model_name, params, numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

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


def with_residual_columns(df, residual, context_cols, resid_cols):
    out = df[context_cols].copy().reset_index(drop=True)
    R = pd.DataFrame(
        np.asarray(residual, dtype=np.float64),
        columns=resid_cols,
    )
    return pd.concat([out, R], axis=1)


def tune_residualized_model(
    model_name,
    outer_train_df,
    y,
    groups,
    context_cols,
    emb_cols,
    resid_cols,
    seed,
):
    grid = ELASTIC_GRID if model_name == "ElasticNet" else HGB_GRID
    splits = shuffled_group_folds(groups, INNER_FOLDS, seed)

    # Build each inner split with its OWN residualizer.
    prepared = []

    for inner_idx, (tr, va) in enumerate(splits, start=1):
        tr_df = outer_train_df.iloc[tr].reset_index(drop=True)
        va_df = outer_train_df.iloc[va].reset_index(drop=True)

        pre_r, ridge, r_tr = fit_residualizer(
            tr_df,
            context_cols,
            emb_cols,
        )
        r_va, _ = apply_residualizer(
            pre_r,
            ridge,
            va_df,
            context_cols,
            emb_cols,
        )

        Xtr = with_residual_columns(
            tr_df,
            r_tr,
            context_cols,
            resid_cols,
        )
        Xva = with_residual_columns(
            va_df,
            r_va,
            context_cols,
            resid_cols,
        )

        prepared.append((
            tr,
            va,
            Xtr,
            Xva,
        ))

    all_cols = context_cols + resid_cols
    numeric_cols, categorical_cols = split_column_types(
        prepared[0][2],
        all_cols,
    )

    best_score = np.inf
    best_params = None

    for params in grid:
        scores = []

        for tr, va, Xtr, Xva in prepared:
            pipe = make_downstream_pipeline(
                model_name,
                params,
                numeric_cols,
                categorical_cols,
            )

            pipe.fit(Xtr[all_cols], y.iloc[tr])
            pred = pipe.predict(Xva[all_cols])

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

    return best_params, best_score


def bootstrap_increment(
    base_avg,
    new_avg,
    anchor,
    model_name,
    baseline,
    combined,
    b,
    seed,
):
    a = base_avg[
        (base_avg["anchor"] == anchor)
        & (base_avg["model"] == model_name)
        & (base_avg["feature_set"] == baseline)
    ][[ROW_ID, GROUP, "y_true", "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_base"}
    )

    c = new_avg[
        (new_avg["anchor"] == anchor)
        & (new_avg["model"] == model_name)
        & (new_avg["feature_set"] == combined)
    ][[ROW_ID, "y_pred_avg"]].rename(
        columns={"y_pred_avg": "pred_comb"}
    )

    d = a.merge(c, on=ROW_ID, how="inner", validate="one_to_one")

    if len(d) != 1141:
        raise RuntimeError(f"Expected 1141 paired rows, got {len(d)}")

    mb = participant_balanced_metrics(
        d["y_true"], d["pred_base"], d[GROUP]
    )
    mc = participant_balanced_metrics(
        d["y_true"], d["pred_comb"], d[GROUP]
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
        sampled = rng.choice(pids, size=len(pids), replace=True)

        pieces = []
        for j, pid in enumerate(sampled):
            z = by_pid[pid].copy()
            z["boot_group"] = f"{pid}__draw{j}"
            pieces.append(z)

        z = pd.concat(pieces, ignore_index=True)

        zb = participant_balanced_metrics(
            z["y_true"], z["pred_base"], z["boot_group"]
        )
        zc = participant_balanced_metrics(
            z["y_true"], z["pred_comb"], z["boot_group"]
        )

        store["delta_mae_pb"].append(zb["mae_pb"] - zc["mae_pb"])
        store["delta_rmse_pb"].append(zb["rmse_pb"] - zc["rmse_pb"])
        store["delta_r2_rep"].append(zc["r2_rep"] - zb["r2_rep"])

    ci = {}

    for key, vals in store.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]

        ci[f"{key}_ci_low"] = float(np.quantile(arr, 0.025))
        ci[f"{key}_ci_high"] = float(np.quantile(arr, 0.975))
        ci[f"{key}_bootstrap_positive_prob"] = float(np.mean(arr > 0))
        ci[f"{key}_bootstrap_n"] = int(len(arr))

    return observed, ci


# =============================================================================
# 1. Load and lock all inputs
# =============================================================================
for fp in [
    START_TABLE,
    END_TABLE,
    DICT,
    START_EMB,
    END_EMB,
    BASE_FOLD,
    BASE_PRED,
    BASE_AVG,
    BASE_SUMMARY,
]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

start_sha = sha256_file(START_EMB)
end_sha = sha256_file(END_EMB)

if start_sha != EXPECTED_START_EMB_SHA256:
    raise SystemExit("START mean768 hash differs from Script-24 lock.")
if end_sha != EXPECTED_END_EMB_SHA256:
    raise SystemExit("END mean768 hash differs from Script-24 lock.")

start_df = pd.read_csv(START_TABLE).sort_values(ROW_ID).reset_index(drop=True)
end_df = pd.read_csv(END_TABLE).sort_values(ROW_ID).reset_index(drop=True)
fd = pd.read_csv(DICT)

if len(start_df) != 1141 or len(end_df) != 1141:
    raise SystemExit("Expected 1141 rows per anchor.")
if start_df[GROUP].nunique() != 34:
    raise SystemExit("Expected 34 participants.")

if not np.array_equal(
    start_df[ROW_ID].astype(str).to_numpy(),
    end_df[ROW_ID].astype(str).to_numpy(),
):
    raise SystemExit("START/END row IDs differ.")

if not np.allclose(
    start_df[TARGET].to_numpy(float),
    end_df[TARGET].to_numpy(float),
):
    raise SystemExit("START/END targets differ.")

zs = np.load(START_EMB, allow_pickle=False)
ze = np.load(END_EMB, allow_pickle=False)

s_ids = zs["row_id"].astype(str)
e_ids = ze["row_id"].astype(str)
s_emb = zs["embedding"].astype(np.float32)
e_emb = ze["embedding"].astype(np.float32)

locked_ids = start_df[ROW_ID].astype(str).to_numpy()

if not np.array_equal(s_ids, locked_ids):
    raise SystemExit("START embedding row IDs mismatch.")
if not np.array_equal(e_ids, locked_ids):
    raise SystemExit("END embedding row IDs mismatch.")
if s_emb.shape != (1141, 768) or e_emb.shape != (1141, 768):
    raise SystemExit("Unexpected embedding shape.")
if not np.isfinite(s_emb).all() or not np.isfinite(e_emb).all():
    raise SystemExit("Non-finite frozen embedding.")

emb_cols = [f"normwear_mean_{j:03d}" for j in range(768)]
resid_cols = [f"normwear_resid_{j:03d}" for j in range(768)]

start_df = pd.concat(
    [start_df, pd.DataFrame(s_emb, columns=emb_cols)],
    axis=1,
)
end_df = pd.concat(
    [end_df, pd.DataFrame(e_emb, columns=emb_cols)],
    axis=1,
)

# =============================================================================
# 2. Recreate exact 18f context sets
# =============================================================================
role_map = {}
for role in fd["role"].dropna().unique():
    role_map[role] = fd.loc[
        fd["role"] == role,
        "column",
    ].tolist()

forbidden = {
    ROW_ID,
    GROUP,
    TARGET,
    "borg_change_10s",
    "target_time_sec",
    "trial_duration_sec_qc",
    "known_corrupt_shoulder_imu_trial_qc",
}

core_cols = [
    c for c in role_map.get("context_core", [])
    if c in start_df.columns and c not in forbidden
]

extended_cols = []
for role in ["context_core", "context_extended"]:
    extended_cols.extend(role_map.get(role, []))

extended_cols = [
    c for c in dict.fromkeys(extended_cols)
    if c in start_df.columns and c not in forbidden
]

if len(core_cols) != 5:
    raise SystemExit(f"Expected 5 core context columns, found {len(core_cols)}")
if len(extended_cols) != 14:
    raise SystemExit(f"Expected 14 extended context columns, found {len(extended_cols)}")

CONTEXT_BY_COMBINED = {
    "M5a_core_plus_normwear_residual_mean768": core_cols,
    "M5b_extended_plus_normwear_residual_mean768": extended_cols,
}

# =============================================================================
# 3. Locked baseline outputs + exact split validation
# =============================================================================
base_fold = pd.read_csv(BASE_FOLD)
base_pred = pd.read_csv(BASE_PRED)
base_avg = pd.read_csv(BASE_AVG)
base_summary = pd.read_csv(BASE_SUMMARY)

outer_split_map = {}

for repeat in range(OUTER_REPEATS):
    outer_seed = BASE_SEED + repeat * 10000
    splits = shuffled_group_folds(
        start_df[GROUP],
        OUTER_FOLDS,
        outer_seed,
    )

    for outer_fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        current_test_ids = set(
            start_df.iloc[te_idx][ROW_ID].astype(str)
        )

        for model_name in ["ElasticNet", "HGB"]:
            for baseline in ["C0a_core_context", "C0b_extended_context"]:
                q = base_pred[
                    (base_pred["repeat"] == repeat + 1)
                    & (base_pred["outer_fold"] == outer_fold)
                    & (base_pred["anchor"] == "START")
                    & (base_pred["model"] == model_name)
                    & (base_pred["feature_set"] == baseline)
                ]

                if current_test_ids != set(q[ROW_ID].astype(str)):
                    raise SystemExit(
                        f"Outer split mismatch: R{repeat+1}F{outer_fold} "
                        f"{model_name} {baseline}"
                    )

        tr_groups = set(start_df.iloc[tr_idx][GROUP].astype(str))
        te_groups = set(start_df.iloc[te_idx][GROUP].astype(str))

        if tr_groups & te_groups:
            raise RuntimeError("Outer participant leakage.")

        outer_split_map[(repeat + 1, outer_fold)] = (tr_idx, te_idx)

# =============================================================================
# 4. Resume support
# =============================================================================
def read_partial(fp):
    if fp.exists() and fp.stat().st_size > 0:
        return pd.read_csv(fp)
    return pd.DataFrame()

pf = read_partial(PARTIAL_FOLD)
pp = read_partial(PARTIAL_PRED)
ph = read_partial(PARTIAL_HYPER)
pr = read_partial(PARTIAL_RESID)

fold_rows = pf.to_dict("records") if len(pf) else []
pred_rows = pp.to_dict("records") if len(pp) else []
hyper_rows = ph.to_dict("records") if len(ph) else []
resid_rows = pr.to_dict("records") if len(pr) else []

completed = set()

if len(pf):
    for _, r in pf.iterrows():
        completed.add((
            int(r["repeat"]),
            int(r["outer_fold"]),
            str(r["anchor"]),
            str(r["model"]),
            str(r["feature_set"]),
        ))

TOTAL_TASKS = 25 * 2 * 2 * 2

print(f"Resume: {len(completed)}/{TOTAL_TASKS} tasks already complete.")

anchors = {
    "START": start_df,
    "END": end_df,
}

# =============================================================================
# 5. Full M5 nested evaluation
# =============================================================================
t0 = time.time()
fresh = 0

for repeat in range(1, OUTER_REPEATS + 1):
    outer_seed = BASE_SEED + (repeat - 1) * 10000

    for outer_fold in range(1, OUTER_FOLDS + 1):
        tr_idx, te_idx = outer_split_map[(repeat, outer_fold)]

        for anchor, df in anchors.items():
            outer_train = df.iloc[tr_idx].reset_index(drop=True)
            outer_test = df.iloc[te_idx].reset_index(drop=True)

            y_train = outer_train[TARGET].astype(float).reset_index(drop=True)
            y_test = outer_test[TARGET].astype(float).reset_index(drop=True)
            g_train = outer_train[GROUP].astype(str).reset_index(drop=True)
            g_test = outer_test[GROUP].astype(str).reset_index(drop=True)

            for model_name in ["ElasticNet", "HGB"]:
                for fs_idx, combined in enumerate(CONTEXT_BY_COMBINED.keys()):
                    key = (
                        repeat,
                        outer_fold,
                        anchor,
                        model_name,
                        combined,
                    )

                    if key in completed:
                        continue

                    context_cols = CONTEXT_BY_COMBINED[combined]

                    inner_seed = (
                        outer_seed
                        + outer_fold * 1000
                        + 700
                        + fs_idx * 10
                        + (1 if model_name == "ElasticNet" else 2)
                    )

                    best_params, inner_mae = tune_residualized_model(
                        model_name,
                        outer_train,
                        y_train,
                        g_train,
                        context_cols,
                        emb_cols,
                        resid_cols,
                        inner_seed,
                    )

                    # Fit residualizer on FULL OUTER TRAIN ONLY.
                    pre_r, ridge, r_train = fit_residualizer(
                        outer_train,
                        context_cols,
                        emb_cols,
                    )

                    r_test, pred_emb_test = apply_residualizer(
                        pre_r,
                        ridge,
                        outer_test,
                        context_cols,
                        emb_cols,
                    )

                    Xtr = with_residual_columns(
                        outer_train,
                        r_train,
                        context_cols,
                        resid_cols,
                    )
                    Xte = with_residual_columns(
                        outer_test,
                        r_test,
                        context_cols,
                        resid_cols,
                    )

                    all_cols = context_cols + resid_cols
                    numeric_cols, categorical_cols = split_column_types(
                        Xtr,
                        all_cols,
                    )

                    pipe = make_downstream_pipeline(
                        model_name,
                        best_params,
                        numeric_cols,
                        categorical_cols,
                    )

                    pipe.fit(Xtr[all_cols], y_train)
                    pred = pipe.predict(Xte[all_cols])

                    if not np.isfinite(pred).all():
                        raise RuntimeError(
                            f"Non-finite predictions: {key}"
                        )

                    metrics = participant_balanced_metrics(
                        y_test,
                        pred,
                        g_test,
                    )

                    # Outcome-free residualizer diagnostics.
                    emb_test = outer_test[emb_cols].to_numpy(dtype=np.float64)
                    mse_emb = float(np.mean((emb_test - pred_emb_test) ** 2))
                    var_emb = float(np.var(emb_test))
                    residual_energy_ratio = (
                        float(np.mean(r_test ** 2) / np.mean(emb_test ** 2))
                        if np.mean(emb_test ** 2) > 0
                        else np.nan
                    )
                    context_explained_r2_embed = (
                        1.0 - mse_emb / var_emb
                        if var_emb > 0
                        else np.nan
                    )

                    fold_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": combined,
                        "baseline_feature_set": COMBINED_TO_BASELINE[combined],
                        "n_context_features": len(context_cols),
                        "n_residual_features": 768,
                        "n_train_rows": len(outer_train),
                        "n_test_rows": len(outer_test),
                        "n_train_participants": g_train.nunique(),
                        "n_test_participants": g_test.nunique(),
                        "inner_selected_mae_pb": inner_mae,
                        "prediction_abs_max": float(np.max(np.abs(pred))),
                        **metrics,
                    })

                    hyper_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": combined,
                        "selected_params_json": json.dumps(
                            best_params,
                            sort_keys=True,
                        ),
                        "inner_selected_mae_pb": inner_mae,
                    })

                    resid_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": combined,
                        "residualizer_ridge_alpha": RESID_RIDGE_ALPHA,
                        "residualizer_train_participants": g_train.nunique(),
                        "residualizer_test_participants": g_test.nunique(),
                        "outer_test_embedding_context_explained_r2": context_explained_r2_embed,
                        "outer_test_residual_energy_ratio": residual_energy_ratio,
                    })

                    meta = outer_test[
                        [ROW_ID, GROUP, "task_label", "current_time_sec", TARGET]
                    ].reset_index(drop=True)

                    for j in range(len(meta)):
                        pred_rows.append({
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "anchor": anchor,
                            "model": model_name,
                            "feature_set": combined,
                            ROW_ID: meta.loc[j, ROW_ID],
                            GROUP: meta.loc[j, GROUP],
                            "task_label": meta.loc[j, "task_label"],
                            "current_time_sec": meta.loc[j, "current_time_sec"],
                            "y_true": float(meta.loc[j, TARGET]),
                            "y_pred": float(pred[j]),
                        })

                    completed.add(key)
                    fresh += 1

                    pd.DataFrame(fold_rows).to_csv(PARTIAL_FOLD, index=False)
                    pd.DataFrame(pred_rows).to_csv(PARTIAL_PRED, index=False)
                    pd.DataFrame(hyper_rows).to_csv(PARTIAL_HYPER, index=False)
                    pd.DataFrame(resid_rows).to_csv(PARTIAL_RESID, index=False)

                    elapsed = time.time() - t0
                    sec = elapsed / max(fresh, 1)
                    remain = TOTAL_TASKS - len(completed)

                    print(
                        f"Completed {len(completed)}/{TOTAL_TASKS} | "
                        f"{anchor} {model_name} {combined} R{repeat}F{outer_fold} | "
                        f"fresh mean {sec:.1f}s/task | ETA {remain*sec/60:.1f} min"
                    )

# =============================================================================
# 6. Finalize new predictions
# =============================================================================
fold_df = pd.DataFrame(fold_rows)
pred_df = pd.DataFrame(pred_rows)
hyper_df = pd.DataFrame(hyper_rows)
resid_df = pd.DataFrame(resid_rows)

if len(fold_df) != TOTAL_TASKS:
    raise RuntimeError(f"Expected {TOTAL_TASKS} task rows, found {len(fold_df)}")

expected_pred = 1141 * 5 * 2 * 2 * 2
if len(pred_df) != expected_pred:
    raise RuntimeError(
        f"Expected {expected_pred} prediction rows, found {len(pred_df)}"
    )

if not np.isfinite(pred_df["y_pred"].to_numpy(float)).all():
    raise RuntimeError("Non-finite final predictions.")

fold_df.to_csv(FOLD_OUT, index=False)
pred_df.to_csv(PRED_OUT, index=False)
hyper_df.to_csv(HYPER_OUT, index=False)
resid_df.to_csv(RESID_AUDIT_OUT, index=False)

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

if not (avg["n_oof_predictions"] == 5).all():
    raise RuntimeError("Every row must have exactly five repeated OOF predictions.")

avg.to_csv(AVG_OUT, index=False)

# =============================================================================
# 7. Absolute summary
# =============================================================================
new_summary_rows = []

for (anchor, model_name, fs_name), g in avg.groupby(
    ["anchor", "model", "feature_set"]
):
    m = participant_balanced_metrics(
        g["y_true"],
        g["y_pred_avg"],
        g[GROUP],
    )

    new_summary_rows.append({
        "anchor": anchor,
        "source": "26_new_residualized_normwear",
        "model": model_name,
        "feature_set": fs_name,
        "n_rows": len(g),
        "n_participants": g[GROUP].nunique(),
        **m,
    })

new_summary = pd.DataFrame(new_summary_rows)

locked_subset = base_summary[
    base_summary["feature_set"].isin(
        ["C0a_core_context", "C0b_extended_context"]
    )
].copy()
locked_subset["source"] = "18f_locked_baseline"

summary_cols = [
    "anchor",
    "source",
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

summary_df = pd.concat(
    [
        locked_subset[summary_cols],
        new_summary[summary_cols],
    ],
    ignore_index=True,
)

summary_df.to_csv(SUMMARY_OUT, index=False)

# =============================================================================
# 8. Incremental value
# =============================================================================
inc_rows = []

for anchor_idx, anchor in enumerate(["START", "END"]):
    for model_idx, model_name in enumerate(["ElasticNet", "HGB"]):
        for comp_idx, combined in enumerate(CONTEXT_BY_COMBINED.keys()):
            baseline = COMBINED_TO_BASELINE[combined]

            obs, ci = bootstrap_increment(
                base_avg,
                avg,
                anchor,
                model_name,
                baseline,
                combined,
                BOOTSTRAP_B,
                BASE_SEED
                + 290000
                + anchor_idx * 10000
                + model_idx * 1000
                + comp_idx,
            )

            f0 = base_fold[
                (base_fold["anchor"] == anchor)
                & (base_fold["model"] == model_name)
                & (base_fold["feature_set"] == baseline)
            ][
                ["repeat", "outer_fold", "mae_pb", "rmse_pb", "r2_rep"]
            ]

            f1 = fold_df[
                (fold_df["anchor"] == anchor)
                & (fold_df["model"] == model_name)
                & (fold_df["feature_set"] == combined)
            ][
                ["repeat", "outer_fold", "mae_pb", "rmse_pb", "r2_rep"]
            ]

            paired = f0.merge(
                f1,
                on=["repeat", "outer_fold"],
                suffixes=("_base", "_comb"),
                validate="one_to_one",
            )

            inc_rows.append({
                "anchor": anchor,
                "model": model_name,
                "representation": "NormWear_IMU_context_residualized_mean768",
                "comparison": COMPARISON_LABEL[combined],
                "baseline_feature_set": baseline,
                "combined_feature_set": combined,
                "baseline_source": "18f_locked",
                "n_rows": 1141,
                "n_participants": 34,
                **obs,
                **ci,
                "fold_delta_mae_pb_mean": float(
                    (paired["mae_pb_base"] - paired["mae_pb_comb"]).mean()
                ),
                "fold_delta_mae_pb_positive_fraction": float(
                    (paired["mae_pb_base"] > paired["mae_pb_comb"]).mean()
                ),
                "fold_delta_rmse_pb_mean": float(
                    (paired["rmse_pb_base"] - paired["rmse_pb_comb"]).mean()
                ),
                "fold_delta_rmse_pb_positive_fraction": float(
                    (paired["rmse_pb_base"] > paired["rmse_pb_comb"]).mean()
                ),
                "fold_delta_r2_rep_mean": float(
                    (paired["r2_rep_comb"] - paired["r2_rep_base"]).mean()
                ),
                "fold_delta_r2_rep_positive_fraction": float(
                    (paired["r2_rep_comb"] > paired["r2_rep_base"]).mean()
                ),
            })

inc_df = pd.DataFrame(inc_rows)
inc_df.to_csv(INCREMENT_OUT, index=False)

# =============================================================================
# 9. Alignment robustness
# =============================================================================
robust_rows = []

for model_name in ["ElasticNet", "HGB"]:
    for combined in CONTEXT_BY_COMBINED.keys():
        label = COMPARISON_LABEL[combined]

        s = inc_df[
            (inc_df["model"] == model_name)
            & (inc_df["comparison"] == label)
        ].set_index("anchor")

        dms = float(s.loc["START", "delta_mae_pb"])
        dme = float(s.loc["END", "delta_mae_pb"])
        rms = float(s.loc["START", "delta_rmse_pb"])
        rme = float(s.loc["END", "delta_rmse_pb"])
        r2s = float(s.loc["START", "delta_r2_rep"])
        r2e = float(s.loc["END", "delta_r2_rep"])

        robust_rows.append({
            "model": model_name,
            "representation": "NormWear_IMU_context_residualized_mean768",
            "comparison": label,
            "start_delta_mae_pb": dms,
            "end_delta_mae_pb": dme,
            "mae_direction_consistent": int(np.sign(dms) == np.sign(dme)),
            "mae_both_positive": int(dms > 0 and dme > 0),
            "mae_both_negative": int(dms < 0 and dme < 0),
            "start_mae_ci_excludes_zero_positive": int(
                s.loc["START", "delta_mae_pb_ci_low"] > 0
            ),
            "end_mae_ci_excludes_zero_positive": int(
                s.loc["END", "delta_mae_pb_ci_low"] > 0
            ),
            "start_mae_ci_excludes_zero_negative": int(
                s.loc["START", "delta_mae_pb_ci_high"] < 0
            ),
            "end_mae_ci_excludes_zero_negative": int(
                s.loc["END", "delta_mae_pb_ci_high"] < 0
            ),
            "start_delta_rmse_pb": rms,
            "end_delta_rmse_pb": rme,
            "rmse_direction_consistent": int(np.sign(rms) == np.sign(rme)),
            "start_delta_r2_rep": r2s,
            "end_delta_r2_rep": r2e,
            "r2_direction_consistent": int(np.sign(r2s) == np.sign(r2e)),
            "start_bootstrap_positive_prob_mae": float(
                s.loc["START", "delta_mae_pb_bootstrap_positive_prob"]
            ),
            "end_bootstrap_positive_prob_mae": float(
                s.loc["END", "delta_mae_pb_bootstrap_positive_prob"]
            ),
            "start_fold_positive_fraction_mae": float(
                s.loc["START", "fold_delta_mae_pb_positive_fraction"]
            ),
            "end_fold_positive_fraction_mae": float(
                s.loc["END", "fold_delta_mae_pb_positive_fraction"]
            ),
        })

robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(ROBUST_OUT, index=False)

# =============================================================================
# 10. Final audit
# =============================================================================
elapsed = time.time() - t0

lines = []
add = lines.append

add("DATASET C — CONTEXT-RESIDUALIZED FROZEN NORMWEAR-IMU NESTED-CV AUDIT")
add("=" * 118)
add(f"Rows per anchor: {len(start_df)}")
add(f"Participants: {start_df[GROUP].nunique()}")
add("Scientific question: after removing context-predictable embedding components,")
add("does the remaining frozen NormWear representation provide incremental predictive value?")
add(f"Outer validation: {OUTER_REPEATS} repeats x {OUTER_FOLDS} participant-grouped folds")
add(f"Inner validation: {INNER_FOLDS} participant-grouped folds")
add(f"Bootstrap: {BOOTSTRAP_B} participant-cluster resamples")
add(f"Elapsed minutes: {elapsed/60:.2f}")
add("")

add("1. REPRESENTATION / BASELINE LOCK")
add("-" * 118)
add(f"START mean768 SHA256: {start_sha}")
add(f"END mean768 SHA256: {end_sha}")
add("C0a/C0b baselines are literal 18f locked predictions and are not re-fit.")
add("Exact outer test row IDs were reconstructed and checked against 18f: PASS.")
add("")

add("2. RESIDUALIZATION POLICY")
add("-" * 118)
add(f"Residualizer: multi-output Ridge(alpha={RESID_RIDGE_ALPHA}, fixed).")
add("Context preprocessing inside residualizer: numeric median+StandardScaler; categorical most-frequent+OneHot.")
add("Residual = frozen NormWear mean768 - context-predicted NormWear mean768.")
add("Inner tuning: residualizer is fit on INNER-TRAIN participants only.")
add("Outer final evaluation: residualizer is fit on OUTER-TRAIN participants only.")
add("Outer-test participant embeddings are never used to fit the residualizer.")
add("Residualizer uses no Borg target.")
add("")

add("3. FEATURE SETS")
add("-" * 118)
add(f"M5a = {len(core_cols)} core context + 768 residualized NormWear dimensions.")
add(f"M5b = {len(extended_cols)} extended context + 768 residualized NormWear dimensions.")
add("")

add("4. LEAKAGE CONTROLS")
add("-" * 118)
add("PASS: exact 18f outer participant splits reused.")
add("PASS: outer train/test participants disjoint.")
add("PASS: inner residualization is trained only on inner-training participants.")
add("PASS: outer residualization is trained only on outer-training participants.")
add("PASS: downstream preprocessing/model tuning uses only applicable training participants.")
add("PASS: frozen NormWear embeddings were created before supervised modeling.")
add("")

add("5. OUTCOME-FREE RESIDUALIZER DIAGNOSTICS")
add("-" * 118)
show_resid = (
    resid_df.groupby(["anchor", "feature_set"], as_index=False)
    .agg(
        median_test_embedding_context_r2=("outer_test_embedding_context_explained_r2", "median"),
        mean_test_embedding_context_r2=("outer_test_embedding_context_explained_r2", "mean"),
        median_residual_energy_ratio=("outer_test_residual_energy_ratio", "median"),
        mean_residual_energy_ratio=("outer_test_residual_energy_ratio", "mean"),
    )
)
add(show_resid.to_string(index=False))
add("These are outcome-free representation diagnostics; they are not predictive-value results.")
add("")

add("6. ABSOLUTE MODEL SUMMARY")
add("-" * 118)
add(
    summary_df[
        [
            "anchor", "source", "model", "feature_set",
            "n_rows", "n_participants", "mae_pb", "rmse_pb", "r2_rep"
        ]
    ]
    .sort_values(["anchor", "model", "mae_pb"])
    .to_string(index=False)
)
add("")

add("7. M5 INCREMENTAL VALUE")
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
add(inc_df[show_cols].to_string(index=False))
add("")
add("Sign convention:")
add("  delta_mae_pb > 0 => residualized NormWear improves MAE.")
add("  delta_rmse_pb > 0 => residualized NormWear improves RMSE.")
add("  delta_r2_rep > 0 => residualized NormWear improves pooled R2.")
add("")

add("8. START/END ALIGNMENT ROBUSTNESS")
add("-" * 118)
add(robust_df.to_string(index=False))
add("")
add("Strong rescue requires BOTH START and END positive MAE increments with 95% CI excluding zero,")
add("high bootstrap positive support, majority positive outer folds, and concordant RMSE/R2 direction.")
add("")

add("9. NUMERICAL STABILITY")
add("-" * 118)
stab = (
    fold_df.groupby(["anchor", "model", "feature_set"], as_index=False)
    .agg(
        max_abs_prediction=("prediction_abs_max", "max"),
        median_abs_prediction_max=("prediction_abs_max", "median"),
    )
)
add(stab.to_string(index=False))
add("")

add("10. SCIENTIFIC GUARDRAILS")
add("-" * 118)
add("M4 remains frozen and is not re-tuned.")
add("M5 residualization is outcome-free and fold-local.")
add("No CLS sensitivity is used in this script.")
add("No NormWear fine-tuning/adaptation is used.")
add("No residualizer alpha tuning is used.")
add("No representation or model is changed to chase a positive result.")
add("")

add("11. OUTPUT HASHES")
add("-" * 118)
for fp in [
    FOLD_OUT,
    PRED_OUT,
    AVG_OUT,
    SUMMARY_OUT,
    INCREMENT_OUT,
    ROBUST_OUT,
    HYPER_OUT,
    RESID_AUDIT_OUT,
]:
    add(f"{fp.name} | bytes={fp.stat().st_size} | SHA256={sha256_file(fp)}")
add("")

global_pass = (
    len(fold_df) == TOTAL_TASKS
    and len(pred_df) == expected_pred
    and len(avg) == 1141 * 2 * 2 * 2
    and len(inc_df) == 8
    and len(robust_df) == 4
    and np.isfinite(pred_df["y_pred"].to_numpy(float)).all()
    and np.isfinite(avg["y_pred_avg"].to_numpy(float)).all()
)

add("12. DECISION GATE")
add("-" * 118)
add(f"DATASET C RESIDUALIZED NORMWEAR INCREMENTAL BENCHMARK PASS: {global_pass}")

if global_pass:
    add("Technical evaluation is complete. Interpret scientific evidence from the paired incremental and")
    add("alignment-robustness outputs only.")
else:
    add("DO NOT interpret the scientific result.")

add("=" * 118)
add("END OF M5 RESIDUALIZED NORMWEAR AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

if global_pass:
    for fp in [
        PARTIAL_FOLD,
        PARTIAL_PRED,
        PARTIAL_HYPER,
        PARTIAL_RESID,
    ]:
        if fp.exists():
            fp.unlink()

print("")
print("Created:")
print(" ", AUDIT_OUT)
print(" ", INCREMENT_OUT)
print(" ", ROBUST_OUT)
print("")
print(f"DATASET C RESIDUALIZED NORMWEAR INCREMENTAL BENCHMARK PASS: {global_pass}")
print("Upload these three files to ChatGPT:")
print("  dataset_C_normwear_imu_residualized_nestedcv_audit.txt")
print("  dataset_C_normwear_residualized_incremental_value.csv")
print("  dataset_C_normwear_residualized_alignment_robustness.csv")
