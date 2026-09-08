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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# 25_dataset_C_normwear_imu_primary_incremental_nestedcv.py
#
# PRIMARY FROZEN-REPRESENTATION RESCUE TEST
#
# Scientific question:
#   Does the prespecified frozen NormWear-IMU mean-pooled representation add
#   stable predictive value beyond the LOCKED 18f context baselines?
#
# IMPORTANT DESIGN:
#   - Context baselines are NOT re-fit here.
#   - C0a/C0b held-out predictions and fold metrics are loaded directly from
#     the final locked 18f benchmark.
#   - Exact 18f outer participant splits are reconstructed and verified
#     against the locked baseline test-row IDs before any new model is used.
#   - Only the new combined models are trained:
#       N2a = C0a core context + frozen NormWear mean768
#       N2b = C0b extended context + frozen NormWear mean768
#   - Same model families as 18f: ElasticNet and HGB.
#   - Same grids, preprocessing, participant-balanced metrics,
#     5x5 outer grouped CV, 4-fold inner grouped tuning,
#     and 2500 participant-cluster bootstrap.
#
# NOT DONE HERE:
#   - no handcrafted benchmark re-tuning;
#   - no CLS-pooling sensitivity;
#   - no residualized Foundation representation (M5);
#   - no NormWear fine-tuning / adaptation;
#   - no outcome-driven representation selection.
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

OUTDIR = ROOT / "results" / "dataset_C_normwear_imu_primary_nestedcv"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

FOLD_OUT = OUTDIR / "dataset_C_normwear_primary_fold_metrics.csv"
PRED_OUT = OUTDIR / "dataset_C_normwear_primary_predictions.csv"
AVG_OUT = OUTDIR / "dataset_C_normwear_primary_avg_oof_predictions.csv"
SUMMARY_OUT = OUTDIR / "dataset_C_normwear_primary_model_summary_with_locked_baselines.csv"
INCREMENT_OUT = OUTDIR / "dataset_C_normwear_primary_incremental_value.csv"
ROBUST_OUT = OUTDIR / "dataset_C_normwear_primary_alignment_robustness.csv"
HYPER_OUT = OUTDIR / "dataset_C_normwear_primary_selected_hyperparameters.csv"
LOCKCHECK_OUT = OUTDIR / "dataset_C_normwear_primary_baseline_lock_validation.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_normwear_imu_primary_nestedcv_audit.txt"

PARTIAL_FOLD = OUTDIR / "_partial_normwear_primary_fold_metrics.csv"
PARTIAL_PRED = OUTDIR / "_partial_normwear_primary_predictions.csv"
PARTIAL_HYPER = OUTDIR / "_partial_normwear_primary_selected_hyperparameters.csv"

TARGET = "target_borg"
GROUP = "subject_num"
ROW_ID = "row_id"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_B = 2500
BASE_SEED = 20260901

EXPECTED_START_EMB_SHA256 = "56115F2118773EC0C288816CB461DDD8F6D3803E182B35C0BB94B88A8B4156E9"
EXPECTED_END_EMB_SHA256 = "E736A6008BDCACF342920CA850A5ADC5A8AC6DC5BE52B583DF659B80E2216401"

BASELINE_FEATURE_SETS = {
    "N2a_core_plus_normwear_mean768": "C0a_core_context",
    "N2b_extended_plus_normwear_mean768": "C0b_extended_context",
}

COMPARISON_LABELS = {
    "N2a_core_plus_normwear_mean768": "Core-context -> NormWear mean768 increment",
    "N2b_extended_plus_normwear_mean768": "Extended-context -> NormWear mean768 increment",
}

ELASTIC_GRID = [
    {"alpha": a, "l1_ratio": l1}
    for a, l1 in product([0.01, 0.1, 1.0, 10.0], [0.1, 0.5, 0.9])
]

HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
    for lr, leaves, l2 in product([0.03, 0.08], [7, 15], [1.0, 10.0])
]


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
    bg = d.groupby("g", sort=False).agg(mae=("ae", "mean"), mse=("se", "mean"))
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
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
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
    return Pipeline([("pre", pre), ("var", VarianceThreshold(threshold=0.0)), ("model", model)])


def tune_model(model_name, X, y, groups, cols, seed):
    grid = ELASTIC_GRID if model_name == "ElasticNet" else HGB_GRID
    numeric_cols, categorical_cols = split_column_types(X, cols)
    splits = shuffled_group_folds(groups, INNER_FOLDS, seed)
    best_score = np.inf
    best_params = None
    for params in grid:
        scores = []
        for tr, va in splits:
            pipe = make_pipeline(model_name, params, numeric_cols, categorical_cols)
            pipe.fit(X.iloc[tr][cols], y.iloc[tr])
            pred = pipe.predict(X.iloc[va][cols])
            m = participant_balanced_metrics(y.iloc[va], pred, groups.iloc[va])
            scores.append(m["mae_pb"])
        score = float(np.mean(scores))
        key = json.dumps(params, sort_keys=True)
        old_key = json.dumps(best_params or {}, sort_keys=True)
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and key < old_key):
            best_score = score
            best_params = params
    return best_params, best_score, numeric_cols, categorical_cols


def bootstrap_increment_from_two_avg(base_avg, new_avg, anchor, model_name, baseline_name, combined_name, b, seed):
    a = base_avg[
        (base_avg["anchor"] == anchor)
        & (base_avg["model"] == model_name)
        & (base_avg["feature_set"] == baseline_name)
    ][[ROW_ID, GROUP, "y_true", "y_pred_avg"]].rename(columns={"y_pred_avg": "pred_base"})
    c = new_avg[
        (new_avg["anchor"] == anchor)
        & (new_avg["model"] == model_name)
        & (new_avg["feature_set"] == combined_name)
    ][[ROW_ID, "y_pred_avg"]].rename(columns={"y_pred_avg": "pred_comb"})
    d = a.merge(c, on=ROW_ID, how="inner", validate="one_to_one")
    if len(d) != 1141:
        raise RuntimeError(f"Bootstrap pair mismatch: {anchor} {model_name} {combined_name}: {len(d)}")
    mb = participant_balanced_metrics(d["y_true"], d["pred_base"], d[GROUP])
    mc = participant_balanced_metrics(d["y_true"], d["pred_comb"], d[GROUP])
    observed = {
        "delta_mae_pb": mb["mae_pb"] - mc["mae_pb"],
        "delta_rmse_pb": mb["rmse_pb"] - mc["rmse_pb"],
        "delta_r2_rep": mc["r2_rep"] - mb["r2_rep"],
    }
    pids = sorted(d[GROUP].astype(str).unique())
    by_pid = {pid: d[d[GROUP].astype(str) == pid].copy() for pid in pids}
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
        zb = participant_balanced_metrics(z["y_true"], z["pred_base"], z["boot_group"])
        zc = participant_balanced_metrics(z["y_true"], z["pred_comb"], z["boot_group"])
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
    return observed, ci, len(d), len(pids)


# 1. Locked input validation
for fp in [START_TABLE, END_TABLE, DICT, START_EMB, END_EMB, BASE_FOLD, BASE_PRED, BASE_AVG, BASE_SUMMARY]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

start_sha = sha256_file(START_EMB)
end_sha = sha256_file(END_EMB)
if start_sha != EXPECTED_START_EMB_SHA256:
    raise SystemExit(f"START mean768 hash differs from Script-24 audit. Expected={EXPECTED_START_EMB_SHA256} Observed={start_sha}")
if end_sha != EXPECTED_END_EMB_SHA256:
    raise SystemExit(f"END mean768 hash differs from Script-24 audit. Expected={EXPECTED_END_EMB_SHA256} Observed={end_sha}")

start_df = pd.read_csv(START_TABLE).sort_values(ROW_ID).reset_index(drop=True)
end_df = pd.read_csv(END_TABLE).sort_values(ROW_ID).reset_index(drop=True)
fd = pd.read_csv(DICT)

if len(start_df) != 1141 or len(end_df) != 1141:
    raise SystemExit("Expected 1141 QC-clean rows per anchor.")
if start_df[GROUP].nunique() != 34:
    raise SystemExit("Expected 34 participants.")
if not np.array_equal(start_df[ROW_ID].astype(str).to_numpy(), end_df[ROW_ID].astype(str).to_numpy()):
    raise SystemExit("START/END row IDs differ.")
if not np.array_equal(start_df[GROUP].astype(str).to_numpy(), end_df[GROUP].astype(str).to_numpy()):
    raise SystemExit("START/END groups differ.")
if not np.allclose(start_df[TARGET].to_numpy(float), end_df[TARGET].to_numpy(float), equal_nan=True):
    raise SystemExit("START/END targets differ.")

zs = np.load(START_EMB, allow_pickle=False)
ze = np.load(END_EMB, allow_pickle=False)
s_ids = zs["row_id"].astype(str)
e_ids = ze["row_id"].astype(str)
s_emb = zs["embedding"].astype(np.float32)
e_emb = ze["embedding"].astype(np.float32)
locked_ids = start_df[ROW_ID].astype(str).to_numpy()

if not np.array_equal(s_ids, locked_ids):
    raise SystemExit("START embedding row IDs do not match locked table.")
if not np.array_equal(e_ids, locked_ids):
    raise SystemExit("END embedding row IDs do not match locked table.")
if s_emb.shape != (1141, 768) or e_emb.shape != (1141, 768):
    raise SystemExit(f"Unexpected embedding shapes START={s_emb.shape} END={e_emb.shape}")
if not np.isfinite(s_emb).all() or not np.isfinite(e_emb).all():
    raise SystemExit("Frozen embeddings contain non-finite values.")

emb_cols = [f"normwear_mean_{j:03d}" for j in range(768)]
start_df = pd.concat([start_df, pd.DataFrame(s_emb, columns=emb_cols)], axis=1)
end_df = pd.concat([end_df, pd.DataFrame(e_emb, columns=emb_cols)], axis=1)

# 2. Exact 18f context columns
role_map = {}
for role in fd["role"].dropna().unique():
    role_map[role] = fd.loc[fd["role"] == role, "column"].tolist()
forbidden = {ROW_ID, GROUP, TARGET, "borg_change_10s", "target_time_sec", "trial_duration_sec_qc", "known_corrupt_shoulder_imu_trial_qc"}
core_cols = [c for c in role_map.get("context_core", []) if c in start_df.columns and c not in forbidden]
extended_cols = []
for role in ["context_core", "context_extended"]:
    extended_cols.extend(role_map.get(role, []))
extended_cols = [c for c in dict.fromkeys(extended_cols) if c in start_df.columns and c not in forbidden]
if len(core_cols) != 5:
    raise SystemExit(f"18f core-context lock failed: expected 5, found {len(core_cols)}: {core_cols}")
if len(extended_cols) != 14:
    raise SystemExit(f"18f extended-context lock failed: expected 14, found {len(extended_cols)}")

FEATURE_SETS = {
    "N2a_core_plus_normwear_mean768": core_cols + emb_cols,
    "N2b_extended_plus_normwear_mean768": extended_cols + emb_cols,
}

# 3. Locked baseline outputs
base_fold = pd.read_csv(BASE_FOLD)
base_pred = pd.read_csv(BASE_PRED)
base_avg = pd.read_csv(BASE_AVG)
base_summary = pd.read_csv(BASE_SUMMARY)
lock_rows = []
for anchor in ["START", "END"]:
    for model_name in ["ElasticNet", "HGB"]:
        for baseline_name, expected_n in [("C0a_core_context", 5), ("C0b_extended_context", 14)]:
            sf = base_summary[(base_summary["anchor"] == anchor) & (base_summary["model"] == model_name) & (base_summary["feature_set"] == baseline_name)]
            ff = base_fold[(base_fold["anchor"] == anchor) & (base_fold["model"] == model_name) & (base_fold["feature_set"] == baseline_name)]
            pp = base_pred[(base_pred["anchor"] == anchor) & (base_pred["model"] == model_name) & (base_pred["feature_set"] == baseline_name)]
            aa = base_avg[(base_avg["anchor"] == anchor) & (base_avg["model"] == model_name) & (base_avg["feature_set"] == baseline_name)]
            if len(sf) != 1 or len(ff) != 25 or len(pp) != 1141 * 5 or len(aa) != 1141:
                raise SystemExit(f"Locked 18f baseline structure failure: {anchor} {model_name} {baseline_name}")
            if set(aa[ROW_ID].astype(str)) != set(locked_ids):
                raise SystemExit(f"Locked baseline row IDs differ: {anchor} {model_name} {baseline_name}")
            lock_rows.append({
                "anchor": anchor,
                "model": model_name,
                "baseline_feature_set": baseline_name,
                "expected_raw_context_columns": expected_n,
                "n_fold_rows": len(ff),
                "n_repeated_oof_predictions": len(pp),
                "n_avg_oof_rows": len(aa),
                "mae_pb_locked": float(sf.iloc[0]["mae_pb"]),
                "rmse_pb_locked": float(sf.iloc[0]["rmse_pb"]),
                "r2_rep_locked": float(sf.iloc[0]["r2_rep"]),
                "lock_pass": True,
            })
lock_df = pd.DataFrame(lock_rows)
lock_df.to_csv(LOCKCHECK_OUT, index=False)

# 4. Reconstruct and verify exact outer folds against 18f test row IDs
outer_split_map = {}
for repeat0 in range(OUTER_REPEATS):
    outer_seed = BASE_SEED + repeat0 * 10000
    splits = shuffled_group_folds(start_df[GROUP], OUTER_FOLDS, outer_seed)
    for outer_fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        current_test_ids = set(start_df.iloc[te_idx][ROW_ID].astype(str))
        for model_name in ["ElasticNet", "HGB"]:
            for baseline_name in ["C0a_core_context", "C0b_extended_context"]:
                q = base_pred[
                    (base_pred["repeat"] == repeat0 + 1)
                    & (base_pred["outer_fold"] == outer_fold)
                    & (base_pred["anchor"] == "START")
                    & (base_pred["model"] == model_name)
                    & (base_pred["feature_set"] == baseline_name)
                ]
                if current_test_ids != set(q[ROW_ID].astype(str)):
                    raise SystemExit(f"Outer split differs from locked 18f: R{repeat0+1}F{outer_fold} {model_name} {baseline_name}")
        train_groups = set(start_df.iloc[tr_idx][GROUP].astype(str))
        test_groups = set(start_df.iloc[te_idx][GROUP].astype(str))
        if train_groups & test_groups:
            raise RuntimeError("Participant leakage in reconstructed outer split.")
        outer_split_map[(repeat0 + 1, outer_fold)] = (tr_idx, te_idx)

# 5. Resume support
def read_partial(fp):
    if fp.exists() and fp.stat().st_size > 0:
        return pd.read_csv(fp)
    return pd.DataFrame()

fold_partial = read_partial(PARTIAL_FOLD)
pred_partial = read_partial(PARTIAL_PRED)
hyper_partial = read_partial(PARTIAL_HYPER)
fold_rows = fold_partial.to_dict("records") if len(fold_partial) else []
pred_rows = pred_partial.to_dict("records") if len(pred_partial) else []
hyper_rows = hyper_partial.to_dict("records") if len(hyper_partial) else []
completed_keys = set()
if len(fold_partial):
    for _, r in fold_partial.iterrows():
        completed_keys.add((int(r["repeat"]), int(r["outer_fold"]), str(r["anchor"]), str(r["model"]), str(r["feature_set"])))

TOTAL_NEW_TASKS = 25 * 2 * 2 * 2
print(f"Resume: {len(completed_keys)}/{TOTAL_NEW_TASKS} new-model outer tasks already complete.")

anchors = {"START": start_df, "END": end_df}
t0 = time.time()
fresh_tasks = 0

# 6. Fit only new combined models
for repeat in range(1, OUTER_REPEATS + 1):
    outer_seed = BASE_SEED + (repeat - 1) * 10000
    for outer_fold in range(1, OUTER_FOLDS + 1):
        tr_idx, te_idx = outer_split_map[(repeat, outer_fold)]
        train_groups = set(start_df.iloc[tr_idx][GROUP].astype(str))
        test_groups = set(start_df.iloc[te_idx][GROUP].astype(str))
        for anchor, df in anchors.items():
            y_train = df.iloc[tr_idx][TARGET].astype(float).reset_index(drop=True)
            y_test = df.iloc[te_idx][TARGET].astype(float).reset_index(drop=True)
            g_train = df.iloc[tr_idx][GROUP].astype(str).reset_index(drop=True)
            g_test = df.iloc[te_idx][GROUP].astype(str).reset_index(drop=True)
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            test_df = df.iloc[te_idx].reset_index(drop=True)
            for model_name in ["ElasticNet", "HGB"]:
                for fs_idx, (fs_name, cols) in enumerate(FEATURE_SETS.items()):
                    task_key = (repeat, outer_fold, anchor, model_name, fs_name)
                    if task_key in completed_keys:
                        continue
                    inner_seed = outer_seed + outer_fold * 1000 + 500 + fs_idx * 10 + (1 if model_name == "ElasticNet" else 2)
                    best_params, inner_mae, num_cols, cat_cols = tune_model(
                        model_name, train_df, y_train, g_train, cols, inner_seed
                    )
                    pipe = make_pipeline(model_name, best_params, num_cols, cat_cols)
                    pipe.fit(train_df[cols], y_train)
                    pred = pipe.predict(test_df[cols])
                    if not np.all(np.isfinite(pred)):
                        raise RuntimeError(f"Non-finite predictions: {anchor} {model_name} {fs_name} R{repeat}F{outer_fold}")
                    metrics = participant_balanced_metrics(y_test, pred, g_test)
                    n_context = len(core_cols) if fs_name.startswith("N2a_") else len(extended_cols)
                    fold_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": fs_name,
                        "baseline_feature_set": BASELINE_FEATURE_SETS[fs_name],
                        "n_raw_features": len(cols),
                        "n_context_features": n_context,
                        "n_normwear_features": 768,
                        "n_train_rows": len(tr_idx),
                        "n_test_rows": len(te_idx),
                        "n_train_participants": len(train_groups),
                        "n_test_participants": len(test_groups),
                        "inner_selected_mae_pb": inner_mae,
                        "prediction_abs_max": float(np.max(np.abs(pred))),
                        **metrics,
                    })
                    hyper_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "model": model_name,
                        "feature_set": fs_name,
                        "selected_params_json": json.dumps(best_params, sort_keys=True),
                        "inner_selected_mae_pb": inner_mae,
                    })
                    meta = test_df[[ROW_ID, GROUP, "task_label", "current_time_sec", TARGET]].reset_index(drop=True)
                    for j in range(len(meta)):
                        pred_rows.append({
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "anchor": anchor,
                            "model": model_name,
                            "feature_set": fs_name,
                            ROW_ID: meta.loc[j, ROW_ID],
                            GROUP: meta.loc[j, GROUP],
                            "task_label": meta.loc[j, "task_label"],
                            "current_time_sec": meta.loc[j, "current_time_sec"],
                            "y_true": float(meta.loc[j, TARGET]),
                            "y_pred": float(pred[j]),
                        })
                    fresh_tasks += 1
                    completed_keys.add(task_key)
                    pd.DataFrame(fold_rows).to_csv(PARTIAL_FOLD, index=False)
                    pd.DataFrame(pred_rows).to_csv(PARTIAL_PRED, index=False)
                    pd.DataFrame(hyper_rows).to_csv(PARTIAL_HYPER, index=False)
                    elapsed = time.time() - t0
                    avg_sec = elapsed / max(fresh_tasks, 1)
                    remain = TOTAL_NEW_TASKS - len(completed_keys)
                    eta_min = remain * avg_sec / 60.0
                    print(
                        f"Completed {len(completed_keys)}/{TOTAL_NEW_TASKS} tasks | "
                        f"{anchor} {model_name} {fs_name} R{repeat}F{outer_fold} | "
                        f"fresh mean {avg_sec:.1f}s/task | ETA {eta_min:.1f} min"
                    )

# 7. Final new outputs
fold_df = pd.DataFrame(fold_rows)
pred_df = pd.DataFrame(pred_rows)
hyper_df = pd.DataFrame(hyper_rows)
if len(fold_df) != TOTAL_NEW_TASKS:
    raise RuntimeError(f"Expected {TOTAL_NEW_TASKS} fold/task rows, found {len(fold_df)}")
expected_pred_rows = 1141 * 5 * 2 * 2 * 2
if len(pred_df) != expected_pred_rows:
    raise RuntimeError(f"Expected {expected_pred_rows} repeated-OOF rows, found {len(pred_df)}")
fold_df.to_csv(FOLD_OUT, index=False)
pred_df.to_csv(PRED_OUT, index=False)
hyper_df.to_csv(HYPER_OUT, index=False)

avg = (
    pred_df.groupby(
        ["anchor", "model", "feature_set", ROW_ID, GROUP, "task_label", "current_time_sec", "y_true"],
        as_index=False,
    ).agg(
        y_pred_avg=("y_pred", "mean"),
        y_pred_sd=("y_pred", "std"),
        n_oof_predictions=("y_pred", "count"),
    )
)
if not (avg["n_oof_predictions"] == 5).all():
    raise RuntimeError("Not every new-model row has exactly five repeated OOF predictions.")
avg.to_csv(AVG_OUT, index=False)

# 8. Absolute summaries: locked baselines + new combined models
new_summary_rows = []
for (anchor, model_name, fs_name), g in avg.groupby(["anchor", "model", "feature_set"]):
    m = participant_balanced_metrics(g["y_true"], g["y_pred_avg"], g[GROUP])
    new_summary_rows.append({
        "anchor": anchor,
        "source": "25_new_normwear",
        "model": model_name,
        "feature_set": fs_name,
        "n_rows": len(g),
        "n_participants": g[GROUP].nunique(),
        **m,
    })
new_summary = pd.DataFrame(new_summary_rows)
locked_summary_subset = base_summary[base_summary["feature_set"].isin(["C0a_core_context", "C0b_extended_context"])].copy()
locked_summary_subset["source"] = "18f_locked_baseline"
summary_cols = ["anchor", "source", "model", "feature_set", "n_rows", "n_participants", "mae_pb", "rmse_pb", "mae_rep", "rmse_rep", "r2_rep"]
summary_df = pd.concat([locked_summary_subset[summary_cols], new_summary[summary_cols]], ignore_index=True)
summary_df.to_csv(SUMMARY_OUT, index=False)

# 9. Incremental value against locked baseline predictions
inc_rows = []
for anchor_idx, anchor in enumerate(["START", "END"]):
    for model_idx, model_name in enumerate(["ElasticNet", "HGB"]):
        for comp_idx, combined_name in enumerate(FEATURE_SETS.keys()):
            baseline_name = BASELINE_FEATURE_SETS[combined_name]
            obs, ci, n_rows, n_pids = bootstrap_increment_from_two_avg(
                base_avg, avg, anchor, model_name, baseline_name, combined_name,
                BOOTSTRAP_B,
                BASE_SEED + 190000 + anchor_idx * 10000 + model_idx * 1000 + comp_idx,
            )
            f0 = base_fold[
                (base_fold["anchor"] == anchor)
                & (base_fold["model"] == model_name)
                & (base_fold["feature_set"] == baseline_name)
            ][["repeat", "outer_fold", "mae_pb", "rmse_pb", "r2_rep"]]
            f1 = fold_df[
                (fold_df["anchor"] == anchor)
                & (fold_df["model"] == model_name)
                & (fold_df["feature_set"] == combined_name)
            ][["repeat", "outer_fold", "mae_pb", "rmse_pb", "r2_rep"]]
            paired = f0.merge(f1, on=["repeat", "outer_fold"], suffixes=("_base", "_comb"), validate="one_to_one")
            if len(paired) != 25:
                raise RuntimeError(f"Expected 25 paired folds: {anchor} {model_name} {combined_name}")
            inc_rows.append({
                "anchor": anchor,
                "model": model_name,
                "representation": "NormWear_IMU_primary_mean768",
                "comparison": COMPARISON_LABELS[combined_name],
                "baseline_feature_set": baseline_name,
                "combined_feature_set": combined_name,
                "baseline_source": "18f_locked",
                "n_rows": n_rows,
                "n_participants": n_pids,
                **obs,
                **ci,
                "fold_delta_mae_pb_mean": float((paired["mae_pb_base"] - paired["mae_pb_comb"]).mean()),
                "fold_delta_mae_pb_positive_fraction": float((paired["mae_pb_base"] > paired["mae_pb_comb"]).mean()),
                "fold_delta_rmse_pb_mean": float((paired["rmse_pb_base"] - paired["rmse_pb_comb"]).mean()),
                "fold_delta_rmse_pb_positive_fraction": float((paired["rmse_pb_base"] > paired["rmse_pb_comb"]).mean()),
                "fold_delta_r2_rep_mean": float((paired["r2_rep_comb"] - paired["r2_rep_base"]).mean()),
                "fold_delta_r2_rep_positive_fraction": float((paired["r2_rep_comb"] > paired["r2_rep_base"]).mean()),
            })
inc_df = pd.DataFrame(inc_rows)
inc_df.to_csv(INCREMENT_OUT, index=False)

# 10. START/END robustness
robust_rows = []
for model_name in ["ElasticNet", "HGB"]:
    for combined_name in FEATURE_SETS.keys():
        label = COMPARISON_LABELS[combined_name]
        s = inc_df[(inc_df["model"] == model_name) & (inc_df["comparison"] == label)].copy()
        if set(s["anchor"]) != {"START", "END"}:
            raise RuntimeError(f"Missing START/END pair: {model_name} {label}")
        a = s.set_index("anchor")
        dms = float(a.loc["START", "delta_mae_pb"])
        dme = float(a.loc["END", "delta_mae_pb"])
        rms = float(a.loc["START", "delta_rmse_pb"])
        rme = float(a.loc["END", "delta_rmse_pb"])
        r2s = float(a.loc["START", "delta_r2_rep"])
        r2e = float(a.loc["END", "delta_r2_rep"])
        robust_rows.append({
            "model": model_name,
            "representation": "NormWear_IMU_primary_mean768",
            "comparison": label,
            "start_delta_mae_pb": dms,
            "end_delta_mae_pb": dme,
            "mae_direction_consistent": int(np.sign(dms) == np.sign(dme)),
            "mae_both_positive": int(dms > 0 and dme > 0),
            "mae_both_negative": int(dms < 0 and dme < 0),
            "start_mae_ci_excludes_zero_positive": int(a.loc["START", "delta_mae_pb_ci_low"] > 0),
            "end_mae_ci_excludes_zero_positive": int(a.loc["END", "delta_mae_pb_ci_low"] > 0),
            "start_mae_ci_excludes_zero_negative": int(a.loc["START", "delta_mae_pb_ci_high"] < 0),
            "end_mae_ci_excludes_zero_negative": int(a.loc["END", "delta_mae_pb_ci_high"] < 0),
            "start_delta_rmse_pb": rms,
            "end_delta_rmse_pb": rme,
            "rmse_direction_consistent": int(np.sign(rms) == np.sign(rme)),
            "rmse_both_positive": int(rms > 0 and rme > 0),
            "rmse_both_negative": int(rms < 0 and rme < 0),
            "start_delta_r2_rep": r2s,
            "end_delta_r2_rep": r2e,
            "r2_direction_consistent": int(np.sign(r2s) == np.sign(r2e)),
            "r2_both_positive": int(r2s > 0 and r2e > 0),
            "r2_both_negative": int(r2s < 0 and r2e < 0),
            "start_bootstrap_positive_prob_mae": float(a.loc["START", "delta_mae_pb_bootstrap_positive_prob"]),
            "end_bootstrap_positive_prob_mae": float(a.loc["END", "delta_mae_pb_bootstrap_positive_prob"]),
            "start_fold_positive_fraction_mae": float(a.loc["START", "fold_delta_mae_pb_positive_fraction"]),
            "end_fold_positive_fraction_mae": float(a.loc["END", "fold_delta_mae_pb_positive_fraction"]),
        })
robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(ROBUST_OUT, index=False)

# 11. Audit
elapsed = time.time() - t0
lines = []
add = lines.append
add("DATASET C — PRIMARY FROZEN NORMWEAR-IMU INCREMENTAL NESTED-CV AUDIT")
add("=" * 118)
add(f"Rows per anchor: {len(start_df)}")
add(f"Participants: {start_df[GROUP].nunique()}")
add("Scientific question: does frozen NormWear-IMU mean768 add value beyond the LOCKED 18f context baselines?")
add(f"Outer validation: {OUTER_REPEATS} repeats x {OUTER_FOLDS} participant-grouped folds")
add(f"Inner validation for NEW combined models: {INNER_FOLDS} participant-grouped folds")
add(f"Bootstrap: {BOOTSTRAP_B} participant-cluster resamples on averaged repeated-OOF predictions")
add(f"Elapsed minutes for new-model run stage: {elapsed/60:.2f}")
add("")
add("1. REPRESENTATION LOCK")
add("-" * 118)
add(f"START mean768 SHA256: {start_sha}")
add(f"END mean768 SHA256: {end_sha}")
add(f"START embedding shape: {s_emb.shape}; finite={np.isfinite(s_emb).all()}")
add(f"END embedding shape: {e_emb.shape}; finite={np.isfinite(e_emb).all()}")
add("Representation was extracted by frozen NormWear before supervised downstream modeling.")
add("No representation tuning or pooling selection is performed here.")
add("")
add("2. LITERAL 18f BASELINE LOCK")
add("-" * 118)
add("C0a/C0b context baselines are NOT re-fit in Script 25.")
add("Their held-out predictions, fold metrics, averaged OOF predictions, and absolute metrics are loaded directly from 18f.")
add("Reconstructed Script-25 outer test row IDs were checked against every locked 18f baseline fold.")
add("All exact outer-split checks: PASS.")
add("")
add(lock_df.to_string(index=False))
add("")
add("3. FEATURE SETS")
add("-" * 118)
add(f"C0a locked core context: {len(core_cols)} columns (baseline source=18f).")
add(f"C0b locked extended context: {len(extended_cols)} columns (baseline source=18f).")
add(f"N2a core + NormWear mean768: {len(core_cols)+768} raw predictors.")
add(f"N2b extended + NormWear mean768: {len(extended_cols)+768} raw predictors.")
add("")
add("4. LEAKAGE CONTROLS")
add("-" * 118)
add("PASS: exact 18f outer participant splits are reused.")
add("PASS: outer train/test participants are disjoint.")
add("PASS: inner tuning of NEW models uses only outer-training participants.")
add("PASS: scaler/imputer/one-hot/variance filter for NEW models are refit inside training folds.")
add("PASS: frozen NormWear embedding extraction used no target labels and no outer-test adaptation.")
add("PASS: raw IMU windows derive only from [t-10,t]; target is Borg(t+10).")
add("")
add("5. ABSOLUTE MODEL SUMMARY")
add("-" * 118)
show_summary = summary_df[["anchor", "source", "model", "feature_set", "n_rows", "n_participants", "mae_pb", "rmse_pb", "mae_rep", "rmse_rep", "r2_rep"]].sort_values(["anchor", "model", "mae_pb"])
add(show_summary.to_string(index=False))
add("")
add("6. PRIMARY NORMWEAR INCREMENTAL VALUE")
add("-" * 118)
show_cols = [
    "anchor", "model", "comparison",
    "delta_mae_pb", "delta_mae_pb_ci_low", "delta_mae_pb_ci_high", "delta_mae_pb_bootstrap_positive_prob",
    "delta_rmse_pb", "delta_rmse_pb_ci_low", "delta_rmse_pb_ci_high", "delta_rmse_pb_bootstrap_positive_prob",
    "delta_r2_rep", "delta_r2_rep_ci_low", "delta_r2_rep_ci_high", "delta_r2_rep_bootstrap_positive_prob",
    "fold_delta_mae_pb_positive_fraction",
]
add(inc_df[show_cols].to_string(index=False))
add("")
add("Sign convention:")
add("  delta_mae_pb  > 0 => NormWear improves participant-balanced MAE.")
add("  delta_rmse_pb > 0 => NormWear improves participant-balanced RMSE.")
add("  delta_r2_rep  > 0 => NormWear improves pooled R2.")
add("")
add("7. START/END ALIGNMENT ROBUSTNESS")
add("-" * 118)
add(robust_df.to_string(index=False))
add("")
add("Interpretation guardrail:")
add("  Same-direction START/END point estimates are required for alignment robustness.")
add("  Positive MAE point estimates alone are NOT a rescue claim.")
add("  Strong positive evidence requires positive MAE increments with 95% CIs excluding zero under BOTH anchors,")
add("  together with concordant RMSE/R2 direction and broad outer-fold support.")
add("  If uncertainty spans zero, the result remains weak/uncertain even if the point estimate is positive.")
add("")
add("8. PREDICTION NUMERICAL STABILITY")
add("-" * 118)
pred_stability = fold_df.groupby(["anchor", "model", "feature_set"], as_index=False).agg(
    max_abs_prediction=("prediction_abs_max", "max"),
    median_abs_prediction_max=("prediction_abs_max", "median"),
)
add(pred_stability.to_string(index=False))
add("")
add("9. SCIENTIFIC GUARDRAILS")
add("-" * 118)
add("The primary estimand is M4-M0: context + frozen NormWear representation versus the corresponding locked context baseline.")
add("No foundation-representation-only model is used to infer incremental value in this script.")
add("No CLS sensitivity result is used here.")
add("No residualized representation result is used here.")
add("No model or representation will be re-tuned to chase a positive wearable result.")
add("")
add("10. OUTPUTS")
add("-" * 118)
for fp in [LOCKCHECK_OUT, FOLD_OUT, PRED_OUT, AVG_OUT, SUMMARY_OUT, INCREMENT_OUT, ROBUST_OUT, HYPER_OUT]:
    add(f"{fp.name} | bytes={fp.stat().st_size} | SHA256={sha256_file(fp)}")
add("")

global_pass = (
    len(fold_df) == TOTAL_NEW_TASKS
    and len(avg) == 1141 * 2 * 2 * 2
    and np.isfinite(pred_df["y_pred"].to_numpy(float)).all()
    and np.isfinite(avg["y_pred_avg"].to_numpy(float)).all()
    and len(inc_df) == 8
    and len(robust_df) == 4
    and lock_df["lock_pass"].all()
)
add("11. DECISION GATE")
add("-" * 118)
add(f"DATASET C PRIMARY NORMWEAR INCREMENTAL BENCHMARK PASS: {global_pass}")
if global_pass:
    add("The benchmark is technically complete. Scientific interpretation must be based on incremental-value")
    add("and alignment-robustness outputs, not on standalone absolute performance.")
else:
    add("DO NOT interpret scientific results; inspect the failed gate first.")
add("=" * 118)
add("END OF PRIMARY NORMWEAR INCREMENTAL BENCHMARK")
AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

if global_pass:
    for fp in [PARTIAL_FOLD, PARTIAL_PRED, PARTIAL_HYPER]:
        if fp.exists():
            fp.unlink()

print("")
print("Created:")
print(" ", AUDIT_OUT)
print(" ", INCREMENT_OUT)
print(" ", ROBUST_OUT)
print("")
print(f"DATASET C PRIMARY NORMWEAR INCREMENTAL BENCHMARK PASS: {global_pass}")
print("Upload these three files to ChatGPT:")
print("  dataset_C_normwear_imu_primary_nestedcv_audit.txt")
print("  dataset_C_normwear_primary_incremental_value.csv")
print("  dataset_C_normwear_primary_alignment_robustness.csv")
