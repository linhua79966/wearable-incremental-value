from pathlib import Path
import json
import time
import hashlib
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]

DERIVED = ROOT / "derived" / "dataset_C_phase1"
START_TABLE = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_TABLE = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

NW_DIR = ROOT / "derived" / "dataset_C_normwear_imu"
START_EMB = NW_DIR / "dataset_C_normwear_imu_start_cls768.npz"
END_EMB = NW_DIR / "dataset_C_normwear_imu_end_cls768.npz"

SRC27 = ROOT / "results" / "dataset_C_normwear_imu_cls_sensitivity_nestedcv"
HYPER_IN = SRC27 / "dataset_C_normwear_cls_sensitivity_selected_hyperparameters.csv"

OUTDIR = SRC27 / "convergence_audit_27c"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR = ROOT / "audit"
AUDITDIR.mkdir(parents=True, exist_ok=True)

FIT_OUT = OUTDIR / "dataset_C_normwear_cls_problem_candidate_recovery_fit_audit.csv"
TASK_OUT = OUTDIR / "dataset_C_normwear_cls_problem_candidate_recovery_task_summary.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_normwear_cls_problem_candidate_recovery_audit_27c.txt"

PARTIAL_FIT = OUTDIR / "_partial_recovery_fit_audit.csv"
PARTIAL_TASK = OUTDIR / "_partial_recovery_task_summary.csv"

TARGET = "target_borg"
GROUP = "subject_num"
ROW_ID = "row_id"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BASE_SEED = 20260901

RECOVERY_MAX_ITER = 300000
PROBLEM_ALPHA = 0.01
PROBLEM_L1 = 0.1

EXPECTED_START_EMB_SHA256 = "499775E1FA69E9FBF57FEEF42E3D81034990CA7DE08D6FF9A96AF7F0589F07E5"
EXPECTED_END_EMB_SHA256 = "B9511E4F4C81CD3807153191E38A06AE7EA19DF44C838BAE6E0586656D83466C"
EXPECTED_HYPER_SHA256 = "C57CDAF3F495BC41AC0DDE70050479B35366E76D4C80655DDE6907810C95DB50"


def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def participant_balanced_mae(y_true, y_pred, groups):
    d = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_pred, dtype=float),
        "g": np.asarray(groups).astype(str),
    })
    d["ae"] = np.abs(d["y"] - d["p"])
    return float(d.groupby("g", sort=False)["ae"].mean().mean())


def shuffled_group_folds(groups, n_splits, seed):
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique = groups.unique().tolist()
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
        mask = groups.isin(gs).to_numpy()
        splits.append((idx[~mask], idx[mask]))
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


def make_pipe(numeric_cols, categorical_cols):
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

    model = ElasticNet(
        alpha=PROBLEM_ALPHA,
        l1_ratio=PROBLEM_L1,
        max_iter=RECOVERY_MAX_ITER,
        random_state=BASE_SEED,
        selection="cyclic",
    )

    return Pipeline([
        ("pre", pre),
        ("var", VarianceThreshold(threshold=0.0)),
        ("model", model),
    ])


def fit_capture(pipe, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y)
    conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter = int(pipe.named_steps["model"].n_iter_)
    return {
        "convergence_warning": int(bool(conv)),
        "n_convergence_warnings": int(len(conv)),
        "n_iter": n_iter,
        "hit_recovery_max_iter": int(n_iter >= RECOVERY_MAX_ITER),
        "warning_message": " | ".join(str(w.message) for w in conv),
    }


def read_partial(fp):
    return pd.read_csv(fp) if fp.exists() and fp.stat().st_size > 0 else pd.DataFrame()


for fp in [START_TABLE, END_TABLE, DICT, START_EMB, END_EMB, HYPER_IN]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

start_sha = sha256_file(START_EMB)
end_sha = sha256_file(END_EMB)
hyper_sha = sha256_file(HYPER_IN)

if start_sha != EXPECTED_START_EMB_SHA256:
    raise SystemExit(f"START CLS hash mismatch: {start_sha}")
if end_sha != EXPECTED_END_EMB_SHA256:
    raise SystemExit(f"END CLS hash mismatch: {end_sha}")
if hyper_sha != EXPECTED_HYPER_SHA256:
    raise SystemExit(f"Script-27 hyperparameter hash mismatch: {hyper_sha}")

start_df = pd.read_csv(START_TABLE).sort_values(ROW_ID).reset_index(drop=True)
end_df = pd.read_csv(END_TABLE).sort_values(ROW_ID).reset_index(drop=True)
fd = pd.read_csv(DICT)

if len(start_df) != 1141 or len(end_df) != 1141:
    raise SystemExit("Expected 1141 rows per anchor.")
if start_df[GROUP].nunique() != 34:
    raise SystemExit("Expected 34 participants.")
if not np.array_equal(start_df[ROW_ID].astype(str).to_numpy(), end_df[ROW_ID].astype(str).to_numpy()):
    raise SystemExit("START/END row IDs differ.")
if not np.allclose(start_df[TARGET].to_numpy(float), end_df[TARGET].to_numpy(float), equal_nan=True):
    raise SystemExit("START/END targets differ.")

zs = np.load(START_EMB, allow_pickle=False)
ze = np.load(END_EMB, allow_pickle=False)
locked_ids = start_df[ROW_ID].astype(str).to_numpy()

for ids, emb, tag in [
    (zs["row_id"].astype(str), zs["embedding"].astype(np.float32), "START"),
    (ze["row_id"].astype(str), ze["embedding"].astype(np.float32), "END"),
]:
    if not np.array_equal(ids, locked_ids):
        raise SystemExit(f"{tag} CLS embedding row IDs mismatch.")
    if emb.shape != (1141, 768):
        raise SystemExit(f"{tag} unexpected embedding shape {emb.shape}.")
    if not np.isfinite(emb).all():
        raise SystemExit(f"{tag} non-finite CLS embedding.")

emb_cols = [f"normwear_cls_{j:03d}" for j in range(768)]
start_df = pd.concat([start_df, pd.DataFrame(zs["embedding"].astype(np.float32), columns=emb_cols)], axis=1)
end_df = pd.concat([end_df, pd.DataFrame(ze["embedding"].astype(np.float32), columns=emb_cols)], axis=1)

role_map = {
    role: fd.loc[fd["role"] == role, "column"].tolist()
    for role in fd["role"].dropna().unique()
}

forbidden = {
    ROW_ID, GROUP, TARGET, "borg_change_10s", "target_time_sec",
    "trial_duration_sec_qc", "known_corrupt_shoulder_imu_trial_qc",
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

if len(core_cols) != 5 or len(extended_cols) != 14:
    raise SystemExit(f"Context lock failed: core={len(core_cols)} extended={len(extended_cols)}")

FEATURE_SETS = {
    "S1a_core_plus_normwear_cls768": core_cols + emb_cols,
    "S1b_extended_plus_normwear_cls768": extended_cols + emb_cols,
}

hyper = pd.read_csv(HYPER_IN)
hyper = hyper[hyper["model"].astype(str) == "ElasticNet"].copy()
if len(hyper) != 100:
    raise SystemExit(f"Expected 100 ElasticNet selected rows, found {len(hyper)}.")

fit_partial = read_partial(PARTIAL_FIT)
task_partial = read_partial(PARTIAL_TASK)

fit_rows = fit_partial.to_dict("records") if len(fit_partial) else []
task_rows = task_partial.to_dict("records") if len(task_partial) else []

completed = set()
if len(task_partial):
    for _, r in task_partial.iterrows():
        completed.add((int(r["repeat"]), int(r["outer_fold"]), str(r["anchor"]), str(r["feature_set"])))

TOTAL_TASKS = 100
print(f"Resume: {len(completed)}/{TOTAL_TASKS} recovery tasks already complete.")

anchors = {"START": start_df, "END": end_df}
t0 = time.time()
fresh = 0

for repeat in range(1, OUTER_REPEATS + 1):
    outer_seed = BASE_SEED + (repeat - 1) * 10000
    outer_splits = shuffled_group_folds(start_df[GROUP], OUTER_FOLDS, outer_seed)

    for outer_fold, (tr_idx, _) in enumerate(outer_splits, start=1):
        for anchor, df in anchors.items():
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            y_train = train_df[TARGET].astype(float).reset_index(drop=True)
            g_train = train_df[GROUP].astype(str).reset_index(drop=True)

            for fs_idx, (fs_name, cols) in enumerate(FEATURE_SETS.items()):
                key = (repeat, outer_fold, anchor, fs_name)
                if key in completed:
                    continue

                h = hyper[
                    (hyper["repeat"] == repeat)
                    & (hyper["outer_fold"] == outer_fold)
                    & (hyper["anchor"].astype(str) == anchor)
                    & (hyper["feature_set"].astype(str) == fs_name)
                ]
                if len(h) != 1:
                    raise RuntimeError(f"Selected-hyperparameter row mismatch: {key}")

                selected_params = json.loads(h.iloc[0]["selected_params_json"])
                selected_score = float(h.iloc[0]["inner_selected_mae_pb"])

                numeric_cols, categorical_cols = split_column_types(train_df, cols)
                inner_seed = outer_seed + outer_fold * 1000 + 500 + fs_idx * 10 + 1
                inner_splits = shuffled_group_folds(g_train, INNER_FOLDS, inner_seed)

                fold_scores = []
                n_warnings = 0
                n_hitmax = 0
                max_niter = 0

                for inner_fold, (itr, iva) in enumerate(inner_splits, start=1):
                    pipe = make_pipe(numeric_cols, categorical_cols)
                    cap = fit_capture(pipe, train_df.iloc[itr][cols], y_train.iloc[itr])

                    pred = pipe.predict(train_df.iloc[iva][cols])
                    score = participant_balanced_mae(
                        y_train.iloc[iva],
                        pred,
                        g_train.iloc[iva],
                    )

                    fold_scores.append(score)
                    n_warnings += cap["convergence_warning"]
                    n_hitmax += cap["hit_recovery_max_iter"]
                    max_niter = max(max_niter, cap["n_iter"])

                    fit_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "feature_set": fs_name,
                        "inner_fold": inner_fold,
                        "alpha": PROBLEM_ALPHA,
                        "l1_ratio": PROBLEM_L1,
                        "recovery_max_iter": RECOVERY_MAX_ITER,
                        "mae_pb": score,
                        **cap,
                    })

                recovered_score = float(np.mean(fold_scores))
                recovered_beats_selected = int(recovered_score < selected_score - 1e-12)

                task_rows.append({
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "anchor": anchor,
                    "feature_set": fs_name,
                    "original_selected_params_json": json.dumps(selected_params, sort_keys=True),
                    "original_selected_inner_mae_pb": selected_score,
                    "recovered_problem_candidate_inner_mae_pb": recovered_score,
                    "recovered_minus_selected_mae": recovered_score - selected_score,
                    "recovered_problem_candidate_beats_selected": recovered_beats_selected,
                    "recovered_inner_convergence_warning_count": int(n_warnings),
                    "recovered_inner_hit_max_iter_count": int(n_hitmax),
                    "recovered_inner_max_n_iter": int(max_niter),
                })

                completed.add(key)
                fresh += 1

                pd.DataFrame(fit_rows).to_csv(PARTIAL_FIT, index=False)
                pd.DataFrame(task_rows).to_csv(PARTIAL_TASK, index=False)

                elapsed = time.time() - t0
                mean_sec = elapsed / max(fresh, 1)
                remain = TOTAL_TASKS - len(completed)
                eta_min = remain * mean_sec / 60.0

                print(
                    f"Recovered {len(completed)}/{TOTAL_TASKS} | "
                    f"{anchor} {fs_name} R{repeat}F{outer_fold} | "
                    f"warnings={n_warnings} | max_n_iter={max_niter} | "
                    f"beats_selected={recovered_beats_selected} | ETA {eta_min:.1f} min"
                )

fit_df = pd.DataFrame(fit_rows)
task_df = pd.DataFrame(task_rows)

if len(task_df) != TOTAL_TASKS:
    raise RuntimeError(f"Expected {TOTAL_TASKS} task rows, found {len(task_df)}.")
if len(fit_df) != TOTAL_TASKS * INNER_FOLDS:
    raise RuntimeError(f"Expected {TOTAL_TASKS * INNER_FOLDS} fit rows, found {len(fit_df)}.")

fit_df.to_csv(FIT_OUT, index=False)
task_df.to_csv(TASK_OUT, index=False)

all_recovered_converged = bool(
    (task_df["recovered_inner_convergence_warning_count"] == 0).all()
    and (task_df["recovered_inner_hit_max_iter_count"] == 0).all()
)

n_beats = int(task_df["recovered_problem_candidate_beats_selected"].sum())
selection_stable = bool(all_recovered_converged and n_beats == 0)

min_margin = float(task_df["recovered_minus_selected_mae"].min())
median_margin = float(task_df["recovered_minus_selected_mae"].median())

lines = []
add = lines.append
add("DATASET C — SCRIPT 27 PROBLEM-CANDIDATE RECOVERY AUDIT (27c)")
add("=" * 118)
add("Purpose: recover only the one non-converged ElasticNet grid candidate (alpha=0.01, l1_ratio=0.1).")
add("Exact same inner folds/objective/preprocessing are used; only max_iter ceiling is increased for numerical convergence.")
add("No split, representation, baseline, target, metric, alpha, or l1_ratio is changed.")
add(f"Recovery max_iter ceiling: {RECOVERY_MAX_ITER}")
add(f"Audited outer ElasticNet tasks: {len(task_df)}")
add(f"Recovered inner fits: {len(fit_df)}")
add("")
add("1. INPUT LOCKS")
add("-" * 118)
add(f"START CLS768 SHA256: {start_sha}")
add(f"END CLS768 SHA256: {end_sha}")
add(f"Script-27 hyperparameter SHA256: {hyper_sha}")
add("")
add("2. RECOVERY CONVERGENCE")
add("-" * 118)
add(f"Recovered fits with ConvergenceWarning: {int(fit_df['convergence_warning'].sum())}/{len(fit_df)}")
add(f"Recovered fits hitting recovery max_iter: {int(fit_df['hit_recovery_max_iter'].sum())}/{len(fit_df)}")
add(f"Maximum recovered n_iter: {int(fit_df['n_iter'].max())}")
add(f"ALL RECOVERED FITS CONVERGED: {all_recovered_converged}")
add("")
add("3. HYPERPARAMETER-SELECTION STABILITY")
add("-" * 118)
add(f"Tasks where recovered problem candidate beats the originally selected candidate: {n_beats}/100")
add(f"Minimum recovered-minus-selected inner MAE: {min_margin:.12f}")
add(f"Median recovered-minus-selected inner MAE: {median_margin:.12f}")
add(f"SCRIPT-27 ELASTICNET SELECTION STABLE AFTER RECOVERY: {selection_stable}")
add("")
add("4. DECISION")
add("-" * 118)
if selection_stable:
    add("The only non-converged Script-27 grid candidate was recovered numerically and did not beat the originally selected candidate in any task.")
    add("Therefore the prior ConvergenceWarnings do not alter Script-27 ElasticNet hyperparameter selection.")
    add("No Script-27 rerun is required for this convergence issue.")
else:
    add("Do NOT lock the Script-27 ElasticNet scientific result yet.")
    if not all_recovered_converged:
        add("At least one recovered fit still failed to converge under the recovery ceiling.")
    if n_beats > 0:
        add("At least one recovered problem candidate outperformed the originally selected candidate.")
    add("Any correction must remain numerical and uniform; do not tune alpha/l1_ratio to chase result direction.")
add("")
add("Outputs:")
add(f"  {FIT_OUT}")
add(f"  {TASK_OUT}")
add("=" * 118)

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Created:")
print(" ", AUDIT_OUT)
print(" ", TASK_OUT)
print("")
print(f"ALL RECOVERED FITS CONVERGED: {all_recovered_converged}")
print(f"TASKS WHERE RECOVERED CANDIDATE BEATS SELECTED: {n_beats}/100")
print(f"SCRIPT-27 ELASTICNET SELECTION STABLE AFTER RECOVERY: {selection_stable}")
