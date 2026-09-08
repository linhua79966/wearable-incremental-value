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


# =============================================================================
# Script 33d — TARGETED numerical recovery for remaining Script-33/33c
# ElasticNet convergence failures.
#
# It reads Script 33c's 400-fit audit, keeps the 287 already-converged fits
# unchanged, and re-runs ONLY the remaining failed fits with a larger
# max_iter ceiling. No alpha/l1_ratio/split/metric/preprocessing/representation
# is changed.
#
# This is a numerical-validity audit only. It does NOT re-fit Script 33
# downstream outer models and does NOT chase result direction.
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DERIVED = ROOT / "derived" / "dataset_C_phase1"
START_TABLE = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_TABLE = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

EMB_DIR = ROOT / "derived" / "dataset_C_moment_imu"
START_EMB = EMB_DIR / "dataset_C_moment1base_imu_start_mean768.npz"
END_EMB = EMB_DIR / "dataset_C_moment1base_imu_end_mean768.npz"

SRC33 = ROOT / "results" / "dataset_C_moment1base_imu_primary_nestedcv"
HYPER_IN = SRC33 / "dataset_C_moment1base_primary_selected_hyperparameters.csv"

SRC33C = SRC33 / "convergence_audit_33c"
FIT33C_IN = SRC33C / "dataset_C_moment1base_problem_candidate_recovery_fit_audit.csv"
TASK33C_IN = SRC33C / "dataset_C_moment1base_problem_candidate_recovery_task_summary.csv"

OUTDIR = SRC33 / "convergence_audit_33d"
OUTDIR.mkdir(parents=True, exist_ok=True)

AUDITDIR = ROOT / "audit"
AUDITDIR.mkdir(parents=True, exist_ok=True)

FIT_OUT = OUTDIR / "dataset_C_moment1base_targeted_recovery_fit_audit.csv"
TASK_OUT = OUTDIR / "dataset_C_moment1base_targeted_recovery_task_summary.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_moment1base_targeted_convergence_recovery_audit_33d.txt"

TARGET = "target_borg"
GROUP = "subject_num"
ROW_ID = "row_id"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BASE_SEED = 20260901

PROBLEM_ALPHA = 0.01
PROBLEM_L1 = 0.1

# Numerical ceiling only. Objective/hyperparameters/tolerance stay unchanged.
RECOVERY_MAX_ITER = 1_000_000

EXPECTED_START_EMB_SHA256 = "56B29145E68DDDDEBDE934456B4D6BB431CCAD2FB854C607CE1349526CBF2256"
EXPECTED_END_EMB_SHA256 = "D44F9E25E82C5DDD6140D983734D47451407A59DA46A7F559603352EC0BD750F"
EXPECTED_HYPER_SHA256 = "2FE067D71D1BCCE36F42FDD5D619D5155525DA2D3381A24D73718FC5613C64B5"

FEATURE_SET_ORDER = [
    "M6a_core_plus_moment1base_mean768",
    "M6b_extended_plus_moment1base_mean768",
]


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

    if len(unique) < n_splits:
        raise ValueError(f"Need >= {n_splits} groups; found {len(unique)}.")

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


# -----------------------------------------------------------------------------
# 1. Input locks
# -----------------------------------------------------------------------------

required_files = [
    START_TABLE,
    END_TABLE,
    DICT,
    START_EMB,
    END_EMB,
    HYPER_IN,
    FIT33C_IN,
    TASK33C_IN,
]

for fp in required_files:
    if not fp.exists():
        raise SystemExit(f"Missing required input: {fp}")

start_sha = sha256_file(START_EMB)
end_sha = sha256_file(END_EMB)
hyper_sha = sha256_file(HYPER_IN)

if start_sha != EXPECTED_START_EMB_SHA256:
    raise SystemExit(f"START MOMENT mean768 hash mismatch: {start_sha}")

if end_sha != EXPECTED_END_EMB_SHA256:
    raise SystemExit(f"END MOMENT mean768 hash mismatch: {end_sha}")

if hyper_sha != EXPECTED_HYPER_SHA256:
    raise SystemExit(f"Script-33 hyperparameter hash mismatch: {hyper_sha}")

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
    equal_nan=True,
):
    raise SystemExit("START/END targets differ.")

zs = np.load(START_EMB, allow_pickle=False)
ze = np.load(END_EMB, allow_pickle=False)

locked_ids = start_df[ROW_ID].astype(str).to_numpy()
s_ids = zs["row_id"].astype(str)
e_ids = ze["row_id"].astype(str)
s_emb = zs["embedding"].astype(np.float32)
e_emb = ze["embedding"].astype(np.float32)

if not np.array_equal(s_ids, locked_ids):
    raise SystemExit("START MOMENT embedding row IDs mismatch.")

if not np.array_equal(e_ids, locked_ids):
    raise SystemExit("END MOMENT embedding row IDs mismatch.")

if s_emb.shape != (1141, 768) or e_emb.shape != (1141, 768):
    raise SystemExit("Unexpected MOMENT embedding shape.")

if not np.isfinite(s_emb).all() or not np.isfinite(e_emb).all():
    raise SystemExit("Non-finite MOMENT embedding.")

emb_cols = [f"moment1base_mean_{j:03d}" for j in range(768)]

start_df = pd.concat(
    [start_df, pd.DataFrame(s_emb, columns=emb_cols)],
    axis=1,
)

end_df = pd.concat(
    [end_df, pd.DataFrame(e_emb, columns=emb_cols)],
    axis=1,
)

role_map = {
    role: fd.loc[fd["role"] == role, "column"].tolist()
    for role in fd["role"].dropna().unique()
}

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
    c
    for c in role_map.get("context_core", [])
    if c in start_df.columns and c not in forbidden
]

extended_cols = []
for role in ["context_core", "context_extended"]:
    extended_cols.extend(role_map.get(role, []))

extended_cols = [
    c
    for c in dict.fromkeys(extended_cols)
    if c in start_df.columns and c not in forbidden
]

if len(core_cols) != 5 or len(extended_cols) != 14:
    raise SystemExit(
        f"Context lock failed: core={len(core_cols)} extended={len(extended_cols)}"
    )

FEATURE_SETS = {
    "M6a_core_plus_moment1base_mean768": core_cols + emb_cols,
    "M6b_extended_plus_moment1base_mean768": extended_cols + emb_cols,
}

hyper = pd.read_csv(HYPER_IN)
hyper = hyper[hyper["model"].astype(str) == "ElasticNet"].copy()

if len(hyper) != 100:
    raise SystemExit(f"Expected 100 Script-33 ElasticNet rows, found {len(hyper)}.")

fit33c = pd.read_csv(FIT33C_IN)
task33c = pd.read_csv(TASK33C_IN)

if len(fit33c) != 400:
    raise SystemExit(f"Expected 400 Script-33c fit rows, found {len(fit33c)}.")

if len(task33c) != 100:
    raise SystemExit(f"Expected 100 Script-33c task rows, found {len(task33c)}.")

# Guard: Script 33c must represent only the known problem candidate.
if not np.allclose(fit33c["alpha"].to_numpy(float), PROBLEM_ALPHA):
    raise SystemExit("Script-33c alpha lock mismatch.")

if not np.allclose(fit33c["l1_ratio"].to_numpy(float), PROBLEM_L1):
    raise SystemExit("Script-33c l1_ratio lock mismatch.")

failed_mask = (
    fit33c["convergence_warning"].astype(int).eq(1)
    | fit33c["hit_recovery_max_iter"].astype(int).eq(1)
)

n_failed_33c = int(failed_mask.sum())

if n_failed_33c != 113:
    raise SystemExit(
        f"Expected 113 Script-33c failed fits from audit, found {n_failed_33c}."
    )

# -----------------------------------------------------------------------------
# 2. Targeted rerun only for Script-33c failed fits
# -----------------------------------------------------------------------------

anchors = {
    "START": start_df,
    "END": end_df,
}

final_rows = []
rerun_count = 0
t0 = time.time()

# Create lookup of failed fits.
failed_keys = set()

for _, r in fit33c.loc[failed_mask].iterrows():
    failed_keys.add((
        int(r["repeat"]),
        int(r["outer_fold"]),
        str(r["anchor"]),
        str(r["feature_set"]),
        int(r["inner_fold"]),
    ))

# Keep all already-converged Script-33c fits exactly as-is.
for _, r in fit33c.loc[~failed_mask].iterrows():
    d = r.to_dict()

    final_rows.append({
        "repeat": int(d["repeat"]),
        "outer_fold": int(d["outer_fold"]),
        "anchor": str(d["anchor"]),
        "feature_set": str(d["feature_set"]),
        "inner_fold": int(d["inner_fold"]),
        "alpha": PROBLEM_ALPHA,
        "l1_ratio": PROBLEM_L1,

        "source": "33c_already_converged",
        "original_33c_mae_pb": float(d["mae_pb"]),
        "original_33c_n_iter": int(d["n_iter"]),
        "original_33c_convergence_warning": int(d["convergence_warning"]),

        "final_mae_pb": float(d["mae_pb"]),
        "final_convergence_warning": 0,
        "final_n_convergence_warnings": 0,
        "final_n_iter": int(d["n_iter"]),
        "final_hit_recovery_max_iter": 0,
        "recovery_max_iter": 300000,
        "warning_message": "",
    })

for repeat in range(1, OUTER_REPEATS + 1):
    outer_seed = BASE_SEED + (repeat - 1) * 10000
    outer_splits = shuffled_group_folds(
        start_df[GROUP],
        OUTER_FOLDS,
        outer_seed,
    )

    for outer_fold, (tr_idx, _) in enumerate(outer_splits, start=1):

        for anchor, df in anchors.items():

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            y_train = train_df[TARGET].astype(float).reset_index(drop=True)
            g_train = train_df[GROUP].astype(str).reset_index(drop=True)

            for fs_idx, fs_name in enumerate(FEATURE_SET_ORDER):

                cols = FEATURE_SETS[fs_name]

                numeric_cols, categorical_cols = split_column_types(
                    train_df,
                    cols,
                )

                inner_seed = (
                    outer_seed
                    + outer_fold * 1000
                    + 500
                    + fs_idx * 10
                    + 1
                )

                inner_splits = shuffled_group_folds(
                    g_train,
                    INNER_FOLDS,
                    inner_seed,
                )

                for inner_fold, (itr, iva) in enumerate(inner_splits, start=1):

                    key = (
                        repeat,
                        outer_fold,
                        anchor,
                        fs_name,
                        inner_fold,
                    )

                    if key not in failed_keys:
                        continue

                    old = fit33c[
                        (fit33c["repeat"] == repeat)
                        & (fit33c["outer_fold"] == outer_fold)
                        & (fit33c["anchor"].astype(str) == anchor)
                        & (fit33c["feature_set"].astype(str) == fs_name)
                        & (fit33c["inner_fold"] == inner_fold)
                    ]

                    if len(old) != 1:
                        raise RuntimeError(
                            f"Script-33c fit row mismatch for {key}: {len(old)}"
                        )

                    old_row = old.iloc[0]

                    pipe = make_pipe(
                        numeric_cols,
                        categorical_cols,
                    )

                    cap = fit_capture(
                        pipe,
                        train_df.iloc[itr][cols],
                        y_train.iloc[itr],
                    )

                    pred = pipe.predict(
                        train_df.iloc[iva][cols]
                    )

                    score = participant_balanced_mae(
                        y_train.iloc[iva],
                        pred,
                        g_train.iloc[iva],
                    )

                    final_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "feature_set": fs_name,
                        "inner_fold": inner_fold,
                        "alpha": PROBLEM_ALPHA,
                        "l1_ratio": PROBLEM_L1,

                        "source": "33d_targeted_rerun",
                        "original_33c_mae_pb": float(old_row["mae_pb"]),
                        "original_33c_n_iter": int(old_row["n_iter"]),
                        "original_33c_convergence_warning": int(old_row["convergence_warning"]),

                        "final_mae_pb": float(score),
                        "final_convergence_warning": int(cap["convergence_warning"]),
                        "final_n_convergence_warnings": int(cap["n_convergence_warnings"]),
                        "final_n_iter": int(cap["n_iter"]),
                        "final_hit_recovery_max_iter": int(cap["hit_recovery_max_iter"]),
                        "recovery_max_iter": RECOVERY_MAX_ITER,
                        "warning_message": cap["warning_message"],
                    })

                    rerun_count += 1

                    elapsed = time.time() - t0
                    mean_sec = elapsed / max(rerun_count, 1)
                    remaining = n_failed_33c - rerun_count
                    eta_min = remaining * mean_sec / 60.0

                    print(
                        f"Targeted recovery {rerun_count}/{n_failed_33c} | "
                        f"{anchor} {fs_name} R{repeat}F{outer_fold} I{inner_fold} | "
                        f"warning={cap['convergence_warning']} | "
                        f"n_iter={cap['n_iter']} | "
                        f"ETA {eta_min:.1f} min"
                    )

# -----------------------------------------------------------------------------
# 3. Reassemble all 400 candidate fits and recompute 100 task means
# -----------------------------------------------------------------------------

fit_final = pd.DataFrame(final_rows)

key_cols = [
    "repeat",
    "outer_fold",
    "anchor",
    "feature_set",
    "inner_fold",
]

if len(fit_final) != 400:
    raise RuntimeError(
        f"Expected 400 final fit rows, found {len(fit_final)}."
    )

if fit_final.duplicated(key_cols).any():
    dup = fit_final.loc[
        fit_final.duplicated(key_cols, keep=False),
        key_cols,
    ]
    raise RuntimeError(
        f"Duplicate final fit keys detected:\n{dup.head(20)}"
    )

if rerun_count != n_failed_33c:
    raise RuntimeError(
        f"Expected {n_failed_33c} targeted reruns, performed {rerun_count}."
    )

fit_final = fit_final.sort_values(key_cols).reset_index(drop=True)
fit_final.to_csv(FIT_OUT, index=False)

task_rows = []

task_key_cols = [
    "repeat",
    "outer_fold",
    "anchor",
    "feature_set",
]

for task_key, g in fit_final.groupby(task_key_cols, sort=False):

    repeat, outer_fold, anchor, feature_set = task_key

    if len(g) != INNER_FOLDS:
        raise RuntimeError(
            f"Expected 4 inner folds for {task_key}; found {len(g)}."
        )

    h = hyper[
        (hyper["repeat"] == int(repeat))
        & (hyper["outer_fold"] == int(outer_fold))
        & (hyper["anchor"].astype(str) == str(anchor))
        & (hyper["feature_set"].astype(str) == str(feature_set))
    ]

    if len(h) != 1:
        raise RuntimeError(
            f"Script-33 selected hyperparameter row mismatch for {task_key}."
        )

    selected_score = float(h.iloc[0]["inner_selected_mae_pb"])
    selected_params = json.loads(h.iloc[0]["selected_params_json"])

    candidate_score = float(g["final_mae_pb"].mean())
    margin = candidate_score - selected_score

    task_rows.append({
        "repeat": int(repeat),
        "outer_fold": int(outer_fold),
        "anchor": str(anchor),
        "feature_set": str(feature_set),

        "original_selected_params_json": json.dumps(
            selected_params,
            sort_keys=True,
        ),

        "original_selected_inner_mae_pb": selected_score,
        "fully_recovered_problem_candidate_inner_mae_pb": candidate_score,
        "recovered_minus_selected_mae": margin,

        "recovered_problem_candidate_beats_selected":
            int(candidate_score < selected_score - 1e-12),

        "n_inner_folds": int(len(g)),
        "n_targeted_33d_reruns":
            int((g["source"] == "33d_targeted_rerun").sum()),

        "final_convergence_warning_count":
            int(g["final_convergence_warning"].sum()),

        "final_hit_max_iter_count":
            int(g["final_hit_recovery_max_iter"].sum()),

        "max_final_n_iter":
            int(g["final_n_iter"].max()),
    })

task_df = pd.DataFrame(task_rows)

if len(task_df) != 100:
    raise RuntimeError(
        f"Expected 100 final task rows, found {len(task_df)}."
    )

task_df.to_csv(TASK_OUT, index=False)

# -----------------------------------------------------------------------------
# 4. Decision
# -----------------------------------------------------------------------------

remaining_warning_fits = int(
    fit_final["final_convergence_warning"].sum()
)

remaining_hitmax_fits = int(
    fit_final["final_hit_recovery_max_iter"].sum()
)

all_fits_converged = bool(
    remaining_warning_fits == 0
    and remaining_hitmax_fits == 0
)

n_beats = int(
    task_df[
        "recovered_problem_candidate_beats_selected"
    ].sum()
)

selection_stable = bool(
    all_fits_converged
    and n_beats == 0
)

min_margin = float(
    task_df["recovered_minus_selected_mae"].min()
)

median_margin = float(
    task_df["recovered_minus_selected_mae"].median()
)

max_final_n_iter = int(
    fit_final["final_n_iter"].max()
)

# -----------------------------------------------------------------------------
# 5. Audit
# -----------------------------------------------------------------------------

lines = []
add = lines.append

add("DATASET C — SCRIPT 33 MOMENT-1-BASE TARGETED CONVERGENCE RECOVERY AUDIT (33d)")
add("=" * 118)
add(
    "Purpose: retain the 287 already-converged Script-33c fits unchanged and "
    "re-run only the 113 remaining non-converged inner fits."
)
add(
    "No split, representation, baseline, target, metric, alpha, l1_ratio, "
    "preprocessing, tolerance, or model family is changed."
)
add(f"Problem candidate: alpha={PROBLEM_ALPHA}, l1_ratio={PROBLEM_L1}")
add(f"Script-33c failed fits targeted: {n_failed_33c}")
add(f"33d numerical max_iter ceiling for targeted fits: {RECOVERY_MAX_ITER}")
add("")

add("1. INPUT LOCKS")
add("-" * 118)
add(f"START mean768 SHA256: {start_sha}")
add(f"END mean768 SHA256: {end_sha}")
add(f"Script-33 hyperparameter SHA256: {hyper_sha}")
add(f"Script-33c fit rows loaded: {len(fit33c)}")
add(f"Script-33c task rows loaded: {len(task33c)}")
add("")

add("2. TARGETED RECOVERY")
add("-" * 118)
add(f"Script-33c already-converged fits retained unchanged: {400 - n_failed_33c}/400")
add(f"Script-33c failed fits re-run: {rerun_count}/{n_failed_33c}")
add(f"Remaining ConvergenceWarning fits after 33d: {remaining_warning_fits}/400")
add(f"Remaining fits hitting 33d max_iter: {remaining_hitmax_fits}/400")
add(f"Maximum final n_iter: {max_final_n_iter}")
add(f"ALL PROBLEM-CANDIDATE FITS NOW CONVERGED: {all_fits_converged}")
add("")

add("3. HYPERPARAMETER-SELECTION STABILITY")
add("-" * 118)
add(
    "Tasks where the now-recovered problem candidate beats the originally "
    f"selected Script-33 candidate: {n_beats}/100"
)
add(
    f"Minimum recovered-minus-selected inner MAE: {min_margin:.12f}"
)
add(
    f"Median recovered-minus-selected inner MAE: {median_margin:.12f}"
)
add(
    "SCRIPT-33 ELASTICNET SELECTION STABLE AFTER TARGETED RECOVERY: "
    f"{selection_stable}"
)
add("")

add("4. DECISION")
add("-" * 118)

if selection_stable:
    add(
        "The only problematic ElasticNet grid candidate is now fully converged "
        "and does not beat the originally selected candidate in any of the 100 "
        "ElasticNet outer tasks."
    )
    add(
        "The original Script-33 ElasticNet hyperparameter selection is therefore "
        "numerically secure; no Script-33 rerun is required."
    )
    add(
        "The MOMENT M6 scientific result may now be interpreted together with "
        "the already-PASS Script-33 technical benchmark."
    )

elif not all_fits_converged:
    add(
        "Do NOT lock the Script-33 ElasticNet scientific result yet: one or more "
        "targeted problem-candidate fits still did not converge."
    )
    add(
        "Any next correction must remain numerical and uniform for the same "
        "alpha=0.01, l1_ratio=0.1 candidate; do not change the scientific grid "
        "or chase result direction."
    )

elif n_beats > 0:
    add(
        "Do NOT lock Script-33 selection: at least one fully converged problem "
        "candidate outperformed the originally selected candidate."
    )
    add(
        "A uniform numerical correction of the affected tuning path would then "
        "be required before scientific interpretation."
    )

add("")
add("Outputs:")
add(f"  {FIT_OUT}")
add(f"  {TASK_OUT}")
add("=" * 118)

AUDIT_OUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("")
print("Created:")
print(" ", AUDIT_OUT)
print(" ", TASK_OUT)
print("")
print(f"ALL PROBLEM-CANDIDATE FITS NOW CONVERGED: {all_fits_converged}")
print(f"TASKS WHERE RECOVERED CANDIDATE BEATS SELECTED: {n_beats}/100")
print(
    "SCRIPT-33 ELASTICNET SELECTION STABLE AFTER TARGETED RECOVERY: "
    f"{selection_stable}"
)
print("")
print(
    "Upload dataset_C_moment1base_targeted_convergence_recovery_audit_33d.txt "
    "to ChatGPT."
)
