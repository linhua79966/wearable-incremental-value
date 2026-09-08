from pathlib import Path
import json
import time
import hashlib
import warnings
from itertools import product

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
# 18g_dataset_C_handcrafted_elasticnet_convergence_audit.py
#
# READ-ONLY NUMERICAL VALIDITY AUDIT FOR FINAL QC-CLEAN SCRIPT 18f
#
# Purpose:
#   Reproduce the ElasticNet path of Script 18f without changing any model,
#   hyperparameter, split, feature set, preprocessing, max_iter, or scientific
#   result.
#
# It checks:
#   1) all 12 ElasticNet candidates x 4 inner folds for every final 18f
#      ElasticNet outer task:
#          5 repeats x 5 outer folds x 2 anchors x 5 feature sets
#          = 250 tasks;
#      therefore 250 x 12 x 4 = 12,000 inner candidate fits;
#   2) the originally selected configuration on every full outer-training set
#      = 250 final refits;
#   3) ConvergenceWarning and n_iter_ for every fit;
#   4) exact reproduction of the originally selected hyperparameters;
#   5) exact reproduction of the stored selected inner MAE;
#   6) reproduction of the stored Script-18f outer-test predictions.
#
# IMPORTANT:
#   - NO hyperparameter is changed.
#   - NO larger max_iter is used.
#   - NO scientific output is overwritten.
#   - NO result is re-selected to chase a favorable direction.
#   - If non-selected candidates fail to converge, this audit only records
#     that fact; any later recovery must be a separate numerical audit.
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DERIVED = ROOT / "derived" / "dataset_C_phase1"
START_TABLE = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_TABLE = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

SRC18 = ROOT / "results" / "dataset_C_phase1_dual_anchor_nestedcv_qc"
HYPER_IN = SRC18 / "dataset_C_qc_dual_anchor_selected_hyperparameters.csv"
PRED_IN = SRC18 / "dataset_C_qc_dual_anchor_predictions.csv"

OUTDIR = SRC18 / "convergence_audit_18g"
OUTDIR.mkdir(parents=True, exist_ok=True)

AUDITDIR = ROOT / "audit"
AUDITDIR.mkdir(parents=True, exist_ok=True)

FIT_OUT = OUTDIR / "dataset_C_18f_elasticnet_convergence_fit_audit.csv"
CAND_OUT = OUTDIR / "dataset_C_18f_elasticnet_convergence_candidate_summary.csv"
TASK_OUT = OUTDIR / "dataset_C_18f_elasticnet_convergence_task_summary.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_18f_elasticnet_convergence_audit_18g.txt"

PARTIAL_FIT = OUTDIR / "_partial_18g_fit_audit.csv"
PARTIAL_CAND = OUTDIR / "_partial_18g_candidate_summary.csv"
PARTIAL_TASK = OUTDIR / "_partial_18g_task_summary.csv"

TARGET = "target_borg"
GROUP = "subject_num"
ROW_ID = "row_id"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BASE_SEED = 20260901
MAX_ITER = 30000

FEATURE_SET_ROLES = {
    "C0a_core_context": ["context_core"],
    "C0b_extended_context": ["context_core", "context_extended"],
    "C1_wearable_only": ["wearable_feature"],
    "C2a_core_plus_wearable": ["context_core", "wearable_feature"],
    "C2b_extended_plus_wearable": [
        "context_core", "context_extended", "wearable_feature"
    ],
}

ELASTIC_GRID = [
    {"alpha": a, "l1_ratio": l1}
    for a, l1 in product([0.01, 0.1, 1.0, 10.0], [0.1, 0.5, 0.9])
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


def make_elastic_pipeline(params, numeric_cols, categorical_cols):
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
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]),
            categorical_cols,
        ))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    model = ElasticNet(
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
        max_iter=MAX_ITER,
        random_state=BASE_SEED,
        selection="cyclic",
    )

    return Pipeline([
        ("pre", pre),
        ("var", VarianceThreshold(threshold=0.0)),
        ("model", model),
    ])


def fit_with_convergence_capture(pipe, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y)

    conv = [
        w for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]

    model = pipe.named_steps["model"]
    n_iter_raw = model.n_iter_
    n_iter = (
        int(n_iter_raw)
        if np.ndim(n_iter_raw) == 0
        else int(np.max(n_iter_raw))
    )

    return {
        "convergence_warning": int(len(conv) > 0),
        "n_convergence_warnings": int(len(conv)),
        "n_iter": n_iter,
        "hit_max_iter": int(n_iter >= MAX_ITER),
        "warning_message": " | ".join(str(w.message) for w in conv),
    }


def json_key(d):
    return json.dumps(d, sort_keys=True)


def read_partial(fp):
    if fp.exists() and fp.stat().st_size > 0:
        return pd.read_csv(fp)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 1. Locked input checks
# -----------------------------------------------------------------------------
required = [
    START_TABLE, END_TABLE, DICT, HYPER_IN, PRED_IN
]
for fp in required:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

start_sha = sha256_file(START_TABLE)
end_sha = sha256_file(END_TABLE)
dict_sha = sha256_file(DICT)
hyper_sha = sha256_file(HYPER_IN)
pred_sha = sha256_file(PRED_IN)

start_df = pd.read_csv(START_TABLE).sort_values(ROW_ID).reset_index(drop=True)
end_df = pd.read_csv(END_TABLE).sort_values(ROW_ID).reset_index(drop=True)
fd = pd.read_csv(DICT)

if len(start_df) != 1141 or len(end_df) != 1141:
    raise SystemExit(
        f"Expected 1141 rows per anchor, found "
        f"START={len(start_df)} END={len(end_df)}."
    )

if start_df[GROUP].nunique() != 34 or end_df[GROUP].nunique() != 34:
    raise SystemExit("Expected 34 participants in each anchor.")

if not np.array_equal(
    start_df[ROW_ID].astype(str).to_numpy(),
    end_df[ROW_ID].astype(str).to_numpy(),
):
    raise SystemExit("START/END row IDs differ.")

if not np.array_equal(
    start_df[GROUP].astype(str).to_numpy(),
    end_df[GROUP].astype(str).to_numpy(),
):
    raise SystemExit("START/END participant grouping differs.")

if not np.allclose(
    start_df[TARGET].to_numpy(float),
    end_df[TARGET].to_numpy(float),
    equal_nan=True,
):
    raise SystemExit("START/END targets differ.")

role_map = {}
for role in fd["role"].dropna().unique():
    role_map[role] = fd.loc[
        fd["role"] == role, "column"
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

FEATURE_SETS = {}
for fs_name, roles in FEATURE_SET_ROLES.items():
    cols = []
    for role in roles:
        cols.extend(role_map.get(role, []))

    cols = [
        c for c in dict.fromkeys(cols)
        if c in start_df.columns and c not in forbidden
    ]

    if not cols:
        raise SystemExit(f"No columns found for feature set {fs_name}.")

    FEATURE_SETS[fs_name] = cols

expected_counts = {
    "C0a_core_context": 5,
    "C0b_extended_context": 14,
    "C1_wearable_only": 396,
    "C2a_core_plus_wearable": 401,
    "C2b_extended_plus_wearable": 410,
}
for fs, n in expected_counts.items():
    if len(FEATURE_SETS[fs]) != n:
        raise SystemExit(
            f"Feature-count lock failed for {fs}: "
            f"expected {n}, found {len(FEATURE_SETS[fs])}"
        )

hyper = pd.read_csv(HYPER_IN)
hyper = hyper[hyper["model"].astype(str) == "ElasticNet"].copy()

TOTAL_TASKS = (
    OUTER_REPEATS
    * OUTER_FOLDS
    * 2
    * len(FEATURE_SETS)
)

if len(hyper) != TOTAL_TASKS:
    raise SystemExit(
        f"Expected {TOTAL_TASKS} Script-18f ElasticNet selected rows, "
        f"found {len(hyper)}."
    )

pred18 = pd.read_csv(PRED_IN)
pred18 = pred18[
    pred18["model"].astype(str) == "ElasticNet"
].copy()

# -----------------------------------------------------------------------------
# 2. Reconstruct the exact Script-18f outer splits
# -----------------------------------------------------------------------------
outer_split_map = {}

for repeat0 in range(OUTER_REPEATS):
    outer_seed = BASE_SEED + repeat0 * 10000
    splits = shuffled_group_folds(
        start_df[GROUP],
        OUTER_FOLDS,
        outer_seed,
    )

    for outer_fold, (tr_idx, te_idx) in enumerate(
        splits,
        start=1,
    ):
        outer_split_map[
            (repeat0 + 1, outer_fold)
        ] = (tr_idx, te_idx)

anchors = {
    "START": start_df,
    "END": end_df,
}

# -----------------------------------------------------------------------------
# 3. Resume
# -----------------------------------------------------------------------------
fit_partial = read_partial(PARTIAL_FIT)
cand_partial = read_partial(PARTIAL_CAND)
task_partial = read_partial(PARTIAL_TASK)

fit_rows = (
    fit_partial.to_dict("records")
    if len(fit_partial)
    else []
)
cand_rows = (
    cand_partial.to_dict("records")
    if len(cand_partial)
    else []
)
task_rows = (
    task_partial.to_dict("records")
    if len(task_partial)
    else []
)

completed = set()

if len(task_partial):
    for _, r in task_partial.iterrows():
        completed.add((
            int(r["repeat"]),
            int(r["outer_fold"]),
            str(r["anchor"]),
            str(r["feature_set"]),
        ))

print(
    f"Resume: {len(completed)}/{TOTAL_TASKS} "
    f"ElasticNet outer tasks already audited."
)

t0 = time.time()
fresh = 0

# -----------------------------------------------------------------------------
# 4. Full-grid convergence audit + original selected final refit
# -----------------------------------------------------------------------------
for repeat in range(1, OUTER_REPEATS + 1):
    outer_seed = BASE_SEED + (repeat - 1) * 10000

    for outer_fold in range(1, OUTER_FOLDS + 1):
        tr_idx, te_idx = outer_split_map[
            (repeat, outer_fold)
        ]

        for anchor, df in anchors.items():
            train_df = df.iloc[
                tr_idx
            ].reset_index(drop=True)

            test_df = df.iloc[
                te_idx
            ].reset_index(drop=True)

            y_train = train_df[
                TARGET
            ].astype(float).reset_index(drop=True)

            g_train = train_df[
                GROUP
            ].astype(str).reset_index(drop=True)

            for fs_idx, (fs_name, cols) in enumerate(
                FEATURE_SETS.items()
            ):
                task_key = (
                    repeat,
                    outer_fold,
                    anchor,
                    fs_name,
                )

                if task_key in completed:
                    continue

                h = hyper[
                    (hyper["repeat"] == repeat)
                    & (hyper["outer_fold"] == outer_fold)
                    & (
                        hyper["anchor"].astype(str)
                        == anchor
                    )
                    & (
                        hyper["feature_set"].astype(str)
                        == fs_name
                    )
                ]

                if len(h) != 1:
                    raise RuntimeError(
                        f"Selected-hyperparameter row mismatch: "
                        f"{task_key}; rows={len(h)}"
                    )

                stored_selected = json.loads(
                    h.iloc[0]["selected_params_json"]
                )
                stored_inner_mae = float(
                    h.iloc[0]["inner_selected_mae_pb"]
                )

                numeric_cols, categorical_cols = (
                    split_column_types(
                        train_df,
                        cols,
                    )
                )

                # Exact Script-18f seed formula:
                # outer_seed + outer_fold*1000 + fs_idx*10 + 1
                inner_seed = (
                    outer_seed
                    + outer_fold * 1000
                    + fs_idx * 10
                    + 1
                )

                inner_splits = shuffled_group_folds(
                    g_train,
                    INNER_FOLDS,
                    inner_seed,
                )

                candidate_scores = []
                candidate_warning_counts = {}
                candidate_hitmax_counts = {}
                candidate_max_niter = {}

                for params in ELASTIC_GRID:
                    pkey = json_key(params)

                    scores = []
                    warn_count = 0
                    hit_count = 0
                    max_niter = 0

                    for inner_fold, (itr, iva) in enumerate(
                        inner_splits,
                        start=1,
                    ):
                        pipe = make_elastic_pipeline(
                            params,
                            numeric_cols,
                            categorical_cols,
                        )

                        cap = fit_with_convergence_capture(
                            pipe,
                            train_df.iloc[
                                itr
                            ][cols],
                            y_train.iloc[itr],
                        )

                        pred = pipe.predict(
                            train_df.iloc[
                                iva
                            ][cols]
                        )

                        score = participant_balanced_mae(
                            y_train.iloc[iva],
                            pred,
                            g_train.iloc[iva],
                        )

                        scores.append(score)
                        warn_count += (
                            cap["convergence_warning"]
                        )
                        hit_count += (
                            cap["hit_max_iter"]
                        )
                        max_niter = max(
                            max_niter,
                            cap["n_iter"],
                        )

                        fit_rows.append({
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "anchor": anchor,
                            "feature_set": fs_name,
                            "stage": "inner_candidate",
                            "inner_fold": inner_fold,
                            "params_json": pkey,
                            "is_originally_selected_params": int(
                                pkey
                                == json_key(stored_selected)
                            ),
                            "mae_pb": score,
                            **cap,
                        })

                    mean_score = float(
                        np.mean(scores)
                    )

                    candidate_scores.append((
                        mean_score,
                        pkey,
                        params,
                    ))

                    candidate_warning_counts[
                        pkey
                    ] = warn_count

                    candidate_hitmax_counts[
                        pkey
                    ] = hit_count

                    candidate_max_niter[
                        pkey
                    ] = max_niter

                # Exact Script-18f minimization + deterministic
                # JSON tie-break.
                candidate_scores_sorted = sorted(
                    candidate_scores,
                    key=lambda x: (
                        x[0],
                        x[1],
                    ),
                )

                (
                    reproduced_score,
                    reproduced_key,
                    reproduced_params,
                ) = candidate_scores_sorted[0]

                for (
                    mean_score,
                    pkey,
                    params,
                ) in candidate_scores:
                    cand_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "anchor": anchor,
                        "feature_set": fs_name,
                        "params_json": pkey,
                        "mean_inner_mae_pb": mean_score,
                        "n_inner_convergence_warnings":
                            candidate_warning_counts[pkey],
                        "n_inner_hit_max_iter":
                            candidate_hitmax_counts[pkey],
                        "max_inner_n_iter":
                            candidate_max_niter[pkey],
                        "is_originally_selected": int(
                            pkey
                            == json_key(stored_selected)
                        ),
                        "is_reproduced_best": int(
                            pkey
                            == reproduced_key
                        ),
                    })

                # Refit ONLY the original stored selected params on
                # the full outer-training data.
                final_pipe = make_elastic_pipeline(
                    stored_selected,
                    numeric_cols,
                    categorical_cols,
                )

                final_cap = fit_with_convergence_capture(
                    final_pipe,
                    train_df[cols],
                    y_train,
                )

                final_pred = final_pipe.predict(
                    test_df[cols]
                )

                q = pred18[
                    (pred18["repeat"] == repeat)
                    & (
                        pred18["outer_fold"]
                        == outer_fold
                    )
                    & (
                        pred18["anchor"].astype(str)
                        == anchor
                    )
                    & (
                        pred18["feature_set"].astype(str)
                        == fs_name
                    )
                ][
                    [ROW_ID, "y_pred"]
                ].copy()

                z = test_df[
                    [ROW_ID]
                ].copy()

                z["y_pred_reproduced"] = (
                    final_pred
                )

                m = z.merge(
                    q,
                    on=ROW_ID,
                    how="inner",
                    validate="one_to_one",
                )

                if len(m) != len(test_df):
                    raise RuntimeError(
                        f"Prediction row mismatch: "
                        f"{task_key}; "
                        f"stored={len(m)} "
                        f"expected={len(test_df)}"
                    )

                max_pred_diff = float(
                    np.max(np.abs(
                        m["y_pred_reproduced"]
                        .to_numpy(float)
                        - m["y_pred"]
                        .to_numpy(float)
                    ))
                )

                fit_rows.append({
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "anchor": anchor,
                    "feature_set": fs_name,
                    "stage": "outer_final_selected",
                    "inner_fold": 0,
                    "params_json": json_key(
                        stored_selected
                    ),
                    "is_originally_selected_params": 1,
                    "mae_pb": np.nan,
                    **final_cap,
                })

                selected_key = json_key(
                    stored_selected
                )

                selected_inner_warn = int(
                    candidate_warning_counts[
                        selected_key
                    ]
                )

                selected_inner_hitmax = int(
                    candidate_hitmax_counts[
                        selected_key
                    ]
                )

                selected_max_niter = int(
                    candidate_max_niter[
                        selected_key
                    ]
                )

                task_rows.append({
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "anchor": anchor,
                    "feature_set": fs_name,
                    "stored_selected_params_json":
                        selected_key,
                    "reproduced_best_params_json":
                        reproduced_key,
                    "selected_params_match": int(
                        selected_key
                        == reproduced_key
                    ),
                    "stored_inner_selected_mae_pb":
                        stored_inner_mae,
                    "reproduced_inner_selected_mae_pb":
                        reproduced_score,
                    "inner_mae_abs_diff": abs(
                        stored_inner_mae
                        - reproduced_score
                    ),
                    "selected_inner_convergence_warning_count":
                        selected_inner_warn,
                    "selected_inner_hit_max_iter_count":
                        selected_inner_hitmax,
                    "selected_inner_max_n_iter":
                        selected_max_niter,
                    "outer_final_convergence_warning":
                        final_cap[
                            "convergence_warning"
                        ],
                    "outer_final_hit_max_iter":
                        final_cap[
                            "hit_max_iter"
                        ],
                    "outer_final_n_iter":
                        final_cap[
                            "n_iter"
                        ],
                    "max_abs_prediction_diff_vs_script18f":
                        max_pred_diff,
                    "any_grid_inner_convergence_warning":
                        int(
                            sum(
                                candidate_warning_counts
                                .values()
                            ) > 0
                        ),
                    "n_grid_inner_convergence_warnings":
                        int(
                            sum(
                                candidate_warning_counts
                                .values()
                            )
                        ),
                })

                completed.add(task_key)
                fresh += 1

                # Resume checkpoints after each outer task.
                pd.DataFrame(
                    fit_rows
                ).to_csv(
                    PARTIAL_FIT,
                    index=False,
                )

                pd.DataFrame(
                    cand_rows
                ).to_csv(
                    PARTIAL_CAND,
                    index=False,
                )

                pd.DataFrame(
                    task_rows
                ).to_csv(
                    PARTIAL_TASK,
                    index=False,
                )

                elapsed = time.time() - t0
                avg_sec = (
                    elapsed / max(fresh, 1)
                )
                remain = (
                    TOTAL_TASKS
                    - len(completed)
                )
                eta = (
                    remain
                    * avg_sec
                    / 60.0
                )

                print(
                    f"Audited "
                    f"{len(completed)}/"
                    f"{TOTAL_TASKS} | "
                    f"{anchor} {fs_name} "
                    f"R{repeat}F{outer_fold} | "
                    f"grid warnings="
                    f"{sum(candidate_warning_counts.values())} | "
                    f"selected-inner warnings="
                    f"{selected_inner_warn} | "
                    f"final warning="
                    f"{final_cap['convergence_warning']} | "
                    f"ETA {eta:.1f} min"
                )

# -----------------------------------------------------------------------------
# 5. Final outputs and decision gate
# -----------------------------------------------------------------------------
fit_df = pd.DataFrame(fit_rows)
cand_df = pd.DataFrame(cand_rows)
task_df = pd.DataFrame(task_rows)

if len(task_df) != TOTAL_TASKS:
    raise RuntimeError(
        f"Expected {TOTAL_TASKS} task rows, "
        f"found {len(task_df)}."
    )

if len(cand_df) != (
    TOTAL_TASKS * len(ELASTIC_GRID)
):
    raise RuntimeError(
        f"Expected "
        f"{TOTAL_TASKS * len(ELASTIC_GRID)} "
        f"candidate rows, "
        f"found {len(cand_df)}."
    )

expected_fit = TOTAL_TASKS * (
    len(ELASTIC_GRID)
    * INNER_FOLDS
    + 1
)

if len(fit_df) != expected_fit:
    raise RuntimeError(
        f"Expected {expected_fit} fit rows, "
        f"found {len(fit_df)}."
    )

fit_df.to_csv(
    FIT_OUT,
    index=False,
)

cand_df.to_csv(
    CAND_OUT,
    index=False,
)

task_df.to_csv(
    TASK_OUT,
    index=False,
)

selected_path_pass = bool(
    (
        task_df[
            "selected_inner_convergence_warning_count"
        ] == 0
    ).all()
    and (
        task_df[
            "outer_final_convergence_warning"
        ] == 0
    ).all()
    and (
        task_df[
            "selected_inner_hit_max_iter_count"
        ] == 0
    ).all()
    and (
        task_df[
            "outer_final_hit_max_iter"
        ] == 0
    ).all()
)

full_grid_pass = bool(
    (
        fit_df[
            "convergence_warning"
        ] == 0
    ).all()
    and (
        fit_df[
            "hit_max_iter"
        ] == 0
    ).all()
)

reproduction_pass = bool(
    (
        task_df[
            "selected_params_match"
        ] == 1
    ).all()
    and (
        task_df[
            "inner_mae_abs_diff"
        ] <= 1e-10
    ).all()
    and (
        task_df[
            "max_abs_prediction_diff_vs_script18f"
        ] <= 1e-10
    ).all()
)

by_param = (
    fit_df[
        fit_df["stage"]
        == "inner_candidate"
    ]
    .groupby(
        "params_json",
        as_index=False,
    )
    .agg(
        n_fits=("params_json", "size"),
        n_convergence_warnings=(
            "convergence_warning",
            "sum",
        ),
        n_hit_max_iter=(
            "hit_max_iter",
            "sum",
        ),
        max_n_iter=("n_iter", "max"),
    )
    .sort_values(
        [
            "n_convergence_warnings",
            "n_hit_max_iter",
        ],
        ascending=False,
    )
)

by_feature = (
    task_df
    .groupby(
        [
            "anchor",
            "feature_set",
        ],
        as_index=False,
    )
    .agg(
        n_tasks=("feature_set", "size"),
        selected_inner_warning_tasks=(
            "selected_inner_convergence_warning_count",
            lambda x: int((x > 0).sum()),
        ),
        final_warning_tasks=(
            "outer_final_convergence_warning",
            "sum",
        ),
        any_grid_warning_tasks=(
            "any_grid_inner_convergence_warning",
            "sum",
        ),
        max_selected_inner_n_iter=(
            "selected_inner_max_n_iter",
            "max",
        ),
        max_outer_final_n_iter=(
            "outer_final_n_iter",
            "max",
        ),
        max_inner_mae_abs_diff=(
            "inner_mae_abs_diff",
            "max",
        ),
        max_prediction_abs_diff=(
            "max_abs_prediction_diff_vs_script18f",
            "max",
        ),
    )
)

summary = pd.DataFrame({
    "metric": [
        "tasks",
        "tasks_selected_inner_any_warning",
        "tasks_outer_final_warning",
        "tasks_any_grid_warning",
        "max_inner_mae_abs_diff",
        "max_prediction_abs_diff",
    ],
    "value": [
        len(task_df),
        int((
            task_df[
                "selected_inner_convergence_warning_count"
            ] > 0
        ).sum()),
        int(
            task_df[
                "outer_final_convergence_warning"
            ].sum()
        ),
        int((
            task_df[
                "any_grid_inner_convergence_warning"
            ] > 0
        ).sum()),
        float(
            task_df[
                "inner_mae_abs_diff"
            ].max()
        ),
        float(
            task_df[
                "max_abs_prediction_diff_vs_script18f"
            ].max()
        ),
    ],
})

lines = []
add = lines.append

add(
    "DATASET C — FINAL 18f HANDCRAFTED "
    "ELASTICNET CONVERGENCE AUDIT (18g)"
)
add("=" * 118)
add(
    "Purpose: read-only numerical reproduction audit. "
    "No model, grid, split, feature set, preprocessing, "
    "max_iter, or scientific result was changed."
)
add(
    f"ElasticNet max_iter reproduced exactly: "
    f"{MAX_ITER}"
)
add(
    f"Audited outer ElasticNet tasks: "
    f"{len(task_df)}"
)
add(
    f"Inner candidate fits: "
    f"{TOTAL_TASKS * len(ELASTIC_GRID) * INNER_FOLDS}"
)
add(
    f"Outer final selected refits: "
    f"{TOTAL_TASKS}"
)
add("")

add("1. INPUT LOCKS")
add("-" * 118)
add(f"START QC table SHA256: {start_sha}")
add(f"END QC table SHA256: {end_sha}")
add(f"Feature dictionary SHA256: {dict_sha}")
add(f"Script-18f hyperparameter CSV SHA256: {hyper_sha}")
add(f"Script-18f prediction CSV SHA256: {pred_sha}")
add(
    f"Rows per anchor: START={len(start_df)}, "
    f"END={len(end_df)}"
)
add(
    f"Participants: START={start_df[GROUP].nunique()}, "
    f"END={end_df[GROUP].nunique()}"
)
add("START/END identical row IDs: True")
add("START/END identical participant sequence: True")
add("START/END identical targets: True")
add(
    "Feature counts: "
    + ", ".join(
        f"{k}={len(v)}"
        for k, v in FEATURE_SETS.items()
    )
)
add("")

add("2. REPRODUCTION")
add("-" * 118)
add(summary.to_string(index=False))
add(
    f"REPRODUCTION PASS: "
    f"{reproduction_pass}"
)
add("")

add("3. SELECTED-PATH CONVERGENCE")
add("-" * 118)
add(
    "Selected path = four inner fits of the originally "
    "selected params plus the final full outer-training refit."
)
add(
    "Tasks with >=1 selected-inner ConvergenceWarning: "
    f"{int((task_df['selected_inner_convergence_warning_count'] > 0).sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    "Tasks with final outer-refit ConvergenceWarning: "
    f"{int(task_df['outer_final_convergence_warning'].sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    "Tasks with selected-inner fit hitting max_iter: "
    f"{int((task_df['selected_inner_hit_max_iter_count'] > 0).sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    "Tasks with final refit hitting max_iter: "
    f"{int(task_df['outer_final_hit_max_iter'].sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    f"SELECTED-PATH CONVERGENCE PASS: "
    f"{selected_path_pass}"
)
add("")

add("4. FULL-GRID CONVERGENCE")
add("-" * 118)
add(
    "All-grid inner ConvergenceWarning fits: "
    f"{int(fit_df.loc[fit_df['stage']=='inner_candidate', 'convergence_warning'].sum())}"
    f"/{TOTAL_TASKS * len(ELASTIC_GRID) * INNER_FOLDS}"
)
add(
    "Tasks with at least one grid warning: "
    f"{int((task_df['any_grid_inner_convergence_warning'] > 0).sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    f"FULL-GRID CONVERGENCE PASS: "
    f"{full_grid_pass}"
)
add("")

add("5. WARNINGS BY HYPERPARAMETER")
add("-" * 118)
add(by_param.to_string(index=False))
add("")

add("6. ANCHOR / FEATURE-SET SUMMARY")
add("-" * 118)
add(by_feature.to_string(index=False))
add("")

add("7. DECISION")
add("-" * 118)

if (
    reproduction_pass
    and selected_path_pass
    and full_grid_pass
):
    add(
        "All final Script-18f ElasticNet paths and all "
        "grid candidates reproduced and converged under "
        "the original max_iter=30000."
    )
    add(
        "The 18f ElasticNet numerical-convergence gate is "
        "fully PASS. No recovery audit and no scientific "
        "rerun are required."
    )

elif (
    reproduction_pass
    and selected_path_pass
    and not full_grid_pass
):
    add(
        "The originally selected Script-18f ElasticNet "
        "paths reproduced and converged, but one or more "
        "non-selected grid candidates did not."
    )
    add(
        "Do NOT change the 18f scientific benchmark. "
        "Review the warning-by-hyperparameter table first."
    )
    add(
        "If the warnings are concentrated in one or a small "
        "number of candidates, perform a separate targeted "
        "numerical-recovery audit using the exact same "
        "alpha/l1_ratio, folds, preprocessing, and objective, "
        "increasing only the numerical iteration ceiling."
    )

else:
    add(
        "One or more originally selected Script-18f "
        "ElasticNet paths failed the convergence and/or "
        "exact-reproduction gate."
    )
    add(
        "Do NOT lock the Dataset C ElasticNet handcrafted "
        "result until the numerical issue is resolved."
    )
    add(
        "Do NOT tune alpha/l1_ratio or alter the split/grid "
        "to chase a result; any correction must be a "
        "numerical-validity correction only."
    )

add("")
add("Outputs:")
add(f"  {FIT_OUT}")
add(f"  {CAND_OUT}")
add(f"  {TASK_OUT}")
add(f"  {AUDIT_OUT}")
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
print(
    f"REPRODUCTION PASS: "
    f"{reproduction_pass}"
)
print(
    f"SELECTED-PATH CONVERGENCE PASS: "
    f"{selected_path_pass}"
)
print(
    f"FULL-GRID CONVERGENCE PASS: "
    f"{full_grid_pass}"
)
