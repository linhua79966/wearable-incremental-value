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
# 07b_dataset_A_direct_elasticnet_convergence_audit.py
#
# READ-ONLY NUMERICAL VALIDITY AUDIT FOR SCRIPT 07
#
# Purpose:
#   Reproduce the ElasticNet path of the locked Dataset-A direct benchmark
#   without changing any model, grid, split, feature set, preprocessing,
#   max_iter, or scientific result.
#
# Checks:
#   - 5 repeats x 5 outer folds x 5 feature sets = 125 ElasticNet tasks
#   - 125 x 12 candidates x 4 inner folds = 6,000 inner fits
#   - 125 final outer-training selected refits
#   - ConvergenceWarning and n_iter_ for every fit
#   - exact selected-parameter reproduction
#   - exact selected inner participant-balanced MAE reproduction
#   - reproduction of stored Script-07 outer-test predictions
#
# IMPORTANT:
#   This script is diagnostic only. It does not overwrite scientific outputs,
#   increase max_iter, retune the grid, or select a result by outcome direction.
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DERIVED = ROOT / "derived" / "dataset_A_phase1"
TABLE = DERIVED / "dataset_A_phase1_modeling_table.csv"
DICT = DERIVED / "dataset_A_phase1_feature_dictionary.csv"

SRC07 = ROOT / "results" / "dataset_A_phase1_nestedcv"
HYPER_IN = SRC07 / "dataset_A_nestedcv_selected_hyperparameters.csv"
PRED_IN = SRC07 / "dataset_A_nestedcv_predictions.csv"

OUTDIR = SRC07 / "convergence_audit_07b"
OUTDIR.mkdir(parents=True, exist_ok=True)

AUDITDIR = ROOT / "audit"
AUDITDIR.mkdir(parents=True, exist_ok=True)

FIT_OUT = OUTDIR / "dataset_A_07_elasticnet_convergence_fit_audit.csv"
CAND_OUT = OUTDIR / "dataset_A_07_elasticnet_convergence_candidate_summary.csv"
TASK_OUT = OUTDIR / "dataset_A_07_elasticnet_convergence_task_summary.csv"
AUDIT_OUT = AUDITDIR / "dataset_A_07_elasticnet_convergence_audit_07b.txt"

PARTIAL_FIT = OUTDIR / "_partial_07b_fit_audit.csv"
PARTIAL_CAND = OUTDIR / "_partial_07b_candidate_summary.csv"
PARTIAL_TASK = OUTDIR / "_partial_07b_task_summary.csv"

TARGET = "physical_fatigue_final"
GROUP = "participant_id"
ROW_ID = "source_file"

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
BASE_SEED = 20260831
MAX_ITER = 30000

FEATURE_SET_ROLES = {
    "M0a_core_context": ["context_core"],
    "M0b_extended_context": ["context_core", "context_extended"],
    "M1_wearable_only": ["wearable_feature"],
    "M2a_core_plus_wearable": ["context_core", "wearable_feature"],
    "M2b_extended_plus_wearable": [
        "context_core", "context_extended", "wearable_feature"
    ],
}

EXPECTED_FEATURE_COUNTS = {
    "M0a_core_context": 5,
    "M0b_extended_context": 9,
    "M1_wearable_only": 336,
    "M2a_core_plus_wearable": 341,
    "M2b_extended_plus_wearable": 345,
}

ELASTIC_GRID = [
    {"alpha": a, "l1_ratio": l1}
    for a, l1 in product(
        [0.01, 0.1, 1.0, 10.0],
        [0.1, 0.5, 0.9],
    )
]

CATEGORICAL_CANDIDATES = {"task", "gender"}


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
    return float(
        d.groupby("g", sort=False)["ae"]
        .mean()
        .mean()
    )


def shuffled_group_folds(groups, n_splits, seed):
    groups = (
        pd.Series(groups)
        .astype(str)
        .reset_index(drop=True)
    )
    unique = groups.unique().tolist()

    if len(unique) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups; "
            f"found {len(unique)}."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    sizes = groups.value_counts().to_dict()
    fold_groups = [
        set() for _ in range(n_splits)
    ]
    fold_sizes = [0] * n_splits

    for g in unique:
        j = int(np.argmin(fold_sizes))
        fold_groups[j].add(g)
        fold_sizes[j] += int(sizes[g])

    all_idx = np.arange(len(groups))
    splits = []

    for gs in fold_groups:
        mask = groups.isin(gs).to_numpy()
        splits.append((
            all_idx[~mask],
            all_idx[mask],
        ))

    return splits


def make_elastic_pipeline(
    params,
    numeric_cols,
    categorical_cols,
):
    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
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
        (
            "var",
            VarianceThreshold(threshold=0.0),
        ),
        ("model", model),
    ])


def fit_capture(pipe, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )
        pipe.fit(X, y)

    conv = [
        w for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    n_iter_raw = pipe.named_steps[
        "model"
    ].n_iter_

    n_iter = (
        int(n_iter_raw)
        if np.ndim(n_iter_raw) == 0
        else int(np.max(n_iter_raw))
    )

    return {
        "convergence_warning":
            int(len(conv) > 0),
        "n_convergence_warnings":
            int(len(conv)),
        "n_iter": n_iter,
        "hit_max_iter":
            int(n_iter >= MAX_ITER),
        "warning_message":
            " | ".join(
                str(w.message) for w in conv
            ),
    }


def json_key(d):
    return json.dumps(d, sort_keys=True)


def read_partial(fp):
    if fp.exists() and fp.stat().st_size > 0:
        return pd.read_csv(fp)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 1. Locked inputs
# -----------------------------------------------------------------------------
required = [
    TABLE,
    DICT,
    HYPER_IN,
    PRED_IN,
]

for fp in required:
    if not fp.exists():
        raise SystemExit(
            f"Missing required input: {fp}"
        )

table_sha = sha256_file(TABLE)
dict_sha = sha256_file(DICT)
hyper_sha = sha256_file(HYPER_IN)
pred_sha = sha256_file(PRED_IN)

df = pd.read_csv(TABLE)
fd = pd.read_csv(DICT)

if len(df) != 372:
    raise SystemExit(
        f"Expected 372 Dataset-A repetitions; "
        f"found {len(df)}."
    )

if df[GROUP].nunique() != 43:
    raise SystemExit(
        f"Expected 43 participant labels; "
        f"found {df[GROUP].nunique()}."
    )

if df.columns.duplicated().any():
    raise SystemExit(
        "Duplicate columns in modeling table."
    )

if df[ROW_ID].duplicated().any():
    raise SystemExit(
        "source_file is not unique."
    )

if df[TARGET].isna().any():
    raise SystemExit(
        "Target contains missing values."
    )

role_map = {
    role: fd.loc[
        fd["role"] == role,
        "column",
    ].tolist()
    for role in
    fd["role"].dropna().unique()
}

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

feature_columns = {}

for fs_name, roles in FEATURE_SET_ROLES.items():
    cols = []

    for role in roles:
        cols.extend(
            role_map.get(role, [])
        )

    cols = [
        c for c in dict.fromkeys(cols)
        if c in df.columns
    ]

    cols = [
        c for c in cols
        if c not in forbidden
        and not c.startswith("qc__")
    ]

    if not cols:
        raise SystemExit(
            f"No columns found for {fs_name}."
        )

    feature_columns[fs_name] = cols

for fs_name, expected in (
    EXPECTED_FEATURE_COUNTS.items()
):
    observed = len(
        feature_columns[fs_name]
    )

    if observed != expected:
        raise SystemExit(
            f"Feature-count lock failed for "
            f"{fs_name}: expected={expected}, "
            f"found={observed}"
        )

hyper = pd.read_csv(HYPER_IN)
hyper = hyper[
    hyper["model"].astype(str)
    == "ElasticNet"
].copy()

TOTAL_TASKS = (
    OUTER_REPEATS
    * OUTER_FOLDS
    * len(feature_columns)
)

if len(hyper) != TOTAL_TASKS:
    raise SystemExit(
        f"Expected {TOTAL_TASKS} "
        f"stored ElasticNet hyper rows; "
        f"found {len(hyper)}."
    )

pred07 = pd.read_csv(PRED_IN)
pred07 = pred07[
    pred07["model"].astype(str)
    == "ElasticNet"
].copy()

# -----------------------------------------------------------------------------
# 2. Exact outer splits
# -----------------------------------------------------------------------------
outer_map = {}

for repeat0 in range(OUTER_REPEATS):
    outer_seed = (
        BASE_SEED
        + repeat0 * 1000
    )

    splits = shuffled_group_folds(
        df[GROUP],
        OUTER_FOLDS,
        outer_seed,
    )

    for outer_fold, (
        tr_idx,
        te_idx,
    ) in enumerate(
        splits,
        start=1,
    ):
        outer_map[
            (
                repeat0 + 1,
                outer_fold,
            )
        ] = (
            tr_idx,
            te_idx,
        )

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
            str(r["feature_set"]),
        ))

print(
    f"Resume: {len(completed)}/"
    f"{TOTAL_TASKS} "
    f"ElasticNet outer tasks "
    f"already audited."
)

t0 = time.time()
fresh = 0

# -----------------------------------------------------------------------------
# 4. Exact full-grid audit
# -----------------------------------------------------------------------------
for repeat in range(
    1,
    OUTER_REPEATS + 1,
):
    outer_seed = (
        BASE_SEED
        + (repeat - 1) * 1000
    )

    for outer_fold in range(
        1,
        OUTER_FOLDS + 1,
    ):
        tr_idx, te_idx = outer_map[
            (repeat, outer_fold)
        ]

        train = (
            df.iloc[tr_idx]
            .reset_index(drop=True)
        )

        test = (
            df.iloc[te_idx]
            .reset_index(drop=True)
        )

        y_train = (
            train[TARGET]
            .astype(float)
            .reset_index(drop=True)
        )

        g_train = (
            train[GROUP]
            .astype(str)
            .reset_index(drop=True)
        )

        for fs_name, cols in (
            feature_columns.items()
        ):
            task_key = (
                repeat,
                outer_fold,
                fs_name,
            )

            if task_key in completed:
                continue

            h = hyper[
                (hyper["repeat"] == repeat)
                & (
                    hyper["outer_fold"]
                    == outer_fold
                )
                & (
                    hyper["feature_set"]
                    .astype(str)
                    == fs_name
                )
            ]

            if len(h) != 1:
                raise RuntimeError(
                    f"Stored hyper row mismatch "
                    f"for {task_key}: "
                    f"{len(h)} rows"
                )

            stored_selected = json.loads(
                h.iloc[0][
                    "selected_params_json"
                ]
            )

            stored_inner_mae = float(
                h.iloc[0][
                    "inner_selected_mae_pb"
                ]
            )

            cat_cols = [
                c for c in cols
                if c in
                CATEGORICAL_CANDIDATES
            ]

            num_cols = [
                c for c in cols
                if c not in
                CATEGORICAL_CANDIDATES
            ]

            X_train = (
                train[cols]
                .reset_index(drop=True)
            )

            X_test = (
                test[cols]
                .reset_index(drop=True)
            )

            # Exact Script-07 seed.
            inner_seed = (
                outer_seed
                + outer_fold * 100
                + 1
            )

            inner_splits = (
                shuffled_group_folds(
                    g_train,
                    INNER_FOLDS,
                    inner_seed,
                )
            )

            candidate_scores = []
            warning_counts = {}
            hitmax_counts = {}
            max_niters = {}

            for params in ELASTIC_GRID:
                pkey = json_key(params)

                fold_scores = []
                warn_count = 0
                hit_count = 0
                max_niter = 0

                for inner_fold, (
                    itr,
                    iva,
                ) in enumerate(
                    inner_splits,
                    start=1,
                ):
                    pipe = (
                        make_elastic_pipeline(
                            params,
                            num_cols,
                            cat_cols,
                        )
                    )

                    cap = fit_capture(
                        pipe,
                        X_train.iloc[itr],
                        y_train.iloc[itr],
                    )

                    pred = pipe.predict(
                        X_train.iloc[iva]
                    )

                    score = (
                        participant_balanced_mae(
                            y_train.iloc[iva],
                            pred,
                            g_train.iloc[iva],
                        )
                    )

                    fold_scores.append(
                        score
                    )

                    warn_count += (
                        cap[
                            "convergence_warning"
                        ]
                    )

                    hit_count += (
                        cap[
                            "hit_max_iter"
                        ]
                    )

                    max_niter = max(
                        max_niter,
                        cap["n_iter"],
                    )

                    fit_rows.append({
                        "repeat": repeat,
                        "outer_fold":
                            outer_fold,
                        "feature_set":
                            fs_name,
                        "stage":
                            "inner_candidate",
                        "inner_fold":
                            inner_fold,
                        "params_json":
                            pkey,
                        "is_originally_selected_params":
                            int(
                                pkey
                                == json_key(
                                    stored_selected
                                )
                            ),
                        "mae_pb":
                            score,
                        **cap,
                    })

                mean_score = float(
                    np.mean(fold_scores)
                )

                candidate_scores.append(
                    (
                        mean_score,
                        pkey,
                        params,
                    )
                )

                warning_counts[
                    pkey
                ] = warn_count

                hitmax_counts[
                    pkey
                ] = hit_count

                max_niters[
                    pkey
                ] = max_niter

            candidate_scores_sorted = (
                sorted(
                    candidate_scores,
                    key=lambda x: (
                        x[0],
                        x[1],
                    ),
                )
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
                    "outer_fold":
                        outer_fold,
                    "feature_set":
                        fs_name,
                    "params_json":
                        pkey,
                    "mean_inner_mae_pb":
                        mean_score,
                    "n_inner_convergence_warnings":
                        warning_counts[
                            pkey
                        ],
                    "n_inner_hit_max_iter":
                        hitmax_counts[
                            pkey
                        ],
                    "max_inner_n_iter":
                        max_niters[
                            pkey
                        ],
                    "is_originally_selected":
                        int(
                            pkey
                            == json_key(
                                stored_selected
                            )
                        ),
                    "is_reproduced_best":
                        int(
                            pkey
                            == reproduced_key
                        ),
                })

            final_pipe = (
                make_elastic_pipeline(
                    stored_selected,
                    num_cols,
                    cat_cols,
                )
            )

            final_cap = fit_capture(
                final_pipe,
                X_train,
                y_train,
            )

            reproduced_pred = (
                final_pipe.predict(
                    X_test
                )
            )

            q = pred07[
                (pred07["repeat"] == repeat)
                & (
                    pred07["outer_fold"]
                    == outer_fold
                )
                & (
                    pred07["feature_set"]
                    .astype(str)
                    == fs_name
                )
            ][
                [
                    ROW_ID,
                    "y_pred",
                ]
            ].copy()

            z = test[
                [ROW_ID]
            ].copy()

            z[
                "y_pred_reproduced"
            ] = reproduced_pred

            m = z.merge(
                q,
                on=ROW_ID,
                how="inner",
                validate="one_to_one",
            )

            if len(m) != len(test):
                raise RuntimeError(
                    f"Prediction-row mismatch "
                    f"for {task_key}: "
                    f"stored={len(m)}, "
                    f"expected={len(test)}"
                )

            max_pred_diff = float(
                np.max(
                    np.abs(
                        m[
                            "y_pred_reproduced"
                        ].to_numpy(float)
                        -
                        m[
                            "y_pred"
                        ].to_numpy(float)
                    )
                )
            )

            fit_rows.append({
                "repeat": repeat,
                "outer_fold":
                    outer_fold,
                "feature_set":
                    fs_name,
                "stage":
                    "outer_final_selected",
                "inner_fold": 0,
                "params_json":
                    json_key(
                        stored_selected
                    ),
                "is_originally_selected_params":
                    1,
                "mae_pb": np.nan,
                **final_cap,
            })

            selected_key = (
                json_key(
                    stored_selected
                )
            )

            task_rows.append({
                "repeat": repeat,
                "outer_fold":
                    outer_fold,
                "feature_set":
                    fs_name,
                "stored_selected_params_json":
                    selected_key,
                "reproduced_best_params_json":
                    reproduced_key,
                "selected_params_match":
                    int(
                        selected_key
                        == reproduced_key
                    ),
                "stored_inner_selected_mae_pb":
                    stored_inner_mae,
                "reproduced_inner_selected_mae_pb":
                    reproduced_score,
                "inner_mae_abs_diff":
                    abs(
                        stored_inner_mae
                        - reproduced_score
                    ),
                "selected_inner_convergence_warning_count":
                    int(
                        warning_counts[
                            selected_key
                        ]
                    ),
                "selected_inner_hit_max_iter_count":
                    int(
                        hitmax_counts[
                            selected_key
                        ]
                    ),
                "selected_inner_max_n_iter":
                    int(
                        max_niters[
                            selected_key
                        ]
                    ),
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
                "max_abs_prediction_diff_vs_script07":
                    max_pred_diff,
                "any_grid_inner_convergence_warning":
                    int(
                        sum(
                            warning_counts
                            .values()
                        ) > 0
                    ),
                "n_grid_inner_convergence_warnings":
                    int(
                        sum(
                            warning_counts
                            .values()
                        )
                    ),
            })

            completed.add(task_key)
            fresh += 1

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

            elapsed = (
                time.time()
                - t0
            )

            avg_sec = (
                elapsed
                / max(fresh, 1)
            )

            remaining = (
                TOTAL_TASKS
                - len(completed)
            )

            eta_min = (
                remaining
                * avg_sec
                / 60.0
            )

            print(
                f"Audited "
                f"{len(completed)}/"
                f"{TOTAL_TASKS} | "
                f"{fs_name} "
                f"R{repeat}F{outer_fold} | "
                f"grid warnings="
                f"{sum(warning_counts.values())} | "
                f"selected-inner warnings="
                f"{warning_counts[selected_key]} | "
                f"final warning="
                f"{final_cap['convergence_warning']} | "
                f"ETA {eta_min:.1f} min"
            )

# -----------------------------------------------------------------------------
# 5. Decision
# -----------------------------------------------------------------------------
fit_df = pd.DataFrame(fit_rows)
cand_df = pd.DataFrame(cand_rows)
task_df = pd.DataFrame(task_rows)

if len(task_df) != TOTAL_TASKS:
    raise RuntimeError(
        f"Expected {TOTAL_TASKS} "
        f"task rows; "
        f"found {len(task_df)}."
    )

if len(cand_df) != (
    TOTAL_TASKS
    * len(ELASTIC_GRID)
):
    raise RuntimeError(
        f"Expected "
        f"{TOTAL_TASKS * len(ELASTIC_GRID)} "
        f"candidate rows; "
        f"found {len(cand_df)}."
    )

expected_fit = (
    TOTAL_TASKS
    * (
        len(ELASTIC_GRID)
        * INNER_FOLDS
        + 1
    )
)

if len(fit_df) != expected_fit:
    raise RuntimeError(
        f"Expected {expected_fit} "
        f"fit rows; "
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
            "max_abs_prediction_diff_vs_script07"
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
        n_fits=(
            "params_json",
            "size",
        ),
        n_convergence_warnings=(
            "convergence_warning",
            "sum",
        ),
        n_hit_max_iter=(
            "hit_max_iter",
            "sum",
        ),
        max_n_iter=(
            "n_iter",
            "max",
        ),
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
        "feature_set",
        as_index=False,
    )
    .agg(
        n_tasks=(
            "feature_set",
            "size",
        ),
        selected_inner_warning_tasks=(
            "selected_inner_convergence_warning_count",
            lambda x:
                int((x > 0).sum()),
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
            "max_abs_prediction_diff_vs_script07",
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
                "max_abs_prediction_diff_vs_script07"
            ].max()
        ),
    ],
})

lines = []
add = lines.append

add(
    "DATASET A — SCRIPT 07 DIRECT "
    "ELASTICNET CONVERGENCE AUDIT (07b)"
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
add(
    f"Dataset-A modeling table SHA256: "
    f"{table_sha}"
)
add(
    f"Feature dictionary SHA256: "
    f"{dict_sha}"
)
add(
    f"Script-07 hyperparameter CSV SHA256: "
    f"{hyper_sha}"
)
add(
    f"Script-07 prediction CSV SHA256: "
    f"{pred_sha}"
)
add(
    f"Rows: {len(df)}; "
    f"participant labels: "
    f"{df[GROUP].nunique()}"
)
add(
    "Feature counts: "
    + ", ".join(
        f"{k}={len(v)}"
        for k, v
        in feature_columns.items()
    )
)
add("")

add("2. REPRODUCTION")
add("-" * 118)
add(
    summary.to_string(
        index=False
    )
)
add(
    f"REPRODUCTION PASS: "
    f"{reproduction_pass}"
)
add("")

add(
    "3. SELECTED-PATH CONVERGENCE"
)
add("-" * 118)
add(
    "Selected path = four inner fits "
    "of the originally selected params "
    "plus the final full outer-training refit."
)
add(
    "Tasks with >=1 selected-inner "
    "ConvergenceWarning: "
    f"{int((task_df['selected_inner_convergence_warning_count'] > 0).sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    "Tasks with final outer-refit "
    "ConvergenceWarning: "
    f"{int(task_df['outer_final_convergence_warning'].sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    "Tasks with selected-inner "
    "fit hitting max_iter: "
    f"{int((task_df['selected_inner_hit_max_iter_count'] > 0).sum())}"
    f"/{TOTAL_TASKS}"
)
add(
    "Tasks with final refit "
    "hitting max_iter: "
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
    "All-grid inner "
    "ConvergenceWarning fits: "
    f"{int(fit_df.loc[fit_df['stage']=='inner_candidate','convergence_warning'].sum())}"
    f"/{TOTAL_TASKS * len(ELASTIC_GRID) * INNER_FOLDS}"
)
add(
    "Tasks with at least one "
    "grid warning: "
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
add(
    by_param.to_string(
        index=False
    )
)
add("")

add("6. FEATURE-SET SUMMARY")
add("-" * 118)
add(
    by_feature.to_string(
        index=False
    )
)
add("")

add("7. DECISION")
add("-" * 118)

if (
    reproduction_pass
    and selected_path_pass
    and full_grid_pass
):
    add(
        "All Script-07 ElasticNet paths "
        "and all grid candidates reproduced "
        "and converged under the original "
        "max_iter=30000."
    )
    add(
        "The Dataset-A direct ElasticNet "
        "convergence gate is fully PASS. "
        "No recovery audit and no scientific "
        "rerun are required."
    )

elif (
    reproduction_pass
    and selected_path_pass
    and not full_grid_pass
):
    add(
        "The originally selected Script-07 "
        "ElasticNet paths reproduced and "
        "converged, but one or more non-selected "
        "grid candidates did not."
    )
    add(
        "Do NOT change the Script-07 scientific "
        "benchmark. Review warnings by candidate "
        "before any targeted numerical recovery."
    )

else:
    add(
        "One or more originally selected "
        "Script-07 ElasticNet paths failed "
        "the exact-reproduction and/or "
        "convergence gate."
    )
    add(
        "Do NOT lock the Dataset-A direct "
        "ElasticNet result until the numerical "
        "issue is resolved. Do not retune "
        "alpha/l1_ratio or change splits."
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
