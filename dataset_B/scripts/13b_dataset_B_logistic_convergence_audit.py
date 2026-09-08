from pathlib import Path
import json, hashlib, warnings, time, sys
from itertools import product

import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# 13b_dataset_B_logistic_convergence_audit.py
#
# READ-ONLY numerical validity audit for the locked Dataset-B Script 13
# LogisticRegression branch.
#
# Scientific settings reproduced exactly:
#   - 329 rows, 19 workers, 16 positive onset intervals
#   - 5 repeats x 5 outer StratifiedGroupKFold
#   - deterministic class-feasibility seed search
#   - 4 inner StratifiedGroupKFold
#   - 7 feature sets
#   - Logistic grid: C in {0.01,0.1,1,10} x penalty in {l1,l2}
#   - solver='liblinear', max_iter=10000, class_weight=None
#   - inner objective = mean PR-AUC across valid grouped inner folds
#
# Audit workload:
#   5 x 5 x 7 = 175 Logistic outer tasks
#   175 x 8 candidates x 4 inner folds = 5,600 inner candidate fits
#   175 final selected outer-training refits
#
# No model/grid/split/seed/preprocessing/max_iter/result is changed.
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT/"derived"/"dataset_B_phase1"
TABLE = DERIVED/"dataset_B_lagged_onset_modeling_table.csv"
DICT = DERIVED/"dataset_B_lagged_onset_feature_dictionary.csv"

SRC = ROOT/"results"/"dataset_B_phase1_nestedcv"
HYPER_IN = SRC/"dataset_B_nestedcv_selected_hyperparameters.csv"
PRED_IN = SRC/"dataset_B_nestedcv_predictions.csv"
FOLD_IN = SRC/"dataset_B_nestedcv_fold_metrics.csv"

OUTDIR = SRC/"convergence_audit_13b"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR = ROOT/"audit"
AUDITDIR.mkdir(parents=True, exist_ok=True)

FIT_OUT = OUTDIR/"dataset_B_13_logistic_convergence_fit_audit.csv"
TASK_OUT = OUTDIR/"dataset_B_13_logistic_convergence_task_summary.csv"
AUDIT_OUT = AUDITDIR/"dataset_B_13_logistic_convergence_audit_13b.txt"

TARGET="fatigue_onset_next_interval"
GROUP="worker_index"
BASE_SEED=20260901
OUTER_REPEATS=5
OUTER_FOLDS=5
INNER_FOLDS=4
MAX_ITER=10000

GRID=[
    {"C":c,"penalty":penalty}
    for c,penalty in product([0.01,0.1,1.0,10.0],["l1","l2"])
]

EXPECTED_COUNTS={
    "B0a_core_context":2,
    "B0b_extended_context":6,
    "B0c_extended_plus_restingHR":7,
    "B1_dynamic_wearable_only":63,
    "B2a_core_plus_wearable":65,
    "B2b_extended_plus_wearable":69,
    "B2c_extended_restingHR_plus_wearable":70,
}

def sha(fp):
    h=hashlib.sha256()
    with open(fp,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest().upper()

def safe_ap(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if len(np.unique(y))<2:
        return np.nan
    return float(average_precision_score(y,p))

def make_valid_sgkf_splits(X,y,groups,n_splits,seed,max_attempts=500):
    y=pd.Series(y).reset_index(drop=True)
    groups=pd.Series(groups).astype(str).reset_index(drop=True)
    for attempt in range(max_attempts):
        rs=seed+attempt
        cv=StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rs,
        )
        splits=list(cv.split(X,y,groups))
        ok=True
        for tr,va in splits:
            if y.iloc[tr].nunique()<2 or y.iloc[va].nunique()<2:
                ok=False
                break
        if ok:
            return splits,rs
    raise RuntimeError(
        f"Could not construct valid {n_splits}-fold SGKF "
        f"after {max_attempts} attempts."
    )

def make_pipe(params,num_cols,cat_cols):
    trs=[]
    if num_cols:
        trs.append((
            "num",
            Pipeline([
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        trs.append((
            "cat",
            Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
            ]),
            cat_cols,
        ))
    pre=ColumnTransformer(
        transformers=trs,
        remainder="drop",
        sparse_threshold=0.0,
    )
    model=LogisticRegression(
        C=float(params["C"]),
        penalty=str(params["penalty"]),
        solver="liblinear",
        max_iter=MAX_ITER,
        class_weight=None,
        random_state=BASE_SEED,
    )
    return Pipeline([
        ("pre",pre),
        ("var",VarianceThreshold(threshold=0.0)),
        ("model",model),
    ])

def fitcap(pipe,X,y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        pipe.fit(X,y)
    conv=[w for w in caught if issubclass(w.category,ConvergenceWarning)]
    raw=np.atleast_1d(pipe.named_steps["model"].n_iter_)
    n_iter=int(np.max(raw))
    return {
        "convergence_warning":int(bool(conv)),
        "n_convergence_warnings":int(len(conv)),
        "n_iter":n_iter,
        "hit_max_iter":int(n_iter>=MAX_ITER),
        "warning_message":" | ".join(str(w.message) for w in conv),
    }

def key(d):
    return json.dumps(d,sort_keys=True)

for fp in [TABLE,DICT,HYPER_IN,PRED_IN,FOLD_IN]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

df=pd.read_csv(TABLE)
fd=pd.read_csv(DICT)
hyper=pd.read_csv(HYPER_IN)
pred0=pd.read_csv(PRED_IN)
fold0=pd.read_csv(FOLD_IN)

if len(df)!=329 or df[GROUP].nunique()!=19 or int(df[TARGET].sum())!=16:
    raise SystemExit(
        f"Dataset-B lock failed: rows={len(df)}, workers={df[GROUP].nunique()}, "
        f"events={int(df[TARGET].sum())}"
    )

# Stable row identity exactly as Script 13.
df["row_id"]=(
    df["task_specific_id"].astype(str)+"__"+
    df["prediction_time_min"].astype(str)+"__"+
    df["target_interval_end_min"].astype(str)
)
if df["row_id"].duplicated().any():
    raise SystemExit("Constructed row_id is not unique.")

role={}
for r in fd["role"].dropna().unique():
    role[r]=fd.loc[fd["role"]==r,"column"].tolist()

core=[c for c in role.get("context_core",[]) if c in df.columns]
ext=[c for c in role.get("context_extended",[]) if c in df.columns]
rest=[c for c in role.get("baseline_physiology",[]) if c in df.columns]
wear=[c for c in role.get("wearable_feature",[]) if c in df.columns]

FEATURE_SETS={
    "B0a_core_context":core,
    "B0b_extended_context":list(dict.fromkeys(core+ext)),
    "B0c_extended_plus_restingHR":list(dict.fromkeys(core+ext+rest)),
    "B1_dynamic_wearable_only":wear,
    "B2a_core_plus_wearable":list(dict.fromkeys(core+wear)),
    "B2b_extended_plus_wearable":list(dict.fromkeys(core+ext+wear)),
    "B2c_extended_restingHR_plus_wearable":list(dict.fromkeys(core+ext+rest+wear)),
}

for name,n in EXPECTED_COUNTS.items():
    if len(FEATURE_SETS[name])!=n:
        raise SystemExit(
            f"Feature-count lock failed for {name}: "
            f"expected={n}, found={len(FEATURE_SETS[name])}"
        )

cat_candidates={"task","sex"}

hyper=hyper[hyper["model"].astype(str)=="Logistic"].copy()
pred0=pred0[pred0["model"].astype(str)=="Logistic"].copy()
fold0=fold0[fold0["model"].astype(str)=="Logistic"].copy()

TOTAL_TASKS=OUTER_REPEATS*OUTER_FOLDS*len(FEATURE_SETS)
if len(hyper)!=TOTAL_TASKS:
    raise SystemExit(f"Expected {TOTAL_TASKS} Logistic hyper rows; found {len(hyper)}")

fit_rows=[]
task_rows=[]
t0=time.time()
done=0

for repeat in range(1,OUTER_REPEATS+1):
    repeat_seed=BASE_SEED+(repeat-1)*10000
    outer_splits,used_outer_seed=make_valid_sgkf_splits(
        df,
        df[TARGET].astype(int),
        df[GROUP].astype(str),
        OUTER_FOLDS,
        repeat_seed,
    )

    # Compare reproduced outer seed to all stored Logistic fold rows for this repeat.
    stored_outer_seeds=set(
        fold0.loc[fold0["repeat"]==repeat,"outer_split_seed"].astype(int).tolist()
    )
    if stored_outer_seeds!={int(used_outer_seed)}:
        raise RuntimeError(
            f"Outer split seed mismatch repeat {repeat}: "
            f"reproduced={used_outer_seed}, stored={sorted(stored_outer_seeds)}"
        )

    for outer_fold,(tr_idx,te_idx) in enumerate(outer_splits,1):
        train=df.iloc[tr_idx].reset_index(drop=True)
        test=df.iloc[te_idx].reset_index(drop=True)
        y_train=train[TARGET].astype(int).reset_index(drop=True)
        g_train=train[GROUP].astype(str).reset_index(drop=True)

        for fs_idx,(fs_name,cols) in enumerate(FEATURE_SETS.items()):
            h=hyper[
                (hyper["repeat"]==repeat)&
                (hyper["outer_fold"]==outer_fold)&
                (hyper["feature_set"].astype(str)==fs_name)
            ]
            if len(h)!=1:
                raise RuntimeError(f"Hyper row mismatch: R{repeat}F{outer_fold} {fs_name}")

            stored_params=json.loads(h.iloc[0]["selected_params_json"])
            stored_inner_ap=float(h.iloc[0]["inner_selected_pr_auc"])

            cat_cols=[c for c in cols if c in cat_candidates]
            num_cols=[c for c in cols if c not in cat_candidates]
            Xtr=train[cols].reset_index(drop=True)
            Xte=test[cols].reset_index(drop=True)

            inner_seed=(
                repeat_seed+
                outer_fold*1000+
                fs_idx*10+
                1
            )
            inner_splits,used_inner_seed=make_valid_sgkf_splits(
                Xtr,y_train,g_train,INNER_FOLDS,inner_seed
            )

            sf=fold0[
                (fold0["repeat"]==repeat)&
                (fold0["outer_fold"]==outer_fold)&
                (fold0["feature_set"].astype(str)==fs_name)
            ]
            if len(sf)!=1:
                raise RuntimeError(f"Stored fold row mismatch: R{repeat}F{outer_fold} {fs_name}")
            stored_inner_seed=int(sf.iloc[0]["inner_split_seed"])
            if int(used_inner_seed)!=stored_inner_seed:
                raise RuntimeError(
                    f"Inner seed mismatch R{repeat}F{outer_fold} {fs_name}: "
                    f"reproduced={used_inner_seed}, stored={stored_inner_seed}"
                )

            scores={}
            warn_by={}
            hit_by={}
            maxiter_by={}

            for params in GRID:
                pk=key(params)
                aps=[]
                sw=sh=mx=0
                for inner_fold,(itr,iva) in enumerate(inner_splits,1):
                    p=make_pipe(params,num_cols,cat_cols)
                    cap=fitcap(p,Xtr.iloc[itr],y_train.iloc[itr])
                    prob=p.predict_proba(Xtr.iloc[iva])[:,1]
                    ap=safe_ap(y_train.iloc[iva],prob)
                    if np.isfinite(ap):
                        aps.append(ap)

                    sw+=cap["convergence_warning"]
                    sh+=cap["hit_max_iter"]
                    mx=max(mx,cap["n_iter"])

                    fit_rows.append({
                        "repeat":repeat,
                        "outer_fold":outer_fold,
                        "feature_set":fs_name,
                        "stage":"inner_candidate",
                        "inner_fold":inner_fold,
                        "params_json":pk,
                        "is_originally_selected":int(pk==key(stored_params)),
                        "pr_auc":ap,
                        **cap,
                    })

                score=float(np.mean(aps)) if aps else -np.inf
                scores[pk]=score
                warn_by[pk]=sw
                hit_by[pk]=sh
                maxiter_by[pk]=mx

            reproduced_key,reproduced_ap=min(
                scores.items(),
                key=lambda kv:(-kv[1],kv[0])
            )

            final=make_pipe(stored_params,num_cols,cat_cols)
            fcap=fitcap(final,Xtr,y_train)
            prob=final.predict_proba(Xte)[:,1]

            fit_rows.append({
                "repeat":repeat,
                "outer_fold":outer_fold,
                "feature_set":fs_name,
                "stage":"outer_final_selected",
                "inner_fold":0,
                "params_json":key(stored_params),
                "is_originally_selected":1,
                "pr_auc":np.nan,
                **fcap,
            })

            q=pred0[
                (pred0["repeat"]==repeat)&
                (pred0["outer_fold"]==outer_fold)&
                (pred0["feature_set"].astype(str)==fs_name)
            ][["row_id","y_prob"]].copy()

            z=test[["row_id"]].copy()
            z["prob_reproduced"]=prob
            m=z.merge(q,on="row_id",how="inner",validate="one_to_one")
            if len(m)!=len(test):
                raise RuntimeError(
                    f"Prediction-row mismatch R{repeat}F{outer_fold} {fs_name}: "
                    f"stored={len(m)}, expected={len(test)}"
                )
            pdiff=float(np.max(np.abs(
                m["prob_reproduced"].to_numpy(float)-
                m["y_prob"].to_numpy(float)
            )))

            skey=key(stored_params)
            task_rows.append({
                "repeat":repeat,
                "outer_fold":outer_fold,
                "feature_set":fs_name,
                "reproduced_outer_seed":int(used_outer_seed),
                "reproduced_inner_seed":int(used_inner_seed),
                "stored_selected_params_json":skey,
                "reproduced_best_params_json":reproduced_key,
                "selected_params_match":int(skey==reproduced_key),
                "stored_inner_selected_pr_auc":stored_inner_ap,
                "reproduced_inner_selected_pr_auc":reproduced_ap,
                "inner_pr_auc_abs_diff":abs(stored_inner_ap-reproduced_ap),
                "selected_inner_warning_count":int(warn_by[skey]),
                "selected_inner_hitmax_count":int(hit_by[skey]),
                "selected_inner_max_n_iter":int(maxiter_by[skey]),
                "final_warning":int(fcap["convergence_warning"]),
                "final_hitmax":int(fcap["hit_max_iter"]),
                "final_n_iter":int(fcap["n_iter"]),
                "max_prediction_abs_diff":pdiff,
                "any_grid_warning":int(sum(warn_by.values())>0),
                "n_grid_warnings":int(sum(warn_by.values())),
            })

            done+=1
            if done%10==0 or done==TOTAL_TASKS:
                print(
                    f"Audited {done}/{TOTAL_TASKS} Logistic tasks | "
                    f"elapsed {(time.time()-t0)/60:.1f} min"
                )

fit=pd.DataFrame(fit_rows)
tasks=pd.DataFrame(task_rows)

expected_fits=TOTAL_TASKS*(len(GRID)*INNER_FOLDS+1)
if len(tasks)!=TOTAL_TASKS or len(fit)!=expected_fits:
    raise RuntimeError(
        f"Audit size mismatch: tasks={len(tasks)}/{TOTAL_TASKS}, "
        f"fits={len(fit)}/{expected_fits}"
    )

fit.to_csv(FIT_OUT,index=False)
tasks.to_csv(TASK_OUT,index=False)

repro=bool(
    (tasks["selected_params_match"]==1).all() and
    (tasks["inner_pr_auc_abs_diff"]<=1e-12).all() and
    (tasks["max_prediction_abs_diff"]<=1e-12).all()
)
selected=bool(
    (tasks["selected_inner_warning_count"]==0).all() and
    (tasks["selected_inner_hitmax_count"]==0).all() and
    (tasks["final_warning"]==0).all() and
    (tasks["final_hitmax"]==0).all()
)
full=bool(
    (fit["convergence_warning"]==0).all() and
    (fit["hit_max_iter"]==0).all()
)

by_param=(
    fit[fit["stage"]=="inner_candidate"]
    .groupby("params_json",as_index=False)
    .agg(
        n_fits=("params_json","size"),
        n_convergence_warnings=("convergence_warning","sum"),
        n_hit_max_iter=("hit_max_iter","sum"),
        max_n_iter=("n_iter","max"),
    )
    .sort_values(["n_convergence_warnings","n_hit_max_iter"],ascending=False)
)

by_feature=(
    tasks.groupby("feature_set",as_index=False)
    .agg(
        n_tasks=("feature_set","size"),
        selected_inner_warning_tasks=("selected_inner_warning_count",lambda x:int((x>0).sum())),
        final_warning_tasks=("final_warning","sum"),
        any_grid_warning_tasks=("any_grid_warning","sum"),
        max_selected_inner_n_iter=("selected_inner_max_n_iter","max"),
        max_final_n_iter=("final_n_iter","max"),
        max_inner_pr_auc_abs_diff=("inner_pr_auc_abs_diff","max"),
        max_prediction_abs_diff=("max_prediction_abs_diff","max"),
    )
)

L=[]
A=L.append
A("DATASET B — SCRIPT 13 LOGISTIC CONVERGENCE AUDIT (13b)")
A("="*118)
A("Purpose: read-only numerical reproduction audit. No model, grid, split, class-feasibility seed search, feature set, preprocessing, max_iter, or scientific result was changed.")
A(f"LogisticRegression solver=liblinear; max_iter reproduced exactly: {MAX_ITER}")
A(f"Audited outer Logistic tasks: {TOTAL_TASKS}")
A(f"Inner candidate fits: {TOTAL_TASKS*len(GRID)*INNER_FOLDS}")
A(f"Outer final selected refits: {TOTAL_TASKS}")
A("")
A("1. INPUT LOCKS"); A("-"*118)
A(f"Modeling table SHA256: {sha(TABLE)}")
A(f"Feature dictionary SHA256: {sha(DICT)}")
A(f"Script-13 hyperparameter CSV SHA256: {sha(HYPER_IN)}")
A(f"Script-13 prediction CSV SHA256: {sha(PRED_IN)}")
A(f"Script-13 fold-metrics CSV SHA256: {sha(FOLD_IN)}")
A(f"Rows={len(df)}; workers={df[GROUP].nunique()}; positive onset intervals={int(df[TARGET].sum())}")
A("Feature counts: "+", ".join(f"{k}={len(v)}" for k,v in FEATURE_SETS.items()))
A("")
A("2. AUDIT ENVIRONMENT"); A("-"*118)
A(f"Python: {sys.version.split()[0]}")
A(f"NumPy: {np.__version__}")
A(f"Pandas: {pd.__version__}")
A(f"scikit-learn: {sklearn.__version__}")
A("")
A("3. REPRODUCTION"); A("-"*118)
A(f"Max selected inner PR-AUC abs diff: {tasks.inner_pr_auc_abs_diff.max():.12e}")
A(f"Max outer-test probability abs diff: {tasks.max_prediction_abs_diff.max():.12e}")
A("All stored outer split seeds reproduced exactly: True")
A("All stored inner split seeds reproduced exactly: True")
A(f"REPRODUCTION PASS: {repro}")
A("")
A("4. SELECTED-PATH CONVERGENCE"); A("-"*118)
A(f"Tasks with selected-inner ConvergenceWarning: {int((tasks.selected_inner_warning_count>0).sum())}/{TOTAL_TASKS}")
A(f"Tasks with final-refit ConvergenceWarning: {int(tasks.final_warning.sum())}/{TOTAL_TASKS}")
A(f"Tasks with selected-inner fit hitting max_iter: {int((tasks.selected_inner_hitmax_count>0).sum())}/{TOTAL_TASKS}")
A(f"Tasks with final refit hitting max_iter: {int(tasks.final_hitmax.sum())}/{TOTAL_TASKS}")
A(f"SELECTED-PATH CONVERGENCE PASS: {selected}")
A("")
A("5. FULL-GRID CONVERGENCE"); A("-"*118)
A(f"All-grid inner ConvergenceWarning fits: {int(fit.loc[fit.stage=='inner_candidate','convergence_warning'].sum())}/{TOTAL_TASKS*len(GRID)*INNER_FOLDS}")
A(f"Tasks with at least one grid warning: {int((tasks.any_grid_warning>0).sum())}/{TOTAL_TASKS}")
A(f"FULL-GRID CONVERGENCE PASS: {full}")
A("")
A("6. WARNINGS BY HYPERPARAMETER"); A("-"*118)
A(by_param.to_string(index=False)); A("")
A("7. FEATURE-SET SUMMARY"); A("-"*118)
A(by_feature.to_string(index=False)); A("")
A("8. DECISION"); A("-"*118)
if repro and selected and full:
    A("All Script-13 Logistic paths and all grid candidates reproduced and converged under the original liblinear max_iter=10000.")
    A("The Dataset-B Logistic numerical-convergence gate is fully PASS. No recovery audit and no scientific rerun are required.")
elif repro and selected and not full:
    A("The originally selected Script-13 Logistic paths reproduced and converged, but one or more non-selected grid candidates did not.")
    A("Do not alter the scientific benchmark. Inspect non-selected candidate warnings before any targeted numerical recovery.")
else:
    A("One or more originally selected Script-13 Logistic paths failed exact reproduction and/or convergence.")
    A("Do not lock the Dataset-B Logistic result until the numerical issue is resolved. Do not change C/penalty, splits, or class-feasibility rules to chase a result.")
A("")
A("Outputs:")
A(f"  {FIT_OUT}")
A(f"  {TASK_OUT}")
A(f"  {AUDIT_OUT}")
A("="*118)

AUDIT_OUT.write_text("\n".join(L),encoding="utf-8")

print("")
print(f"REPRODUCTION PASS: {repro}")
print(f"SELECTED-PATH CONVERGENCE PASS: {selected}")
print(f"FULL-GRID CONVERGENCE PASS: {full}")
print(f"Audit: {AUDIT_OUT}")
