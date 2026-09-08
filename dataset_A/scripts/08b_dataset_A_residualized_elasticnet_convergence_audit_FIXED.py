from pathlib import Path
import json, time, hashlib, warnings
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

# READ-ONLY convergence audit for Script 08.
# No grid, split, residual construction, preprocessing, max_iter, or scientific
# result is changed.
ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT/"derived"/"dataset_A_phase1"/"dataset_A_phase1_modeling_table.csv"
DICT = ROOT/"derived"/"dataset_A_phase1"/"dataset_A_phase1_feature_dictionary.csv"
SRC = ROOT/"results"/"dataset_A_phase1_residualized"
HYPER = SRC/"dataset_A_residualized_hyperparameters.csv"
PRED = SRC/"dataset_A_residualized_predictions.csv"
OUTDIR = SRC/"convergence_audit_08b"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR = ROOT/"audit"
AUDITDIR.mkdir(parents=True, exist_ok=True)

FIT_OUT = OUTDIR/"dataset_A_08_elasticnet_convergence_fit_audit.csv"
TASK_OUT = OUTDIR/"dataset_A_08_elasticnet_convergence_task_summary.csv"
AUDIT_OUT = AUDITDIR/"dataset_A_08_elasticnet_convergence_audit_08b.txt"

TARGET="physical_fatigue_final"
GROUP="participant_id"
ROW_ID="source_file"
OUTER_REPEATS=5
OUTER_FOLDS=5
INNER_FOLDS=4
BASE_SEED=20260901
MAX_ITER=30000
GRID=[{"alpha":a,"l1_ratio":l1} for a,l1 in product([0.01,0.1,1.0,10.0],[0.1,0.5,0.9])]

def sha(fp):
    h=hashlib.sha256()
    with open(fp,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest().upper()

def pb_mae(y,p,g):
    d=pd.DataFrame({"y":np.asarray(y,float),"p":np.asarray(p,float),"g":np.asarray(g).astype(str)})
    d["ae"]=np.abs(d.y-d.p)
    return float(d.groupby("g",sort=False).ae.mean().mean())

def group_folds(groups,n_splits,seed):
    groups=pd.Series(groups).astype(str).reset_index(drop=True)
    u=groups.unique().tolist()
    rng=np.random.default_rng(seed); rng.shuffle(u)
    sizes=groups.value_counts().to_dict()
    fg=[set() for _ in range(n_splits)]; fs=[0]*n_splits
    for x in u:
        j=int(np.argmin(fs)); fg[j].add(x); fs[j]+=int(sizes[x])
    idx=np.arange(len(groups)); out=[]
    for s in fg:
        m=groups.isin(s).to_numpy()
        out.append((idx[~m],idx[m]))
    return out

def pipe(params,num,cat):
    tr=[]
    if num:
        tr.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),
                                   ("scaler",StandardScaler())]),num))
    if cat:
        tr.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                   ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
    pre=ColumnTransformer(tr,remainder="drop",sparse_threshold=0.0)
    model=ElasticNet(alpha=float(params["alpha"]),l1_ratio=float(params["l1_ratio"]),
                     max_iter=MAX_ITER,random_state=BASE_SEED,selection="cyclic")
    return Pipeline([("pre",pre),("var",VarianceThreshold(0.0)),("model",model)])

def fitcap(p,X,y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",ConvergenceWarning)
        p.fit(X,y)
    cw=[x for x in w if issubclass(x.category,ConvergenceWarning)]
    n=int(np.max(np.atleast_1d(p.named_steps["model"].n_iter_)))
    return int(bool(cw)), n, int(n>=MAX_ITER)

def key(d): return json.dumps(d,sort_keys=True)

for fp in [TABLE,DICT,HYPER,PRED]:
    if not fp.exists(): raise SystemExit(f"Missing required input: {fp}")

df=pd.read_csv(TABLE)
fd=pd.read_csv(DICT)
hyp=pd.read_csv(HYPER)
pred0=pd.read_csv(PRED)

if len(df)!=372 or df[GROUP].nunique()!=43:
    raise SystemExit(f"Dataset-A lock failed: rows={len(df)}, participants={df[GROUP].nunique()}")
if df[ROW_ID].duplicated().any() or df[TARGET].isna().any():
    raise SystemExit("Dataset-A row/target lock failed.")

context=fd.loc[fd.role=="context_core","column"].tolist()
context=[c for c in context if c in df.columns]
wear=fd.loc[fd.role=="wearable_feature","column"].tolist()
wear=[c for c in wear if c in df.columns]
cat_ctx=[c for c in ["task","gender"] if c in context]
num_ctx=[c for c in context if c not in cat_ctx]

def starts(c,pfx): return any(c.startswith(x) for x in pfx)
families={
 "Physiology":[c for c in wear if starts(c,("sensor__hr__","sensor__hr_processed__","sensor__hrv__","sensor__rr__","sensor__ecg__","sensor__temperature__"))],
 "ChestAcceleration":[c for c in wear if starts(c,("sensor__acc_x__","sensor__acc_y__","sensor__acc_z__","sensor__chest_acc_"))],
 "IMUAcceleration":[c for c in wear if c.startswith("sensor__imu")],
 "AllWearable":list(wear),
}
if len(context)!=5 or any(len(v)==0 for v in families.values()):
    raise SystemExit("Context/family lock failed.")

hyp=hyp[hyp.model.astype(str)=="ElasticNet"].copy()
hctx=hyp[hyp.stage.astype(str)=="context"].copy()
hres=hyp[hyp.stage.astype(str)=="residual"].copy()
if len(hctx)!=25 or len(hres)!=100:
    raise SystemExit(f"Hyperparameter lock failed: context={len(hctx)}, residual={len(hres)}")
pred0=pred0[pred0.model.astype(str)=="ElasticNet"].copy()

fit_rows=[]; task_rows=[]
t0=time.time(); done=0

for repeat in range(1,6):
    outer_seed=BASE_SEED+1000*(repeat-1)
    outer=group_folds(df[GROUP],5,outer_seed)
    for outer_fold,(tr,te) in enumerate(outer,1):
        train=df.iloc[tr].reset_index(drop=True)
        test=df.iloc[te].reset_index(drop=True)
        y=train[TARGET].astype(float).reset_index(drop=True)
        g=train[GROUP].astype(str).reset_index(drop=True)
        Xc=train[context].reset_index(drop=True)
        Xct=test[context].reset_index(drop=True)

        # ---- context tuning
        hc=hctx[(hctx.repeat==repeat)&(hctx.outer_fold==outer_fold)]
        if len(hc)!=1: raise RuntimeError("Context hyper row mismatch")
        sp=json.loads(hc.iloc[0].selected_params_json)
        stored=float(hc.iloc[0].inner_mae_pb)
        ctx_seed=outer_seed+outer_fold*100+1
        ins=group_folds(g,4,ctx_seed)

        scores={}
        for par in GRID:
            pk=key(par); vals=[]
            for k,(itr,iva) in enumerate(ins,1):
                m=pipe(par,num_ctx,cat_ctx)
                cw,ni,hit=fitcap(m,Xc.iloc[itr],y.iloc[itr])
                pp=m.predict(Xc.iloc[iva])
                vals.append(pb_mae(y.iloc[iva],pp,g.iloc[iva]))
                fit_rows.append([repeat,outer_fold,"context_tuning","None","inner",k,pk,cw,ni,hit])
            scores[pk]=float(np.mean(vals))
        reproduced=min(scores.items(),key=lambda z:(z[1],z[0]))
        ctx_match=(reproduced[0]==key(sp))
        ctx_mae_diff=abs(reproduced[1]-stored)

        # ---- context final
        mctx=pipe(sp,num_ctx,cat_ctx)
        ctx_fw,ctx_fn,ctx_fhit=fitcap(mctx,Xc,y)
        ctx_test=mctx.predict(Xct)
        fit_rows.append([repeat,outer_fold,"context_final","None","final",0,key(sp),ctx_fw,ctx_fn,ctx_fhit])

        # ---- context cross-fit used to build residual target
        coof=np.full(len(train),np.nan)
        cross_warn=cross_hit=0; cross_max=0
        for k,(itr,iva) in enumerate(group_folds(g,4,ctx_seed+50),1):
            mm=pipe(sp,num_ctx,cat_ctx)
            cw,ni,hit=fitcap(mm,Xc.iloc[itr],y.iloc[itr])
            coof[iva]=mm.predict(Xc.iloc[iva])
            cross_warn+=cw; cross_hit+=hit; cross_max=max(cross_max,ni)
            fit_rows.append([repeat,outer_fold,"context_crossfit","None","crossfit",k,key(sp),cw,ni,hit])
        if np.isnan(coof).any(): raise RuntimeError("Context cross-fit contains NaN")
        resid=pd.Series(y.to_numpy()-coof,name="context_residual")

        # context prediction reproduction
        q=pred0[(pred0.repeat==repeat)&(pred0.outer_fold==outer_fold)&
                (pred0.variant.astype(str)=="ContextOnly")][[ROW_ID,"y_pred"]]
        z=test[[ROW_ID]].copy(); z["p"]=ctx_test
        mm=z.merge(q,on=ROW_ID,validate="one_to_one")
        ctx_pred_diff=float(np.max(np.abs(mm.p.to_numpy()-mm.y_pred.to_numpy())))

        family_detail=[]
        for fi,(fam,cols) in enumerate(families.items()):
            hr=hres[(hres.repeat==repeat)&(hres.outer_fold==outer_fold)&
                    (hres.sensor_family.astype(str)==fam)]
            if len(hr)!=1: raise RuntimeError(f"Residual hyper row mismatch: {fam}")
            rp=json.loads(hr.iloc[0].selected_params_json)
            rstored=float(hr.iloc[0].inner_mae_pb)
            rseed=outer_seed+outer_fold*1000+fi*10+3
            rins=group_folds(g,4,rseed)
            Xw=train[cols].reset_index(drop=True); Xwt=test[cols].reset_index(drop=True)

            rscores={}; warn_by={}; hit_by={}; max_by={}
            for par in GRID:
                pk=key(par); vals=[]; sw=sh=mx=0
                for k,(itr,iva) in enumerate(rins,1):
                    mr=pipe(par,cols,[])
                    cw,ni,hit=fitcap(mr,Xw.iloc[itr],resid.iloc[itr])
                    pp=mr.predict(Xw.iloc[iva])
                    vals.append(pb_mae(resid.iloc[iva],pp,g.iloc[iva]))
                    sw+=cw; sh+=hit; mx=max(mx,ni)
                    fit_rows.append([repeat,outer_fold,"residual_tuning",fam,"inner",k,pk,cw,ni,hit])
                rscores[pk]=float(np.mean(vals)); warn_by[pk]=sw; hit_by[pk]=sh; max_by[pk]=mx

            rbest=min(rscores.items(),key=lambda z:(z[1],z[0]))
            mr=pipe(rp,cols,[])
            fw,fn,fhit=fitcap(mr,Xw,resid)
            final_pred=ctx_test+mr.predict(Xwt)
            fit_rows.append([repeat,outer_fold,"residual_final",fam,"final",0,key(rp),fw,fn,fhit])

            variant=f"Residualized_{fam}"
            q=pred0[(pred0.repeat==repeat)&(pred0.outer_fold==outer_fold)&
                    (pred0.variant.astype(str)==variant)&
                    (pred0.sensor_family.astype(str)==fam)][[ROW_ID,"y_pred"]]
            z=test[[ROW_ID]].copy(); z["p"]=final_pred
            mm=z.merge(q,on=ROW_ID,validate="one_to_one")
            pdiff=float(np.max(np.abs(mm.p.to_numpy()-mm.y_pred.to_numpy())))

            family_detail.append({
                "family":fam,
                "match":int(rbest[0]==key(rp)),
                "mae_diff":abs(rbest[1]-rstored),
                "selected_inner_warning_count":warn_by[key(rp)],
                "selected_inner_hitmax_count":hit_by[key(rp)],
                "selected_inner_max_n_iter":max_by[key(rp)],
                "final_warning":fw,
                "final_hitmax":fhit,
                "final_n_iter":fn,
                "prediction_diff":pdiff,
                "any_grid_warning":int(sum(warn_by.values())>0),
            })

        task_rows.append({
            "repeat":repeat,"outer_fold":outer_fold,
            "context_match":int(ctx_match),
            "context_mae_diff":ctx_mae_diff,
            "context_selected_inner_warning_count":
                sum(r[7] for r in fit_rows if r[0]==repeat and r[1]==outer_fold and r[2]=="context_tuning" and r[6]==key(sp)),
            "context_crossfit_warning_count":cross_warn,
            "context_crossfit_hitmax_count":cross_hit,
            "context_crossfit_max_n_iter":cross_max,
            "context_final_warning":ctx_fw,
            "context_final_hitmax":ctx_fhit,
            "context_final_n_iter":ctx_fn,
            "context_prediction_diff":ctx_pred_diff,
            "family_detail":json.dumps(family_detail,sort_keys=True),
        })
        done+=1
        print(f"Audited {done}/25 outer tasks | R{repeat}F{outer_fold} | elapsed {(time.time()-t0)/60:.1f} min")

fit=pd.DataFrame(fit_rows,columns=[
    "repeat","outer_fold","component","sensor_family","stage","inner_fold",
    "params_json","convergence_warning","n_iter","hit_max_iter"
])
tasks=pd.DataFrame(task_rows)

fam=[]
for _,r in tasks.iterrows():
    for x in json.loads(r.family_detail):
        fam.append({"repeat":int(r["repeat"]),"outer_fold":int(r["outer_fold"]),**x})
fam=pd.DataFrame(fam)

repro=bool(
    (tasks.context_match==1).all() and (tasks.context_mae_diff<=1e-10).all() and
    (tasks.context_prediction_diff<=1e-10).all() and
    (fam.match==1).all() and (fam.mae_diff<=1e-10).all() and
    (fam.prediction_diff<=1e-10).all()
)
selected=bool(
    (tasks.context_selected_inner_warning_count==0).all() and
    (tasks.context_crossfit_warning_count==0).all() and
    (tasks.context_crossfit_hitmax_count==0).all() and
    (tasks.context_final_warning==0).all() and
    (tasks.context_final_hitmax==0).all() and
    (fam.selected_inner_warning_count==0).all() and
    (fam.selected_inner_hitmax_count==0).all() and
    (fam.final_warning==0).all() and
    (fam.final_hitmax==0).all()
)
full=bool((fit.convergence_warning==0).all() and (fit.hit_max_iter==0).all())

fit.to_csv(FIT_OUT,index=False)
pd.concat([
    tasks.drop(columns=["family_detail"]).assign(component="context"),
    fam.assign(component="residual_family")
],ignore_index=True,sort=False).to_csv(TASK_OUT,index=False)

bycomp=fit.groupby(["component","stage"],as_index=False).agg(
    n_fits=("stage","size"),
    warnings=("convergence_warning","sum"),
    hit_max_iter=("hit_max_iter","sum"),
    max_n_iter=("n_iter","max")
)
byfam=fam.groupby("family",as_index=False).agg(
    n_tasks=("family","size"),
    selected_inner_warning_tasks=("selected_inner_warning_count",lambda x:int((x>0).sum())),
    final_warning_tasks=("final_warning","sum"),
    max_selected_inner_n_iter=("selected_inner_max_n_iter","max"),
    max_final_n_iter=("final_n_iter","max"),
    max_mae_diff=("mae_diff","max"),
    max_prediction_diff=("prediction_diff","max"),
)

lines=[]
A=lines.append
A("DATASET A — SCRIPT 08 OUTCOME-RESIDUALIZED ELASTICNET CONVERGENCE AUDIT (08b)")
A("="*118)
A("Purpose: read-only numerical reproduction audit. No model, grid, split, context definition, sensor family, residual construction, preprocessing, max_iter, or scientific result was changed.")
A(f"ElasticNet max_iter reproduced exactly: {MAX_ITER}")
A("Context tuning inner fits: 1200")
A("Context selected-parameter cross-fit fits: 100")
A("Context final refits: 25")
A("Residual-family tuning inner fits: 4800")
A("Residual-family final refits: 100")
A(f"Total ElasticNet fits audited: {len(fit)}")
A("")
A("1. INPUT LOCKS"); A("-"*118)
A(f"Modeling table SHA256: {sha(TABLE)}")
A(f"Feature dictionary SHA256: {sha(DICT)}")
A(f"Script-08 hyperparameter CSV SHA256: {sha(HYPER)}")
A(f"Script-08 prediction CSV SHA256: {sha(PRED)}")
A(f"Rows={len(df)}; participant labels={df[GROUP].nunique()}")
A("Core context: "+", ".join(context))
A("Sensor family counts: "+", ".join(f"{k}={len(v)}" for k,v in families.items()))
A("")
A("2. REPRODUCTION"); A("-"*118)
A(f"Max context inner-MAE abs diff: {tasks.context_mae_diff.max():.12e}")
A(f"Max context prediction abs diff: {tasks.context_prediction_diff.max():.12e}")
A(f"Max residual inner-MAE abs diff: {fam.mae_diff.max():.12e}")
A(f"Max residual prediction abs diff: {fam.prediction_diff.max():.12e}")
A(f"REPRODUCTION PASS: {repro}")
A("")
A("3. SELECTED-PATH CONVERGENCE"); A("-"*118)
A(f"Context tasks with selected-inner warning: {int((tasks.context_selected_inner_warning_count>0).sum())}/25")
A(f"Context tasks with cross-fit warning: {int((tasks.context_crossfit_warning_count>0).sum())}/25")
A(f"Context tasks with final-refit warning: {int(tasks.context_final_warning.sum())}/25")
A(f"Residual-family tasks with selected-inner warning: {int((fam.selected_inner_warning_count>0).sum())}/100")
A(f"Residual-family tasks with final-refit warning: {int(fam.final_warning.sum())}/100")
A(f"SELECTED-PATH CONVERGENCE PASS: {selected}")
A("")
A("4. FULL-FIT CONVERGENCE"); A("-"*118)
A(f"All audited ConvergenceWarning fits: {int(fit.convergence_warning.sum())}/{len(fit)}")
A(f"All audited fits hitting max_iter: {int(fit.hit_max_iter.sum())}/{len(fit)}")
A(f"FULL-GRID CONVERGENCE PASS: {full}")
A("")
A("5. FITS BY COMPONENT"); A("-"*118); A(bycomp.to_string(index=False)); A("")
A("6. SENSOR-FAMILY SUMMARY"); A("-"*118); A(byfam.to_string(index=False)); A("")
A("7. DECISION"); A("-"*118)
if repro and selected and full:
    A("All Script-08 ElasticNet context, cross-fit, and residual-family paths reproduced and converged under the original max_iter=30000.")
    A("The Dataset-A outcome-residualized ElasticNet convergence gate is fully PASS. No recovery audit and no scientific rerun are required.")
elif repro and selected and not full:
    A("The originally selected Script-08 ElasticNet paths reproduced and converged, but one or more non-selected grid candidates did not.")
    A("Do not alter the scientific benchmark. Inspect non-selected candidate warnings before any targeted numerical recovery.")
else:
    A("One or more selected Script-08 paths failed exact reproduction and/or convergence.")
    A("Do not lock the residualized ElasticNet result until the numerical issue is resolved. Do not retune the grid or change splits.")
A("")
A("Outputs:")
A(f"  {FIT_OUT}")
A(f"  {TASK_OUT}")
A(f"  {AUDIT_OUT}")
A("="*118)

AUDIT_OUT.write_text("\n".join(lines),encoding="utf-8")
print("")
print(f"REPRODUCTION PASS: {repro}")
print(f"SELECTED-PATH CONVERGENCE PASS: {selected}")
print(f"FULL-GRID CONVERGENCE PASS: {full}")
print(f"Audit: {AUDIT_OUT}")
