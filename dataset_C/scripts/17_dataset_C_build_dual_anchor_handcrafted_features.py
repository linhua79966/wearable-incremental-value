
from pathlib import Path
import re, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DS=ROOT/"data"/"dataset_C_shoulder_rotation"/"raw"/"WSD4FEDSRM"
SR=DS/"EMG, IMU, and PPG data"
BORG=DS/"Borg data"/"borg_data.csv"
DEMO=DS/"Demographic and antropometric data"/"demographic.csv"
BODY=DS/"Demographic and antropometric data"/"body_composition.csv"
OUT=ROOT/"derived"/"dataset_C_phase1"; OUT.mkdir(parents=True,exist_ok=True)
AUD=ROOT/"audit"; AUD.mkdir(parents=True,exist_ok=True)

START_OUT=OUT/"dataset_C_handcrafted_start_anchor.csv"
END_OUT=OUT/"dataset_C_handcrafted_end_anchor.csv"
DICT_OUT=OUT/"dataset_C_handcrafted_feature_dictionary.csv"
PAIR_OUT=OUT/"dataset_C_forecast_pair_manifest.csv"
AUDIT_OUT=AUD/"dataset_C_handcrafted_build_audit.txt"

EMG_HZ=1000
IMU_HZ=100
WIN=10

TASK={
"task1_35i":("30-40_ internal rotation","internal",35),
"task2_45i":("40-50_ internal rotation","internal",45),
"task3_55i":("50-60_ internal rotation","internal",55),
"task4_35e":("30-40_ external rotation","external",35),
"task5_45e":("40-50_ external rotation","external",45),
"task6_55e":("50-60_ external rotation","external",55),
}
EMGS=["anterior_deltoid.csv","infraspinatus.csv","latissimus_dorsi.csv",
      "pectoralis_major.csv","posterior_deltoid.csv","upper_trapezius.csv"]
LOCS=["Forearm","Hand","Pelvis","Shoulder","Sternum","Upper arm"]

def norm(c): return re.sub(r"\s+","",str(c).strip().lower())
def safe(s): return re.sub(r"[^a-z0-9]+","_",str(s).strip().lower()).strip("_")
def subjnum(s):
    m=re.search(r"(\d+)",str(s)); return int(m.group(1)) if m else None

def readnum(fp):
    x=pd.read_csv(fp)
    for c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.to_numpy(dtype=float)

def b_start(t,rate,n):
    a=int(round((t-WIN)*rate)); b=int(round(t*rate))
    return (a,b) if a>=0 and b<=n and b>a else None

def b_end(t,dur,rate,n):
    a=int(round(n-(dur-(t-WIN))*rate))
    b=int(round(n-(dur-t)*rate))
    return (a,b) if a>=0 and b<=n and b>a else None

def stats(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return {k:np.nan for k in ["mean","std","rms","median","iqr","range","slope"]}
    q25,q50,q75=np.quantile(x,[.25,.5,.75])
    if len(x)>1:
        tt=np.linspace(-.5,.5,len(x)); den=np.sum((tt-tt.mean())**2)
        sl=np.sum((tt-tt.mean())*(x-x.mean()))/den if den>0 else np.nan
    else: sl=np.nan
    return dict(mean=float(x.mean()),std=float(x.std()),rms=float(np.sqrt(np.mean(x*x))),
                median=float(q50),iqr=float(q75-q25),range=float(x.max()-x.min()),
                slope=float(sl) if np.isfinite(sl) else np.nan)

def emgstats(x):
    d=stats(x); x=np.asarray(x,float); x=x[np.isfinite(x)]
    d["mav"]=float(np.mean(np.abs(x))) if len(x) else np.nan
    d["waveform_length_norm"]=float(np.mean(np.abs(np.diff(x)))) if len(x)>1 else np.nan
    d["zero_cross_rate"]=float(np.mean((x[:-1]*x[1:])<0)) if len(x)>1 else np.nan
    return d

def add(row,prefix,d):
    for k,v in d.items(): row[f"wearable__{prefix}__{k}"]=v

def vec(row,prefix,a):
    for j,ax in enumerate(["x","y","z"]): add(row,f"{prefix}_{ax}",stats(a[:,j]))
    ok=np.all(np.isfinite(a),axis=1)
    mag=np.sqrt(np.sum(a[ok]**2,axis=1)) if ok.any() else np.array([])
    add(row,f"{prefix}_magnitude",stats(mag))

# ---------- Borg exact +10 s pairs ----------
b=pd.read_csv(BORG)
L={norm(c):c for c in b.columns}
sc,tc,dc,ec=L["subject"],L["task_order"],L["length_of_trial_(sec)"],L["end_of_trial"]
b[sc]=b[sc].ffill()
tcols={}
for c in b.columns:
    m=re.fullmatch(r"(\d+)_sec",norm(c))
    if m: tcols[int(m.group(1))]=c

prs=[]
for _,r in b.iterrows():
    sn=subjnum(r[sc]); task=str(r[tc]).strip()
    dur=float(pd.to_numeric(pd.Series([r[dc]]),errors="coerce").iloc[0])
    vals={}
    for t,c in tcols.items():
        v=pd.to_numeric(pd.Series([r[c]]),errors="coerce").iloc[0]
        if pd.notna(v): vals[float(t)]=float(v)
    ev=pd.to_numeric(pd.Series([r[ec]]),errors="coerce").iloc[0]
    if pd.notna(ev) and np.isclose(dur%10,0,atol=1e-9): vals[dur]=float(ev)
    folder,direction,load=TASK[task]
    for t in sorted(vals):
        if t>=10 and (t+10) in vals:
            prs.append(dict(row_id=f"S{sn:02d}__{task}__t{int(t)}",subject_num=sn,
                task_label=task,task_folder=folder,rotation_direction=direction,
                load_band_mid_pct_mvic=load,trial_duration_sec=dur,current_time_sec=t,
                target_time_sec=t+10,current_borg=vals[t],target_borg=vals[t+10],
                borg_change_10s=vals[t+10]-vals[t]))
pairs=pd.DataFrame(prs)
pairs.to_csv(PAIR_OUT,index=False)

# ---------- compact baseline context ----------
ctx=pd.DataFrame({"subject_num":sorted(pairs.subject_num.unique())})
extcols=[]

def merge_meta(path,prefix):
    global ctx,extcols
    if not path.exists(): return
    x=pd.read_csv(path); x.columns=[str(c).strip() for c in x.columns]
    subj=[c for c in x.columns if "subject" in safe(c) or "participant" in safe(c)]
    if subj:
        x[subj[0]]=x[subj[0]].ffill(); x["subject_num"]=x[subj[0]].apply(subjnum)
    elif len(x)==34:
        x["subject_num"]=np.arange(1,35)
    else: return
    keep=[]; ren={}
    for c in x.columns:
        if c=="subject_num" or c in subj: continue
        z=safe(c)
        if z in {"date","time"}: continue
        if path.name.lower()=="demographic.csv":
            allowed=z in {"group","age","sex","height","dominant_hand",
                          "what_kind_of_exercise_do_you_participate_in",
                          "how_often_do_you_exercise_per_week"}
        else:
            allowed=any(k in z for k in ["mass","bmi","body_fat","muscle","visceral"])
        if allowed:
            keep.append(c); ren[c]=f"context_ext__{prefix}__{z}"
    y=x[["subject_num"]+keep].drop_duplicates("subject_num").rename(columns=ren)
    ctx=ctx.merge(y,on="subject_num",how="left",validate="one_to_one")
    extcols += list(ren.values())

merge_meta(DEMO,"demo"); merge_meta(BODY,"body")

# ---------- sensor feature extraction ----------
start_rows=[]; end_rows=[]; groups=list(pairs.groupby(["subject_num","task_label"])); t0=time.time()

for gi,((sn,task),g) in enumerate(groups,1):
    folder,_,_=TASK[task]; td=SR/folder/f"Subject {sn}"; dur=float(g.trial_duration_sec.iloc[0])
    emg={}
    for fn in EMGS: emg[fn]=readnum(td/"EMG data"/fn)[:,0]
    imu={}
    for loc in LOCS:
        lf=safe(loc)
        for kind in ["acc","gyr"]:
            corrupt=(sn==3 and task=="task1_35i" and loc=="Shoulder")
            fp=td/"IMU data"/loc/f"{kind}_{lf}.csv"
            imu[(loc,kind)]=None if corrupt else readnum(fp)[:,:3]

    for _,pr in g.iterrows():
        t=float(pr.current_time_sec); ok=True; cache={"start":{},"end":{}}
        for fn,a in emg.items():
            s=b_start(t,EMG_HZ,len(a)); e=b_end(t,dur,EMG_HZ,len(a))
            cache["start"][("e",fn)]=s; cache["end"][("e",fn)]=e
            ok &= s is not None and e is not None
        for key,a in imu.items():
            if a is None: continue
            s=b_start(t,IMU_HZ,len(a)); e=b_end(t,dur,IMU_HZ,len(a))
            cache["start"][("i",key)]=s; cache["end"][("i",key)]=e
            ok &= s is not None and e is not None

        if not ok: continue

        base=dict(row_id=pr.row_id,subject_num=sn,task_label=task,
                  rotation_direction=pr.rotation_direction,
                  load_band_mid_pct_mvic=pr.load_band_mid_pct_mvic,
                  current_time_sec=t,current_borg=pr.current_borg,
                  target_time_sec=pr.target_time_sec,target_borg=pr.target_borg,
                  borg_change_10s=pr.borg_change_10s,trial_duration_sec_qc=dur,
                  known_corrupt_shoulder_imu_trial_qc=int(sn==3 and task=="task1_35i"))

        for anchor,dest in [("start",start_rows),("end",end_rows)]:
            row=dict(base)
            for fn,a in emg.items():
                aa,bb=cache[anchor][("e",fn)]
                add(row,"emg_"+safe(Path(fn).stem),emgstats(a[aa:bb]))
            for (loc,kind),a in imu.items():
                pref=f"imu_{safe(loc)}_{kind}"
                if a is None:
                    for ax in ["x","y","z","magnitude"]:
                        for st in ["mean","std","rms","median","iqr","range","slope"]:
                            row[f"wearable__{pref}_{ax}__{st}"]=np.nan
                else:
                    aa,bb=cache[anchor][("i",(loc,kind))]
                    vec(row,pref,a[aa:bb])
            dest.append(row)

    if gi%20==0 or gi==len(groups):
        print(f"Processed trials {gi}/{len(groups)}")

S=pd.DataFrame(start_rows).sort_values("row_id").reset_index(drop=True)
E=pd.DataFrame(end_rows).sort_values("row_id").reset_index(drop=True)
if list(S.row_id)!=list(E.row_id): raise RuntimeError("Start/end anchor row mismatch")
S=S.merge(ctx,on="subject_num",how="left",validate="many_to_one")
E=E.merge(ctx,on="subject_num",how="left",validate="many_to_one")
S.to_csv(START_OUT,index=False); E.to_csv(END_OUT,index=False)

# ---------- dictionary ----------
D=[]
for c in S.columns:
    if c in ["row_id","subject_num"]: role,fam="identifier","identifier"
    elif c=="target_borg": role,fam="primary_outcome","outcome"
    elif c=="borg_change_10s": role,fam="secondary_outcome","outcome"
    elif c in ["task_label","rotation_direction","load_band_mid_pct_mvic","current_time_sec","current_borg"]:
        role,fam="context_core","current_state_task_exposure"
    elif c.startswith("context_ext__"): role,fam="context_extended","baseline_demographic"
    elif c.startswith("wearable__emg_"): role,fam="wearable_feature","emg"
    elif c.startswith("wearable__imu_") and "_acc_" in c: role,fam="wearable_feature","imu_acceleration"
    elif c.startswith("wearable__imu_") and "_gyr_" in c: role,fam="wearable_feature","imu_gyroscope"
    elif c in ["target_time_sec","trial_duration_sec_qc","known_corrupt_shoulder_imu_trial_qc"]:
        role,fam="quality_control_only","quality_control"
    else: role,fam="other","other"
    D.append(dict(column=c,role=role,family=fam,
                  primary_predictor_eligible=int(role in {"context_core","context_extended","wearable_feature"})))
D=pd.DataFrame(D); D.to_csv(DICT_OUT,index=False)

# ---------- audit ----------
lines=[]; A=lines.append
A("DATASET C — DUAL-ANCHOR HANDCRAFTED FEATURE BUILD AUDIT")
A("="*118)
A(f"Candidate exact +10 s Borg pairs: {len(pairs)}")
A(f"Pairs retained under BOTH start/end sensor-window anchors: {len(S)}")
A(f"Pairs dropped by dual-anchor eligibility: {len(pairs)-len(S)}")
A(f"Participants retained: {S.subject_num.nunique()}")
A(f"Trials represented: {S[['subject_num','task_label']].drop_duplicates().shape[0]}")
A("")
A("1. FIXED DESIGN")
A("-"*118)
A(f"EMG nominal rate: {EMG_HZ} Hz")
A(f"IMU nominal rate: {IMU_HZ} Hz")
A(f"Prior wearable window: {WIN} s")
A("Primary wearable modalities: six-channel EMG + six-location IMU acceleration/gyroscope.")
A("PPG excluded from primary table because it was not hardware/firmware synchronized with EMG/IMU.")
A("Magnetometer excluded from primary table because of magnetic-field sensitivity and limited necessity.")
A("")
A("2. ALIGNMENT SENSITIVITY")
A("-"*118)
A("Start anchor: first sensor sample is mapped to exercise time 0.")
A("End anchor: last sensor sample is mapped to Borg trial end time.")
A("Only rows whose required 10-s EMG/IMU windows exist under BOTH anchors are retained.")
A("Thus both anchor analyses use identical Borg targets and participants.")
A("")
A("3. FORECAST TARGET")
A("-"*118)
A("At time t, use Borg(t) + task/load/time context + wearable samples assigned to [t-10,t].")
A("Target: Borg(t+10). No wearable samples from (t,t+10] are used.")
A("")
A("4. OUTCOME DISTRIBUTION")
A("-"*118)
for c in ["current_borg","target_borg","borg_change_10s"]:
    s=pd.to_numeric(S[c],errors="coerce")
    A(f"{c}: n={s.notna().sum()}, unique={s.nunique()}, min={s.min()}, median={s.median()}, mean={s.mean():.4f}, max={s.max()}")
A("")
A("5. FEATURE COUNTS")
A("-"*118)
for role in ["context_core","context_extended","wearable_feature"]:
    A(f"{role}: {int((D.role==role).sum())}")
for fam in ["emg","imu_acceleration","imu_gyroscope"]:
    A(f"{fam}: {int((D.family==fam).sum())}")
A("")
A("6. MISSINGNESS")
A("-"*118)
pred=D.loc[D.primary_predictor_eligible==1,"column"].tolist()
for lab,df in [("START",S),("END",E)]:
    m=df[pred].isna().mean().sort_values(ascending=False)
    A(f"[{lab}] >20% missing: {int((m>.2).sum())}/{len(m)}")
    A(f"[{lab}] 100% missing: {int((m>=1).sum())}/{len(m)}")
    for c,r in m.head(12).items(): A(f"  {c}: {r:.4f}")
A("")
A("7. KNOWN CORRUPTION")
A("-"*118)
cr=S[S.known_corrupt_shoulder_imu_trial_qc==1]
A(f"Subject 3 / task1_35i forecast rows retained: {len(cr)}")
A("Shoulder ACC/GYR features in those rows are NaN by design; all other modalities remain.")
A("Any imputation must be fit within training folds only.")
A("")
A("8. NEXT BENCHMARK")
A("-"*118)
A("Run identical repeated participant-grouped nested regression on START and END tables.")
A("C0a Core Context; C0b Core+Extended; C1 Wearable only; C2a Core+Wearable; C2b Extended+Wearable.")
A("Primary metrics: participant-balanced MAE/RMSE and pooled R^2.")
A("Primary quantity: paired incremental value of wearable over the corresponding context baseline.")
A("Call an increment alignment-robust only if its direction is consistent under both anchors.")
A("="*118)
A("END OF AUDIT")
AUDIT_OUT.write_text("\n".join(lines),encoding="utf-8")

print("Created:")
for fp in [START_OUT,END_OUT,DICT_OUT,PAIR_OUT,AUDIT_OUT]: print(" ",fp)
print(f"Elapsed minutes: {(time.time()-t0)/60:.2f}")
print("Upload dataset_C_handcrafted_build_audit.txt to ChatGPT.")
