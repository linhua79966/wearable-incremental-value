
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived" / "dataset_C_phase1"
RESULTS = ROOT / "results" / "dataset_C_phase1_dual_anchor_nestedcv"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

START = DERIVED / "dataset_C_handcrafted_start_anchor.csv"
END = DERIVED / "dataset_C_handcrafted_end_anchor.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

TXT_OUT = AUDIT / "dataset_C_elasticnet_numerical_diagnostic.txt"
CSV_OUT = AUDIT / "dataset_C_elasticnet_problematic_features.csv"

GROUP = "subject_num"
BASE_SEED = 20260901
OUTER_REPEATS = 5
OUTER_FOLDS = 5

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
    out = []
    for gs in fold_groups:
        mask = groups.isin(gs).to_numpy()
        out.append((idx[~mask], idx[mask]))
    return out

start = pd.read_csv(START).sort_values("row_id").reset_index(drop=True)
end = pd.read_csv(END).sort_values("row_id").reset_index(drop=True)
fd = pd.read_csv(DICT)

wear = fd.loc[fd["role"] == "wearable_feature", "column"].tolist()
wear = [c for c in wear if c in start.columns]

records = []

for anchor, df in [("START", start), ("END", end)]:
    # raw global non-finite check
    for c in wear:
        x = pd.to_numeric(df[c], errors="coerce")
        records.append({
            "anchor": anchor,
            "repeat": 0,
            "outer_fold": 0,
            "feature": c,
            "train_std": np.nan,
            "test_max_abs_z": np.nan,
            "test_max_abs_raw": float(np.nanmax(np.abs(x.to_numpy(dtype=float)))) if x.notna().any() else np.nan,
            "nonfinite_count": int((~np.isfinite(x.to_numpy(dtype=float))).sum()),
            "diagnostic_type": "global_raw",
        })

    for rep in range(OUTER_REPEATS):
        seed = BASE_SEED + rep * 10000
        splits = shuffled_group_folds(df[GROUP], OUTER_FOLDS, seed)

        for fold, (tr, te) in enumerate(splits, 1):
            trdf = df.iloc[tr]
            tedf = df.iloc[te]

            for c in wear:
                xtr = pd.to_numeric(trdf[c], errors="coerce").to_numpy(dtype=float)
                xte = pd.to_numeric(tedf[c], errors="coerce").to_numpy(dtype=float)

                finite_tr = xtr[np.isfinite(xtr)]
                if len(finite_tr) == 0:
                    med = 0.0
                    sd = 0.0
                else:
                    med = float(np.median(finite_tr))
                    xtr_imp = np.where(np.isfinite(xtr), xtr, med)
                    sd = float(np.std(xtr_imp, ddof=0))

                xte_imp = np.where(np.isfinite(xte), xte, med)

                if sd > 0 and np.isfinite(sd):
                    z = (xte_imp - np.mean(np.where(np.isfinite(xtr), xtr, med))) / sd
                    maxz = float(np.max(np.abs(z))) if len(z) else np.nan
                else:
                    maxz = np.nan

                records.append({
                    "anchor": anchor,
                    "repeat": rep + 1,
                    "outer_fold": fold,
                    "feature": c,
                    "train_std": sd,
                    "test_max_abs_z": maxz,
                    "test_max_abs_raw": float(np.max(np.abs(xte_imp))) if len(xte_imp) else np.nan,
                    "nonfinite_count": int((~np.isfinite(xte)).sum()),
                    "diagnostic_type": "outer_fold_scaling",
                })

rec = pd.DataFrame(records)

fold_rec = rec[rec["diagnostic_type"] == "outer_fold_scaling"].copy()
fold_rec["tiny_train_std"] = (
    np.isfinite(fold_rec["train_std"])
    & (fold_rec["train_std"] > 0)
    & (fold_rec["train_std"] < 1e-10)
)

# Rank primarily by maximum held-out z-score.
rank = (
    fold_rec.groupby(["anchor", "feature"], as_index=False)
    .agg(
        max_test_abs_z=("test_max_abs_z", "max"),
        median_test_abs_z=("test_max_abs_z", "median"),
        min_train_std=("train_std", "min"),
        max_test_abs_raw=("test_max_abs_raw", "max"),
        tiny_std_fold_count=("tiny_train_std", "sum"),
        nonfinite_test_count=("nonfinite_count", "sum"),
    )
    .sort_values(["anchor", "max_test_abs_z"], ascending=[True, False])
)

rank.to_csv(CSV_OUT, index=False)

lines = []
A = lines.append
A("DATASET C — ELASTICNET NUMERICAL-STABILITY DIAGNOSTIC")
A("=" * 118)
A(f"Wearable features inspected: {len(wear)}")
A(f"Rows per anchor: {len(start)}")
A(f"Participants: {start[GROUP].nunique()}")
A("")
A("Purpose:")
A("  Diagnose the catastrophic START-anchor ElasticNet predictions observed in benchmark 18.")
A("  This script does NOT tune or refit outcome models.")
A("  It only reproduces outer participant splits and audits training-fold feature scale.")
A("")

for anchor in ["START", "END"]:
    z = rank[rank["anchor"] == anchor]
    A(f"1. {anchor} — TOP HELD-OUT STANDARDIZED MAGNITUDES")
    A("-" * 118)
    A(z.head(30).to_string(index=False))
    A("")

    for th in [10, 100, 1000, 1e4, 1e6, 1e10]:
        n = int((z["max_test_abs_z"] > th).sum())
        A(f"Features with max held-out |z| > {th:g}: {n}/{len(z)}")
    A(f"Features with any tiny training std (<1e-10): {int((z['tiny_std_fold_count']>0).sum())}/{len(z)}")
    A("")

A("2. CROSS-ANCHOR COMPARISON")
A("-" * 118)
s = rank[rank.anchor=="START"][["feature","max_test_abs_z","min_train_std"]].rename(
    columns={"max_test_abs_z":"start_max_z","min_train_std":"start_min_std"})
e = rank[rank.anchor=="END"][["feature","max_test_abs_z","min_train_std"]].rename(
    columns={"max_test_abs_z":"end_max_z","min_train_std":"end_min_std"})
m = s.merge(e,on="feature",how="outer")
m["z_ratio_start_over_end"] = m["start_max_z"] / m["end_max_z"].replace(0,np.nan)
m = m.sort_values("z_ratio_start_over_end",ascending=False)
A(m.head(40).to_string(index=False))
A("")

A("3. DECISION RULE")
A("-" * 118)
A("If START contains very large held-out standardized values while END does not,")
A("the ElasticNet failure is a numerical/extrapolation instability caused by fold-specific")
A("near-zero variance or extreme held-out values, not a scientifically meaningful negative increment.")
A("")
A("In that case, benchmark 18 START-ElasticNet must be marked invalid and replaced by a")
A("pre-specified stability-safe linear baseline (e.g., foldwise robust scaling plus near-zero-variance")
A("filtering) applied identically to START and END, without altering HGB results.")
A("")
A("=" * 118)
A("END OF DIAGNOSTIC")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Numerical diagnostic complete.")
print(f"  {TXT_OUT}")
print(f"  {CSV_OUT}")
print("")
print("Upload dataset_C_elasticnet_numerical_diagnostic.txt to ChatGPT.")
