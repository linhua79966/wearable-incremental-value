# Wearable Incremental Value in Repetitive Work

Analysis and reproducibility code for the manuscript:

**Incremental predictive value of wearable sensing beyond worker and task context in repetitive work**

Release candidate: **v1.0.0**

## Study question

The project evaluates whether wearable sensing provides **incremental predictive value** beyond a specified decision-time baseline containing worker-state, task, exposure, and/or other context available at prediction time.

The repository is organized around three secondary-analysis datasets:

- **Dataset A** — final physical-fatigue regression and a supporting outcome-residualized sensor-family analysis.
- **Dataset B** — next-interval fatigue-onset prediction across an ordered hierarchy of decision-time baselines.
- **Dataset C** — exact +10-s Borg forecasting with paired temporal-alignment analysis and frozen IMU-representation robustness analyses.

The scientific target is the performance increment from **baseline** to **baseline + wearable information**, under participant-separated held-out evaluation. Wearable-only predictability is not treated as equivalent to incremental system value.

## Repository layout

```text
dataset_A/
  scripts/
dataset_B/
  scripts/
dataset_C/
  scripts/
data/
pretrained/
reproducibility/
  environments/
results/
figures/
configs/
docs/
```

## Data policy

This repository does **not** redistribute the original third-party raw datasets.

Users must obtain Dataset A, Dataset B, and Dataset C from their respective original repositories/publications. The final public release will provide exact source citations and access information in `data/README.md`.

The public release also excludes:
- row-level participant prediction tables;
- participant/worker/subject identifiers;
- raw or reconstructed wearable waveforms;
- derived embedding NPZ files;
- pretrained model checkpoints.

A separate Zenodo record is being prepared for author-generated aggregate/fold-level results, feature metadata, and reproducibility audits.

## Environments

Three release-preparation environment snapshots are stored under:

```text
reproducibility/environments/
```

Direct dependency snapshots are also provided as:

```text
requirements-core.txt
requirements-normwear.txt
requirements-moment.txt
```

These document the current reproducibility environments used during release preparation. They must not be interpreted as evidence that every original analysis or audit was historically executed under exactly those versions unless separately documented.

## Portability

Public-release scripts resolve the repository root relative to the script location:

```python
ROOT = Path(__file__).resolve().parents[2]
```

The original private development-machine path has been removed from the staged code. This release-only portability change does not alter model settings, feature definitions, participant splits, random seeds, outcome definitions, or reported scientific results.

See `PORTABILITY.md`.

## Suggested workflow

The script numbering preserves the analytical chronology.

### Dataset A

Core/final analysis sequence:

```text
06b -> 07 -> 08
```

Supporting QC and numerical-reproducibility scripts are also provided.

### Dataset B

Core/final analysis sequence:

```text
12 -> 13
```

Temporal-semantics and convergence audits are provided alongside the main scripts.

### Dataset C — handcrafted

Core/final analysis sequence:

```text
15b -> 16 -> 17 -> 18c -> 18d -> 18e -> 18f
```

### Dataset C — NormWear robustness

```text
19 -> 23 -> 24 -> 25 / 26 / 27
```

Scripts 27b and 27c document numerical convergence/recovery checks.

### Dataset C — MOMENT-1-base robustness

```text
30 -> 31 -> 32 -> 33
```

Scripts 33b–33d document convergence and targeted recovery audits.

## Pretrained model policy

Pretrained weights are **not redistributed**.

The final release documentation records the pretrained model identifier, immutable revision, relevant package version, extraction policy, and model/output SHA256 values needed to identify the exact evaluated representation.

## Reproducibility boundary

The repository supports reproduction of the reported analysis workflow once the source datasets are obtained under their original terms.

The released code does not imply that:
- the source datasets may be redistributed by downstream users;
- frozen pretrained representations exhaust all possible representation choices;
- the reported participant-cluster bootstrap intervals represent external-validation uncertainty;
- predictive increment is a causal or deployment-utility estimate.

## License

Author-generated code in this repository is released under the **MIT License**.

Licensing of the original datasets, pretrained models, and third-party software remains governed by their respective source licenses.

## Citation

`CITATION.cff.template` is included during pre-publication preparation. It will be converted to the final `CITATION.cff` after the ordered author list, GitHub URL, and Zenodo software DOI are finalized.

## Release status

**Do not cite this staging directory yet.**
The immutable public software DOI will be assigned after the GitHub v1.0.0 release is archived in Zenodo.
