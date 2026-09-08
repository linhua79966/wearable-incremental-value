# Portability

The public-release scripts resolve the repository root relative to each script:

```python
ROOT = Path(__file__).resolve().parents[2]
```

This replaces the private development-machine path used during the original analyses. No model settings, feature definitions, validation splits, seeds, outcome definitions, or reported numerical results were changed by this release-only portability edit.

The original third-party datasets are not redistributed in this repository. Users should obtain them from their respective source repositories/publications and place them in the locations documented in the final data-access instructions before reproducing the workflows.

Pretrained model weights are likewise not redistributed. The final release documentation records model identifiers, immutable revisions, package versions, and relevant SHA256 values.
