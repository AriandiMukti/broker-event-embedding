# Publication validation

Validation date: 17 November 2026.

The sanitized source copy was tested with Python 3.10.2 on Windows using the direct dependency versions in `requirements-verified.txt`.

```text
python -m unittest discover -v
Ran 105 tests
OK (skipped=12)
```

93 tests passed. The 12 skipped checks require local input files, trained models, dashboards, or completed historical runs; those resources are intentionally excluded. A skip is not a successful real-data verification.

The executed tests cover synthetic data parsing, temporal alignment, train-only vocabulary/scalers, purged folds, causal convolutions, set invariance and padding, target masking, optimization, prediction replay, baseline controls, and uncertainty/statistical procedures. No real-data training run, historical performance table, or profitability result was reproduced for publication.

The source repository had no Git metadata or enclosing Git history. A fresh publication history is used. Original launch scripts, notes, logs, generated output, datasets, model files, and nested source snapshots are excluded; personal absolute paths in retained source are replaced with portable project-relative defaults. Legacy artifact-packaging scripts are excluded to avoid bundling private runs.

## Security audit

- The source inventory covered 31,470 files. Every inventoried file was hashed and scanned for credential, email, and personal-path patterns. Generated binary formats were conservatively excluded rather than treated as safe based on string scanning.
- Gitleaks v8.30.1 scanned the original directory. Its 32 matches were reviewed: 29 were SHA256 source-manifest values and three were duplicated source-code syntax. No confirmed credential was identified by those checks.
- The final publication allowlist contains 78 UTF-8 source/documentation files. Checks found no credential assignments, private absolute paths, email addresses, unexpected file types, or symlinks. The authorized public GitHub identity is used only in Git metadata.
- Gitleaks found zero findings in the sanitized publication files and zero findings in the initial remote history. The initial remote commit contained only a repository-title README; its email metadata used GitHub noreply addresses.
- Two tests containing literal observations from excluded real files and specific private-data observations in a legacy chart/test were removed. The final test result above is after those edits.

These are scoped verification results, not a guarantee that pattern detection can recognize every possible secret. No data-bearing artifact or known unresolved sensitive finding is approved for publication.

Installation into a fresh environment was not tested. The version snapshot records installed direct dependencies; the CUDA-specific PyTorch build may require its matching package index or a platform-appropriate CPU build. Historical real-data experiments and their numeric results remain unverified in the public release.
