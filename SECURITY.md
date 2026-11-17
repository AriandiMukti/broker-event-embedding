# Publication security and data handling

This repository is a sanitized source-only export with fresh Git history. The original working directory is preserved. Its raw datasets, frozen snapshots, database files, training runs, weights, caches, logs, generated reports, local launch scripts, and original research notes are excluded.

Publication review covers the source inventory, credential/PII/path pattern scanning, reviewed source changes, and the final Git object inventory. Automated scans have detection limits and cannot prove that arbitrary confidential information is absent. Only allowlisted text source and reviewed documentation are eligible for publication.

Do not commit private datasets, account sessions, credentials, model artifacts, or generated research results without a separate review. Output configurations and source fingerprints can disclose filesystem paths. Load only checkpoints and pickle files from trusted sources: some serialization formats can execute code.

The ignore rules reduce accidental additions; they do not remove files already tracked. Review `git diff --cached` and `git ls-files` before every publication. If credentials are ever exposed, revoke/rotate them immediately and remove them from all reachable Git history.
