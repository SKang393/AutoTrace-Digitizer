# Marker scientific gate seals

Each future public or confirmation evaluation creates exactly one canonical
directory keyed by task, gate revision, and candidate hashes. The immutable
opened record additionally binds the frozen split-manifest hash,
gate-configuration hash, and evaluator-source bundle hash.

`opened.json` is created atomically before inference. Its existence rejects the
same candidate/revision pair regardless of the requested report output path.
`result.json` is then created atomically and binds the report hash. Both files
are release evidence and must be committed with the evaluator and split
configuration that produced them.

Historical 2026-08-04 evaluations predate this mechanism. They remain failed,
unsealed evidence and are explicitly retired from replay. They must never be
upgraded into canonical evidence retroactively.
