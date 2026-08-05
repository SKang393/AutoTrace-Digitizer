# Official PP-OCRv5 archive audit

This tool verifies the pinned tag, commit tree, official documentation blobs,
documented archive URLs, exact archives, and extracted member hashes. It accepts
artifact terms through either of two fully bound routes: exact structured terms
inside each archive, or the immutable official PaddlePaddle model repository
whose model card scopes Apache-2.0 to byte-identical inference files. The
repository route validates the exact owner, repository, revision, public and
ungated state, complete file inventory, model-card SHA-256, model identity,
license field, contradiction scan, revision-bound Git blob IDs, LFS pointer and
content identities, and all three BOS payload hashes for each model. The saved
API must include `?blobs=true`; malformed sibling records and any missing,
altered, extra, duplicated, gated, or mismatched evidence fail closed.

The archive-embedded route requires a structured `artifact-terms.json` that
identifies the exact model, uses an allowlisted SPDX license, affirmatively
grants redistribution and commercial use with boolean `true`, covers every
archive file, and names a reviewed `NOTICE` member with an exact SHA-256. Legal
keyword presence alone never grants permission. Conversion also requires
exactly one complete audit for every pinned candidate.
Every reviewed JSON input rejects duplicate object keys at any depth, including
artifact terms and saved tag and commit responses. Archive auditing also rejects
duplicate normalized file paths before reading payload bytes or constructing the
member inventory.
Because the payload targets Windows, every file member must use canonical POSIX
spelling and Windows-safe segments. Absolute paths, backslashes, repeated
separators, dot segments, trailing dots or spaces, alternate data stream colons,
invalid Windows characters, and reserved device names are rejected. Duplicate
identity uses a case-insensitive canonical key.
The pinned archive SHA-256 must match before `tarfile.open` runs. The member
preflight covers directories and files together, rejects duplicate identity
across all members, and permits only regular files and canonical safe
directories. Links, devices, FIFOs, sparse or contiguous entries, and unknown
types are rejected before any payload read.
Any nonempty `TarInfo.sparse` map is rejected independently of the type byte, so
PAX GNU sparse metadata cannot disguise a sparse payload as regular type `b'0'`.

Downloaded archives, extracted payloads, and generated JSON reports stay under
ignored `runs/`.

```powershell
python -m ml.ocr.official_bakeoff.audit_archives `
  --archives ml/ocr/official_bakeoff/runs/archives `
  --source ml/ocr/official_bakeoff/runs/source `
  --model-license-evidence ml/ocr/official_bakeoff/runs/huggingface `
  --output ml/ocr/official_bakeoff/runs/archive-audit.json
```

Exit code `0` means these source bytes are license-cleared for conversion only.
It does not approve an ONNX conversion, benchmark, manifest, runtime provider,
notice bundle, packaging discovery, or production composition. Exit code `2`
means provenance is blocked and conversion must not proceed.
