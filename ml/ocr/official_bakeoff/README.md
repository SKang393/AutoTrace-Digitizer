# Official PP-OCRv5 archive audit

This tool verifies the pinned tag, commit tree, official documentation blobs,
documented archive URLs, exact archives, and extracted member hashes. Conversion
requires a structured `artifact-terms.json` that identifies the exact model,
uses an allowlisted SPDX license, affirmatively grants redistribution and
commercial use with boolean `true`, covers every archive file, and names a
reviewed `NOTICE` member with an exact SHA-256. That notice must identify the
same model, license, and grants and contain no proprietary, copyleft,
noncommercial, research-only, or forbidden-use contradiction. Legal keyword
presence alone never grants permission. After UTF-8 decoding, the complete set
of nonempty notice lines must exactly equal the four required scoped lines.
Additional prose, changed spelling, conditions, homoglyphs, and duplicate lines
fail closed. Conversion also requires exactly one audit with complete matching
metadata for every pinned candidate, with no missing, extra, or duplicate audit.
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
  --output ml/ocr/official_bakeoff/runs/archive-audit.json
```

Exit code `2` means provenance is blocked and conversion must not proceed.
