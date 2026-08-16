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

## Locked Windows conversion and parity gate

The converter runs only from the ignored project-local CPython 3.11.9
toolchain. `requirements-conversion.txt` binds every installed distribution by
version and SHA-256. The selected Paddle nightly is the exact minimum Windows
build accepted by Paddle2ONNX 2.0.2rc3. The three rejected stable-toolchain
attempts and the selected-toolchain intake are recorded in ignored local
evidence under `runs/toolchain/`.

Create a clean environment and install it offline from the audited wheelhouse:

```powershell
& artifacts/toolchains/python-3.11.9/python.exe -m venv `
  artifacts/toolchains/ppocr-conversion-locked-py311

& artifacts/toolchains/ppocr-conversion-locked-py311/Scripts/python.exe -m pip install `
  --no-index `
  --find-links ml/ocr/official_bakeoff/runs/toolchain/wheelhouse `
  --require-hashes `
  --requirement ml/ocr/official_bakeoff/requirements-conversion.txt
```

Convert both exact audited source directories twice, require byte-identical
ONNX outputs, and compare 16 deterministic raw tensor inputs per model between
Paddle CPU and ONNX Runtime CPU:

```powershell
& artifacts/toolchains/ppocr-conversion-locked-py311/Scripts/python.exe `
  -m ml.ocr.official_bakeoff.convert_models `
  --audit ml/ocr/official_bakeoff/runs/archive-audit.json `
  --source ml/ocr/official_bakeoff/runs/extracted `
  --output ml/ocr/official_bakeoff/runs/conversion `
  --report ml/ocr/official_bakeoff/runs/conversion/report.json `
  --toolchain-root artifacts/toolchains/ppocr-conversion-locked-py311 `
  --converter artifacts/toolchains/ppocr-conversion-locked-py311/Scripts/paddle2onnx.exe `
  --wheelhouse ml/ocr/official_bakeoff/runs/toolchain/wheelhouse `
  --toolchain-intake ml/ocr/official_bakeoff/runs/toolchain/INTAKE.json `
  --python-installer artifacts/toolchains/downloads/python-3.11.9-amd64.exe
```

The gate verifies the upstream audit, exact extracted inventory, signed Python
installer intake, venv and converter hashes, all 27 locked distributions, exact
bootstrap versions, every installed `RECORD` file, one selected wheel per lock
entry, and the absence of unexpected packages. It also verifies the CPU
provider, two independent converter runs, byte reproducibility, reviewed exact
converter-warning profiles, ONNX full check, exact opset 11, no external tensor
payloads, representative dynamic input shapes, and maximum raw tensor
difference of `1e-4`. It deletes a stale model output before each conversion so
an earlier ONNX file cannot satisfy a failed run. Exit code `0` remains
conversion-only evidence.

The report explicitly keeps `production_approved` and `release_ready` false.
No model manifest, notice bundle, production-model-store entry, or application
package is created by this command.

## Frozen public and sealed production evaluation

`PRODUCTION_GATE_PROTOCOL.json` freezes the public synthetic renderer,
degradations, validation and sealed-test counts, thresholds, and one-evaluation
budget before model inference. Generated images, predictions, and reports stay
under ignored `runs/production-gate/`. Chandler and all private images are
forbidden.

The ambiguity family renders common `O/0`, `l/1`, and `I/1` confusions in
`display_text` while `truth_text` stores the normalized numeric meaning. This is
intentional scientific truth, not a model-specific accommodation. The current
C# `ProductionOcrApprovalGate.TruthMatchesFamily` cannot represent this
separation and therefore remains a mandatory production blocker. The evaluator
records that blocker and cannot emit a production pass until the reviewed gate
is corrected.

Freeze once, then verify the exact bytes without regenerating them:

```powershell
& artifacts/toolchains/ppocr-conversion-locked-py311/Scripts/python.exe `
  -m ml.ocr.official_bakeoff.production_evaluate freeze `
  --output-root ml/ocr/official_bakeoff/runs/production-gate/frozen

& artifacts/toolchains/ppocr-conversion-locked-py311/Scripts/python.exe `
  -m ml.ocr.official_bakeoff.production_evaluate verify-freeze `
  --frozen-root ml/ocr/official_bakeoff/runs/production-gate/frozen
```

Run the official detector and recognizer once against that fixed split:

```powershell
& artifacts/toolchains/ppocr-conversion-locked-py311/Scripts/python.exe `
  -m ml.ocr.official_bakeoff.production_evaluate evaluate `
  --frozen-root ml/ocr/official_bakeoff/runs/production-gate/frozen `
  --conversion-report ml/ocr/official_bakeoff/runs/conversion/report.json `
  --source-root ml/ocr/official_bakeoff/runs/extracted `
  --output-root ml/ocr/official_bakeoff/runs/production-gate/evaluation
```

The default evaluation remains fail-closed because it has no checksum-bound
downstream marker-creation evidence. A later integrated CPU workflow may supply
`--marker-evidence` using schema
`graphreader.ocr-marker-creation-results.v1`. The command still fails when any
exact match, CER, role, detection, parity, marker, split-integrity, or C#
contract gate is missing.

## Preregistered structure-consensus composition

The first official-model evaluation used a dense connected-component
postprocessor rather than the production DB postprocessor and exposed the
complete split. It remains a recorded failed experiment and must not be rerun,
tuned, or rewritten as production evidence.

`STRUCTURE_CONSENSUS_GATE_PROTOCOL.json` preregisters one distinct production
composition before any new fixture generation or model inference. The exact
DB detector must agree with an independent connected-component candidate that
carries explicit non-structure evidence. A candidate can filter a model box but
can never create or replace one. Greedy one-to-one matching also prevents
duplicate model boxes from sharing one candidate. Axis, tick, and divider
masking is checksum-bound, while recognition crops continue to use immutable
original bytes.

The new protocol forbids the exposed 220-case split and Chandler, freezes a new
public validation and inference-locked sealed split, retains the original OCR
quality thresholds, adds zero-duplicate and zero-exclusion-hit requirements,
and permits exactly one evaluation. The evaluator and C# approval gate were
frozen and checksum-bound before any model execution. They rederive the
one-to-one structure consensus, truth matching, duplicate counts, source BGR
pixels, and axis/tick/divider-masked detector BGR pixels from embedded fixture
bytes and frozen mask geometry. The workflow now matches the production
vertical-glyph grouping and rejects every embedded resource above the C# gate's
8 MiB limit before report creation. It no longer accepts a precomputed marker
result file. Because the three-candidate marker-center budget is exhausted and
the selected P3 candidate remains rejected, the official OCR run must record
`marker_creation_evaluated=false` and cannot approve even if its OCR metrics
pass.

The authoritative 500-case split was frozen once before inference at SHA-256
`8685a3dfcb8212f612115c20d0f70437e0738fa1c4d86743cfd0e50bc5a41a8d`.
Its 1,732,541-byte fixture archive has SHA-256
`6bd83d6b05918a6829a99e4cfecbb5af0e0629c458242398429856baae5e02b2`,
and post-freeze verification passed. `requirements-structure-consensus.txt`
extends the exact conversion lock with evaluation-only
`opencv-python-headless` `4.10.0.84`; the reviewed wheel SHA-256 is
`afcf28bd1209dd58810d33defb622b325d3cbe49dcd7a43a902982c33e5fad05`.

The single authorized official composition execution then failed closed on the
first detector inference because the exact detector output violated the frozen
probability-tensor range contract. The evaluator emitted
`BLOCKED: Detector output is not a probability tensor.` before creating its
output root or report. Exact execution-source commit
`7fa6abee5deaf7c17ad19169928290b96a65ce2a` preserves every reviewed source
hash used by that consumed attempt.
The consumed run must not be rerun, repaired, or tuned against these exposed
fixtures. The ignored failure record has SHA-256
`b14dd36632224254933fad9c826ab80298e25371a0e32040bcab66a4684fd4e0`.
Production approval, manifests, model-store promotion, package discovery, and
release authorization therefore remain false.

The production runtime now also supports a separately selected
`probability_with_1e-5_clamp` activation for future preregistered compositions.
It clamps only finite output drift within `1e-5` of `[0,1]`, records the fixed
tolerance in cache and request provenance, and rejects non-finite or larger
drift. The original strict `probability` activation is unchanged. This repair
does not alter, reopen, or approve the consumed 500-case attempt. A new run
requires a disjoint split, a new frozen protocol, and a new one-run seal.

## Preregistered bounded-probability composition V2

`STRUCTURE_CONSENSUS_V2_GATE_PROTOCOL.json` freezes the distinct
`graph-structure-consensus-bounded-v2` defect class before any V2 fixture
generation or official-model inference. Protocol SHA-256 is
`0ee2ec0ef4a9f2f7f7f373da7389b84513f254c8642e4ddd5fd5427518d5e133`;
workflow SHA-256 is
`b2a52e925f9d06714d7a346ac929c60c8a81cf101893147657011e70325a18d7`.
The frozen workflow retains the exact official model bytes, BGR preprocessing,
DB postprocessing, structure consensus, thresholds, and CPU provider. Its only
candidate change is the separately selected fixed `1e-5` probability-boundary
activation.

The protocol binds both exposed predecessor splits and their complete source
hash inventories. V2 uses new case IDs and render index offset `100003`; its
single freeze must prove zero prior source-hash and case-ID overlap.
Fixture generation remains at zero until this preregistration is committed.
Official evaluation remains at zero until a separately committed post-freeze
`STRUCTURE_CONSENSUS_V2_EVALUATION_CONFIG.json` binds the exact split, archive,
source bundle, model pair, gate configuration, and canonical one-run seal.
Chandler and every private image remain prohibited.
