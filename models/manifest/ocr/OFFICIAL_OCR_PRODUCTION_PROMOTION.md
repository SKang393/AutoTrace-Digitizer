<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Official PP-OCRv5 production promotion contract

## Current decision

The official PP-OCRv5 pair has completed its one authorized public synthetic
evaluation and failed the fixed production gates. There is no tracked OCR model
manifest, no approved benchmark report, and no OCR payload in the production
model store or a release package. The reviewed Apache-2.0 license and ONNX
change-notice bundle exists, but it has not been bound through a model-store
index or package. This is an intentional fail-closed state.

| Task | Model | Converted ONNX SHA-256 | Current status |
| --- | --- | --- | --- |
| OCR detection | `PP-OCRv5_mobile_det` | `d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb` | Conversion parity passed; public evaluation failed |
| OCR recognition | `en_PP-OCRv5_mobile_rec` | `7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743` | Conversion parity passed; public evaluation failed |

Both models must bind the same checksum-exact production gate report. Neither
model may be promoted by itself.

## Fixed evaluation result

The split was frozen before inference with 100 validation text cases, 100
sealed-test text cases, and 20 exclusions. No private image or Chandler data was
used. The split SHA-256 is
`1fc3b2e72f89cbfb0d8854ec8701368e7ae764cbd5c6fef17b7e497d06ec9f09`.
Its 532,313-byte fixture archive SHA-256 is
`69eeeff73f4cfd2dd6580ad9538f1a89527f8e5b320ce6a9cd7155d2bd22ea99`.
The post-run freeze verification passed.

The ignored report at
`ml/ocr/official_bakeoff/runs/production-gate/evaluation/report.json` has
SHA-256 `3fceb688d44a2f4c322e40bdfa3dd09ca74ec6d5742ab257ae102ccf76e1c44b`
and records `status=fail`, `production_approval=false`, and
`release_eligible=false`. Validation exact match was `0.0`, CER `1.0`, and role
accuracy `0.03`. Sealed-test exact match was `0.0`, CER `1.0`, and role accuracy
`0.02`. Detection exact-count rate was `18/220`, or
`0.08181818181818182`. The detector matched only 3 of 100 validation text cases
and 2 of 100 sealed text cases, produced 195 false regions across text cases,
and produced false regions on 7 of 20 exclusion fixtures. Independent composed
marker-stage evidence was not available, so `marker_creation_evaluated=false`.
ONNX parity remained passing at maximum absolute error
`2.205371856689453e-06`.

## Evidence that must exist before approval

The following evidence is conjunctive. A passing item cannot waive or replace
another item.

1. The public validation and inference-locked sealed splits are frozen before
   model execution. Every fixture byte is checksum-bound to its split record,
   and no private image or Chandler data is present.
2. The exact converted ONNX pair runs on the CPU provider over the frozen
   fixture bytes. Predictions identify the detector and recognizer SHA-256
   values, split SHA-256, evaluator SHA-256, provider, and per-case source
   SHA-256.
3. Validation and sealed-test exact match are each at least `0.90`; CER is at
   most `0.05`; role accuracy is at least `0.90`; detection exact-count rate is
   `1.0`; and marker creation from OCR text is zero.
4. At least 16 direct parity pairs per model establish ONNX maximum absolute
   error at most `1e-4`. The runtime report identifies CPU execution for both
   payloads.
5. The report embeds and hashes the reviewed evaluator, execution workflow,
   fixture ZIP, sealed split, direct core predictions, final predictions,
   runtime results, and independently composed marker-stage results required by
   `graphreader-ocr-public-gate-v1`. Reported aggregate metrics must equal
   metrics derived from those resources.
6. The reviewed Apache-2.0 license and ONNX change-notice files exist at
   `LICENSES/PaddlePaddle-PP-OCRv5-Models-Apache-2.0.txt` and
   `LICENSES/PaddlePaddle-PP-OCRv5-Models-Notice.txt`, with SHA-256 values
   `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
   and `8d81f5d0c58547cce471c24f82efe768a9d907d06764f67e90cc680c6d777729`.
   The notice paths must still be checksum-bound by the model-store index and
   verified in both packages. Repository-level or archive-name-only licensing
   is not enough.
7. Both schema-version 1 manifests declare the exact payload SHA-256, exact
   tensor and preprocessing contracts, commercial use, redistribution, CPU
   provider support, and one shared production approval report.
8. The ignored production model store contains only canonical indexed
   resources. Each manifest, ONNX payload, notice, and benchmark report passes
   checksum validation, and the resolver successfully discovers both tasks.
9. The Windows packaging audit discovers both OCR tasks, packages the same
   payload bytes and report, verifies notices and checksums, and still passes
   the forbidden-file scan. Installer and portable staging must resolve the
   same model identities.

The runtime gate now rehashes every embedded resource, opens the fixture ZIP,
matches every fixture byte to its split record, binds direct detector and
recognizer tensor identities, and requires runtime provenance for the frozen
execution workflow and CPU provider. It also requires independently composed
marker-stage results for every fixture. The current official report predates
that composed marker run and failed the accuracy and exclusion thresholds, so
it cannot satisfy the strengthened gate and must not be rewritten as passing.

## Fail-closed manifest transition

Do not add a tracked OCR manifest while the public gate inputs are missing.
After those inputs exist, an approval-false manifest may be used for local
evaluation only. It must bind the reviewed license and notice checksums above,
and it must contain no benchmark where all of these values are simultaneously
true:

```text
status = pass
release_eligible = true
production_approval = true
```

Promotion is one atomic evidence change for both tasks. Each final manifest
must contain exactly one such approval benchmark with:

```text
profile = graphreader-ocr-public-gate-v1
evidence_path
evidence_sha256
evaluator_source_sha256
sealed_split_sha256
predictions_sha256
runtime_results_sha256
sealed_test_exact_match
sealed_test_cer
onnx_max_abs_error
```

The two `evidence_sha256` values must be identical. Any model identity,
checksum, resource, provider, notice, threshold, store-resolution, or packaging
failure returns both tasks to unavailable. It must not emit release artifacts.

## Production store layout

The final ignored store is rooted at `artifacts/production-model-store`. Its
package index must resolve each resource beneath the canonical task identity
and version paths enforced by `ProductionModelStore`:

```text
manifest/<model-id>/<version>/manifest.json
runtime/<model-id>/<version>/<declared-onnx-file>
notices/<model-id>/<version>/<notice-file>
evidence/<model-id>/<version>/<shared-report-file>
```

The source-built conversion outputs remain ignored research artifacts until
the full transition above passes. A conversion report, payload hash, or CPU
smoke by itself is not production approval.
