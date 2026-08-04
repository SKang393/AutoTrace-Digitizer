# Project numeric OCR V1

This directory contains a source-only preregistration for a project-trained
graph-numeric recognizer. It is distinct from the exhausted CTC, spatial V2, and
canonical-slot V3 experiments. Generated data and artifacts remain ignored.

Verify the frozen protocol, split fingerprints, and source binding:

```powershell
python -m ml.ocr.project_numeric_v1.verify_preregistration
```

The committed-main gate is intentionally expected to fail until the complete
preregistration is reviewed and committed:

```powershell
python -m ml.ocr.project_numeric_v1.verify_preregistration --require-committed
```

Candidate 1 completed training once, but its original process stopped in the
validation reporter. It must not retrain. After the evaluator recovery is
reviewed and committed, the exact retained checkpoint may be evaluated once:

```powershell
python -m ml.ocr.project_numeric_v1.train --resume-evaluation-only
```

Candidates 2 and 3 are reserved, not executable. No model is approved or
package eligible by this preregistration.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->
