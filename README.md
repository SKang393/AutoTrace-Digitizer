# Graph Auto Reader

<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

Graph Auto Reader is a Windows-native desktop application for extracting
editable, phase-labeled data from single-case design graphs.

It is designed for research figures containing small markers, multiple phases,
stacked panels, incomplete session labels, old scans, and degraded printing.

## Application type

Graph Auto Reader is a local WPF desktop application. It is not a browser,
Electron application, WebView shell, or cloud-hosted digitizer.

Core image processing, OCR, model inference, review, project saving, and export
run on the local Windows computer.

## Core workflow

1. Import an image, a batch of images, or a PDF article.
2. Separate multi-panel figures into tabs.
3. Enhance difficult scans locally at 2× when needed.
4. Detect axes, tick values, session spacing, marker centers, series, legends,
   annotations, and phase dividers.
5. Review and correct the editable overlay.
6. Export intervention-specific, phase-labeled CSV files.

## Planned capabilities

- immutable original-image preservation
- Real-ESRGAN-assisted 2× enhancement
- automatic and manual three-anchor calibration
- filled and open marker recognition
- marker shape and series separation
- solid, dashed, and dotted phase-divider detection
- baseline, intervention, maintenance, and generalization export
- shared-baseline handling for multiple intervention series
- PDF figure extraction and stacked-panel separation
- batch image tabs
- fixed side-panel magnifier
- light, dark, and system themes
- autosave and project recovery
- multilingual UI architecture
- CPU inference and supported Windows GPU acceleration

## Windows distributions

Each public release provides:

- a Windows installer;
- a Windows portable ZIP.

Both contain the same application version, required models, contracts, and
license notices. The portable build is extract-and-run and stores mutable data
inside its own `Data` folder.

## Scientific safeguards

Graph Auto Reader preserves the original image and records every crop, deskew,
scale, and model operation. Automated output remains editable. Low-confidence
results are surfaced for review rather than silently treated as final data.

## Export

The default per-series CSV contains:

```csv
x_value,y_value,phase
```

An optional audit export includes source pixel coordinates, marker identity,
confidence, session-value provenance, and review status.

## Privacy

Article PDFs, graph images, projects, and inference remain on the local computer
unless the user explicitly exports or shares them.

## License

Original Graph Auto Reader source code is licensed under the Apache License 2.0.

Copyright 2026 Sungwoo Kang.

Third-party libraries and models retain their respective licenses. Real-ESRGAN
components and official model files retain their applicable upstream notices,
including the BSD 3-Clause license where applicable.

## Repository layout

Production code is split into independently owned modules:

- `src/GraphReader.App`: Windows-native WPF application shell
- `src/GraphReader.Domain`: contracts and shared domain abstractions
- `src/GraphReader.Imaging`: raster transforms and cache
- `src/GraphReader.Pdf`: PDF and panel extraction
- `src/GraphReader.SuperResolution`: enhancement adapter
- `src/GraphReader.Axis`: axes, ticks, calibration, and session lattice
- `src/GraphReader.Ocr`: text detection and recognition
- `src/GraphReader.Markers`: marker detection and series grouping
- `src/GraphReader.Legends`: legends, annotations, and participant metadata
- `src/GraphReader.Phases`: phase dividers and semantics
- `src/GraphReader.Export`: CSV, audit, and save formats
- `src/GraphReader.Inference`: local inference providers and scheduling

Frozen schemas live under `contracts/`. Windows installer and portable
definitions share one publish stage under `packaging/`.

## Foundation build

The repository requires the .NET 10 SDK. From the repository root:

```powershell
dotnet restore
dotnet build -c Release
dotnet test -c Release
```

The current internal version is defined centrally in `Directory.Build.props`. A pre-release version is not evidence that
a public release or functional digitization workflow exists.
