// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Services;
using System.IO;

namespace GraphReader.App.Integration;

public static class ProductionStageAvailabilityRegistry
{
    public static IReadOnlyList<AutomaticStageStatus> Current { get; } = Create(
        localEnhancementConfigured: false,
        modelRoot: null,
        reviewedPdfiumConfigured: false);

    public static IReadOnlyList<AutomaticStageStatus> Create(
        bool localEnhancementConfigured,
        string? modelRoot = null,
        bool reviewedPdfiumConfigured = false)
    {
        bool packageIndexPresent = !string.IsNullOrWhiteSpace(modelRoot) &&
            File.Exists(Path.Combine(modelRoot, "production-model-index.json"));
        string modelEvidence = packageIndexPresent
            ? "A package index exists, but no checksum-resolved approved default model set is composed."
            : "No production-model-index.json is installed in the application model root.";
        string pdfEvidence = reviewedPdfiumConfigured
            ? "A checksum-bound reviewed PDFium renderer is configured for PDF import."
            : "No checksum-bound reviewed PDFium approval is configured; scanned-PDF rendering remains unavailable.";
        return
        [
        localEnhancementConfigured
            ? new AutomaticStageStatus(
                "enhancement",
                AutomaticStageState.Experimental,
                "Official realesr-animevideov3 x2 is configured for local evaluation only. Runtime and model checksums are verified before use; public redistribution is not approved.")
            : new AutomaticStageStatus(
                "enhancement",
                AutomaticStageState.Unavailable,
                "No approved Real-ESRGAN runtime and model payload are installed. The original image remains editable."),
        new(
            "axis",
            AutomaticStageState.Unavailable,
            "The deterministic axis adapter is implemented, but its OpenCvSharp native runtime remains blocked by the linked-library provenance audit."),
        new(
            "ocr",
            AutomaticStageState.Unavailable,
            $"OCR detection, text recognition, and graph-numeric recognition do not have a composed checksum-approved default set. {modelEvidence}"),
        new(
            "markers",
            AutomaticStageState.Unavailable,
            $"Marker detection has no composed checksum-approved center and shape/fill model set. {modelEvidence}"),
        new(
            "legends",
            AutomaticStageState.Unavailable,
            $"Legend reasoning requires approved OCR and marker evidence. {pdfEvidence}"),
        new(
            "phases",
            AutomaticStageState.Unavailable,
            "Phase reasoning is implemented but requires approved axis, OCR, and marker evidence."),
        ];
    }
}
