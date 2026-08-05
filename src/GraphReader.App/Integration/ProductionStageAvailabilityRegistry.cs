// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Services;

namespace GraphReader.App.Integration;

public static class ProductionStageAvailabilityRegistry
{
    public static IReadOnlyList<AutomaticStageStatus> Current { get; } = Create(
        localEnhancementConfigured: false,
        modelAvailability: null,
        reviewedPdfiumConfigured: false);

    public static IReadOnlyList<AutomaticStageStatus> Create(
        bool localEnhancementConfigured,
        ProductionModelAvailabilitySnapshot? modelAvailability = null,
        bool reviewedPdfiumConfigured = false)
    {
        modelAvailability ??= ProductionModelAvailabilitySnapshot.Missing(
            "No production-model-index.json is installed in the application model root.");
        IReadOnlySet<string> approvedTasks = modelAvailability.ApprovedCpuTasks;
        bool ocrApproved = HasTasks(approvedTasks, "ocr_detection", "ocr_recognition");
        bool markersApproved = HasTasks(approvedTasks, "marker_center", "marker_classifier");
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
            "The deterministic axis adapter and reviewed source-built OpenCV provenance exist, but mandatory clean-machine runtime approval is not installed."),
        ocrApproved
            ? new AutomaticStageStatus(
                "ocr",
                AutomaticStageState.Approved,
                $"OCR detection and recognition have checksum-resolved CPU-approved payloads. {modelAvailability.Evidence}")
            : new AutomaticStageStatus(
                "ocr",
                AutomaticStageState.Unavailable,
                $"OCR requires checksum-resolved CPU-approved ocr_detection and ocr_recognition payloads. Missing: {MissingTasks(approvedTasks, "ocr_detection", "ocr_recognition")}. {modelAvailability.Evidence}"),
        markersApproved
            ? new AutomaticStageStatus(
                "markers",
                AutomaticStageState.Approved,
                $"Marker center and shape/fill classification have checksum-resolved CPU-approved payloads. {modelAvailability.Evidence}")
            : new AutomaticStageStatus(
                "markers",
                AutomaticStageState.Unavailable,
                $"Markers require checksum-resolved CPU-approved marker_center and marker_classifier payloads. Missing: {MissingTasks(approvedTasks, "marker_center", "marker_classifier")}. {modelAvailability.Evidence}"),
        new(
            "legends",
            AutomaticStageState.Unavailable,
            ocrApproved && markersApproved
                ? $"OCR and marker model evidence is approved, but no production legend detection adapter is composed. {pdfEvidence}"
                : $"Legend reasoning requires approved OCR and marker evidence. {pdfEvidence}"),
        new(
            "phases",
            AutomaticStageState.Unavailable,
            ocrApproved && markersApproved
                ? "OCR and marker model evidence is approved, but no production phase detection adapter is composed and axis clean-machine approval is absent."
                : "Phase reasoning is implemented but requires approved axis, OCR, and marker evidence."),
        ];
    }

    private static bool HasTasks(IReadOnlySet<string> approvedTasks, params string[] requiredTasks) =>
        requiredTasks.All(approvedTasks.Contains);

    private static string MissingTasks(IReadOnlySet<string> approvedTasks, params string[] requiredTasks) =>
        string.Join(", ", requiredTasks.Where(task => !approvedTasks.Contains(task)));
}
