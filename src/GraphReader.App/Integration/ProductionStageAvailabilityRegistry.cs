// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Services;
using System.Collections.ObjectModel;

namespace GraphReader.App.Integration;

public sealed class ProductionDetectionAdapterAvailabilitySnapshot
{
    private static readonly HashSet<string> AllowedStages = new(
        ["axis", "ocr", "markers", "legends", "phases"],
        StringComparer.Ordinal);

    public ProductionDetectionAdapterAvailabilitySnapshot(
        IEnumerable<string> approvedStages,
        string evidence)
    {
        ArgumentNullException.ThrowIfNull(approvedStages);
        ArgumentException.ThrowIfNullOrWhiteSpace(evidence);
        var stages = new HashSet<string>(approvedStages, StringComparer.Ordinal);
        if (!stages.IsSubsetOf(AllowedStages))
        {
            throw new ArgumentException(
                "Production detection adapter stages must use registered stage IDs.",
                nameof(approvedStages));
        }

        ApprovedStages = new ReadOnlySet<string>(stages);
        Evidence = evidence;
    }

    public IReadOnlySet<string> ApprovedStages { get; }

    public string Evidence { get; }

    public static ProductionDetectionAdapterAvailabilitySnapshot Missing(string evidence) =>
        new([], evidence);
}

public static class ProductionStageAvailabilityRegistry
{
    public static IReadOnlyList<AutomaticStageStatus> Current { get; } = Create(
        localEnhancementConfigured: false,
        modelAvailability: null,
        reviewedPdfiumConfigured: false);

    public static IReadOnlyList<AutomaticStageStatus> Create(
        bool localEnhancementConfigured,
        ProductionModelAvailabilitySnapshot? modelAvailability = null,
        bool reviewedPdfiumConfigured = false,
        ProductionRuntimeAvailabilitySnapshot? runtimeAvailability = null,
        bool inferenceRuntimeConfigured = true,
        ProductionDetectionAdapterAvailabilitySnapshot? adapterAvailability = null)
    {
        modelAvailability ??= ProductionModelAvailabilitySnapshot.Missing(
            "No production-model-index.json is installed in the application model root.");
        runtimeAvailability ??= ProductionRuntimeAvailabilitySnapshot.Missing(
            "No release-approved OpenCV runtime evidence is installed in the application root.");
        adapterAvailability ??= ProductionDetectionAdapterAvailabilitySnapshot.Missing(
            "No approved production detection adapter is composed.");
        IReadOnlySet<string> approvedTasks = modelAvailability.ApprovedCpuTasks;
        IReadOnlySet<string> approvedAdapterStages = adapterAvailability.ApprovedStages;
        bool ocrModelsApproved = HasTasks(approvedTasks, "ocr_detection", "ocr_recognition");
        bool markerModelsApproved = HasTasks(approvedTasks, "marker_center", "marker_classifier");
        bool axisApproved = runtimeAvailability.AxisApproved && approvedAdapterStages.Contains("axis");
        bool ocrApproved = inferenceRuntimeConfigured &&
            ocrModelsApproved &&
            approvedAdapterStages.Contains("ocr");
        bool markersApproved = inferenceRuntimeConfigured &&
            markerModelsApproved &&
            approvedAdapterStages.Contains("markers");
        bool legendsApproved = ocrApproved &&
            markersApproved &&
            approvedAdapterStages.Contains("legends");
        bool phasesApproved = axisApproved &&
            ocrApproved &&
            markersApproved &&
            approvedAdapterStages.Contains("phases");
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
        axisApproved
            ? new AutomaticStageStatus(
                "axis",
                AutomaticStageState.Approved,
                $"{runtimeAvailability.Evidence} {adapterAvailability.Evidence}")
            : new AutomaticStageStatus(
                "axis",
                AutomaticStageState.Unavailable,
                runtimeAvailability.AxisApproved
                    ? $"The exact OpenCV runtime is approved, but no approved production axis adapter is composed. {adapterAvailability.Evidence}"
                    : $"The deterministic axis adapter requires exact reviewed OpenCV provenance and mandatory clean-machine runtime approval. {runtimeAvailability.Evidence}"),
        ocrApproved
            ? new AutomaticStageStatus(
                "ocr",
                AutomaticStageState.Approved,
                $"OCR detection and recognition have checksum-resolved CPU-approved payloads. {modelAvailability.Evidence}")
            : new AutomaticStageStatus(
                "ocr",
                AutomaticStageState.Unavailable,
                ocrModelsApproved
                    ? inferenceRuntimeConfigured
                        ? $"OCR models and the bounded local ONNX Runtime are approved, but no approved production OCR adapter is composed. {adapterAvailability.Evidence}"
                        : "OCR models are approved, but the bounded local ONNX Runtime with mandatory CPU fallback is unavailable."
                    : $"OCR requires checksum-resolved CPU-approved ocr_detection and ocr_recognition payloads. Missing: {MissingTasks(approvedTasks, "ocr_detection", "ocr_recognition")}. {modelAvailability.Evidence}"),
        markersApproved
            ? new AutomaticStageStatus(
                "markers",
                AutomaticStageState.Approved,
                $"Marker center and shape/fill classification have checksum-resolved CPU-approved payloads. {modelAvailability.Evidence}")
            : new AutomaticStageStatus(
                "markers",
                AutomaticStageState.Unavailable,
                markerModelsApproved
                    ? inferenceRuntimeConfigured
                        ? $"Marker models and the bounded local ONNX Runtime are approved, but no approved production marker adapter is composed. {adapterAvailability.Evidence}"
                        : "Marker models are approved, but the bounded local ONNX Runtime with mandatory CPU fallback is unavailable."
                    : $"Markers require checksum-resolved CPU-approved marker_center and marker_classifier payloads. Missing: {MissingTasks(approvedTasks, "marker_center", "marker_classifier")}. {modelAvailability.Evidence}"),
        legendsApproved
            ? new AutomaticStageStatus(
                "legends",
                AutomaticStageState.Approved,
                $"Approved OCR, marker, and legend adapters are composed. {adapterAvailability.Evidence} {pdfEvidence}")
            : new AutomaticStageStatus(
                "legends",
                AutomaticStageState.Unavailable,
                ocrApproved && markersApproved
                    ? $"OCR and marker evidence is approved, but no approved production legend adapter is composed. {adapterAvailability.Evidence} {pdfEvidence}"
                    : $"Legend reasoning requires approved composed OCR and marker adapters. {pdfEvidence}"),
        phasesApproved
            ? new AutomaticStageStatus(
                "phases",
                AutomaticStageState.Approved,
                $"Approved axis, OCR, marker, and phase adapters are composed. {adapterAvailability.Evidence}")
            : new AutomaticStageStatus(
                "phases",
                AutomaticStageState.Unavailable,
                axisApproved && ocrApproved && markersApproved
                    ? $"Axis, OCR, and marker evidence is approved, but no approved production phase adapter is composed. {adapterAvailability.Evidence}"
                    : "Phase reasoning requires approved composed axis, OCR, and marker adapters."),
        ];
    }

    private static bool HasTasks(IReadOnlySet<string> approvedTasks, params string[] requiredTasks) =>
        requiredTasks.All(approvedTasks.Contains);

    private static string MissingTasks(IReadOnlySet<string> approvedTasks, params string[] requiredTasks) =>
        string.Join(", ", requiredTasks.Where(task => !approvedTasks.Contains(task)));
}
