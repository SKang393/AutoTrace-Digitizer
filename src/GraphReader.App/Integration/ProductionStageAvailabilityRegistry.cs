// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Services;

namespace GraphReader.App.Integration;

public static class ProductionStageAvailabilityRegistry
{
    public static IReadOnlyList<AutomaticStageStatus> Current { get; } = Create(localEnhancementConfigured: false);

    public static IReadOnlyList<AutomaticStageStatus> Create(bool localEnhancementConfigured) =>
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
            "OCR detection, text recognition, and graph-numeric recognition do not have checksum-approved redistributable production models."),
        new(
            "markers",
            AutomaticStageState.Unavailable,
            "The marker-center candidate failed production acceptance and no approved marker shape/fill classifier exists."),
        new(
            "legends",
            AutomaticStageState.Unavailable,
            "Legend reasoning is implemented but requires approved OCR and marker evidence."),
        new(
            "phases",
            AutomaticStageState.Unavailable,
            "Phase reasoning is implemented but requires approved axis, OCR, and marker evidence."),
    ];
}
