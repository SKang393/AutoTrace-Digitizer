// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Services;

namespace GraphReader.App.Integration;

public static class ProductionStageAvailabilityRegistry
{
    public static IReadOnlyList<AutomaticStageStatus> Current { get; } =
    [
        new(
            "enhancement",
            AutomaticStageState.Unavailable,
            "No approved Real-ESRGAN runtime, model payload, and Graph Auto Reader benchmark set are installed."),
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
