// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;

namespace GraphReader.Ocr;

/// <summary>
/// Creates the bounded P1 precision-repair candidate over the unchanged V8
/// payloads. This factory is available only to direct candidate validation and
/// does not grant model approval or enter normal Auto Detect composition.
/// </summary>
public static class OcrV9CandidateCompositionFactory
{
    public const string CandidateCompositionId =
        OcrV8ProductionCompositionOptions.CandidateV9CompositionId;
    public const double OfficialMinimumConfidence = 0.65;

    public static OcrV8ProductionCompositionPipeline Create(
        InferenceRuntime runtime,
        OcrV8ProductionPayloadSet payloads,
        IReadOnlyList<InferenceProvider> allowedProviders,
        bool bypassCache = false)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(payloads);
        var providers = OcrV8ProductionCompositionFactory.ValidateProviderPolicy(allowedProviders);
        OcrV8ProductionCompositionFactory.ValidatePayloads(payloads);

        var detector = new LocalOnnxProposalTextRegionDetector(
            runtime,
            new LocalOnnxProposalTextRegionDetectorOptions(payloads.Detector)
            {
                StageVersion = "0.0.21-v10-p2",
                AllowedProviders = providers,
                BypassCache = bypassCache,
            });
        return OcrV8ProductionCompositionFactory.CreateWithDetector(
            runtime,
            payloads,
            providers,
            detector,
            bypassCache,
            new OcrV8ProductionCompositionOptions
            {
                CompositionId = CandidateCompositionId,
                StageVersion = "0.0.21-v9-p1",
                OfficialMinimumConfidence = OfficialMinimumConfidence,
            });
    }
}
