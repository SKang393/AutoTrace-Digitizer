// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;

namespace GraphReader.Ocr;

/// <summary>
/// Creates the exact V11 proposal-role candidate with the unchanged reviewed
/// recognition payloads. This is a direct composition-gate seam only. It does
/// not expose the candidate through normal Auto Detect or grant model approval.
/// </summary>
public static class OcrV11CandidateCompositionFactory
{
    public const string DetectorModelId = "graph-text-composite-proposal-role-v11-p2";
    public const string DetectorSha256 =
        "af13b387140d70946b23ff7349fed82649fde95eb6f6cabe90179b2914a16631";
    public const string CandidateCompositionId =
        OcrV8ProductionCompositionOptions.CandidateV11CompositionId;

    public static OcrV8ProductionCompositionPipeline Create(
        InferenceRuntime runtime,
        OcrV8ProductionPayloadSet payloads,
        IReadOnlyList<InferenceProvider> allowedProviders,
        bool bypassCache = false)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(payloads);
        var providers = OcrV8ProductionCompositionFactory.ValidateProviderPolicy(allowedProviders);
        ValidatePayloads(payloads);

        var detector = new LocalOnnxProposalTextRegionDetector(
            runtime,
            new LocalOnnxProposalTextRegionDetectorOptions(payloads.Detector)
            {
                Contract = OcrProposalClassifierContract.CompositeProposalRoleV11,
                GeometryFeatureCount = 16,
                OutputName = "proposal_role_logits",
                StageVersion = "0.0.21-v11-p2",
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
                StageVersion = "0.0.21-v11",
            });
    }

    public static void ValidatePayloads(OcrV8ProductionPayloadSet payloads)
    {
        ArgumentNullException.ThrowIfNull(payloads);
        OcrV8ProductionCompositionFactory.ValidatePayload(
            payloads.Detector,
            DetectorModelId,
            DetectorSha256,
            nameof(payloads.Detector));
        OcrV8ProductionCompositionFactory.ValidateRecognitionPayloads(payloads);
    }
}
