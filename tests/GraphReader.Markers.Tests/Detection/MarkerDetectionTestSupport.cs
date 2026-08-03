// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Detection;

namespace GraphReader.Markers.Tests.Detection;

internal static class MarkerDetectionTestSupport
{
    internal const int FrameSize = 64;
    internal const int OutputSize = 64;

    internal static readonly ModelIdentity Model = new(
        "graph-marker-center",
        "1.0.0",
        new string('a', 64),
        "marker-center.onnx");

    internal static MarkerModelTensorContract Contract(
        MarkerTensorLayout outputLayout = MarkerTensorLayout.ChannelsFirst) =>
        new(
            "image",
            "heads",
            FrameSize,
            FrameSize,
            3,
            MarkerTensorLayout.ChannelsFirst,
            OutputSize,
            OutputSize,
            3,
            outputLayout,
            0,
            1,
            2,
            MarkerHeadActivation.Identity,
            MarkerHeadActivation.Identity,
            1,
            0,
            1);

    internal static MarkerDetectionOptions Options(
        MarkerTensorLayout outputLayout = MarkerTensorLayout.ChannelsFirst) =>
        new(Contract(outputLayout))
        {
            CenterThreshold = 0.5f,
            ArtifactThreshold = 0.5f,
            MaskThreshold = 0.5f,
            ConsensusToleranceOriginalPixels = 5,
            UnmatchedSourceConfidenceScale = 0.75,
            StageVersion = "0.2.0-test",
            Timeout = TimeSpan.FromSeconds(2),
        };

    internal static MarkerDetectionRequest Request(
        MarkerDetectionOptions? options = null,
        MarkerImageFrame? original = null,
        MarkerPolygon? plot = null,
        MarkerImageFrame? enhanced = null,
        ModelIdentity? model = null,
        string transformChain = "identity")
    {
        return new MarkerDetectionRequest(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            new string('b', 64),
            model ?? Model,
            original ?? Frame(MarkerSourceImage.Original),
            plot ?? MarkerPolygon.FromRectangle(new MarkerRectangle(0, 0, FrameSize, FrameSize)),
            options ?? Options(),
            enhanced,
            MarkerContract.Version,
            transformChain);
    }

    internal static MarkerImageFrame Frame(
        MarkerSourceImage sourceImage,
        MarkerAffineTransform? transform = null,
        MarkerMask? ocrMask = null,
        MarkerMask? artifactMask = null,
        IReadOnlyList<MarkerPoint>? darkPixels = null)
    {
        float[] pixels = Enumerable.Repeat(1f, FrameSize * FrameSize).ToArray();
        if (darkPixels is not null)
        {
            foreach (MarkerPoint point in darkPixels)
            {
                var x = (int)Math.Round(point.X, MidpointRounding.AwayFromZero);
                var y = (int)Math.Round(point.Y, MidpointRounding.AwayFromZero);
                if (x >= 0 && x < FrameSize && y >= 0 && y < FrameSize)
                {
                    pixels[(y * FrameSize) + x] = 0;
                }
            }
        }

        return new MarkerImageFrame(
            FrameSize,
            FrameSize,
            1,
            pixels,
            sourceImage,
            transform ?? MarkerAffineTransform.Identity,
            ocrMask ?? MarkerMask.Empty(FrameSize, FrameSize),
            artifactMask ?? MarkerMask.Empty(FrameSize, FrameSize));
    }

    internal static MarkerMask Mask(params MarkerPoint[] points)
    {
        var values = new float[FrameSize * FrameSize];
        foreach (MarkerPoint point in points)
        {
            var x = (int)Math.Round(point.X, MidpointRounding.AwayFromZero);
            var y = (int)Math.Round(point.Y, MidpointRounding.AwayFromZero);
            values[(y * FrameSize) + x] = 1;
        }

        return new MarkerMask(FrameSize, FrameSize, values);
    }

    internal static InferenceResponse Success(
        IReadOnlyList<HeatmapPeak> peaks,
        MarkerTensorLayout outputLayout = MarkerTensorLayout.ChannelsFirst,
        InferenceProvider provider = InferenceProvider.Fake,
        bool cacheHit = false,
        IReadOnlyList<ProviderAttempt>? attempts = null)
    {
        float[] output = FlatOutput(peaks, outputLayout);
        var timing = new StageTiming(0.1, 0.2, 0.1, 0.4, 0, false, cacheHit);
        var memory = new MemoryDiagnostics(0, 0, 0, 0, 0);
        return new InferenceResponse(
            true,
            new InferenceExecution(output, provider, timing, memory),
            null,
            attempts ?? [new ProviderAttempt(provider, true, null)]);
    }

    internal static InferenceResponse Success(
        IReadOnlyList<HeatmapPeak> peaks,
        MarkerModelTensorContract contract,
        InferenceProvider provider = InferenceProvider.Fake,
        bool cacheHit = false,
        IReadOnlyList<ProviderAttempt>? attempts = null)
    {
        float[] output = FlatOutput(peaks, contract);
        var timing = new StageTiming(0.1, 0.2, 0.1, 0.4, 0, false, cacheHit);
        var memory = new MemoryDiagnostics(0, 0, 0, 0, 0);
        return new InferenceResponse(
            true,
            new InferenceExecution(output, provider, timing, memory),
            null,
            attempts ?? [new ProviderAttempt(provider, true, null)]);
    }

    internal static InferenceResponse Failure(
        string code = "INFERENCE_FAILED",
        bool recoverable = true) =>
        InferenceResponse.Failure(
            new InferenceError(
                code,
                "error",
                "Errors.MarkerInferenceFailed",
                "Deterministic fake inference failure.",
                recoverable,
                "retry"),
            [new ProviderAttempt(InferenceProvider.Cpu, false, "Deterministic fake inference failure.")]);

    internal static float[] FlatOutput(
        IReadOnlyList<HeatmapPeak> peaks,
        MarkerTensorLayout layout = MarkerTensorLayout.ChannelsFirst)
    {
        var values = new float[3 * OutputSize * OutputSize];
        foreach (HeatmapPeak peak in peaks)
        {
            Set(values, layout, 0, peak.X, peak.Y, peak.Confidence);
            Set(values, layout, 1, peak.X, peak.Y, peak.Radius);
            Set(values, layout, 2, peak.X, peak.Y, peak.ArtifactProbability);
        }

        return values;
    }

    internal static float[] FlatOutput(
        IReadOnlyList<HeatmapPeak> peaks,
        MarkerModelTensorContract contract)
    {
        var values = new float[
            contract.OutputChannelCount * contract.OutputWidth * contract.OutputHeight];
        foreach (HeatmapPeak peak in peaks)
        {
            Set(values, contract, contract.CenterChannelIndex, peak.X, peak.Y, peak.Confidence);
            Set(values, contract, contract.RadiusChannelIndex, peak.X, peak.Y, peak.Radius);
            Set(
                values,
                contract,
                contract.ArtifactChannelIndex,
                peak.X,
                peak.Y,
                peak.ArtifactProbability);
        }

        return values;
    }

    internal static MarkerPoint ExpectedCenter(int outputX, int outputY) =>
        new(
            ((outputX + 0.5) * FrameSize / OutputSize) - 0.5,
            ((outputY + 0.5) * FrameSize / OutputSize) - 0.5);

    internal static void AssertNear(MarkerPoint expected, MarkerPoint actual, double tolerance = 0.01)
    {
        Assert.AreEqual(expected.X, actual.X, tolerance, "Unexpected marker x coordinate.");
        Assert.AreEqual(expected.Y, actual.Y, tolerance, "Unexpected marker y coordinate.");
    }

    private static void Set(
        float[] values,
        MarkerTensorLayout layout,
        int channel,
        int x,
        int y,
        float value)
    {
        int index = layout == MarkerTensorLayout.ChannelsFirst
            ? (channel * OutputSize * OutputSize) + (y * OutputSize) + x
            : ((y * OutputSize) + x) * 3 + channel;
        values[index] = value;
    }

    private static void Set(
        float[] values,
        MarkerModelTensorContract contract,
        int channel,
        int x,
        int y,
        float value)
    {
        int spatialIndex = (y * contract.OutputWidth) + x;
        int index = contract.OutputLayout == MarkerTensorLayout.ChannelsFirst
            ? (channel * contract.OutputWidth * contract.OutputHeight) + spatialIndex
            : (spatialIndex * contract.OutputChannelCount) + channel;
        values[index] = value;
    }
}

internal sealed record HeatmapPeak(
    int X,
    int Y,
    float Confidence = 0.95f,
    float Radius = 1.5f,
    float ArtifactProbability = 0.05f);

internal sealed class MarkerInferenceRunnerStub : IMarkerInferenceRunner
{
    private readonly Func<InferenceRequest, CancellationToken, ValueTask<InferenceResponse>> _run;
    private readonly List<InferenceRequest> _requests = [];

    internal MarkerInferenceRunnerStub(params InferenceResponse[] responses)
    {
        var responseQueue = new Queue<InferenceResponse>(responses);
        _run = (_, _) => responseQueue.Count == 0
            ? throw new InvalidOperationException("The fake marker inference response queue was exhausted.")
            : ValueTask.FromResult(responseQueue.Dequeue());
    }

    internal MarkerInferenceRunnerStub(
        Func<InferenceRequest, CancellationToken, ValueTask<InferenceResponse>> run) =>
        _run = run;

    internal IReadOnlyList<InferenceRequest> Requests => _requests;

    public ValueTask<InferenceResponse> RunAsync(
        InferenceRequest request,
        CancellationToken cancellationToken)
    {
        _requests.Add(request);
        cancellationToken.ThrowIfCancellationRequested();
        return _run(request, cancellationToken);
    }
}
