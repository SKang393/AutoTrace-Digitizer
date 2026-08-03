// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.Globalization;
using GraphReader.Inference;

namespace GraphReader.Markers.Detection;

internal sealed record MarkerTensorPreparation(
    InferenceRequest Request,
    string CacheKey,
    MarkerRectangle FrameCrop,
    double PreprocessMilliseconds);

internal sealed record MarkerCandidate(
    MarkerPoint Center,
    double Radius,
    double ArtifactProbability,
    double CenterConfidence,
    MarkerSourceImage SourceImage,
    MarkerReviewState ReviewState = MarkerReviewState.Unreviewed,
    MarkerDisagreementKind Disagreement = MarkerDisagreementKind.None);

internal sealed record MarkerTensorCandidate(
    int X,
    int Y,
    double Radius,
    double ArtifactProbability,
    double CenterConfidence);

internal sealed record MarkerDecodeResult(
    IReadOnlyList<MarkerCandidate> Candidates,
    int RawCandidateCount);

internal static class MarkerTensorPipeline
{
    public static MarkerTensorPreparation Prepare(
        MarkerDetectionRequest request,
        MarkerImageFrame frame,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var contract = request.Options.TensorContract;
        var framePolygon = request.PlotPolygon.Points
            .Select(frame.OriginalToFrame.MapFromOriginal)
            .ToArray();
        var left = Math.Max(0, framePolygon.Min(static point => point.X));
        var top = Math.Max(0, framePolygon.Min(static point => point.Y));
        var right = Math.Min(frame.Width, framePolygon.Max(static point => point.X));
        var bottom = Math.Min(frame.Height, framePolygon.Max(static point => point.Y));
        var crop = new MarkerRectangle(left, top, right - left, bottom - top);
        if (!crop.IsValid)
        {
            throw new MarkerPipelineException(
                "MARKER_PLOT_OUTSIDE_FRAME",
                "The plot polygon does not overlap the detector frame.",
                false,
                "review_plot_region");
        }

        var pixelCount = checked(contract.InputWidth * contract.InputHeight);
        var tensor = new float[checked(pixelCount * contract.InputChannelCount)];
        for (var outputY = 0; outputY < contract.InputHeight; outputY++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var frameY = crop.Top + (((outputY + 0.5) * crop.Height / contract.InputHeight) - 0.5);
            for (var outputX = 0; outputX < contract.InputWidth; outputX++)
            {
                var frameX = crop.Left + (((outputX + 0.5) * crop.Width / contract.InputWidth) - 0.5);
                var spatialIndex = (outputY * contract.InputWidth) + outputX;
                var luminance = SampleChannel(frame, 0, frameX, frameY);
                WriteTensorValue(
                    tensor,
                    spatialIndex,
                    0,
                    pixelCount,
                    1 - luminance);
                WriteTensorValue(
                    tensor,
                    spatialIndex,
                    1,
                    pixelCount,
                    SampleMask(frame.OcrMask, frameX, frameY));
                WriteTensorValue(
                    tensor,
                    spatialIndex,
                    2,
                    pixelCount,
                    SampleMask(frame.ArtifactMask, frameX, frameY));
            }
        }

        IReadOnlyList<long> inputShape =
            [1, contract.InputChannelCount, contract.InputHeight, contract.InputWidth];
        var cacheParameters = CreateCacheParameters(request, frame);
        var inferenceRequest = new InferenceRequest(
            request.Model,
            new InferenceInput(tensor, inputShape, contract.InputName, contract.OutputName),
            new StageCacheMaterial(
                request.InputSha256,
                SerializePolygon(request.PlotPolygon),
                request.TransformChain + "|" + frame.SourceImage + ":" + frame.OriginalToFrame.ToCacheMaterial(),
                MarkerContract.Stage,
                request.Options.StageVersion,
                cacheParameters,
                request.ContractVersion),
            request.Options.Timeout);
        var cacheKey = InferenceCacheKeyDeriver.Derive(inferenceRequest).Value;
        stopwatch.Stop();
        return new MarkerTensorPreparation(
            inferenceRequest,
            cacheKey,
            crop,
            stopwatch.Elapsed.TotalMilliseconds);
    }

    public static MarkerDecodeResult Decode(
        IReadOnlyList<float> output,
        MarkerDetectionRequest request,
        MarkerImageFrame frame,
        MarkerRectangle crop,
        CancellationToken cancellationToken)
    {
        var contract = request.Options.TensorContract;
        var expectedLength = checked(contract.OutputWidth * contract.OutputHeight * contract.OutputChannelCount);
        if (output.Count != expectedLength)
        {
            throw new MarkerPipelineException(
                "MARKER_MODEL_OUTPUT_SHAPE_MISMATCH",
                $"Marker model returned {output.Count} values; the declared flat output requires {expectedLength}.",
                false,
                "select_compatible_model");
        }

        ValidateOutputHeads(output, contract, cancellationToken);
        var tensorCandidates = new List<MarkerTensorCandidate>();
        var rawCandidateCount = 0;
        var outputPixelCount = checked(contract.OutputWidth * contract.OutputHeight);
        var frameScaleX = crop.Width / contract.OutputWidth;
        var frameScaleY = crop.Height / contract.OutputHeight;
        var radiusFrameScale = Math.Sqrt(frameScaleX * frameScaleY);
        for (var y = 0; y < contract.OutputHeight; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var x = 0; x < contract.OutputWidth; x++)
            {
                var centerConfidence = ReadOutput(
                    output,
                    x,
                    y,
                    contract.CenterChannelIndex,
                    contract,
                    outputPixelCount);
                if (!double.IsFinite(centerConfidence) || centerConfidence < request.Options.CenterThreshold)
                {
                    continue;
                }

                rawCandidateCount++;
                if (!IsLocalMaximum(output, x, y, contract, outputPixelCount, request.Options.LocalMaximumWindow))
                {
                    continue;
                }

                var artifactProbability = ReadOutput(
                    output,
                    x,
                    y,
                    contract.ArtifactChannelIndex,
                    contract,
                    outputPixelCount);
                var radiusHead = ReadOutput(output, x, y, contract.RadiusChannelIndex, contract, outputPixelCount);
                if (!double.IsFinite(artifactProbability) || !double.IsFinite(radiusHead) || radiusHead < 0)
                {
                    continue;
                }

                var framePoint = new MarkerPoint(
                    crop.Left + (((x + 0.5) * frameScaleX) - 0.5),
                    crop.Top + (((y + 0.5) * frameScaleY) - 0.5));
                artifactProbability = Math.Max(
                    artifactProbability,
                    Math.Max(
                        SampleMask(frame.OcrMask, framePoint.X, framePoint.Y),
                        SampleMask(frame.ArtifactMask, framePoint.X, framePoint.Y)));
                if (artifactProbability >= request.Options.ArtifactThreshold ||
                    SampleMask(frame.OcrMask, framePoint.X, framePoint.Y) >= request.Options.MaskThreshold ||
                    SampleMask(frame.ArtifactMask, framePoint.X, framePoint.Y) >= request.Options.MaskThreshold)
                {
                    continue;
                }

                tensorCandidates.Add(new MarkerTensorCandidate(
                    x,
                    y,
                    Math.Max(request.Options.MinimumRadiusGridPixels, radiusHead * contract.RadiusScale),
                    Math.Clamp(artifactProbability, 0, 1),
                    Math.Clamp(centerConfidence, 0, 1)));
            }
        }

        var candidates = new List<MarkerCandidate>();
        foreach (var tensorCandidate in ApplyRadiusAwareNms(tensorCandidates, request.Options))
        {
            var framePoint = new MarkerPoint(
                crop.Left + (((tensorCandidate.X + 0.5) * frameScaleX) - 0.5),
                crop.Top + (((tensorCandidate.Y + 0.5) * frameScaleY) - 0.5));
            var originalPoint = frame.OriginalToFrame.MapToOriginal(framePoint);
            if (!request.PlotPolygon.Contains(originalPoint))
            {
                continue;
            }

            var frameRadius = tensorCandidate.Radius * radiusFrameScale;
            var originalRadius = frame.OriginalToFrame.MapFrameRadiusToOriginal(frameRadius);
            if (!double.IsFinite(originalRadius) || originalRadius <= 0)
            {
                continue;
            }

            candidates.Add(new MarkerCandidate(
                originalPoint,
                originalRadius,
                tensorCandidate.ArtifactProbability,
                tensorCandidate.CenterConfidence,
                frame.SourceImage));
        }

        return new MarkerDecodeResult(
            MarkerCollections.Freeze(candidates),
            rawCandidateCount);
    }

    public static double Distance(MarkerPoint left, MarkerPoint right) =>
        Math.Sqrt(DistanceSquared(left, right));

    public static double DistanceSquared(MarkerPoint left, MarkerPoint right)
    {
        var deltaX = left.X - right.X;
        var deltaY = left.Y - right.Y;
        return (deltaX * deltaX) + (deltaY * deltaY);
    }

    private static IReadOnlyList<MarkerTensorCandidate> ApplyRadiusAwareNms(
        IEnumerable<MarkerTensorCandidate> candidates,
        MarkerDetectionOptions options)
    {
        var accepted = new List<MarkerTensorCandidate>();
        foreach (var candidate in candidates
                     .OrderByDescending(static item => item.CenterConfidence)
                     .ThenBy(static item => item.Y)
                     .ThenBy(static item => item.X))
        {
            var suppressed = accepted.Any(existing =>
            {
                var deltaX = candidate.X - existing.X;
                var deltaY = candidate.Y - existing.Y;
                var distance = Math.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
                var suppressionDistance = Math.Max(
                    options.MinimumSuppressionDistanceGridPixels,
                    options.RadiusSuppressionScale * (candidate.Radius + existing.Radius));
                return distance < suppressionDistance;
            });
            if (!suppressed)
            {
                accepted.Add(candidate);
            }
        }

        return MarkerCollections.Freeze(accepted);
    }

    private static bool IsLocalMaximum(
        IReadOnlyList<float> output,
        int x,
        int y,
        MarkerModelTensorContract contract,
        int outputPixelCount,
        int window)
    {
        var center = ReadOutput(output, x, y, contract.CenterChannelIndex, contract, outputPixelCount);
        var radius = window / 2;
        var minimumY = Math.Max(0, y - radius);
        var maximumY = Math.Min(contract.OutputHeight - 1, y + radius);
        var minimumX = Math.Max(0, x - radius);
        var maximumX = Math.Min(contract.OutputWidth - 1, x + radius);
        for (var neighborY = minimumY; neighborY <= maximumY; neighborY++)
        {
            for (var neighborX = minimumX; neighborX <= maximumX; neighborX++)
            {
                if (ReadOutput(
                        output,
                        neighborX,
                        neighborY,
                        contract.CenterChannelIndex,
                        contract,
                        outputPixelCount) > center)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static void ValidateOutputHeads(
        IReadOnlyList<float> output,
        MarkerModelTensorContract contract,
        CancellationToken cancellationToken)
    {
        var outputPixelCount = checked(contract.OutputWidth * contract.OutputHeight);
        for (var y = 0; y < contract.OutputHeight; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var x = 0; x < contract.OutputWidth; x++)
            {
                var center = ReadOutput(
                    output,
                    x,
                    y,
                    contract.CenterChannelIndex,
                    contract,
                    outputPixelCount);
                var radius = ReadOutput(
                    output,
                    x,
                    y,
                    contract.RadiusChannelIndex,
                    contract,
                    outputPixelCount);
                var artifact = ReadOutput(
                    output,
                    x,
                    y,
                    contract.ArtifactChannelIndex,
                    contract,
                    outputPixelCount);
                if (!double.IsFinite(center) || center < 0 || center > 1 ||
                    !double.IsFinite(radius) || radius < 0 ||
                    !double.IsFinite(artifact) || artifact < 0 || artifact > 1)
                {
                    throw new MarkerPipelineException(
                        "MARKER_MODEL_OUTPUT_INVALID",
                        "Marker output must contain finite activated center/artifact probabilities and nonnegative radius pixels.",
                        false,
                        "select_compatible_model");
                }
            }
        }
    }

    private static Dictionary<string, object?> CreateCacheParameters(
        MarkerDetectionRequest request,
        MarkerImageFrame frame)
    {
        var tensor = request.Options.TensorContract;
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["runtime_revision"] = MarkerContract.RuntimeRevision,
            ["input_channel_order"] = MarkerContract.InputChannelOrder,
            ["output_channel_order"] = MarkerContract.OutputChannelOrder,
            ["grid_alignment"] = MarkerContract.GridAlignment,
            ["source_image"] = frame.SourceImage.ToString(),
            ["frame_width"] = frame.Width,
            ["frame_height"] = frame.Height,
            ["image_channels"] = frame.ChannelCount,
            ["input_width"] = tensor.InputWidth,
            ["input_height"] = tensor.InputHeight,
            ["input_channels"] = tensor.InputChannelCount,
            ["input_layout"] = tensor.InputLayout.ToString(),
            ["output_width"] = tensor.OutputWidth,
            ["output_height"] = tensor.OutputHeight,
            ["output_channels"] = tensor.OutputChannelCount,
            ["output_layout"] = tensor.OutputLayout.ToString(),
            ["center_channel"] = tensor.CenterChannelIndex,
            ["radius_channel"] = tensor.RadiusChannelIndex,
            ["artifact_channel"] = tensor.ArtifactChannelIndex,
            ["center_activation"] = tensor.CenterActivation.ToString(),
            ["artifact_activation"] = tensor.ArtifactActivation.ToString(),
            ["radius_scale"] = tensor.RadiusScale,
            ["normalize_mean"] = tensor.NormalizeMean,
            ["normalize_scale"] = tensor.NormalizeScale,
            ["center_threshold"] = request.Options.CenterThreshold,
            ["artifact_threshold"] = request.Options.ArtifactThreshold,
            ["mask_threshold"] = request.Options.MaskThreshold,
            ["local_maximum_window"] = request.Options.LocalMaximumWindow,
            ["minimum_radius_grid_pixels"] = request.Options.MinimumRadiusGridPixels,
            ["minimum_suppression_distance_grid_pixels"] = request.Options.MinimumSuppressionDistanceGridPixels,
            ["radius_suppression_scale"] = request.Options.RadiusSuppressionScale,
            ["consensus_tolerance"] = request.Options.ConsensusToleranceOriginalPixels,
            ["unmatched_confidence_scale"] = request.Options.UnmatchedSourceConfidenceScale,
        };
    }

    private static string SerializePolygon(MarkerPolygon polygon) => string.Join(
        ';',
        polygon.Points.Select(point =>
            point.X.ToString("R", CultureInfo.InvariantCulture) + "," +
            point.Y.ToString("R", CultureInfo.InvariantCulture)));

    private static void WriteTensorValue(
        float[] tensor,
        int spatialIndex,
        int channel,
        int pixelCount,
        float value)
    {
        var index = (channel * pixelCount) + spatialIndex;
        tensor[index] = value;
    }

    private static double ReadOutput(
        IReadOnlyList<float> output,
        int x,
        int y,
        int channel,
        MarkerModelTensorContract contract,
        int outputPixelCount)
    {
        var spatialIndex = (y * contract.OutputWidth) + x;
        var index = (channel * outputPixelCount) + spatialIndex;
        return output[index];
    }

    private static float SampleChannel(MarkerImageFrame frame, int channel, double x, double y)
    {
        var pixelsPerChannel = checked(frame.Width * frame.Height);
        return SamplePlane(
            frame.ChannelsFirstPixels.Span.Slice(channel * pixelsPerChannel, pixelsPerChannel),
            frame.Width,
            frame.Height,
            x,
            y);
    }

    private static float SampleMask(MarkerMask mask, double x, double y) =>
        SamplePlane(mask.Values.Span, mask.Width, mask.Height, x, y);

    private static float SamplePlane(
        ReadOnlySpan<float> values,
        int width,
        int height,
        double x,
        double y)
    {
        var clampedX = Math.Clamp(x, 0, width - 1d);
        var clampedY = Math.Clamp(y, 0, height - 1d);
        var x0 = (int)Math.Floor(clampedX);
        var y0 = (int)Math.Floor(clampedY);
        var x1 = Math.Min(width - 1, x0 + 1);
        var y1 = Math.Min(height - 1, y0 + 1);
        var xWeight = clampedX - x0;
        var yWeight = clampedY - y0;
        var top = (values[(y0 * width) + x0] * (1 - xWeight)) +
                  (values[(y0 * width) + x1] * xWeight);
        var bottom = (values[(y1 * width) + x0] * (1 - xWeight)) +
                     (values[(y1 * width) + x1] * xWeight);
        return (float)((top * (1 - yWeight)) + (bottom * yWeight));
    }
}

internal sealed class MarkerPipelineException : Exception
{
    public MarkerPipelineException(
        string code,
        string technicalMessage,
        bool recoverable,
        string suggestedAction)
        : base(technicalMessage)
    {
        Code = code;
        Recoverable = recoverable;
        SuggestedAction = suggestedAction;
    }

    public string Code { get; }

    public bool Recoverable { get; }

    public string SuggestedAction { get; }
}
