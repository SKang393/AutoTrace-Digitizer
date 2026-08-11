// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using GraphReader.Inference;

namespace GraphReader.Markers.Detection;

public static class NormalizedMarkerProposalPostprocessContract
{
    public const string Revision = "radial-local-consensus-calibration-v2";
    public const string OutputName = "candidate_predictions";
    public const string OutputLayout = "NC";
    public const string OutputColumns = "marker_probability,offset_x_grid,offset_y_grid,radius_pixels";
    public const int OutputColumnCount = 4;
    public const float SelectedMarkerThreshold = 0.6f;
    public const double MinimumOffset = -0.75;
    public const double MaximumOffset = 0.75;
    public const double MinimumRadius = 2.5;
    public const double MaximumRadius = 8;
    public const double MinimumCenterSeparation = 6.5;
    public const double RadiusSuppressionScale = 1.25;
    public const double GeometryInkThreshold = 0.12;
    public const double GeometryCenterDensityThreshold = 0.28;
}

public sealed record NormalizedMarkerProposalDetectionOptions
{
    public float MarkerThreshold { get; init; } =
        NormalizedMarkerProposalPostprocessContract.SelectedMarkerThreshold;

    public double MinimumCenterSeparation { get; init; } =
        NormalizedMarkerProposalPostprocessContract.MinimumCenterSeparation;

    public double RadiusSuppressionScale { get; init; } =
        NormalizedMarkerProposalPostprocessContract.RadiusSuppressionScale;

    public string StageVersion { get; init; } = "normalized-training-v4-p1";

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);
}

public sealed record NormalizedMarkerProposalDetectionRequest(
    string ProjectId,
    string PanelId,
    string InputSha256,
    ModelIdentity Model,
    MarkerImageFrame OriginalImage,
    NormalizedMarkerProposalDetectionOptions Options,
    int ContractVersion = MarkerContract.Version);

public interface INormalizedMarkerProposalDetectionService
{
    ValueTask<MarkerDetectionResult> DetectAsync(
        NormalizedMarkerProposalDetectionRequest request,
        CancellationToken cancellationToken);
}

/// <summary>
/// CPU-only candidate runtime for the passing normalized-training marker model.
/// This service is not wired into ordinary Auto Detect and does not approve a
/// model, mask provider, manifest, or package payload.
/// </summary>
public sealed class NormalizedMarkerProposalDetector : INormalizedMarkerProposalDetectionService
{
    private static readonly IReadOnlyList<InferenceProvider> CpuOnly = [InferenceProvider.Cpu];
    private readonly IMarkerInferenceRunner inference;

    public NormalizedMarkerProposalDetector(InferenceRuntime runtime)
        : this(new InferenceRuntimeMarkerRunner(runtime))
    {
    }

    public NormalizedMarkerProposalDetector(IMarkerInferenceRunner inference) =>
        this.inference = inference ?? throw new ArgumentNullException(nameof(inference));

    public async ValueTask<MarkerDetectionResult> DetectAsync(
        NormalizedMarkerProposalDetectionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var total = Stopwatch.StartNew();
        string runId = Guid.NewGuid().ToString();
        MarkerDetectionFailure? validationFailure = Validate(request);
        if (validationFailure is not null)
        {
            return Failure(request, runId, validationFailure, total.Elapsed.TotalMilliseconds);
        }

        NormalizedMarkerProposalBatch proposals;
        var preprocess = Stopwatch.StartNew();
        try
        {
            proposals = NormalizedMarkerProposalPreprocessor.Prepare(
                request.OriginalImage,
                cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (ArgumentException exception)
        {
            preprocess.Stop();
            return Failure(
                request,
                runId,
                Invalid(exception.Message),
                total.Elapsed.TotalMilliseconds);
        }

        preprocess.Stop();
        if (proposals.Count == 0)
        {
            total.Stop();
            var emptyTiming = new MarkerDetectionTiming(
                preprocess.Elapsed.TotalMilliseconds,
                0,
                0,
                total.Elapsed.TotalMilliseconds);
            var emptyReport = new MarkerFrameReport(
                MarkerSourceImage.Original,
                string.Empty,
                null,
                Array.Empty<ProviderAttempt>(),
                emptyTiming,
                0,
                0,
                false,
                null);
            return Success(
                request,
                runId,
                Array.Empty<MarkerCandidate>(),
                emptyTiming,
                ["no_eligible_marker_proposals"],
                emptyReport,
                null);
        }

        InferenceRequest inferenceRequest = CreateInferenceRequest(request, proposals);
        string cacheKey = InferenceCacheKeyDeriver.Derive(inferenceRequest).Value;
        InferenceResponse response;
        try
        {
            response = await inference.RunAsync(inferenceRequest, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            total.Stop();
            MarkerDetectionFailure failure = Error(
                "MARKER_INFERENCE_FAILED",
                exception.Message,
                true,
                "retry");
            var failureTiming = new MarkerDetectionTiming(
                preprocess.Elapsed.TotalMilliseconds,
                0,
                0,
                total.Elapsed.TotalMilliseconds);
            return Failure(
                request,
                runId,
                failure,
                total.Elapsed.TotalMilliseconds,
                failureTiming,
                [FrameFailure(cacheKey, failureTiming, null, Array.Empty<ProviderAttempt>(), false, failure)],
                null);
        }

        if (!response.Succeeded || response.Execution is null)
        {
            total.Stop();
            MarkerDetectionFailure failure = response.Error is null
                ? Error(
                    "MARKER_INFERENCE_FAILED",
                    "The normalized marker runtime returned no execution result.",
                    true,
                    "retry")
                : new MarkerDetectionFailure(
                    response.Error.Code,
                    response.Error.Severity,
                    response.Error.UserMessageKey,
                    response.Error.TechnicalMessage,
                    response.Error.Recoverable,
                    response.Error.SuggestedAction);
            var failureTiming = new MarkerDetectionTiming(
                preprocess.Elapsed.TotalMilliseconds,
                0,
                0,
                total.Elapsed.TotalMilliseconds);
            return Failure(
                request,
                runId,
                failure,
                total.Elapsed.TotalMilliseconds,
                failureTiming,
                [FrameFailure(
                    cacheKey,
                    failureTiming,
                    null,
                    response.ProviderAttempts,
                    false,
                    failure)],
                null);
        }

        if (response.Execution.Provider != InferenceProvider.Cpu)
        {
            total.Stop();
            MarkerDetectionFailure failure = Error(
                "MARKER_PROVIDER_UNAPPROVED",
                "Normalized marker candidate execution currently permits the CPU provider only.",
                false,
                "select_cpu_provider");
            var failureTiming = new MarkerDetectionTiming(
                preprocess.Elapsed.TotalMilliseconds,
                response.Execution.Timing.InferenceMilliseconds,
                0,
                total.Elapsed.TotalMilliseconds);
            return Failure(
                request,
                runId,
                failure,
                total.Elapsed.TotalMilliseconds,
                failureTiming,
                [FrameFailure(
                    cacheKey,
                    failureTiming,
                    response.Execution.Provider,
                    response.ProviderAttempts,
                    response.Execution.Timing.CacheHit,
                    failure)],
                response.Execution.Provider);
        }

        MarkerDecodeResult decoded;
        var postprocess = Stopwatch.StartNew();
        try
        {
            decoded = NormalizedMarkerProposalPostprocessor.Decode(
                response.Execution.Output,
                proposals,
                request.OriginalImage,
                request.Options,
                cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (MarkerPipelineException exception)
        {
            postprocess.Stop();
            total.Stop();
            MarkerDetectionFailure failure = Error(
                exception.Code,
                exception.Message,
                exception.Recoverable,
                exception.SuggestedAction);
            var failureTiming = new MarkerDetectionTiming(
                preprocess.Elapsed.TotalMilliseconds,
                response.Execution.Timing.InferenceMilliseconds,
                postprocess.Elapsed.TotalMilliseconds,
                total.Elapsed.TotalMilliseconds);
            return Failure(
                request,
                runId,
                failure,
                total.Elapsed.TotalMilliseconds,
                failureTiming,
                [FrameFailure(
                    cacheKey,
                    failureTiming,
                    response.Execution.Provider,
                    response.ProviderAttempts,
                    response.Execution.Timing.CacheHit,
                    failure)],
                response.Execution.Provider);
        }

        postprocess.Stop();
        total.Stop();
        var timing = new MarkerDetectionTiming(
            preprocess.Elapsed.TotalMilliseconds,
            response.Execution.Timing.InferenceMilliseconds,
            postprocess.Elapsed.TotalMilliseconds,
            total.Elapsed.TotalMilliseconds);
        var report = new MarkerFrameReport(
            MarkerSourceImage.Original,
            cacheKey,
            response.Execution.Provider,
            MarkerCollections.Freeze(response.ProviderAttempts),
            timing,
            decoded.RawCandidateCount,
            decoded.Candidates.Count,
            response.Execution.Timing.CacheHit,
            null);
        return Success(
            request,
            runId,
            decoded.Candidates,
            timing,
            Array.Empty<string>(),
            report,
            response.Execution.Provider);
    }

    private static InferenceRequest CreateInferenceRequest(
        NormalizedMarkerProposalDetectionRequest request,
        NormalizedMarkerProposalBatch proposals)
    {
        var parameters = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["runtime_revision"] = NormalizedMarkerProposalContract.RuntimeRevision,
            ["preprocess_revision"] = NormalizedMarkerProposalContract.PreprocessRevision,
            ["postprocess_revision"] = NormalizedMarkerProposalPostprocessContract.Revision,
            ["input_channel_order"] = NormalizedMarkerProposalContract.InputChannelOrder,
            ["output_columns"] = NormalizedMarkerProposalPostprocessContract.OutputColumns,
            ["proposal_tensor_sha256"] = proposals.TensorSha256,
            ["proposal_count"] = proposals.Count,
            ["patch_size"] = NormalizedMarkerProposalContract.PatchSize,
            ["proposal_stride"] = NormalizedMarkerProposalContract.ProposalStride,
            ["marker_threshold"] = request.Options.MarkerThreshold,
            ["minimum_center_separation"] = request.Options.MinimumCenterSeparation,
            ["radius_suppression_scale"] = request.Options.RadiusSuppressionScale,
            ["frame_width"] = request.OriginalImage.Width,
            ["frame_height"] = request.OriginalImage.Height,
        };
        return new InferenceRequest(
            request.Model,
            new InferenceInput(
                proposals.Tensor,
                proposals.Shape,
                NormalizedMarkerProposalContract.InputName,
                NormalizedMarkerProposalPostprocessContract.OutputName),
            new StageCacheMaterial(
                request.InputSha256,
                "full-frame",
                "identity",
                MarkerContract.Stage,
                request.Options.StageVersion,
                parameters,
                request.ContractVersion),
            request.Options.Timeout,
            CpuOnly);
    }

    private static MarkerDetectionFailure? Validate(NormalizedMarkerProposalDetectionRequest request)
    {
        if (!Guid.TryParse(request.ProjectId, out _) || !Guid.TryParse(request.PanelId, out _))
        {
            return Invalid("ProjectId and PanelId must be UUID strings.");
        }

        if (request.InputSha256 is null || request.InputSha256.Length != 64 ||
            !request.InputSha256.All(Uri.IsHexDigit))
        {
            return Invalid("InputSha256 must contain exactly 64 hexadecimal characters.");
        }

        if (request.ContractVersion != MarkerContract.Version)
        {
            return Invalid($"Marker contract version {request.ContractVersion} is unsupported.");
        }

        try
        {
            request.Model.Validate();
        }
        catch (Exception exception) when (exception is ArgumentException or NullReferenceException)
        {
            return Invalid(exception.Message);
        }

        if (request.Options is null ||
            request.Options.MarkerThreshold !=
                NormalizedMarkerProposalPostprocessContract.SelectedMarkerThreshold ||
            request.Options.MinimumCenterSeparation !=
                NormalizedMarkerProposalPostprocessContract.MinimumCenterSeparation ||
            request.Options.RadiusSuppressionScale !=
                NormalizedMarkerProposalPostprocessContract.RadiusSuppressionScale ||
            string.IsNullOrWhiteSpace(request.Options.StageVersion) ||
            (request.Options.Timeout <= TimeSpan.Zero &&
                request.Options.Timeout != Timeout.InfiniteTimeSpan))
        {
            return Invalid("Normalized marker options do not match the frozen selected contract.");
        }

        MarkerImageFrame frame = request.OriginalImage;
        if (frame is null || frame.SourceImage != MarkerSourceImage.Original ||
            frame.OriginalToFrame != MarkerAffineTransform.Identity ||
            frame.ChannelCount != 1 || frame.Width <= 0 || frame.Height <= 0)
        {
            return Invalid("Normalized marker detection requires an immutable original one-channel frame.");
        }

        return null;
    }

    private static MarkerDetectionResult Success(
        NormalizedMarkerProposalDetectionRequest request,
        string runId,
        IReadOnlyList<MarkerCandidate> candidates,
        MarkerDetectionTiming timing,
        IReadOnlyList<string> warnings,
        MarkerFrameReport report,
        InferenceProvider? provider)
    {
        MarkerCenter[] markers = candidates.Select((candidate, index) => new MarkerCenter(
            $"marker-{index + 1:D4}",
            candidate.Center,
            candidate.Radius,
            candidate.ArtifactProbability,
            candidate.CenterConfidence,
            MarkerSourceImage.Original)).ToArray();
        double confidence = markers.Length == 0
            ? 0
            : markers.Average(static marker => marker.CenterConfidence);
        return new MarkerDetectionResult(
            request.ContractVersion,
            runId,
            request.ProjectId,
            request.PanelId,
            MarkerContract.Stage,
            request.Options.StageVersion,
            request.InputSha256,
            MarkerContract.CoordinateSpace,
            MarkerCollections.Freeze(markers),
            timing,
            Math.Clamp(confidence, 0, 1),
            MarkerCollections.Freeze(warnings),
            [report],
            new MarkerModelReport(
                request.Model.ModelId,
                request.Model.Version,
                request.Model.Sha256,
                provider),
            null);
    }

    private static MarkerDetectionResult Failure(
        NormalizedMarkerProposalDetectionRequest request,
        string runId,
        MarkerDetectionFailure failure,
        double totalMilliseconds,
        MarkerDetectionTiming? timing = null,
        IReadOnlyList<MarkerFrameReport>? frames = null,
        InferenceProvider? provider = null) =>
        new(
            request.ContractVersion,
            runId,
            request.ProjectId,
            request.PanelId,
            MarkerContract.Stage,
            request.Options?.StageVersion ?? string.Empty,
            request.InputSha256,
            MarkerContract.CoordinateSpace,
            Array.Empty<MarkerCenter>(),
            timing ?? new MarkerDetectionTiming(0, 0, 0, totalMilliseconds),
            0,
            Array.Empty<string>(),
            frames ?? Array.Empty<MarkerFrameReport>(),
            new MarkerModelReport(
                request.Model?.ModelId ?? string.Empty,
                request.Model?.Version ?? string.Empty,
                request.Model?.Sha256 ?? string.Empty,
                provider),
            failure);

    private static MarkerFrameReport FrameFailure(
        string cacheKey,
        MarkerDetectionTiming timing,
        InferenceProvider? provider,
        IReadOnlyList<ProviderAttempt> attempts,
        bool cacheHit,
        MarkerDetectionFailure failure) =>
        new(
            MarkerSourceImage.Original,
            cacheKey,
            provider,
            MarkerCollections.Freeze(attempts),
            timing,
            0,
            0,
            cacheHit,
            failure);

    private static MarkerDetectionFailure Invalid(string message) =>
        Error("MARKER_REQUEST_INVALID", message, false, "review_marker_request");

    private static MarkerDetectionFailure Error(
        string code,
        string message,
        bool recoverable,
        string suggestedAction) =>
        new(code, "error", "Errors." + code, message, recoverable, suggestedAction);
}

internal static class NormalizedMarkerProposalPostprocessor
{
    private static readonly double[] RefinementOffsets = [-1, 0, 1];

    public static MarkerDecodeResult Decode(
        IReadOnlyList<float> output,
        NormalizedMarkerProposalBatch proposals,
        MarkerImageFrame frame,
        NormalizedMarkerProposalDetectionOptions options,
        CancellationToken cancellationToken)
    {
        int expected = checked(
            proposals.Count * NormalizedMarkerProposalPostprocessContract.OutputColumnCount);
        if (output.Count != expected)
        {
            throw new MarkerPipelineException(
                "MARKER_MODEL_OUTPUT_SHAPE_MISMATCH",
                $"Normalized marker model returned {output.Count} values; NC output requires {expected}.",
                false,
                "select_compatible_model");
        }

        var candidates = new List<MarkerCandidate>();
        int rawCandidateCount = 0;
        for (var index = 0; index < proposals.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int offset = index * NormalizedMarkerProposalPostprocessContract.OutputColumnCount;
            float probability = output[offset];
            float offsetX = output[offset + 1];
            float offsetY = output[offset + 2];
            float radiusValue = output[offset + 3];
            if (!float.IsFinite(probability) || probability < 0 || probability > 1 ||
                !float.IsFinite(offsetX) ||
                offsetX < NormalizedMarkerProposalPostprocessContract.MinimumOffset ||
                offsetX > NormalizedMarkerProposalPostprocessContract.MaximumOffset ||
                !float.IsFinite(offsetY) ||
                offsetY < NormalizedMarkerProposalPostprocessContract.MinimumOffset ||
                offsetY > NormalizedMarkerProposalPostprocessContract.MaximumOffset ||
                !float.IsFinite(radiusValue) ||
                radiusValue < NormalizedMarkerProposalPostprocessContract.MinimumRadius ||
                radiusValue > NormalizedMarkerProposalPostprocessContract.MaximumRadius)
            {
                throw new MarkerPipelineException(
                    "MARKER_MODEL_OUTPUT_INVALID",
                    "Normalized marker output violates the activated probability, offset, or radius contract.",
                    false,
                    "select_compatible_model");
            }

            if (probability < options.MarkerThreshold)
            {
                continue;
            }

            rawCandidateCount++;
            MarkerPoint proposal = proposals.Coordinates[index];
            double x = proposal.X + (offsetX * NormalizedMarkerProposalContract.ProposalStride);
            double y = proposal.Y + (offsetY * NormalizedMarkerProposalContract.ProposalStride);
            double radius = Math.Clamp(
                (double)radiusValue,
                NormalizedMarkerProposalPostprocessContract.MinimumRadius,
                NormalizedMarkerProposalPostprocessContract.MaximumRadius);
            MarkerPoint? refined = RefineGeometryCenter(frame, x, y, radius);
            if (refined is null)
            {
                continue;
            }

            double artifactProbability = Math.Max(
                WindowMaximum(frame.OcrMask.Values.Span, frame.Width, frame.Height, refined.Value, 2),
                WindowMaximum(frame.ArtifactMask.Values.Span, frame.Width, frame.Height, refined.Value, 2));
            candidates.Add(new MarkerCandidate(
                refined.Value,
                radius,
                artifactProbability,
                probability,
                MarkerSourceImage.Original));
        }

        var accepted = new List<MarkerCandidate>();
        foreach (MarkerCandidate candidate in candidates
            .OrderByDescending(static candidate => candidate.CenterConfidence)
            .ThenBy(static candidate => candidate.Center.Y)
            .ThenBy(static candidate => candidate.Center.X))
        {
            bool suppressed = accepted.Any(current =>
                Distance(candidate.Center, current.Center) < Math.Max(
                    options.MinimumCenterSeparation,
                    options.RadiusSuppressionScale * Math.Max(candidate.Radius, current.Radius)));
            if (!suppressed)
            {
                accepted.Add(candidate);
            }
        }

        return new MarkerDecodeResult(
            MarkerCollections.Freeze(accepted
                .OrderBy(static candidate => candidate.Center.Y)
                .ThenBy(static candidate => candidate.Center.X)
                .ThenByDescending(static candidate => candidate.CenterConfidence)),
            rawCandidateCount);
    }

    private static MarkerPoint? RefineGeometryCenter(
        MarkerImageFrame frame,
        double x,
        double y,
        double radius)
    {
        var center = new MarkerPoint(x, y);
        if (!CenterIsUnmasked(frame, center))
        {
            return null;
        }

        if (HasMarkerGeometryConsensus(frame, center, radius))
        {
            return center;
        }

        MarkerPoint? best = null;
        (double Distance, double AbsY, double AbsX, double Y, double X) bestKey = default;
        bool found = false;
        foreach (double dy in RefinementOffsets)
        {
            foreach (double dx in RefinementOffsets)
            {
                var refined = new MarkerPoint(x + dx, y + dy);
                if (!CenterIsUnmasked(frame, refined) ||
                    !HasMarkerGeometryConsensus(frame, refined, radius))
                {
                    continue;
                }

                var key = ((dx * dx) + (dy * dy), Math.Abs(dy), Math.Abs(dx), dy, dx);
                if (!found || key.CompareTo(bestKey) < 0)
                {
                    found = true;
                    bestKey = key;
                    best = refined;
                }
            }
        }

        return best;
    }

    private static bool CenterIsUnmasked(MarkerImageFrame frame, MarkerPoint point)
    {
        int x = checked((int)Math.Round(point.X, MidpointRounding.ToEven));
        int y = checked((int)Math.Round(point.Y, MidpointRounding.ToEven));
        return x >= 0 && x < frame.Width && y >= 0 && y < frame.Height &&
            WindowMaximum(frame.OcrMask.Values.Span, frame.Width, frame.Height, point, 2) <
                NormalizedMarkerProposalContract.MaskRejectionThreshold &&
            WindowMaximum(frame.ArtifactMask.Values.Span, frame.Width, frame.Height, point, 2) <
                NormalizedMarkerProposalContract.MaskRejectionThreshold;
    }

    private static bool HasMarkerGeometryConsensus(
        MarkerImageFrame frame,
        MarkerPoint point,
        double radius)
    {
        int x = checked((int)Math.Round(point.X, MidpointRounding.ToEven));
        int y = checked((int)Math.Round(point.Y, MidpointRounding.ToEven));
        int ringRadius = Math.Max(3, checked((int)Math.Round(radius, MidpointRounding.ToEven)));
        (int X, int Y)[] points =
        [
            (x - ringRadius, y), (x + ringRadius, y),
            (x, y - ringRadius), (x, y + ringRadius),
            (x - ringRadius, y - ringRadius), (x + ringRadius, y - ringRadius),
            (x - ringRadius, y + ringRadius), (x + ringRadius, y + ringRadius),
        ];
        int support = 0;
        ReadOnlySpan<float> luminance = frame.ChannelsFirstPixels.Span;
        foreach ((int sampleX, int sampleY) in points)
        {
            if (sampleX >= 0 && sampleX < frame.Width && sampleY >= 0 && sampleY < frame.Height &&
                1f - luminance[(sampleY * frame.Width) + sampleX] >=
                    NormalizedMarkerProposalPostprocessContract.GeometryInkThreshold)
            {
                support++;
            }
        }

        int left = Math.Max(0, x - 2);
        int top = Math.Max(0, y - 2);
        int right = Math.Min(frame.Width, x + 3);
        int bottom = Math.Min(frame.Height, y + 3);
        float sum = 0;
        int count = 0;
        for (int sampleY = top; sampleY < bottom; sampleY++)
        {
            for (int sampleX = left; sampleX < right; sampleX++)
            {
                sum += 1f - luminance[(sampleY * frame.Width) + sampleX];
                count++;
            }
        }

        float centerDensity = sum / count;
        return support >= 3 ||
            centerDensity >= NormalizedMarkerProposalPostprocessContract.GeometryCenterDensityThreshold;
    }

    private static double WindowMaximum(
        ReadOnlySpan<float> values,
        int width,
        int height,
        MarkerPoint point,
        int radius)
    {
        int centerX = checked((int)Math.Round(point.X, MidpointRounding.ToEven));
        int centerY = checked((int)Math.Round(point.Y, MidpointRounding.ToEven));
        int left = Math.Max(0, centerX - radius);
        int top = Math.Max(0, centerY - radius);
        int right = Math.Min(width, centerX + radius + 1);
        int bottom = Math.Min(height, centerY + radius + 1);
        float maximum = float.NegativeInfinity;
        for (int y = top; y < bottom; y++)
        {
            for (int x = left; x < right; x++)
            {
                maximum = Math.Max(maximum, values[(y * width) + x]);
            }
        }

        return maximum;
    }

    private static double Distance(MarkerPoint left, MarkerPoint right)
    {
        double x = left.X - right.X;
        double y = left.Y - right.Y;
        return Math.Sqrt((x * x) + (y * y));
    }
}
