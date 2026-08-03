// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using GraphReader.Inference;

namespace GraphReader.Markers.Detection;

/// <summary>
/// Local marker-center runtime. It does not own the injected inference runner or model files.
/// </summary>
public sealed class MarkerCenterDetector : IMarkerDetectionService
{
    private readonly IMarkerInferenceRunner _inference;

    public MarkerCenterDetector(InferenceRuntime runtime)
        : this(new InferenceRuntimeMarkerRunner(runtime))
    {
    }

    public MarkerCenterDetector(IMarkerInferenceRunner inference) =>
        _inference = inference ?? throw new ArgumentNullException(nameof(inference));

    public async ValueTask<MarkerDetectionResult> DetectAsync(
        MarkerDetectionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var totalStopwatch = Stopwatch.StartNew();
        var runId = Guid.NewGuid().ToString();
        var validationFailure = Validate(request);
        if (validationFailure is not null)
        {
            return FailureResult(request, runId, validationFailure, totalStopwatch.Elapsed.TotalMilliseconds);
        }

        var frames = request.EnhancedImage is null
            ? new[] { request.OriginalImage }
            : new[] { request.OriginalImage, request.EnhancedImage };
        var runs = new List<MarkerFrameRun>(frames.Length);
        foreach (var frame in frames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            runs.Add(await RunFrameAsync(request, frame, cancellationToken).ConfigureAwait(false));
        }

        var successfulRuns = runs.Where(static run => run.Failure is null).ToArray();
        if (successfulRuns.Length == 0)
        {
            var primaryFailure = runs
                .Select(static run => run.Failure)
                .FirstOrDefault(static failure => failure is not null) ??
                Error(
                    "MARKER_INFERENCE_FAILED",
                    "The marker model returned no usable frame result.",
                    true,
                    "retry");
            totalStopwatch.Stop();
            return FailureResult(
                request,
                runId,
                primaryFailure,
                totalStopwatch.Elapsed.TotalMilliseconds,
                runs.Select(static run => run.Report));
        }

        var warnings = new List<string>();
        foreach (var run in runs.Where(static run => run.Failure is not null))
        {
            warnings.Add($"{run.Frame.SourceImage.ToString().ToLowerInvariant()}_frame_failed:{run.Failure!.Code}");
        }

        var postprocessStopwatch = Stopwatch.StartNew();
        IReadOnlyList<MarkerCandidate> assembled;
        var originalRun = successfulRuns.FirstOrDefault(static run => run.Frame.SourceImage == MarkerSourceImage.Original);
        var enhancedRun = successfulRuns.FirstOrDefault(static run => run.Frame.SourceImage == MarkerSourceImage.Enhanced);
        if (originalRun is not null && enhancedRun is not null)
        {
            assembled = BuildConsensus(
                originalRun.Candidates,
                enhancedRun.Candidates,
                request.Options,
                cancellationToken);
            if (assembled.Any(static candidate => candidate.SourceImage != MarkerSourceImage.Consensus))
            {
                warnings.Add("original_enhanced_disagreement_requires_review");
            }
        }
        else
        {
            var onlyRun = successfulRuns[0];
            var scale = request.EnhancedImage is null
                ? 1d
                : request.Options.UnmatchedSourceConfidenceScale;
            var disagreement = request.EnhancedImage is null
                ? MarkerDisagreementKind.None
                : onlyRun.Frame.SourceImage == MarkerSourceImage.Original
                    ? MarkerDisagreementKind.OriginalOnly
                    : MarkerDisagreementKind.EnhancedOnly;
            assembled = MarkerCollections.Freeze(onlyRun.Candidates.Select(candidate => candidate with
            {
                CenterConfidence = Math.Clamp(candidate.CenterConfidence * scale, 0, 1),
                ReviewState = disagreement == MarkerDisagreementKind.None
                    ? MarkerReviewState.Unreviewed
                    : MarkerReviewState.NeedsReview,
                Disagreement = disagreement,
            }));
            if (onlyRun.Frame.SourceImage == MarkerSourceImage.Enhanced)
            {
                warnings.Add("enhanced_only_evidence_requires_review");
            }
        }

        var markers = MarkerCollections.Freeze(assembled.Select(candidate => new MarkerCenter(
            Guid.NewGuid().ToString(),
            candidate.Center,
            candidate.Radius,
            candidate.ArtifactProbability,
            candidate.CenterConfidence,
            candidate.SourceImage,
            MarkerContract.CoordinateSpace,
            candidate.ReviewState,
            candidate.Disagreement)));
        postprocessStopwatch.Stop();
        totalStopwatch.Stop();
        var preprocessMilliseconds = runs.Sum(static run => run.Report.Timing.PreprocessMilliseconds);
        var inferenceMilliseconds = runs.Sum(static run => run.Report.Timing.InferenceMilliseconds);
        var postprocessMilliseconds =
            runs.Sum(static run => run.Report.Timing.PostprocessMilliseconds) +
            postprocessStopwatch.Elapsed.TotalMilliseconds;
        var providers = successfulRuns
            .Select(static run => run.Report.Provider)
            .Where(static provider => provider is not null)
            .Distinct()
            .ToArray();
        var provider = providers.Length == 1 ? providers[0] : null;
        var confidence = markers.Count == 0
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
            markers,
            new MarkerDetectionTiming(
                preprocessMilliseconds,
                inferenceMilliseconds,
                postprocessMilliseconds,
                totalStopwatch.Elapsed.TotalMilliseconds),
            Math.Clamp(confidence, 0, 1),
            MarkerCollections.Freeze(warnings),
            MarkerCollections.Freeze(runs.Select(static run => run.Report)),
            new MarkerModelReport(
                request.Model.ModelId,
                request.Model.Version,
                request.Model.Sha256,
                provider),
            null);
    }

    private async ValueTask<MarkerFrameRun> RunFrameAsync(
        MarkerDetectionRequest request,
        MarkerImageFrame frame,
        CancellationToken cancellationToken)
    {
        var frameStopwatch = Stopwatch.StartNew();
        MarkerTensorPreparation? preparation = null;
        InferenceResponse? response = null;
        var postprocessMilliseconds = 0d;
        try
        {
            preparation = MarkerTensorPipeline.Prepare(request, frame, cancellationToken);
            response = await _inference.RunAsync(preparation.Request, cancellationToken).ConfigureAwait(false);
            if (!response.Succeeded || response.Execution is null)
            {
                var failure = FromInferenceError(response.Error);
                frameStopwatch.Stop();
                return FrameFailure(
                    frame,
                    preparation,
                    response,
                    failure,
                    frameStopwatch.Elapsed.TotalMilliseconds);
            }

            var postprocessStopwatch = Stopwatch.StartNew();
            var decoded = MarkerTensorPipeline.Decode(
                response.Execution.Output,
                request,
                frame,
                preparation.FrameCrop,
                cancellationToken);
            postprocessStopwatch.Stop();
            postprocessMilliseconds = postprocessStopwatch.Elapsed.TotalMilliseconds;
            frameStopwatch.Stop();
            var report = new MarkerFrameReport(
                frame.SourceImage,
                preparation.CacheKey,
                response.Execution.Provider,
                MarkerCollections.Freeze(response.ProviderAttempts),
                new MarkerDetectionTiming(
                    preparation.PreprocessMilliseconds,
                    response.Execution.Timing.InferenceMilliseconds,
                    postprocessMilliseconds,
                    frameStopwatch.Elapsed.TotalMilliseconds),
                decoded.RawCandidateCount,
                decoded.Candidates.Count,
                response.Execution.Timing.CacheHit,
                null);
            return new MarkerFrameRun(frame, decoded.Candidates, report, null);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (MarkerPipelineException exception)
        {
            frameStopwatch.Stop();
            var failure = Error(
                exception.Code,
                exception.Message,
                exception.Recoverable,
                exception.SuggestedAction);
            return FrameFailure(
                frame,
                preparation,
                response,
                failure,
                frameStopwatch.Elapsed.TotalMilliseconds,
                postprocessMilliseconds);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            frameStopwatch.Stop();
            var failure = Error(
                "MARKER_INFERENCE_FAILED",
                exception.Message,
                true,
                "retry");
            return FrameFailure(
                frame,
                preparation,
                response,
                failure,
                frameStopwatch.Elapsed.TotalMilliseconds,
                postprocessMilliseconds);
        }
    }

    private static MarkerFrameRun FrameFailure(
        MarkerImageFrame frame,
        MarkerTensorPreparation? preparation,
        InferenceResponse? response,
        MarkerDetectionFailure failure,
        double totalMilliseconds,
        double postprocessMilliseconds = 0)
    {
        var report = new MarkerFrameReport(
            frame.SourceImage,
            preparation?.CacheKey ?? string.Empty,
            response?.Execution?.Provider,
            MarkerCollections.Freeze(response?.ProviderAttempts ?? Array.Empty<ProviderAttempt>()),
            new MarkerDetectionTiming(
                preparation?.PreprocessMilliseconds ?? 0,
                response?.Execution?.Timing.InferenceMilliseconds ?? 0,
                postprocessMilliseconds,
                totalMilliseconds),
            0,
            0,
            response?.Execution?.Timing.CacheHit ?? false,
            failure);
        return new MarkerFrameRun(frame, Array.Empty<MarkerCandidate>(), report, failure);
    }

    private static IReadOnlyList<MarkerCandidate> BuildConsensus(
        IReadOnlyList<MarkerCandidate> original,
        IReadOnlyList<MarkerCandidate> enhanced,
        MarkerDetectionOptions options,
        CancellationToken cancellationToken)
    {
        var matches = FindMinimumCostMaximumMatching(
            original,
            enhanced,
            options.ConsensusToleranceOriginalPixels,
            cancellationToken);
        var matchByOriginal = matches.ToDictionary(static match => match.OriginalIndex);
        var usedEnhanced = matches.Select(static match => match.EnhancedIndex).ToHashSet();
        var assembled = new List<MarkerCandidate>(original.Count + enhanced.Count);
        for (var originalIndex = 0; originalIndex < original.Count; originalIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var originalCandidate = original[originalIndex];
            if (!matchByOriginal.TryGetValue(originalIndex, out var match))
            {
                assembled.Add(originalCandidate with
                {
                    CenterConfidence = Math.Clamp(
                        originalCandidate.CenterConfidence * options.UnmatchedSourceConfidenceScale,
                        0,
                        1),
                    ReviewState = MarkerReviewState.NeedsReview,
                    Disagreement = MarkerDisagreementKind.OriginalOnly,
                });
                continue;
            }

            var enhancedCandidate = enhanced[match.EnhancedIndex];
            var originalWeight = Math.Max(originalCandidate.CenterConfidence, double.Epsilon);
            var enhancedWeight = Math.Max(enhancedCandidate.CenterConfidence, double.Epsilon);
            var weight = originalWeight + enhancedWeight;
            var agreementFactor = 1 - (0.2 * Math.Clamp(
                match.Cost / options.ConsensusToleranceOriginalPixels,
                0,
                1));
            assembled.Add(new MarkerCandidate(
                new MarkerPoint(
                    ((originalCandidate.Center.X * originalWeight) +
                     (enhancedCandidate.Center.X * enhancedWeight)) / weight,
                    ((originalCandidate.Center.Y * originalWeight) +
                     (enhancedCandidate.Center.Y * enhancedWeight)) / weight),
                ((originalCandidate.Radius * originalWeight) +
                 (enhancedCandidate.Radius * enhancedWeight)) / weight,
                ((originalCandidate.ArtifactProbability * originalWeight) +
                 (enhancedCandidate.ArtifactProbability * enhancedWeight)) / weight,
                Math.Clamp(
                    ((originalCandidate.CenterConfidence + enhancedCandidate.CenterConfidence) / 2) *
                    agreementFactor,
                    0,
                    1),
                MarkerSourceImage.Consensus));
        }

        for (var index = 0; index < enhanced.Count; index++)
        {
            if (!usedEnhanced.Contains(index))
            {
                assembled.Add(enhanced[index] with
                {
                    CenterConfidence = Math.Clamp(
                        enhanced[index].CenterConfidence * options.UnmatchedSourceConfidenceScale,
                        0,
                        1),
                    ReviewState = MarkerReviewState.NeedsReview,
                    Disagreement = MarkerDisagreementKind.EnhancedOnly,
                });
            }
        }

        return MarkerCollections.Freeze(assembled
            .OrderBy(static candidate => candidate.Center.Y)
            .ThenBy(static candidate => candidate.Center.X)
            .ThenByDescending(static candidate => candidate.CenterConfidence));
    }

    private static IReadOnlyList<ConsensusMatch> FindMinimumCostMaximumMatching(
        IReadOnlyList<MarkerCandidate> original,
        IReadOnlyList<MarkerCandidate> enhanced,
        double tolerance,
        CancellationToken cancellationToken)
    {
        var source = 0;
        var originalOffset = 1;
        var enhancedOffset = originalOffset + original.Count;
        var sink = enhancedOffset + enhanced.Count;
        var graph = Enumerable.Range(0, sink + 1)
            .Select(static _ => new List<FlowEdge>())
            .ToArray();
        for (var originalIndex = 0; originalIndex < original.Count; originalIndex++)
        {
            AddFlowEdge(graph, source, originalOffset + originalIndex, 0);
        }

        for (var enhancedIndex = 0; enhancedIndex < enhanced.Count; enhancedIndex++)
        {
            AddFlowEdge(graph, enhancedOffset + enhancedIndex, sink, 0);
        }

        var originalOrder = Enumerable.Range(0, original.Count)
            .OrderBy(index => original[index].Center.Y)
            .ThenBy(index => original[index].Center.X)
            .ThenByDescending(index => original[index].CenterConfidence)
            .ThenBy(static index => index)
            .ToArray();
        var enhancedOrder = Enumerable.Range(0, enhanced.Count)
            .OrderBy(index => enhanced[index].Center.Y)
            .ThenBy(index => enhanced[index].Center.X)
            .ThenByDescending(index => enhanced[index].CenterConfidence)
            .ThenBy(static index => index)
            .ToArray();
        foreach (var originalIndex in originalOrder)
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var enhancedIndex in enhancedOrder)
            {
                var cost = MarkerTensorPipeline.Distance(
                    original[originalIndex].Center,
                    enhanced[enhancedIndex].Center);
                if (cost <= tolerance)
                {
                    AddFlowEdge(
                        graph,
                        originalOffset + originalIndex,
                        enhancedOffset + enhancedIndex,
                        cost,
                        originalIndex,
                        enhancedIndex);
                }
            }
        }

        while (TryAugmentMinimumCostFlow(graph, source, sink, cancellationToken))
        {
        }

        var matches = new List<ConsensusMatch>();
        for (var originalIndex = 0; originalIndex < original.Count; originalIndex++)
        {
            foreach (var edge in graph[originalOffset + originalIndex])
            {
                if (edge.OriginalIndex == originalIndex && edge.EnhancedIndex >= 0 && edge.Capacity == 0)
                {
                    matches.Add(new ConsensusMatch(originalIndex, edge.EnhancedIndex, edge.Cost));
                }
            }
        }

        return MarkerCollections.Freeze(matches.OrderBy(static match => match.OriginalIndex));
    }

    private static bool TryAugmentMinimumCostFlow(
        IReadOnlyList<List<FlowEdge>> graph,
        int source,
        int sink,
        CancellationToken cancellationToken)
    {
        var distances = Enumerable.Repeat(double.PositiveInfinity, graph.Count).ToArray();
        var priorNodes = Enumerable.Repeat(-1, graph.Count).ToArray();
        var priorEdges = Enumerable.Repeat(-1, graph.Count).ToArray();
        distances[source] = 0;
        for (var iteration = 0; iteration < graph.Count - 1; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var changed = false;
            for (var node = 0; node < graph.Count; node++)
            {
                if (!double.IsFinite(distances[node]))
                {
                    continue;
                }

                for (var edgeIndex = 0; edgeIndex < graph[node].Count; edgeIndex++)
                {
                    var edge = graph[node][edgeIndex];
                    if (edge.Capacity == 0 || edge.To == source)
                    {
                        continue;
                    }

                    var candidateDistance = distances[node] + edge.Cost;
                    var improves = candidateDistance < distances[edge.To] - 1e-12;
                    if (!improves)
                    {
                        continue;
                    }

                    distances[edge.To] = candidateDistance;
                    priorNodes[edge.To] = node;
                    priorEdges[edge.To] = edgeIndex;
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }
        }

        if (priorNodes[sink] < 0)
        {
            return false;
        }

        for (var node = sink; node != source; node = priorNodes[node])
        {
            var priorNode = priorNodes[node];
            var edgeIndex = priorEdges[node];
            var edge = graph[priorNode][edgeIndex];
            edge.Capacity = 0;
            graph[node][edge.ReverseIndex].Capacity = 1;
        }

        return true;
    }

    private static void AddFlowEdge(
        IReadOnlyList<List<FlowEdge>> graph,
        int from,
        int to,
        double cost,
        int originalIndex = -1,
        int enhancedIndex = -1)
    {
        var forward = new FlowEdge(
            to,
            graph[to].Count,
            1,
            cost,
            originalIndex,
            enhancedIndex);
        var reverse = new FlowEdge(from, graph[from].Count, 0, -cost, -1, -1);
        graph[from].Add(forward);
        graph[to].Add(reverse);
    }

    private static MarkerDetectionFailure? Validate(MarkerDetectionRequest request)
    {
        if (request.Model is null || request.OriginalImage is null || request.PlotPolygon is null ||
            request.Options is null)
        {
            return Invalid("Model, original frame, plot polygon, and detection options are required.");
        }

        if (!Guid.TryParse(request.ProjectId, out _) || !Guid.TryParse(request.PanelId, out _))
        {
            return Invalid("ProjectId and PanelId must be UUID strings.");
        }

        if (request.InputSha256 is null ||
            request.InputSha256.Length != 64 ||
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

        var options = request.Options;
        var tensor = options.TensorContract;
        if (tensor is null)
        {
            return Invalid("Explicit marker model tensor metadata is required.");
        }

        if (string.IsNullOrWhiteSpace(tensor.InputName) || string.IsNullOrWhiteSpace(tensor.OutputName) ||
            tensor.InputWidth <= 0 || tensor.InputHeight <= 0 ||
            tensor.OutputWidth <= 0 || tensor.OutputHeight <= 0)
        {
            return Invalid("Marker tensor names and dimensions must be explicitly valid.");
        }

        if (tensor.InputChannelCount != 3 || tensor.OutputChannelCount != 3 ||
            tensor.InputLayout != MarkerTensorLayout.ChannelsFirst ||
            tensor.OutputLayout != MarkerTensorLayout.ChannelsFirst ||
            tensor.InputWidth != tensor.OutputWidth || tensor.InputHeight != tensor.OutputHeight ||
            tensor.CenterChannelIndex != 0 || tensor.RadiusChannelIndex != 1 || tensor.ArtifactChannelIndex != 2 ||
            tensor.CenterActivation != MarkerHeadActivation.Identity ||
            tensor.ArtifactActivation != MarkerHeadActivation.Identity)
        {
            return Invalid(
                "Marker runtime v2 requires one NCHW [1,3,H,W] input and one NCHW [1,3,H,W] output with center, radius, and artifact channels in that order at stride one.");
        }

        if (tensor.RadiusScale != 1 || tensor.NormalizeMean != 0 || tensor.NormalizeScale != 1)
        {
            return Invalid(
                "Marker runtime v2 requires activated [0,1] probability heads, radius pixels, and unnormalized [0,1] input channels.");
        }

        if (!float.IsFinite(options.CenterThreshold) || options.CenterThreshold < 0 || options.CenterThreshold > 1 ||
            !float.IsFinite(options.ArtifactThreshold) || options.ArtifactThreshold < 0 || options.ArtifactThreshold > 1 ||
            !float.IsFinite(options.MaskThreshold) || options.MaskThreshold < 0 || options.MaskThreshold > 1 ||
            options.LocalMaximumWindow != 9 ||
            options.MinimumRadiusGridPixels != 2.5 ||
            options.MinimumSuppressionDistanceGridPixels != 5 ||
            options.RadiusSuppressionScale != 1.25 ||
            !double.IsFinite(options.ConsensusToleranceOriginalPixels) || options.ConsensusToleranceOriginalPixels <= 0 ||
            !double.IsFinite(options.UnmatchedSourceConfidenceScale) || options.UnmatchedSourceConfidenceScale < 0 ||
            options.UnmatchedSourceConfidenceScale > 1 ||
            string.IsNullOrWhiteSpace(options.StageVersion) ||
            (options.Timeout <= TimeSpan.Zero && options.Timeout != Timeout.InfiniteTimeSpan))
        {
            return Invalid(
                "Marker thresholds, the fixed 9x9 radius-aware suppression contract, tolerances, stage version, or timeout are invalid.");
        }

        var originalFailure = ValidateFrame(request.OriginalImage, MarkerSourceImage.Original, tensor);
        if (originalFailure is not null)
        {
            return originalFailure;
        }

        if (request.EnhancedImage is not null)
        {
            var enhancedFailure = ValidateFrame(request.EnhancedImage, MarkerSourceImage.Enhanced, tensor);
            if (enhancedFailure is not null)
            {
                return enhancedFailure;
            }
        }

        return request.PlotPolygon.Bounds.IsValid
            ? null
            : Invalid("PlotPolygon must have finite positive bounds.");
    }

    private static MarkerDetectionFailure? ValidateFrame(
        MarkerImageFrame frame,
        MarkerSourceImage expectedSource,
        MarkerModelTensorContract tensor)
    {
        if (frame.SourceImage != expectedSource)
        {
            return Invalid($"The {expectedSource} frame has an incorrect source-image label.");
        }

        if (frame.OcrMask is null || frame.ArtifactMask is null ||
            frame.Width <= 0 || frame.Height <= 0 || frame.ChannelCount != 1 ||
            !frame.OriginalToFrame.IsInvertible)
        {
            return Invalid($"The {expectedSource} frame dimensions, channels, or inverse transform are invalid.");
        }

        int expectedPixelCount;
        try
        {
            expectedPixelCount = checked(frame.Width * frame.Height);
        }
        catch (OverflowException)
        {
            return Invalid($"The {expectedSource} frame dimensions overflow the supported tensor size.");
        }

        int expectedFrameValueCount;
        try
        {
            expectedFrameValueCount = checked(expectedPixelCount * frame.ChannelCount);
        }
        catch (OverflowException)
        {
            return Invalid($"The {expectedSource} frame channel count overflows the supported tensor size.");
        }

        if (frame.ChannelsFirstPixels.Length != expectedFrameValueCount ||
            frame.OcrMask.Width != frame.Width || frame.OcrMask.Height != frame.Height ||
            frame.OcrMask.Values.Length != expectedPixelCount ||
            frame.ArtifactMask.Width != frame.Width || frame.ArtifactMask.Height != frame.Height ||
            frame.ArtifactMask.Values.Length != expectedPixelCount)
        {
            return Invalid($"The {expectedSource} frame or mask lengths do not match declared dimensions.");
        }

        if (tensor.InputChannelCount != 3)
        {
            return Invalid(
                $"The {expectedSource} frame must supply one luminance channel for the fixed ink, text-mask, and artifact-mask input contract.");
        }

        if (!IsValidProbabilityPlane(frame.ChannelsFirstPixels.Span) ||
            !IsValidMask(frame.OcrMask.Values.Span) ||
            !IsValidMask(frame.ArtifactMask.Values.Span))
        {
            return Invalid($"The {expectedSource} frame contains non-finite pixels or masks outside [0,1].");
        }

        return null;
    }

    private static bool IsValidMask(ReadOnlySpan<float> values)
        => IsValidProbabilityPlane(values);

    private static bool IsValidProbabilityPlane(ReadOnlySpan<float> values)
    {
        foreach (var value in values)
        {
            if (!float.IsFinite(value) || value < 0 || value > 1)
            {
                return false;
            }
        }

        return true;
    }

    private static MarkerDetectionResult FailureResult(
        MarkerDetectionRequest request,
        string runId,
        MarkerDetectionFailure failure,
        double totalMilliseconds,
        IEnumerable<MarkerFrameReport>? frames = null) =>
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
            new MarkerDetectionTiming(0, 0, 0, totalMilliseconds),
            0,
            Array.Empty<string>(),
            MarkerCollections.Freeze(frames ?? Array.Empty<MarkerFrameReport>()),
            new MarkerModelReport(
                request.Model?.ModelId ?? string.Empty,
                request.Model?.Version ?? string.Empty,
                request.Model?.Sha256 ?? string.Empty,
                null),
            failure);

    private static MarkerDetectionFailure FromInferenceError(InferenceError? error) =>
        error is null
            ? Error(
                "MARKER_INFERENCE_FAILED",
                "The local inference runtime returned no marker-center result.",
                true,
                "retry")
            : new MarkerDetectionFailure(
                error.Code,
                error.Severity,
                error.UserMessageKey,
                error.TechnicalMessage,
                error.Recoverable,
                error.SuggestedAction);

    private static MarkerDetectionFailure Invalid(string technicalMessage) =>
        Error("MARKER_REQUEST_INVALID", technicalMessage, false, "review_marker_request");

    private static MarkerDetectionFailure Error(
        string code,
        string technicalMessage,
        bool recoverable,
        string suggestedAction) =>
        new(
            code,
            "error",
            "Errors." + code,
            technicalMessage,
            recoverable,
            suggestedAction);

    private sealed record ConsensusMatch(int OriginalIndex, int EnhancedIndex, double Cost);

    private sealed class FlowEdge
    {
        public FlowEdge(
            int to,
            int reverseIndex,
            int capacity,
            double cost,
            int originalIndex,
            int enhancedIndex)
        {
            To = to;
            ReverseIndex = reverseIndex;
            Capacity = capacity;
            Cost = cost;
            OriginalIndex = originalIndex;
            EnhancedIndex = enhancedIndex;
        }

        public int To { get; }

        public int ReverseIndex { get; }

        public int Capacity { get; set; }

        public double Cost { get; }

        public int OriginalIndex { get; }

        public int EnhancedIndex { get; }
    }

    private sealed record MarkerFrameRun(
        MarkerImageFrame Frame,
        IReadOnlyList<MarkerCandidate> Candidates,
        MarkerFrameReport Report,
        MarkerDetectionFailure? Failure);
}
