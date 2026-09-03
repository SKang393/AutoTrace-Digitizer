// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security.Cryptography;
using System.Text.Json.Serialization;
using GraphReader.Inference;
using GraphReader.Markers.Detection;

namespace GraphReader.App.Integration.Workflow;

internal interface IProposalMarkerInferenceRunner
{
    ValueTask<InferenceResponse> RunAsync(InferenceRequest request, CancellationToken cancellationToken);
}

internal sealed record ProposalMarkerPrediction(
    MarkerPoint Center,
    double Radius,
    double Confidence);

public sealed record ProposalMarkerStageCounters(
    int ProposalGridPositionsConsidered,
    int LowInkRejects,
    int OcrMaskRejects,
    int ArtifactMaskRejects,
    int EmittedProposals,
    int InferenceOutputs,
    [property: JsonPropertyName("outputs_above_0_25")] int OutputsAbove025,
    int DecodedPointsMasked,
    [property: JsonPropertyName("geometry_consensus_rejects_after_1px_refinement_attempts")] int GeometryConsensusRejectsAfterRefinementAttempts,
    int DecodedPointsOutsidePlot,
    int CandidatesBeforeNms,
    int NmsSuppressions,
    int FinalCandidates);

public sealed record ProposalMarkerCandidateDiagnosticResult(
    IReadOnlyList<MarkerCenter> Candidates,
    ProposalMarkerStageCounters StageCounters,
    [property: JsonIgnore] IReadOnlyList<MarkerCenter> PreNmsCandidates);

/// <summary>
/// Candidate-only integration for the checksum-bound runtime-consistency-v2 P2
/// proposal payload. It is intentionally never composed into production unless
/// a future maintainer supplies an explicit approval boundary.
/// </summary>
public sealed class ProductionProposalMarkerCenterAdapter : IProductionMarkerCenterAdapter
{
    public const string CandidateRevision = "marker-center-runtime-consistency-v2";
    public const string CandidateId = "P2";
    public const string ExpectedModelSha256 = "924c555e2f27955c644143125d7abd3b05859ea9928ab9d1e741e0544fa19e8b";
    public const string MultiradiusCandidateRevision = "marker-center-multiradius-geometry-v23";
    public const string MultiradiusCandidateId = "P1";
    public const string ExpectedMultiradiusModelSha256 = "0b413db48f8e6707ee5ec99afff4cd8ec3d25c6b8a8d9f165bd416deb4578a38";
    public const float CenterThreshold = 0.25f;
    public const int PatchSize = 33;
    public const int ProposalStride = 4;
    public const int BatchSize = 256;
    internal const double MinimumCenterSeparationForTesting = 6.5;
    private const float InkSupportThreshold = 0.11f;
    private const float MaskRejectionThreshold = 0.35f;
    private const double MinimumCenterSeparation = 6.5;
    private const double RadiusSuppressionScale = 1.25;
    public const int MaximumDecodedCandidates = 100_000;

    private readonly IProposalMarkerInferenceRunner inference;
    private readonly int maximumDecodedCandidates;
    private readonly bool multiradiusGeometry;

    public static ProductionProposalMarkerCenterAdapter CreateCandidate(
        ModelIdentity model,
        InferenceRuntime runtime)
    {
        VerifyPayload(model);
        ArgumentNullException.ThrowIfNull(runtime);
        return new ProductionProposalMarkerCenterAdapter(
            model,
            new RuntimeProposalMarkerInferenceRunner(runtime));
    }

    public static ProductionProposalMarkerCenterAdapter CreateMultiradiusCandidate(
        ModelIdentity model,
        InferenceRuntime runtime)
    {
        VerifyMultiradiusPayload(model);
        ArgumentNullException.ThrowIfNull(runtime);
        return new ProductionProposalMarkerCenterAdapter(
            model,
            new RuntimeProposalMarkerInferenceRunner(runtime),
            multiradiusGeometry: true);
    }

    internal ProductionProposalMarkerCenterAdapter(
        ModelIdentity model,
        IProposalMarkerInferenceRunner inference,
        bool multiradiusGeometry = false,
        int? maximumDecodedCandidates = null)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        Model.Validate();
        this.multiradiusGeometry = multiradiusGeometry;
        string expectedModelSha256 = multiradiusGeometry ? ExpectedMultiradiusModelSha256 : ExpectedModelSha256;
        if (!string.Equals(Model.Sha256, expectedModelSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(multiradiusGeometry
                ? "The proposal marker payload is not the checksum-bound multiradius V23 P1 model."
                : "The proposal marker payload is not the checksum-bound runtime-consistency-v2 P2 model.");
        }

        this.inference = inference ?? throw new ArgumentNullException(nameof(inference));
        IsApproved = false;
        this.maximumDecodedCandidates = maximumDecodedCandidates ?? MaximumDecodedCandidates;
        if (this.maximumDecodedCandidates <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumDecodedCandidates));
        }
    }

    public string AdapterId => $"graphreader-marker-center-proposal:{Model.Sha256[..12].ToLowerInvariant()}";

    public bool IsApproved { get; }

    public ModelIdentity Model { get; }

    private sealed class RuntimeProposalMarkerInferenceRunner(InferenceRuntime runtime)
        : IProposalMarkerInferenceRunner
    {
        public ValueTask<InferenceResponse> RunAsync(
            InferenceRequest request,
            CancellationToken cancellationToken) =>
            runtime.RunAsync(request, cancellationToken);
    }

    private static void VerifyPayload(ModelIdentity model)
    {
        ArgumentNullException.ThrowIfNull(model);
        model.Validate();
        if (!File.Exists(model.FilePath))
        {
            throw new FileNotFoundException("The checksum-bound proposal marker model is missing.", model.FilePath);
        }

        string actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(model.FilePath)));
        if (!string.Equals(actual, ExpectedModelSha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(model.Sha256, ExpectedModelSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The proposal marker model bytes do not match the checksum-bound P2 payload.");
        }
    }

    private static void VerifyMultiradiusPayload(ModelIdentity model)
    {
        ArgumentNullException.ThrowIfNull(model);
        model.Validate();
        if (!string.Equals(model.ModelId, MultiradiusCandidateRevision, StringComparison.Ordinal) ||
            !string.Equals(model.Version, MultiradiusCandidateId, StringComparison.Ordinal) ||
            !string.Equals(model.Sha256, ExpectedMultiradiusModelSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The proposal marker payload identity is not the checksum-bound multiradius V23 P1 model.");
        }

        if (!File.Exists(model.FilePath))
        {
            throw new FileNotFoundException("The checksum-bound multiradius proposal marker model is missing.", model.FilePath);
        }

        string actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(model.FilePath)));
        if (!string.Equals(actual, ExpectedMultiradiusModelSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The proposal marker model bytes do not match the checksum-bound multiradius V23 P1 payload.");
        }
    }

    public Task<ProductionMarkerCenterEvidence> DetectAsync(
        ProductionWorkflowDetectionRequest request,
        MarkerImageFrame originalImage,
        MarkerPolygon plotPolygon,
        MarkerImageFrame? enhancedImage,
        IReadOnlyList<WorkflowTransformProvenance>? enhancedTransforms,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(originalImage);
        ArgumentNullException.ThrowIfNull(plotPolygon);
        cancellationToken.ThrowIfCancellationRequested();
        throw Failure(
            ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
            "Errors.ModelNotFound",
            $"Marker-center adapter '{AdapterId}' is candidate-only and not production-approved.",
            "Use the candidate-only evaluation method or continue in manual mode.");
    }

    /// <summary>
    /// Runs the P2 candidate for private real-dev diagnosis. This method never
    /// changes or implies the production approval state.
    /// </summary>
    public async Task<IReadOnlyList<MarkerCenter>> DetectCandidateAsync(
        MarkerImageFrame frame,
        MarkerPolygon plotPolygon,
        CancellationToken cancellationToken)
        => (await DetectCandidateWithDiagnosticsAsync(frame, plotPolygon, cancellationToken).ConfigureAwait(false)).Candidates;

    public async Task<ProposalMarkerCandidateDiagnosticResult> DetectCandidateWithDiagnosticsAsync(
        MarkerImageFrame frame,
        MarkerPolygon plotPolygon,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(plotPolygon);
        ValidateFrame(frame);
        cancellationToken.ThrowIfCancellationRequested();

        var counters = new ProposalMarkerStageCounterAccumulator();
        var predictions = new List<ProposalMarkerPrediction>();
        var batch = new List<Proposal>(BatchSize);
        int batchOffset = 0;

        async Task InferBatchAsync(List<Proposal> proposals)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = proposals.Count;
            float[] values = new float[checked(count * 3 * PatchSize * PatchSize)];
            for (int index = 0; index < count; index++)
            {
                proposals[index].Patch.CopyTo(values.AsSpan(index * 3 * PatchSize * PatchSize));
            }

            InferenceResponse response = await inference.RunAsync(
                    new InferenceRequest(
                        Model,
                        new InferenceInput(values, [count, 3, PatchSize, PatchSize], "candidate_patches", "candidate_predictions"),
                        new StageCacheMaterial(
                            "candidate-only",
                            "proposal-patches",
                            frame.SourceImage.ToString(),
                            multiradiusGeometry ? "marker_center_candidate_v23" : "marker_center_candidate_p2",
                            multiradiusGeometry ? MultiradiusCandidateRevision : CandidateRevision,
                            new Dictionary<string, object?>(StringComparer.Ordinal)
                            {
                                ["candidate_id"] = multiradiusGeometry ? MultiradiusCandidateId : CandidateId,
                                ["threshold"] = CenterThreshold,
                                ["batch_offset"] = batchOffset,
                                ["batch_count"] = count,
                            },
                            MarkerContract.Version),
                        TimeSpan.FromSeconds(30),
                        [InferenceProvider.Cpu],
                        BypassCache: true),
                    cancellationToken)
                .ConfigureAwait(false);
            if (!response.Succeeded || response.Execution is null || response.Execution.Provider != InferenceProvider.Cpu)
            {
                throw new InvalidDataException("The proposal marker candidate requires successful CPU inference evidence.");
            }

            IReadOnlyList<float> output = response.Execution.Output;
            if (output.Count != checked(count * 4))
            {
                throw new InvalidDataException("The proposal marker candidate must return [N,4] output.");
            }

            counters.InferenceOutputs += count;

            for (int index = 0; index < count; index++)
            {
                int baseIndex = index * 4;
                float probability = output[baseIndex];
                float offsetX = output[baseIndex + 1];
                float offsetY = output[baseIndex + 2];
                float radius = output[baseIndex + 3];
                if (!float.IsFinite(probability) || probability is < 0 or > 1 ||
                    !float.IsFinite(offsetX) || !float.IsFinite(offsetY) ||
                    !float.IsFinite(radius) || radius < 0)
                {
                    throw new InvalidDataException("The proposal marker candidate returned invalid output values.");
                }

                if (probability < CenterThreshold)
                {
                    continue;
                }

                counters.OutputsAbove025++;

                Proposal proposal = proposals[index];
                double x = proposal.X + (offsetX * ProposalStride);
                double y = proposal.Y + (offsetY * ProposalStride);
                double decodedRadius = Math.Clamp(radius, 2.5, 8.0);
                if (!TryRefine(frame, x, y, decodedRadius, multiradiusGeometry, out MarkerPoint refined, out RefinementFailure failure))
                {
                    if (failure == RefinementFailure.Masked)
                    {
                        counters.DecodedPointsMasked++;
                    }
                    else
                    {
                        counters.GeometryConsensusRejectsAfterRefinementAttempts++;
                    }
                    continue;
                }

                if (plotPolygon.Contains(frame.OriginalToFrame.MapToOriginal(refined)))
                {
                    if (predictions.Count >= maximumDecodedCandidates)
                    {
                        throw new InvalidDataException("The proposal marker candidate exceeded its decoded-candidate limit.");
                    }

                    predictions.Add(new ProposalMarkerPrediction(refined, decodedRadius, probability));
                }
                else
                {
                    counters.DecodedPointsOutsidePlot++;
                }
            }

            batchOffset += count;
        }

        foreach (Proposal proposal in EnumerateProposals(frame, plotPolygon, counters, cancellationToken))
        {
            batch.Add(proposal);
            if (batch.Count == BatchSize)
            {
                await InferBatchAsync(batch).ConfigureAwait(false);
                batch.Clear();
            }
        }
        if (batch.Count > 0)
        {
            await InferBatchAsync(batch).ConfigureAwait(false);
        }

        List<ProposalMarkerPrediction> accepted = ApplyNms(predictions, out int nmsSuppressions);
        counters.CandidatesBeforeNms = predictions.Count;
        counters.NmsSuppressions = nmsSuppressions;
        counters.FinalCandidates = accepted.Count;
        IReadOnlyList<MarkerCenter> preNmsCandidates = predictions
            .Select((candidate, index) => ToMarkerCenter(frame, candidate, index, multiradiusGeometry, "pre-nms"))
            .ToArray();
        IReadOnlyList<MarkerCenter> candidates = accepted
            .OrderBy(candidate => candidate.Center.Y)
            .ThenBy(candidate => candidate.Center.X)
            .ThenByDescending(candidate => candidate.Confidence)
            .Select((candidate, index) => ToMarkerCenter(frame, candidate, index, multiradiusGeometry, suffix: null))
            .ToArray();
        return new ProposalMarkerCandidateDiagnosticResult(candidates, counters.ToRecord(), preNmsCandidates);
    }

    private static MarkerCenter ToMarkerCenter(
        MarkerImageFrame frame,
        ProposalMarkerPrediction candidate,
        int index,
        bool multiradiusGeometry,
        string? suffix)
    {
        string prefix = multiradiusGeometry ? "candidate-v23-p1" : "candidate-p2";
        string markerId = $"{prefix}{(suffix is null ? string.Empty : $"-{suffix}")}-{index.ToString(System.Globalization.CultureInfo.InvariantCulture)}";
        return new MarkerCenter(
            markerId,
            frame.OriginalToFrame.MapToOriginal(candidate.Center),
            frame.OriginalToFrame.MapFrameRadiusToOriginal(candidate.Radius),
            0,
            candidate.Confidence,
            frame.SourceImage,
            MarkerContract.CoordinateSpace);
    }

    private sealed record Proposal(int X, int Y, float[] Patch);

    private enum RefinementFailure
    {
        Masked,
        GeometryConsensus,
    }

    private sealed class ProposalMarkerStageCounterAccumulator
    {
        public int ProposalGridPositionsConsidered;
        public int LowInkRejects;
        public int OcrMaskRejects;
        public int ArtifactMaskRejects;
        public int EmittedProposals;
        public int InferenceOutputs;
        public int OutputsAbove025;
        public int DecodedPointsMasked;
        public int GeometryConsensusRejectsAfterRefinementAttempts;
        public int DecodedPointsOutsidePlot;
        public int CandidatesBeforeNms;
        public int NmsSuppressions;
        public int FinalCandidates;

        public ProposalMarkerStageCounters ToRecord() => new(
            ProposalGridPositionsConsidered,
            LowInkRejects,
            OcrMaskRejects,
            ArtifactMaskRejects,
            EmittedProposals,
            InferenceOutputs,
            OutputsAbove025,
            DecodedPointsMasked,
            GeometryConsensusRejectsAfterRefinementAttempts,
            DecodedPointsOutsidePlot,
            CandidatesBeforeNms,
            NmsSuppressions,
            FinalCandidates);
    }

    private static IEnumerable<Proposal> EnumerateProposals(
        MarkerImageFrame frame,
        MarkerPolygon plotPolygon,
        ProposalMarkerStageCounterAccumulator counters,
        CancellationToken cancellationToken)
    {
        int width = frame.Width;
        int height = frame.Height;
        int gridWidth = (width + ProposalStride - 1) / ProposalStride;
        int gridHeight = (height + ProposalStride - 1) / ProposalStride;
        var framePolygon = plotPolygon.Points.Select(frame.OriginalToFrame.MapFromOriginal).ToArray();
        int minimumX = Math.Max(0, (int)Math.Floor(framePolygon.Min(static point => point.X)) - PatchSize / 2);
        int maximumX = Math.Min(width - 1, (int)Math.Ceiling(framePolygon.Max(static point => point.X)) + PatchSize / 2);
        int minimumY = Math.Max(0, (int)Math.Floor(framePolygon.Min(static point => point.Y)) - PatchSize / 2);
        int maximumY = Math.Min(height - 1, (int)Math.Ceiling(framePolygon.Max(static point => point.Y)) + PatchSize / 2);
        int minimumGridX = Math.Max(0, (minimumX - ProposalStride + 1) / ProposalStride);
        int maximumGridX = Math.Min(gridWidth - 1, maximumX / ProposalStride);
        int minimumGridY = Math.Max(0, (minimumY - ProposalStride + 1) / ProposalStride);
        int maximumGridY = Math.Min(gridHeight - 1, maximumY / ProposalStride);
        ReadOnlyMemory<float> luminance = frame.ChannelsFirstPixels;
        ReadOnlyMemory<float> text = frame.OcrMask.Values;
        ReadOnlyMemory<float> artifact = frame.ArtifactMask.Values;
        for (int gy = minimumGridY; gy <= maximumGridY; gy++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int gx = minimumGridX; gx <= maximumGridX; gx++)
            {
                int x = gx * ProposalStride;
                int y = gy * ProposalStride;
                counters.ProposalGridPositionsConsidered++;
                if (WindowMaxInk(luminance, width, height, x, y, 8) < InkSupportThreshold)
                {
                    counters.LowInkRejects++;
                    continue;
                }
                if (WindowMax(text, width, height, x, y, 2) >= MaskRejectionThreshold)
                {
                    counters.OcrMaskRejects++;
                    continue;
                }
                if (WindowMax(artifact, width, height, x, y, 2) >= MaskRejectionThreshold)
                {
                    counters.ArtifactMaskRejects++;
                    continue;
                }

                float[] patch = new float[checked(3 * PatchSize * PatchSize)];
                int planeSize = PatchSize * PatchSize;
                for (int py = 0; py < PatchSize; py++)
                {
                    for (int px = 0; px < PatchSize; px++)
                    {
                        int sourceX = x + px - (PatchSize / 2);
                        int sourceY = y + py - (PatchSize / 2);
                        int patchIndex = py * PatchSize + px;
                        if ((uint)sourceX < (uint)width && (uint)sourceY < (uint)height)
                        {
                            int sourceIndex = sourceY * width + sourceX;
                            patch[patchIndex] = 1 - luminance.Span[sourceIndex];
                            patch[planeSize + patchIndex] = text.Span[sourceIndex];
                            patch[(2 * planeSize) + patchIndex] = artifact.Span[sourceIndex];
                        }
                    }
                }
                counters.EmittedProposals++;
                yield return new Proposal(x, y, patch);
            }
        }
    }

    private static float WindowMax(ReadOnlyMemory<float> values, int width, int height, int centerX, int centerY, int radius)
    {
        float maximum = 0;
        for (int y = Math.Max(0, centerY - radius); y <= Math.Min(height - 1, centerY + radius); y++)
        {
            for (int x = Math.Max(0, centerX - radius); x <= Math.Min(width - 1, centerX + radius); x++)
            {
                maximum = Math.Max(maximum, values.Span[(y * width) + x]);
            }
        }
        return maximum;
    }

    private static float WindowMaxInk(ReadOnlyMemory<float> luminance, int width, int height, int centerX, int centerY, int radius)
    {
        float maximum = 0;
        for (int y = Math.Max(0, centerY - radius); y <= Math.Min(height - 1, centerY + radius); y++)
        {
            for (int x = Math.Max(0, centerX - radius); x <= Math.Min(width - 1, centerX + radius); x++)
            {
                maximum = Math.Max(maximum, 1 - luminance.Span[(y * width) + x]);
            }
        }
        return maximum;
    }

    private static bool TryRefine(
        MarkerImageFrame frame,
        double x,
        double y,
        double radius,
        bool multiradiusGeometry,
        out MarkerPoint refined,
        out RefinementFailure failure)
    {
        if (!CenterIsUnmasked(frame, x, y))
        {
            refined = default;
            failure = RefinementFailure.Masked;
            return false;
        }

        if (GeometryConsensus(frame, x, y, radius, multiradiusGeometry))
        {
            refined = new MarkerPoint(x, y);
            failure = default;
            return true;
        }

        var candidates = new List<(double Distance, double AbsY, double AbsX, double Dy, double Dx, double X, double Y)>();
        foreach (double dy in new[] { -1d, 0d, 1d })
        {
            foreach (double dx in new[] { -1d, 0d, 1d })
            {
                double candidateX = x + dx;
                double candidateY = y + dy;
                if (CenterIsUnmasked(frame, candidateX, candidateY) &&
                    GeometryConsensus(frame, candidateX, candidateY, radius, multiradiusGeometry))
                {
                    candidates.Add((dx * dx + dy * dy, Math.Abs(dy), Math.Abs(dx), dy, dx, candidateX, candidateY));
                }
            }
        }
        if (candidates.Count > 0)
        {
            var best = candidates.Min();
            refined = new MarkerPoint(best.X, best.Y);
            failure = default;
            return true;
        }
        refined = default;
        failure = RefinementFailure.GeometryConsensus;
        return false;
    }

    private static bool CenterIsUnmasked(MarkerImageFrame frame, double x, double y)
    {
        int ix = (int)Math.Round(x);
        int iy = (int)Math.Round(y);
        return ix >= 0 && iy >= 0 && ix < frame.Width && iy < frame.Height &&
            WindowMax(frame.OcrMask.Values, frame.Width, frame.Height, ix, iy, 2) < MaskRejectionThreshold &&
            WindowMax(frame.ArtifactMask.Values, frame.Width, frame.Height, ix, iy, 2) < MaskRejectionThreshold;
    }

    private static bool GeometryConsensus(
        MarkerImageFrame frame,
        double x,
        double y,
        double radius,
        bool multiradiusGeometry)
    {
        if (multiradiusGeometry)
        {
            for (int ring = 3; ring <= 12; ring++)
            {
                if (GeometryConsensusAtRadius(frame, x, y, ring))
                {
                    return true;
                }
            }

            return false;
        }

        return GeometryConsensusAtRadius(frame, x, y, radius);
    }

    private static bool GeometryConsensusAtRadius(MarkerImageFrame frame, double x, double y, double radius)
    {
        int ix = (int)Math.Round(x);
        int iy = (int)Math.Round(y);
        int ring = Math.Max(3, (int)Math.Round(radius));
        int[][] points =
        [
            [ix - ring, iy], [ix + ring, iy], [ix, iy - ring], [ix, iy + ring],
            [ix - ring, iy - ring], [ix + ring, iy - ring], [ix - ring, iy + ring], [ix + ring, iy + ring],
        ];
        ReadOnlySpan<float> luminance = frame.ChannelsFirstPixels.Span;
        int support = 0;
        foreach (int[] point in points)
        {
            if ((uint)point[0] < (uint)frame.Width && (uint)point[1] < (uint)frame.Height &&
                1 - luminance[(point[1] * frame.Width) + point[0]] >= 0.12f)
            {
                support++;
            }
        }
        int left = Math.Max(0, ix - 2);
        int top = Math.Max(0, iy - 2);
        int right = Math.Min(frame.Width, ix + 3);
        int bottom = Math.Min(frame.Height, iy + 3);
        double sum = 0;
        int count = 0;
        for (int py = top; py < bottom; py++)
        {
            for (int px = left; px < right; px++)
            {
                sum += 1 - luminance[(py * frame.Width) + px];
                count++;
            }
        }
        return support >= 3 || (count > 0 && sum / count >= 0.28);
    }

    private static List<ProposalMarkerPrediction> ApplyNms(
        IEnumerable<ProposalMarkerPrediction> values,
        out int suppressions)
    {
        var accepted = new List<ProposalMarkerPrediction>();
        suppressions = 0;
        var buckets = new Dictionary<(int X, int Y), List<ProposalMarkerPrediction>>();
        foreach (ProposalMarkerPrediction candidate in values
                     .OrderByDescending(static item => item.Confidence)
                     .ThenBy(static item => item.Center.Y)
                     .ThenBy(static item => item.Center.X))
        {
            int bucketX = (int)Math.Floor(candidate.Center.X / MinimumCenterSeparation);
            int bucketY = (int)Math.Floor(candidate.Center.Y / MinimumCenterSeparation);
            bool suppressed = false;
            for (int y = bucketY - 2; y <= bucketY + 2 && !suppressed; y++)
            {
                for (int x = bucketX - 2; x <= bucketX + 2 && !suppressed; x++)
                {
                    if (!buckets.TryGetValue((x, y), out List<ProposalMarkerPrediction>? neighbors))
                    {
                        continue;
                    }

                    suppressed = neighbors.Any(current => Distance(candidate.Center, current.Center) <
                        Math.Max(MinimumCenterSeparation, RadiusSuppressionScale * Math.Max(candidate.Radius, current.Radius)));
                }
            }
            if (suppressed)
            {
                suppressions++;
                continue;
            }

            accepted.Add(candidate);
            buckets.GetValueOrDefault((bucketX, bucketY))?.Add(candidate);
            if (!buckets.ContainsKey((bucketX, bucketY)))
            {
                buckets[(bucketX, bucketY)] = [candidate];
            }
        }

        return accepted;
    }

    private static double Distance(MarkerPoint left, MarkerPoint right) =>
        Math.Sqrt(Math.Pow(left.X - right.X, 2) + Math.Pow(left.Y - right.Y, 2));

    private static void ValidateFrame(MarkerImageFrame frame)
    {
        if (frame.Width <= 0 || frame.Height <= 0 || frame.ChannelCount != 1 ||
            !frame.OriginalToFrame.IsInvertible ||
            frame.ChannelsFirstPixels.Length != checked(frame.Width * frame.Height) ||
            frame.OcrMask.Width != frame.Width || frame.OcrMask.Height != frame.Height ||
            frame.OcrMask.Values.Length != checked(frame.Width * frame.Height) ||
            frame.ArtifactMask.Width != frame.Width || frame.ArtifactMask.Height != frame.Height ||
            frame.ArtifactMask.Values.Length != checked(frame.Width * frame.Height) ||
            !AreNormalized(frame.ChannelsFirstPixels.Span) ||
            !AreNormalized(frame.OcrMask.Values.Span) ||
            !AreNormalized(frame.ArtifactMask.Values.Span))
        {
            throw new ArgumentException("Candidate marker frame must contain one finite luminance plane and matching masks.", nameof(frame));
        }
    }

    private static bool AreNormalized(ReadOnlySpan<float> values)
    {
        foreach (float value in values)
        {
            if (!float.IsFinite(value) || value < 0 || value > 1)
            {
                return false;
            }
        }

        return true;
    }

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction) => new(new ProductionWorkflowFailure(
            code, userMessageKey, technicalMessage, Recoverable: true, suggestedAction));
}
