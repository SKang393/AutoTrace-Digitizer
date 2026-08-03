// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;
using GraphReader.Ocr;

namespace GraphReader.Legends;

public sealed class LegendRegionResolver : ILegendRegionResolver
{
    private const double MinimumOverlapTolerance = -2;

    public IReadOnlyList<LegendRegion> Resolve(
        LegendReasoningRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        if (!IsValidRequest(request))
        {
            return Array.Empty<LegendRegion>();
        }

        LegendGlyphCandidate[] glyphs = request.Glyphs
            .Where(IsValidGlyph)
            .Where(glyph => request.PanelBounds.Contains(glyph.Bounds.Center))
            .GroupBy(static glyph => glyph.GlyphId, StringComparer.Ordinal)
            .Select(static group => group
                .OrderByDescending(static glyph => glyph.Confidence)
                .ThenBy(static glyph => glyph.Bounds.Top)
                .ThenBy(static glyph => glyph.Bounds.Left)
                .First())
            .OrderBy(static glyph => glyph.GlyphId, StringComparer.Ordinal)
            .ToArray();

        LegendTextRegion[] texts = request.TextRegions
            .Where(IsEligibleText)
            .Where(text => request.PanelBounds.Contains(text.Bounds.Center))
            .GroupBy(static text => text.RegionId, StringComparer.Ordinal)
            .Select(static group => group
                .OrderByDescending(static text => text.Confidence)
                .ThenBy(static text => text.Bounds.Top)
                .ThenBy(static text => text.Bounds.Left)
                .First())
            .OrderBy(static text => text.RegionId, StringComparer.Ordinal)
            .ToArray();

        if (glyphs.Length == 0 || texts.Length == 0)
        {
            return Array.Empty<LegendRegion>();
        }

        List<PairCandidate> candidates = [];
        foreach (LegendGlyphCandidate glyph in glyphs)
        {
            cancellationToken.ThrowIfCancellationRequested();

            LegendRegionLocation glyphLocation = Locate(glyph.Bounds.Center, request.PlotBounds);
            foreach (LegendTextRegion text in texts)
            {
                cancellationToken.ThrowIfCancellationRequested();

                LegendRegionLocation textLocation = Locate(text.Bounds.Center, request.PlotBounds);
                if (glyphLocation != textLocation ||
                    !TryScorePair(glyph, text, request.Options, out PairCandidate candidate))
                {
                    continue;
                }

                candidates.Add(candidate with { Location = glyphLocation });
            }
        }

        HashSet<string> assignedGlyphs = new(StringComparer.Ordinal);
        HashSet<string> assignedTexts = new(StringComparer.Ordinal);
        List<ResolvedPair> pairs = [];

        foreach (PairCandidate candidate in candidates
                     .OrderByDescending(static candidate => candidate.Confidence)
                     .ThenBy(static candidate => candidate.VerticalOffset)
                     .ThenBy(static candidate => candidate.HorizontalGap)
                     .ThenBy(static candidate => candidate.Glyph.GlyphId, StringComparer.Ordinal)
                     .ThenBy(static candidate => candidate.Text.RegionId, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (!assignedGlyphs.Add(candidate.Glyph.GlyphId))
            {
                continue;
            }

            if (!assignedTexts.Add(candidate.Text.RegionId))
            {
                assignedGlyphs.Remove(candidate.Glyph.GlyphId);
                continue;
            }

            LegendRectangle bounds = LegendRectangle.Union(candidate.Glyph.Bounds, candidate.Text.Bounds);
            LegendSemanticEvidence semantic = LegendSemanticNormalizer.Normalize(
                candidate.Text.Text,
                candidate.Text.Confidence);
            LegendEntry entry = new(
                StableId(
                    "legend-entry",
                    request.ProjectId,
                    request.PanelId,
                    candidate.Glyph.GlyphId,
                    candidate.Text.RegionId),
                candidate.Glyph.GlyphId,
                candidate.Text.RegionId,
                candidate.Text.Text.Trim(),
                candidate.Glyph.Shape,
                candidate.Glyph.Fill,
                candidate.Confidence,
                request.PanelId,
                LegendEvidenceSource.DetectedLegend,
                semantic);

            pairs.Add(new ResolvedPair(entry, bounds, candidate.Location));
        }

        return BuildRegions(request, pairs, cancellationToken);
    }

    private static IReadOnlyList<LegendRegion> BuildRegions(
        LegendReasoningRequest request,
        IReadOnlyList<ResolvedPair> pairs,
        CancellationToken cancellationToken)
    {
        if (pairs.Count == 0)
        {
            return Array.Empty<LegendRegion>();
        }

        ResolvedPair[] orderedPairs = pairs
            .OrderBy(static pair => pair.Location)
            .ThenBy(static pair => pair.Bounds.Top)
            .ThenBy(static pair => pair.Bounds.Left)
            .ThenBy(static pair => pair.Entry.EntryId, StringComparer.Ordinal)
            .ToArray();

        DisjointSet components = new(orderedPairs.Length);
        for (int leftIndex = 0; leftIndex < orderedPairs.Length; leftIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            for (int rightIndex = leftIndex + 1; rightIndex < orderedPairs.Length; rightIndex++)
            {
                if (BelongsToSameRegion(orderedPairs[leftIndex], orderedPairs[rightIndex], request.Options))
                {
                    components.Union(leftIndex, rightIndex);
                }
            }
        }

        List<LegendRegion> regions = [];
        foreach (IGrouping<int, int> component in Enumerable.Range(0, orderedPairs.Length)
                     .GroupBy(components.Find)
                     .OrderBy(static group => group.Min()))
        {
            cancellationToken.ThrowIfCancellationRequested();

            ResolvedPair[] members = component
                .Select(index => orderedPairs[index])
                .OrderBy(static pair => pair.Bounds.Top)
                .ThenBy(static pair => pair.Bounds.Left)
                .ThenBy(static pair => pair.Entry.EntryId, StringComparer.Ordinal)
                .ToArray();

            LegendRectangle bounds = members[0].Bounds;
            for (int index = 1; index < members.Length; index++)
            {
                bounds = LegendRectangle.Union(bounds, members[index].Bounds);
            }

            string[] entryIds = members
                .Select(static member => member.Entry.EntryId)
                .Order(StringComparer.Ordinal)
                .ToArray();
            regions.Add(new LegendRegion(
                StableId(
                    "legend-region",
                    request.ProjectId,
                    request.PanelId,
                    members[0].Location.ToString(),
                    string.Join('|', entryIds)),
                bounds,
                members[0].Location,
                members.Select(static member => member.Entry),
                members.Average(static member => member.Entry.Confidence)));
        }

        return LegendCollections.Freeze(regions
            .OrderBy(static region => region.Location)
            .ThenBy(static region => region.Bounds.Top)
            .ThenBy(static region => region.Bounds.Left)
            .ThenBy(static region => region.RegionId, StringComparer.Ordinal));
    }

    private static bool TryScorePair(
        LegendGlyphCandidate glyph,
        LegendTextRegion text,
        LegendReasoningOptions options,
        out PairCandidate candidate)
    {
        double verticalOffset = Math.Abs(glyph.Bounds.Center.Y - text.Bounds.Center.Y);
        if (verticalOffset > options.MaximumGlyphTextVerticalOffset)
        {
            candidate = default;
            return false;
        }

        double leftGap = text.Bounds.Left - glyph.Bounds.Right;
        double rightGap = glyph.Bounds.Left - text.Bounds.Right;
        bool textIsRight = text.Bounds.Center.X >= glyph.Bounds.Center.X;
        double horizontalGap = textIsRight ? leftGap : rightGap;
        if (horizontalGap < MinimumOverlapTolerance ||
            horizontalGap > options.MaximumGlyphTextHorizontalGap)
        {
            candidate = default;
            return false;
        }

        double alignmentScore = 1 - (verticalOffset / options.MaximumGlyphTextVerticalOffset);
        double proximityScore = 1 - (Math.Max(0, horizontalGap) / options.MaximumGlyphTextHorizontalGap);
        double orientationScore = textIsRight ? 1 : 0.88;
        double confidence = Math.Clamp(
            (alignmentScore * 0.30) +
            (proximityScore * 0.25) +
            (glyph.Confidence * 0.20) +
            (text.Confidence * 0.20) +
            (orientationScore * 0.05),
            0,
            1);

        if (confidence < options.MinimumPairConfidence)
        {
            candidate = default;
            return false;
        }

        candidate = new PairCandidate(
            glyph,
            text,
            horizontalGap,
            verticalOffset,
            confidence,
            default);
        return true;
    }

    private static bool BelongsToSameRegion(
        ResolvedPair left,
        ResolvedPair right,
        LegendReasoningOptions options)
    {
        if (left.Location != right.Location)
        {
            return false;
        }

        double verticalGap = RectangleGap(left.Bounds.Top, left.Bounds.Bottom, right.Bounds.Top, right.Bounds.Bottom);
        double horizontalGap = RectangleGap(left.Bounds.Left, left.Bounds.Right, right.Bounds.Left, right.Bounds.Right);
        double maximumVerticalGap = Math.Max(
            options.MaximumGlyphTextVerticalOffset * 2,
            Math.Max(left.Bounds.Height, right.Bounds.Height) * 1.5);
        double maximumHorizontalGap = options.MaximumGlyphTextHorizontalGap;

        return verticalGap <= maximumVerticalGap && horizontalGap <= maximumHorizontalGap;
    }

    private static double RectangleGap(double firstStart, double firstEnd, double secondStart, double secondEnd)
    {
        if (firstEnd < secondStart)
        {
            return secondStart - firstEnd;
        }

        return secondEnd < firstStart ? firstStart - secondEnd : 0;
    }

    private static LegendRegionLocation Locate(LegendPoint point, LegendRectangle plotBounds) =>
        plotBounds.Contains(point) ? LegendRegionLocation.InsidePlot : LegendRegionLocation.OutsidePlot;

    private static bool IsValidRequest(LegendReasoningRequest request)
    {
        LegendReasoningOptions options = request.Options;
        return !string.IsNullOrWhiteSpace(request.ProjectId) &&
               !string.IsNullOrWhiteSpace(request.PanelId) &&
               request.ContractVersion == LegendReasoningContract.Version &&
               request.PanelBounds.IsValid &&
               request.PlotBounds.IsValid &&
               IsFinitePositive(options.MaximumGlyphTextHorizontalGap) &&
               IsFinitePositive(options.MaximumGlyphTextVerticalOffset) &&
               IsUnitInterval(options.MinimumPairConfidence);
    }

    private static bool IsValidGlyph(LegendGlyphCandidate glyph) =>
        !string.IsNullOrWhiteSpace(glyph.GlyphId) &&
        glyph.Bounds.IsValid &&
        IsUnitInterval(glyph.Confidence) &&
        glyph.Embedding.All(float.IsFinite);

    private static bool IsEligibleText(LegendTextRegion text) =>
        !string.IsNullOrWhiteSpace(text.RegionId) &&
        text.Bounds.IsValid &&
        !string.IsNullOrWhiteSpace(text.Text) &&
        text.Role == OcrTextRole.LegendText &&
        text.ReviewStatus != OcrReviewStatus.Rejected &&
        IsUnitInterval(text.Confidence);

    private static bool IsFinitePositive(double value) => double.IsFinite(value) && value > 0;

    private static bool IsUnitInterval(double value) => double.IsFinite(value) && value is >= 0 and <= 1;

    private static string StableId(string kind, params string[] components)
    {
        string material = string.Join('\u001F', new[] { kind }.Concat(components));
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(material));
        return new Guid(digest.AsSpan(0, 16)).ToString();
    }

    private readonly record struct PairCandidate(
        LegendGlyphCandidate Glyph,
        LegendTextRegion Text,
        double HorizontalGap,
        double VerticalOffset,
        double Confidence,
        LegendRegionLocation Location);

    private sealed record ResolvedPair(
        LegendEntry Entry,
        LegendRectangle Bounds,
        LegendRegionLocation Location);

    private sealed class DisjointSet
    {
        private readonly int[] _parent;

        public DisjointSet(int count) => _parent = Enumerable.Range(0, count).ToArray();

        public int Find(int value)
        {
            while (_parent[value] != value)
            {
                _parent[value] = _parent[_parent[value]];
                value = _parent[value];
            }

            return value;
        }

        public void Union(int left, int right)
        {
            int leftRoot = Find(left);
            int rightRoot = Find(right);
            if (leftRoot == rightRoot)
            {
                return;
            }

            int minimum = Math.Min(leftRoot, rightRoot);
            int maximum = Math.Max(leftRoot, rightRoot);
            _parent[maximum] = minimum;
        }
    }
}
