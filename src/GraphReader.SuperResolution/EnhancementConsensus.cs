// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.SuperResolution;

public sealed record EnhancementEvidencePoint(
    string EvidenceId,
    EnhancementPoint Location,
    double Confidence);

public sealed record EnhancementConsensusItem(
    string EvidenceId,
    EnhancementPoint? OriginalLocation,
    EnhancementPoint? EnhancedLocationInOriginalPixels,
    double? DisplacementPixels,
    bool RequiresReview,
    string Reason);

public sealed record EnhancementConsensusResult(
    IReadOnlyList<EnhancementConsensusItem> Items,
    bool RequiresReview,
    double ConfidenceMultiplier);

public static class EnhancementConsensus
{
    public static EnhancementConsensusResult Compare(
        IEnumerable<EnhancementEvidencePoint> originalEvidence,
        IEnumerable<EnhancementEvidencePoint> enhancedEvidence,
        EnhancementTransform transform,
        double maximumDisplacementPixels)
    {
        ArgumentNullException.ThrowIfNull(originalEvidence);
        ArgumentNullException.ThrowIfNull(enhancedEvidence);
        ArgumentNullException.ThrowIfNull(transform);
        if (!double.IsFinite(maximumDisplacementPixels) || maximumDisplacementPixels < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumDisplacementPixels));
        }

        Dictionary<string, EnhancementEvidencePoint> original = ToUniqueMap(originalEvidence, nameof(originalEvidence));
        Dictionary<string, EnhancementEvidencePoint> enhanced = ToUniqueMap(enhancedEvidence, nameof(enhancedEvidence));
        string[] evidenceIds = original.Keys
            .Concat(enhanced.Keys)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();
        var items = new List<EnhancementConsensusItem>(evidenceIds.Length);
        foreach (string evidenceId in evidenceIds)
        {
            bool hasOriginal = original.TryGetValue(evidenceId, out EnhancementEvidencePoint? originalPoint);
            bool hasEnhanced = enhanced.TryGetValue(evidenceId, out EnhancementEvidencePoint? enhancedPoint);
            if (!hasOriginal)
            {
                items.Add(new EnhancementConsensusItem(
                    evidenceId,
                    null,
                    transform.ToOriginal(enhancedPoint!.Location),
                    null,
                    RequiresReview: true,
                    "Evidence appears only in the enhanced derivative."));
                continue;
            }

            if (!hasEnhanced)
            {
                items.Add(new EnhancementConsensusItem(
                    evidenceId,
                    originalPoint!.Location,
                    null,
                    null,
                    RequiresReview: true,
                    "Evidence from the original is missing in the enhanced derivative."));
                continue;
            }

            EnhancementPoint mapped = transform.ToOriginal(enhancedPoint!.Location);
            double deltaX = mapped.X - originalPoint!.Location.X;
            double deltaY = mapped.Y - originalPoint.Location.Y;
            double displacement = Math.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
            bool review = displacement > maximumDisplacementPixels;
            items.Add(new EnhancementConsensusItem(
                evidenceId,
                originalPoint.Location,
                mapped,
                displacement,
                review,
                review
                    ? "Original and enhanced evidence disagree beyond tolerance."
                    : "Original and enhanced evidence agree within tolerance."));
        }

        bool requiresReview = items.Any(static item => item.RequiresReview);
        double agreement = items.Count == 0
            ? 0
            : (double)items.Count(static item => !item.RequiresReview) / items.Count;
        return new EnhancementConsensusResult(
            items.AsReadOnly(),
            requiresReview,
            requiresReview ? Math.Max(0.25, agreement) : 1);
    }

    private static Dictionary<string, EnhancementEvidencePoint> ToUniqueMap(
        IEnumerable<EnhancementEvidencePoint> evidence,
        string parameterName)
    {
        var result = new Dictionary<string, EnhancementEvidencePoint>(StringComparer.Ordinal);
        foreach (EnhancementEvidencePoint point in evidence)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(point.EvidenceId);
            if (!double.IsFinite(point.Location.X) || !double.IsFinite(point.Location.Y) ||
                !double.IsFinite(point.Confidence) || point.Confidence is < 0 or > 1)
            {
                throw new ArgumentException("Evidence coordinates and confidence must be finite and confidence must be in [0, 1].", parameterName);
            }

            if (!result.TryAdd(point.EvidenceId, point))
            {
                throw new ArgumentException($"Duplicate evidence ID '{point.EvidenceId}'.", parameterName);
            }
        }

        return result;
    }
}
