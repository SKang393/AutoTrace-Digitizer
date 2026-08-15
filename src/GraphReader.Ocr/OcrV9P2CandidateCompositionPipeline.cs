// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using GraphReader.Inference;

namespace GraphReader.Ocr;

/// <summary>
/// Bounded P2 selection candidate over the immutable, consumed P1 pipeline.
/// It changes only the selected-text confidence floor for regions accepted by
/// the high detector route. It is not available to production composition.
/// </summary>
public sealed class OcrV9P2CandidateCompositionPipeline
{
    public const string CandidateCompositionId =
        "graphreader-v10-selected-confidence-acceptance-composition-v9-p2";
    public const string StageVersion = "0.0.21-v9-p2";
    public const double SelectedTextMinimumConfidence = 0.75;

    private const string P1DirectRoute = "recognizer_confirmed_detector";
    private readonly OcrV8ProductionCompositionPipeline inner;

    public OcrV9P2CandidateCompositionPipeline(OcrV8ProductionCompositionPipeline inner)
    {
        this.inner = inner ?? throw new ArgumentNullException(nameof(inner));
        if (!string.Equals(
                inner.CompositionId,
                OcrV9CandidateCompositionFactory.CandidateCompositionId,
                StringComparison.Ordinal))
        {
            throw new ArgumentException("P2 must wrap the exact immutable P1 candidate.", nameof(inner));
        }
    }

    public string ConfigurationFingerprint => HashStrings(
    [
        CandidateCompositionId,
        StageVersion,
        SelectedTextMinimumConfidence.ToString("R", CultureInfo.InvariantCulture),
        inner.ConfigurationFingerprint,
    ]);

    public async ValueTask<OcrResult> RecognizeAsync(
        OcrRequest request,
        CancellationToken cancellationToken = default)
    {
        OcrResult result = await inner.RecognizeAsync(request, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        var warnings = result.Warnings.ToList();
        warnings.Add("ocr_v9_p2_candidate_composition");
        if (!result.Succeeded)
        {
            return result with
            {
                StageVersion = StageVersion,
                Warnings = OcrCollections.Freeze(warnings.Distinct(StringComparer.Ordinal)),
            };
        }

        OcrRegion[] selected = result.Regions
            .Where(region => Keep(region, result.Warnings, warnings))
            .ToArray();
        HashSet<string> selectedIds = selected
            .Select(static region => region.RegionId)
            .ToHashSet(StringComparer.Ordinal);
        OcrMask[] masks = result.Masks
            .Where(mask => selectedIds.Contains(mask.RegionId))
            .ToArray();
        return result with
        {
            RunId = Guid.NewGuid().ToString(),
            StageVersion = StageVersion,
            Regions = OcrCollections.Freeze(selected),
            Masks = OcrCollections.Freeze(masks),
            Confidence = selected.Length == 0 ? 0 : selected.Average(static region => region.Confidence),
            Warnings = OcrCollections.Freeze(warnings.Distinct(StringComparer.Ordinal)),
        };
    }

    private static bool Keep(
        OcrRegion region,
        IReadOnlyList<string> innerWarnings,
        List<string> outputWarnings)
    {
        string route = $"ocr_v9_candidate_acceptance_route:{region.RegionId}:{P1DirectRoute}";
        if (!innerWarnings.Contains(route, StringComparer.Ordinal))
        {
            outputWarnings.Add($"ocr_v9_p2_candidate_non_direct_route_retained:{region.RegionId}");
            return true;
        }

        double confidence = region.Alternatives
            .Where(alternative => string.Equals(alternative.Text, region.Text, StringComparison.Ordinal))
            .Select(static alternative => alternative.Confidence)
            .DefaultIfEmpty(0)
            .Max();
        if (confidence < SelectedTextMinimumConfidence)
        {
            outputWarnings.Add($"ocr_v9_p2_candidate_selected_confidence_rejected:{region.RegionId}");
            return false;
        }

        outputWarnings.Add($"ocr_v9_p2_candidate_selected_confidence_accepted:{region.RegionId}");
        return true;
    }

    private static string HashStrings(IEnumerable<string> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string value in values)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            hash.AppendData(BitConverter.GetBytes(bytes.Length));
            hash.AppendData(bytes);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
}

public static class OcrV9P2CandidateCompositionFactory
{
    public static OcrV9P2CandidateCompositionPipeline Create(
        InferenceRuntime runtime,
        OcrV8ProductionPayloadSet payloads,
        IReadOnlyList<InferenceProvider> allowedProviders,
        bool bypassCache = false) =>
        new(OcrV9CandidateCompositionFactory.Create(
            runtime,
            payloads,
            allowedProviders,
            bypassCache));
}
