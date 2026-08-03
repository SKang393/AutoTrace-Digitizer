// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;

namespace GraphReader.Legends;

internal static class LegendSemanticNormalizer
{
    public static LegendSemanticEvidence Normalize(string text, double confidence)
    {
        string normalized = NormalizeText(text);
        string[] tokens = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        if (tokens.Any(static token =>
                token.StartsWith("generaliz", StringComparison.Ordinal) ||
                token.StartsWith("generalis", StringComparison.Ordinal)))
        {
            return new LegendSemanticEvidence(
                LegendSemanticHint.Generalization,
                "generalization",
                SemanticConfidence(normalized, "generalization", confidence));
        }

        if (tokens.Any(static token => token.StartsWith("maintenan", StringComparison.Ordinal)))
        {
            return new LegendSemanticEvidence(
                LegendSemanticHint.Maintenance,
                "maintenance",
                SemanticConfidence(normalized, "maintenance", confidence));
        }

        return new LegendSemanticEvidence(LegendSemanticHint.Unknown, normalized, 0);
    }

    private static double SemanticConfidence(string normalized, string canonical, double confidence)
    {
        double lexicalConfidence = string.Equals(normalized, canonical, StringComparison.Ordinal) ? 1 : 0.95;
        return Math.Clamp(confidence * lexicalConfidence, 0, 1);
    }

    private static string NormalizeText(string text)
    {
        StringBuilder builder = new();
        bool pendingSpace = false;
        foreach (char character in text.Normalize(NormalizationForm.FormKD))
        {
            if (char.IsLetterOrDigit(character))
            {
                if (pendingSpace && builder.Length > 0)
                {
                    builder.Append(' ');
                }

                builder.Append(char.ToLowerInvariant(character));
                pendingSpace = false;
            }
            else
            {
                pendingSpace = true;
            }
        }

        return builder.ToString();
    }
}
