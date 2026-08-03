// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Markers.Classification;

public static class MarkerSymbolMap
{
    private static readonly MarkerSymbolDescriptor[,] Descriptors =
    {
        {
            new("●", "Filled circle"),
            new("○", "Open circle"),
            new("◉", "Circle with unknown fill"),
        },
        {
            new("■", "Filled square"),
            new("□", "Open square"),
            new("▣", "Square with unknown fill"),
        },
        {
            new("▲", "Filled triangle up"),
            new("△", "Open triangle up"),
            new("◬", "Triangle up with unknown fill"),
        },
        {
            new("▼", "Filled triangle down"),
            new("▽", "Open triangle down"),
            new("◭", "Triangle down with unknown fill"),
        },
        {
            new("◆", "Filled diamond"),
            new("◇", "Open diamond"),
            new("◈", "Diamond with unknown fill"),
        },
        {
            new("★", "Filled star"),
            new("☆", "Open star"),
            new("✦", "Star with unknown fill"),
        },
        {
            new("✱", "Filled asterisk"),
            new("✳", "Open asterisk"),
            new("∗", "Asterisk with unknown fill"),
        },
        {
            new("✚", "Filled cross"),
            new("✛", "Open cross"),
            new("✕", "Cross with unknown fill"),
        },
        {
            new("⬢", "Filled other marker"),
            new("⬡", "Open other marker"),
            new("?", "Other marker with unknown fill"),
        },
    };

    public static MarkerSymbolDescriptor Describe(MarkerShape shape, MarkerFill fill)
    {
        if (!Enum.IsDefined(shape))
        {
            throw new ArgumentOutOfRangeException(nameof(shape));
        }

        if (!Enum.IsDefined(fill))
        {
            throw new ArgumentOutOfRangeException(nameof(fill));
        }

        return Descriptors[(int)shape, (int)fill];
    }

    public static string GetSymbol(MarkerShape shape, MarkerFill fill) =>
        Describe(shape, fill).Symbol;

    public static string GetAccessibleName(MarkerShape shape, MarkerFill fill) =>
        Describe(shape, fill).AccessibleName;
}
