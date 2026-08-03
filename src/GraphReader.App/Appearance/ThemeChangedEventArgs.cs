// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Appearance;

public sealed class ThemeChangedEventArgs(
    ApplicationTheme theme,
    ApplicationTheme effectiveTheme) : EventArgs
{
    public ApplicationTheme Theme { get; } = theme;

    public ApplicationTheme EffectiveTheme { get; } = effectiveTheme;
}
