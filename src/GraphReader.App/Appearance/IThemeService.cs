// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Appearance;

public interface IThemeService : IDisposable
{
    ApplicationTheme Theme { get; }

    ApplicationTheme EffectiveTheme { get; }

    event EventHandler<ThemeChangedEventArgs>? ThemeChanged;

    void ApplyTheme(ApplicationTheme theme);
}
