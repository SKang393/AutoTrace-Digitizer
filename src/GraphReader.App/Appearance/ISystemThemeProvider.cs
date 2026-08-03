// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Appearance;

public interface ISystemThemeProvider : IDisposable
{
    ApplicationTheme EffectiveTheme { get; }

    event EventHandler? ThemeChanged;
}
