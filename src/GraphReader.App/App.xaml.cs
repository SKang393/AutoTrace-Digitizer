// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows;
using GraphReader.App.Appearance;
using GraphReader.App.Localization;

namespace GraphReader.App;

public partial class App : Application, IDisposable
{
    private ThemeService? _themeService;

    public IThemeService ThemeService =>
        _themeService ?? throw new InvalidOperationException("The application theme service is not initialized.");

    public IThemeService? AvailableThemeService => _themeService;

    public ILocalizationService LocalizationService { get; private set; } = null!;

    protected override void OnStartup(StartupEventArgs e)
    {
        _themeService = new ThemeService(Resources);
        LocalizationService = new LocalizationService(Resources);
        base.OnStartup(e);
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Dispose();
        base.OnExit(e);
    }

    public void Dispose()
    {
        _themeService?.Dispose();
        _themeService = null;
        GC.SuppressFinalize(this);
    }
}
