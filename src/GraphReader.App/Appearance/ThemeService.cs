// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows;

namespace GraphReader.App.Appearance;

public sealed class ThemeService : IThemeService
{
    internal const string LightThemeSource =
        "/GraphReader.App;component/Themes/LightTheme.xaml";
    internal const string DarkThemeSource =
        "/GraphReader.App;component/Themes/DarkTheme.xaml";
    internal const string SystemThemeSource =
        "/GraphReader.App;component/Themes/SystemTheme.xaml";

    private static readonly string[] ThemeSources =
    [
        LightThemeSource,
        DarkThemeSource,
        SystemThemeSource,
    ];

    private readonly ResourceDictionary _resources;
    private readonly ISystemThemeProvider _systemThemeProvider;
    private readonly bool _ownsSystemThemeProvider;
    private bool _disposed;

    public ThemeService(
        ResourceDictionary resources,
        ISystemThemeProvider? systemThemeProvider = null)
    {
        ArgumentNullException.ThrowIfNull(resources);

        _resources = resources;
        _systemThemeProvider = systemThemeProvider ?? new WindowsSystemThemeProvider();
        _ownsSystemThemeProvider = systemThemeProvider is null;
        _systemThemeProvider.ThemeChanged += OnSystemThemeChanged;

        ApplyTheme(ApplicationTheme.System);
    }

    public ApplicationTheme Theme { get; private set; }

    public ApplicationTheme EffectiveTheme { get; private set; }

    public event EventHandler<ThemeChangedEventArgs>? ThemeChanged;

    public void ApplyTheme(ApplicationTheme theme)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ApplicationTheme effectiveTheme = theme == ApplicationTheme.System
            ? NormalizeEffectiveTheme(_systemThemeProvider.EffectiveTheme)
            : NormalizeEffectiveTheme(theme);

        bool changed = Theme != theme || EffectiveTheme != effectiveTheme;
        Theme = theme;
        EffectiveTheme = effectiveTheme;
        ReplaceThemeDictionary(GetThemeSource(effectiveTheme));

        if (changed)
        {
            ThemeChanged?.Invoke(this, new ThemeChangedEventArgs(Theme, EffectiveTheme));
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _systemThemeProvider.ThemeChanged -= OnSystemThemeChanged;
        if (_ownsSystemThemeProvider)
        {
            _systemThemeProvider.Dispose();
        }

        _disposed = true;
    }

    private static ApplicationTheme NormalizeEffectiveTheme(ApplicationTheme theme) =>
        theme == ApplicationTheme.Dark ? ApplicationTheme.Dark : ApplicationTheme.Light;

    private static string GetThemeSource(ApplicationTheme effectiveTheme) =>
        effectiveTheme == ApplicationTheme.Dark ? DarkThemeSource : LightThemeSource;

    private void OnSystemThemeChanged(object? sender, EventArgs e)
    {
        if (Theme == ApplicationTheme.System)
        {
            ApplyTheme(ApplicationTheme.System);
        }
    }

    private void ReplaceThemeDictionary(string source)
    {
        ResourceDictionary replacement = new()
        {
            Source = new Uri(source, UriKind.RelativeOrAbsolute),
        };

        int existingIndex = FindThemeDictionaryIndex();
        if (existingIndex >= 0)
        {
            _resources.MergedDictionaries[existingIndex] = replacement;
            return;
        }

        _resources.MergedDictionaries.Add(replacement);
    }

    private int FindThemeDictionaryIndex()
    {
        for (int index = 0; index < _resources.MergedDictionaries.Count; index++)
        {
            string? source = _resources.MergedDictionaries[index].Source?.OriginalString;
            if (source is not null && ThemeSources.Any(
                    candidate => source.EndsWith(candidate, StringComparison.OrdinalIgnoreCase)
                        || source.EndsWith(
                            candidate[(candidate.LastIndexOf('/') + 1)..],
                            StringComparison.OrdinalIgnoreCase)))
            {
                return index;
            }
        }

        return -1;
    }
}
