// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.ComponentModel;
using System.Windows;
using Microsoft.Win32;

namespace GraphReader.App.Appearance;

public sealed class WindowsSystemThemeProvider : ISystemThemeProvider
{
    private const string PersonalizeKey =
        @"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize";
    private const string AppsUseLightThemeValue = "AppsUseLightTheme";

    private bool _disposed;
    private ApplicationTheme _effectiveTheme;
    private bool _isHighContrast;
    private bool _areClientAreaAnimationsEnabled;

    public WindowsSystemThemeProvider()
    {
        _effectiveTheme = ReadEffectiveTheme();
        _isHighContrast = SystemParameters.HighContrast;
        _areClientAreaAnimationsEnabled = SystemParameters.ClientAreaAnimation;
        SystemParameters.StaticPropertyChanged += OnSystemParametersChanged;
        SystemEvents.UserPreferenceChanged += OnUserPreferenceChanged;
    }

    public ApplicationTheme EffectiveTheme => _effectiveTheme;

    public bool IsHighContrast => _isHighContrast;

    public bool AreClientAreaAnimationsEnabled => _areClientAreaAnimationsEnabled;

    public event EventHandler? ThemeChanged;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        SystemParameters.StaticPropertyChanged -= OnSystemParametersChanged;
        SystemEvents.UserPreferenceChanged -= OnUserPreferenceChanged;
        _disposed = true;
    }

    private static ApplicationTheme ReadEffectiveTheme()
    {
        using RegistryKey? key = Registry.CurrentUser.OpenSubKey(PersonalizeKey);
        object? value = key?.GetValue(AppsUseLightThemeValue);

        return value is int intValue && intValue == 0
            ? ApplicationTheme.Dark
            : ApplicationTheme.Light;
    }

    private void OnSystemParametersChanged(object? sender, PropertyChangedEventArgs e)
    {
        _ = sender;
        _ = e;
        RefreshTheme();
    }

    private void OnUserPreferenceChanged(object sender, UserPreferenceChangedEventArgs e)
    {
        _ = sender;
        _ = e;
        RefreshTheme();
    }

    private void RefreshTheme()
    {
        ApplicationTheme nextTheme = ReadEffectiveTheme();
        bool nextHighContrast = SystemParameters.HighContrast;
        bool nextAnimationsEnabled = SystemParameters.ClientAreaAnimation;
        if (nextTheme == _effectiveTheme &&
            nextHighContrast == _isHighContrast &&
            nextAnimationsEnabled == _areClientAreaAnimationsEnabled)
        {
            return;
        }

        _effectiveTheme = nextTheme;
        _isHighContrast = nextHighContrast;
        _areClientAreaAnimationsEnabled = nextAnimationsEnabled;
        EventHandler? handler = ThemeChanged;
        if (handler is null)
        {
            return;
        }

        Action raiseThemeChanged = () => handler(this, EventArgs.Empty);
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher is not null && !dispatcher.CheckAccess())
        {
            _ = dispatcher.BeginInvoke(raiseThemeChanged);
            return;
        }

        raiseThemeChanged();
    }
}
