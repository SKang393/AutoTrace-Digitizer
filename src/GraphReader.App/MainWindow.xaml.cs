// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Security;
using System.Windows;
using System.Windows.Threading;
using GraphReader.App.Appearance;
using GraphReader.App.Controls;
using GraphReader.App.Localization;
using GraphReader.App.Services;
using GraphReader.App.ViewModels;

namespace GraphReader.App;

public partial class MainWindow : Window
{
    private const double DefaultInspectorWidth = 390;
    private const double MaximumInspectorWidth = 800;
    private const double MinimumInspectorWidth = 340;
    private const string WindowLayoutFileName = "window-layout.txt";
    private readonly IThemeService? _themeService;
    private readonly DispatcherTimer _autosaveTimer;
    private readonly string? _windowLayoutPath;

    public MainWindow()
        : this((Application.Current as App)?.AvailableThemeService)
    {
    }

    public MainWindow(IThemeService? themeService)
    {
        _themeService = themeService;
        InitializeComponent();
        _windowLayoutPath = ResolveWindowLayoutPath();
        InspectorColumn.Width = new GridLength(ReadInspectorWidth(_windowLayoutPath, DefaultInspectorWidth));
        DataContextChanged += OnDataContextChanged;
        App? application = Application.Current as App;
        DataContext = new MainWindowViewModel(
            application?.WorkspaceService ?? new UnavailableWorkspaceService(),
            application?.LocalizationService,
            application?.StartupErrorMessageKey,
            new WindowsWorkspaceDialogService(application?.LocalizationService));
        _autosaveTimer = new DispatcherTimer(DispatcherPriority.Background, Dispatcher)
        {
            Interval = TimeSpan.FromMinutes(1),
        };
        _autosaveTimer.Tick += OnAutosaveTimerTick;
        _autosaveTimer.Start();
        Closed += OnClosed;
    }

    private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
    {
        _ = sender;
        if (e.OldValue is INotifyPropertyChanged previous)
        {
            previous.PropertyChanged -= OnViewModelPropertyChanged;
        }

        if (e.NewValue is MainWindowViewModel current)
        {
            current.PropertyChanged += OnViewModelPropertyChanged;
            _themeService?.ApplyTheme(current.AppearanceMode);
        }
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(MainWindowViewModel.AppearanceMode)
            && sender is MainWindowViewModel viewModel)
        {
            _themeService?.ApplyTheme(viewModel.AppearanceMode);
        }
    }

    private void OnClosed(object? sender, EventArgs e)
    {
        _ = sender;
        _ = e;
        WriteInspectorWidth(_windowLayoutPath, InspectorColumn.ActualWidth);
        _autosaveTimer.Stop();
        _autosaveTimer.Tick -= OnAutosaveTimerTick;
        DataContextChanged -= OnDataContextChanged;
        if (DataContext is IDisposable disposable)
        {
            if (DataContext is INotifyPropertyChanged current)
            {
                current.PropertyChanged -= OnViewModelPropertyChanged;
            }

            disposable.Dispose();
        }
    }

    private async void OnAutosaveTimerTick(object? sender, EventArgs e)
    {
        _ = sender;
        _ = e;
        if (DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.RunTimerAutosaveAsync();
        }
    }

    private async void OnGraphImagePointInvoked(object sender, GraphImagePointEventArgs e)
    {
        _ = sender;
        if (DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.HandleCanvasPointAsync(e.ImagePoint);
        }
    }

    private void OnWindowStateChanged(object? sender, EventArgs e)
    {
        _ = sender;
        _ = e;
        Dispatcher.BeginInvoke(GraphCanvasHost.RecalculateViewport, DispatcherPriority.Loaded);
    }

    private void OnFitGraphClick(object sender, RoutedEventArgs e)
    {
        _ = sender;
        _ = e;
        GraphCanvasHost.ResetView();
    }

    private void OnResetViewClick(object sender, RoutedEventArgs e)
    {
        _ = sender;
        _ = e;
        GraphCanvasHost.ResetView();
    }

    private static double ReadInspectorWidth(string? path, double fallback)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return fallback;
        }

        try
        {
            string value = File.ReadAllText(path).Trim();
            return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double width)
                && double.IsFinite(width)
                ? Math.Clamp(width, MinimumInspectorWidth, MaximumInspectorWidth)
                : fallback;
        }
        catch (Exception exception) when (IsExpectedSettingsException(exception))
        {
            return fallback;
        }
    }

    private static void WriteInspectorWidth(string? path, double width)
    {
        if (string.IsNullOrWhiteSpace(path) || !double.IsFinite(width) || width < MinimumInspectorWidth)
        {
            return;
        }

        try
        {
            string? directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(
                path,
                Math.Clamp(width, MinimumInspectorWidth, MaximumInspectorWidth)
                    .ToString("R", CultureInfo.InvariantCulture));
        }
        catch (Exception exception) when (IsExpectedSettingsException(exception))
        {
            // Layout persistence is intentionally fail-soft and cannot block the workspace.
        }
    }

    private static bool IsExpectedSettingsException(Exception exception) =>
        exception is IOException or UnauthorizedAccessException or SecurityException or ArgumentException or NotSupportedException;

    private static string? ResolveWindowLayoutPath()
    {
        string? settingsRoot = (Application.Current as App)?.ApplicationPaths?.SettingsRoot;
        return string.IsNullOrWhiteSpace(settingsRoot)
            ? null
            : Path.Combine(settingsRoot, WindowLayoutFileName);
    }
}
