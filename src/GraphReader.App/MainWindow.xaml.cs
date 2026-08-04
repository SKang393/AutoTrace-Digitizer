// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.ComponentModel;
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
    private readonly IThemeService? _themeService;
    private readonly DispatcherTimer _autosaveTimer;

    public MainWindow()
        : this((Application.Current as App)?.AvailableThemeService)
    {
    }

    public MainWindow(IThemeService? themeService)
    {
        _themeService = themeService;
        InitializeComponent();
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
}
