// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.ComponentModel;
using System.Windows;
using GraphReader.App.Appearance;
using GraphReader.App.Services;
using GraphReader.App.ViewModels;

namespace GraphReader.App;

public partial class MainWindow : Window
{
    private readonly IThemeService? _themeService;

    public MainWindow()
        : this((Application.Current as App)?.AvailableThemeService)
    {
    }

    public MainWindow(IThemeService? themeService)
    {
        _themeService = themeService;
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
        DataContext = new MainWindowViewModel(
            new FakeWorkspaceService(),
            (Application.Current as App)?.LocalizationService);
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
}
