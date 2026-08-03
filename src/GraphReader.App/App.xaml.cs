// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows;
using GraphReader.App.Appearance;
using GraphReader.App.Integration;
using GraphReader.App.Integration.Runtime;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Localization;
using GraphReader.App.Services;
using GraphReader.Domain;

namespace GraphReader.App;

public partial class App : Application, IDisposable
{
    private ThemeService? _themeService;

    public IThemeService ThemeService =>
        _themeService ?? throw new InvalidOperationException("The application theme service is not initialized.");

    public IThemeService? AvailableThemeService => _themeService;

    public ILocalizationService LocalizationService { get; private set; } = null!;

    public IApplicationPaths? ApplicationPaths { get; private set; }

    public DomainError? StartupError { get; private set; }

    public DomainError? WorkflowStartupError { get; private set; }

    public string? StartupErrorMessageKey { get; private set; }

    public IWorkspaceService WorkspaceService { get; private set; } = new UnavailableWorkspaceService();

    protected override void OnStartup(StartupEventArgs e)
    {
        _themeService = new ThemeService(Resources);
        LocalizationService = new LocalizationService(Resources);
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);
        WorkspaceService = composition.WorkspaceService;
        WorkflowStartupError = composition.StartupError;
        DomainResult<IApplicationPaths> runtimePaths =
            RuntimePathBootstrapper.CreateDefault().Initialize();
        ApplicationPaths = runtimePaths.Value;
        StartupError = runtimePaths.Errors.Count > 0 ? runtimePaths.Errors[0] : null;
        StartupErrorMessageKey = StartupError?.UserMessageKey ?? WorkflowStartupError?.UserMessageKey;
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
