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

    public WorkflowRuntimeEnvironment RuntimeEnvironment { get; private set; }

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        _themeService = new ThemeService(Resources);
        LocalizationService = new LocalizationService(Resources);
        RuntimeEnvironment = RuntimeModeSelector.Select();
        DomainResult<IApplicationPaths> runtimePaths =
            RuntimePathBootstrapper.CreateDefault().Initialize();
        ApplicationPaths = runtimePaths.Value;
        StartupError = runtimePaths.Errors.Count > 0 ? runtimePaths.Errors[0] : null;
        ApplicationCompositionResult composition = await ApplicationComposition.CreateAsync(
            RuntimeEnvironment,
            ApplicationPaths,
            cancellationToken: CancellationToken.None);
        WorkspaceService = composition.WorkspaceService;
        WorkflowStartupError = composition.StartupError;
        StartupErrorMessageKey = StartupError?.UserMessageKey ?? WorkflowStartupError?.UserMessageKey;

        StartupArguments startupArguments = StartupArguments.Parse(e.Args);
        bool portableSmoke = startupArguments.PortableSmoke;
        if (portableSmoke)
        {
            int exitCode = RunPortableSmoke();
            Shutdown(exitCode);
            return;
        }

        if (startupArguments.OpenImagePath is not null &&
            WorkspaceService is IManualWorkspaceService manualWorkspace)
        {
            await manualWorkspace
                .ImportImagesAsync([startupArguments.OpenImagePath], CancellationToken.None);
        }

        var mainWindow = new MainWindow(_themeService);
        MainWindow = mainWindow;
        mainWindow.Show();
    }

    private int RunPortableSmoke()
    {
        if (StartupError is not null || WorkflowStartupError is not null ||
            ApplicationPaths is null ||
            RuntimeEnvironment != WorkflowRuntimeEnvironment.ManualPreview ||
            WorkspaceService is not IRuntimeWorkspaceService runtimeWorkspace ||
            runtimeWorkspace.UsesFakeGraphData ||
            runtimeWorkspace.CreateWorkspace().Count != 0)
        {
            return 2;
        }

        var window = new MainWindow(_themeService)
        {
            ShowActivated = false,
            ShowInTaskbar = false,
        };
        window.Close();
        return 0;
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
