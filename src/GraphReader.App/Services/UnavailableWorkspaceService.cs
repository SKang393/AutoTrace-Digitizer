// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;
using GraphReader.App.ViewModels;

namespace GraphReader.App.Services;

public sealed class UnavailableWorkspaceService : IRuntimeWorkspaceService
{
    public WorkflowRuntimeEnvironment RuntimeEnvironment => WorkflowRuntimeEnvironment.Production;

    public IReadOnlyList<AutomaticStageStatus> AutomaticStages { get; } =
    [
        new("enhancement", AutomaticStageState.Unavailable, "No approved Real-ESRGAN runtime and model are installed."),
        new("axis", AutomaticStageState.Unavailable, "The approved production axis adapter is not connected."),
        new("ocr", AutomaticStageState.Unavailable, "No approved OCR models are installed."),
        new("markers", AutomaticStageState.Unavailable, "No approved marker models are installed."),
        new("legends", AutomaticStageState.Unavailable, "The approved production legend adapter is not connected."),
        new("phases", AutomaticStageState.Unavailable, "The approved production phase adapter is not connected."),
    ];

    public bool UsesFakeGraphData => false;

    public IReadOnlyList<WorkspaceTabViewModel> CreateWorkspace() => [];

    public Task RunStageAsync(WorkflowStage stage, CancellationToken cancellationToken)
    {
        _ = stage;
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromException(
            new InvalidOperationException("Production workflow adapters are not available in this internal build."));
    }
}
