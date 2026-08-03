// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.ViewModels;

namespace GraphReader.App.Services;

public sealed class UnavailableWorkspaceService : IWorkspaceService
{
    public IReadOnlyList<WorkspaceTabViewModel> CreateWorkspace() => [];

    public Task RunStageAsync(WorkflowStage stage, CancellationToken cancellationToken)
    {
        _ = stage;
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromException(
            new InvalidOperationException("Production workflow adapters are not available in this internal build."));
    }
}
