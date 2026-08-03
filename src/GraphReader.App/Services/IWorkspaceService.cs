// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.ViewModels;

namespace GraphReader.App.Services;

public interface IWorkspaceService
{
    IReadOnlyList<WorkspaceTabViewModel> CreateWorkspace();

    Task RunStageAsync(WorkflowStage stage, CancellationToken cancellationToken);
}
