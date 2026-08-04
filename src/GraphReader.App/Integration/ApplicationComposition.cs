// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
using GraphReader.Domain;

namespace GraphReader.App.Integration;

public sealed record ApplicationCompositionResult(
    WorkflowRuntimeEnvironment Environment,
    IWorkspaceService WorkspaceService,
    DomainError? StartupError);

public static class ApplicationComposition
{
    public static ApplicationCompositionResult Create(
        WorkflowRuntimeEnvironment environment,
        IApplicationPaths? applicationPaths = null) =>
        environment switch
        {
            WorkflowRuntimeEnvironment.Production => new ApplicationCompositionResult(
                environment,
                new ManualPreviewWorkspaceService(
                    applicationPaths,
                    runtimeEnvironment: WorkflowRuntimeEnvironment.Production,
                    automaticStages: ProductionStageAvailabilityRegistry.Current),
                null),
            WorkflowRuntimeEnvironment.ManualPreview => new ApplicationCompositionResult(
                environment,
                new ManualPreviewWorkspaceService(applicationPaths),
                null),
            WorkflowRuntimeEnvironment.RecordedFake => new ApplicationCompositionResult(
                environment,
                new FakeWorkspaceService(),
                null),
            _ => throw new ArgumentOutOfRangeException(nameof(environment)),
        };
}
