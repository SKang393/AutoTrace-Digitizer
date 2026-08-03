// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;
using GraphReader.App.Localization;
using GraphReader.App.Services;
using GraphReader.Domain;

namespace GraphReader.App.Integration;

public sealed record ApplicationCompositionResult(
    WorkflowRuntimeEnvironment Environment,
    IWorkspaceService WorkspaceService,
    DomainError? StartupError);

public static class ApplicationComposition
{
    public static ApplicationCompositionResult Create(WorkflowRuntimeEnvironment environment) =>
        environment switch
        {
            WorkflowRuntimeEnvironment.Production => new ApplicationCompositionResult(
                environment,
                new UnavailableWorkspaceService(),
                new DomainError(
                    "PRODUCTION_WORKFLOW_UNAVAILABLE",
                    DomainErrorSeverity.Error,
                    LocalizationKeys.ProductionWorkflowUnavailable,
                    "Approved production workflow adapters and models are not available in this internal build.",
                    Recoverable: true,
                    "install_approved_workflow_assets")),
            WorkflowRuntimeEnvironment.RecordedFake => new ApplicationCompositionResult(
                environment,
                new FakeWorkspaceService(),
                null),
            _ => throw new ArgumentOutOfRangeException(nameof(environment)),
        };
}
