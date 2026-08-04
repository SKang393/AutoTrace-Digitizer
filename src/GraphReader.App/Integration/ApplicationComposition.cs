// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
using GraphReader.Domain;
using GraphReader.SuperResolution;

namespace GraphReader.App.Integration;

public sealed record ApplicationCompositionResult(
    WorkflowRuntimeEnvironment Environment,
    IWorkspaceService WorkspaceService,
    DomainError? StartupError);

public static class ApplicationComposition
{
    public const string RealEsrganManifestEnvironmentVariable = "GRAPHREADER_REALESRGAN_MANIFEST_PATH";
    public const string RealEsrganRuntimeRootEnvironmentVariable = "GRAPHREADER_REALESRGAN_RUNTIME_ROOT";

    public static ApplicationCompositionResult Create(
        WorkflowRuntimeEnvironment environment,
        IApplicationPaths? applicationPaths = null)
    {
        Func<CancellationToken, Task<RealEsrganBackendResolution>>? enhancementResolver =
            CreateLocalEnhancementResolver(applicationPaths);
        IReadOnlyList<AutomaticStageStatus> automaticStages =
            ProductionStageAvailabilityRegistry.Create(enhancementResolver is not null);
        return environment switch
        {
            WorkflowRuntimeEnvironment.Production => new ApplicationCompositionResult(
                environment,
                new ManualPreviewWorkspaceService(
                    applicationPaths,
                    runtimeEnvironment: WorkflowRuntimeEnvironment.Production,
                    automaticStages: automaticStages,
                    enhancementResolver: enhancementResolver),
                null),
            WorkflowRuntimeEnvironment.ManualPreview => new ApplicationCompositionResult(
                environment,
                new ManualPreviewWorkspaceService(
                    applicationPaths,
                    automaticStages: automaticStages,
                    enhancementResolver: enhancementResolver),
                null),
            WorkflowRuntimeEnvironment.RecordedFake => new ApplicationCompositionResult(
                environment,
                new FakeWorkspaceService(),
                null),
            _ => throw new ArgumentOutOfRangeException(nameof(environment)),
        };
    }

    private static Func<CancellationToken, Task<RealEsrganBackendResolution>>? CreateLocalEnhancementResolver(
        IApplicationPaths? applicationPaths)
    {
        string? manifestPath = Environment.GetEnvironmentVariable(RealEsrganManifestEnvironmentVariable);
        string? runtimeRoot = Environment.GetEnvironmentVariable(RealEsrganRuntimeRootEnvironmentVariable);
        if (applicationPaths is null ||
            string.IsNullOrWhiteSpace(manifestPath) ||
            string.IsNullOrWhiteSpace(runtimeRoot))
        {
            return null;
        }

        string cacheRoot = Path.Combine(applicationPaths.CacheRoot, "RealESRGAN");
        return cancellationToken => ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            manifestPath,
            runtimeRoot,
            cacheRoot,
            RealEsrganBackendPurpose.LocalEvaluation,
            cancellationToken);
    }
}
