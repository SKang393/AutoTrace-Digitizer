// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using GraphReader.App.Integration.Workflow;

namespace GraphReader.App.Integration;

public static class RuntimeModeSelector
{
    internal const string BuildMetadataKey = "GraphReaderRuntimeMode";
    private const string RuntimeModeVariable = "GRAPHREADER_RUNTIME_MODE";

    public static WorkflowRuntimeEnvironment Select(
        string? configuredValue = null,
        Assembly? applicationAssembly = null)
    {
        WorkflowRuntimeEnvironment buildDefault = SelectBuildDefault(applicationAssembly);
        if (buildDefault == WorkflowRuntimeEnvironment.Production)
        {
            return WorkflowRuntimeEnvironment.Production;
        }

        string? value = configuredValue ?? Environment.GetEnvironmentVariable(RuntimeModeVariable);
        return string.Equals(value, nameof(WorkflowRuntimeEnvironment.Production), StringComparison.OrdinalIgnoreCase)
            ? WorkflowRuntimeEnvironment.Production
            : WorkflowRuntimeEnvironment.ManualPreview;
    }

    internal static WorkflowRuntimeEnvironment SelectBuildDefault(Assembly? applicationAssembly = null)
    {
        Assembly assembly = applicationAssembly ?? typeof(RuntimeModeSelector).Assembly;
        string[] configuredModes = assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .Where(attribute => string.Equals(attribute.Key, BuildMetadataKey, StringComparison.Ordinal))
            .Select(static attribute => attribute.Value)
            .Where(static value => !string.IsNullOrWhiteSpace(value))
            .Cast<string>()
            .ToArray();
        if (configuredModes.Length != 1)
        {
            return WorkflowRuntimeEnvironment.ManualPreview;
        }

        return string.Equals(
            configuredModes[0],
            nameof(WorkflowRuntimeEnvironment.Production),
            StringComparison.Ordinal)
                ? WorkflowRuntimeEnvironment.Production
                : WorkflowRuntimeEnvironment.ManualPreview;
    }
}
