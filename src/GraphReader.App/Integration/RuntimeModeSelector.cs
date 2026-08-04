// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;

namespace GraphReader.App.Integration;

public static class RuntimeModeSelector
{
    private const string RuntimeModeVariable = "GRAPHREADER_RUNTIME_MODE";

    public static WorkflowRuntimeEnvironment Select(string? configuredValue = null)
    {
        string? value = configuredValue ?? Environment.GetEnvironmentVariable(RuntimeModeVariable);
        return string.Equals(value, nameof(WorkflowRuntimeEnvironment.Production), StringComparison.OrdinalIgnoreCase)
            ? WorkflowRuntimeEnvironment.Production
            : WorkflowRuntimeEnvironment.ManualPreview;
    }
}
