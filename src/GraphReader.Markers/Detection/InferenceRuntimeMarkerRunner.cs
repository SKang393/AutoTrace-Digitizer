// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;

namespace GraphReader.Markers.Detection;

/// <summary>
/// Non-owning adapter over the shared local inference runtime.
/// </summary>
public sealed class InferenceRuntimeMarkerRunner : IMarkerInferenceRunner
{
    private readonly InferenceRuntime _runtime;

    public InferenceRuntimeMarkerRunner(InferenceRuntime runtime) =>
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));

    public ValueTask<InferenceResponse> RunAsync(
        InferenceRequest request,
        CancellationToken cancellationToken) =>
        _runtime.RunAsync(request, cancellationToken);
}
