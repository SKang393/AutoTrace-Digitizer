// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.SuperResolution;

public enum BenchmarkRuntimeCompatibility
{
    ReferenceWeightsOnly,
    NcnnVulkan
}

public sealed record EnhancementBenchmarkModel(
    string ModelId,
    string ModelVersion,
    int NativeScale,
    int OutputScale,
    string ArtifactFormat,
    IReadOnlyList<string> Providers,
    BenchmarkRuntimeCompatibility RuntimeCompatibility);

public sealed record EnhancementBenchmarkCase(
    string CaseId,
    string InputPath,
    string GroundTruthPath);

public sealed record EnhancementBenchmarkWorkItem(
    EnhancementBenchmarkModel Model,
    EnhancementBenchmarkCase TestCase);

public sealed record EnhancementBenchmarkMetrics(
    double MarkerCenterF1,
    double ShapeFillClassificationF1,
    double NumericOcrExactMatch,
    double AxisLocalizationErrorPixels,
    double HallucinatedStructureRate,
    double RuntimeMilliseconds,
    long PeakMemoryBytes);

public sealed record EnhancementBenchmarkObservation(
    EnhancementBenchmarkWorkItem WorkItem,
    EnhancementBenchmarkMetrics Metrics,
    string EvidenceSha256);

public interface IEnhancementBenchmarkRunner
{
    Task<EnhancementBenchmarkObservation> RunAsync(
        EnhancementBenchmarkWorkItem workItem,
        CancellationToken cancellationToken);
}

public sealed class EnhancementBenchmarkHarness
{
    private static readonly IReadOnlyList<string> CpuCuda = Array.AsReadOnly(new[] { "cpu", "cuda" });
    private static readonly IReadOnlyList<string> Vulkan = Array.AsReadOnly(new[] { "vulkan" });
    private static readonly IReadOnlyList<EnhancementBenchmarkModel> Required = Array.AsReadOnly(
        new[]
        {
            new EnhancementBenchmarkModel(
                "RealESRGAN_x2plus",
                "v0.2.1",
                2,
                2,
                "pytorch_pth",
                CpuCuda,
                BenchmarkRuntimeCompatibility.ReferenceWeightsOnly),
            new EnhancementBenchmarkModel(
                "realesr-general-x4v3",
                "v0.2.5.0",
                4,
                2,
                "pytorch_pth",
                CpuCuda,
                BenchmarkRuntimeCompatibility.ReferenceWeightsOnly),
            new EnhancementBenchmarkModel(
                "realesr-animevideov3",
                "v0.2.5.0",
                2,
                2,
                "ncnn_param_bin",
                Vulkan,
                BenchmarkRuntimeCompatibility.NcnnVulkan)
        });

    public static IReadOnlyList<EnhancementBenchmarkModel> RequiredModels => Required;

    public static IReadOnlyList<EnhancementBenchmarkWorkItem> CreatePlan(
        IEnumerable<EnhancementBenchmarkCase> testCases)
    {
        ArgumentNullException.ThrowIfNull(testCases);
        EnhancementBenchmarkCase[] cases = testCases.ToArray();
        if (cases.Length == 0)
        {
            throw new ArgumentException("A fixed benchmark set is required.", nameof(testCases));
        }

        if (cases.Any(static item => string.IsNullOrWhiteSpace(item.CaseId) ||
                                     string.IsNullOrWhiteSpace(item.InputPath) ||
                                     string.IsNullOrWhiteSpace(item.GroundTruthPath)))
        {
            throw new ArgumentException("Every benchmark case needs an ID, input, and ground truth path.", nameof(testCases));
        }

        return Required
            .SelectMany(model => cases.Select(testCase => new EnhancementBenchmarkWorkItem(model, testCase)))
            .ToArray();
    }

    public static async Task<IReadOnlyList<EnhancementBenchmarkObservation>> RunAsync(
        IEnumerable<EnhancementBenchmarkCase> testCases,
        IEnhancementBenchmarkRunner runner,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(runner);
        IReadOnlyList<EnhancementBenchmarkWorkItem> plan = CreatePlan(testCases);
        var observations = new List<EnhancementBenchmarkObservation>(plan.Count);
        foreach (EnhancementBenchmarkWorkItem workItem in plan)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EnhancementBenchmarkObservation observation = await runner.RunAsync(
                workItem,
                cancellationToken).ConfigureAwait(false);
            ValidateObservation(workItem, observation);
            observations.Add(observation);
        }

        return observations.AsReadOnly();
    }

    private static void ValidateObservation(
        EnhancementBenchmarkWorkItem expected,
        EnhancementBenchmarkObservation observation)
    {
        if (observation.WorkItem != expected)
        {
            throw new InvalidDataException("Benchmark runner returned evidence for a different work item.");
        }

        EnhancementBenchmarkMetrics metrics = observation.Metrics;
        ValidateRate(metrics.MarkerCenterF1, nameof(metrics.MarkerCenterF1));
        ValidateRate(metrics.ShapeFillClassificationF1, nameof(metrics.ShapeFillClassificationF1));
        ValidateRate(metrics.NumericOcrExactMatch, nameof(metrics.NumericOcrExactMatch));
        ValidateRate(metrics.HallucinatedStructureRate, nameof(metrics.HallucinatedStructureRate));
        if (!double.IsFinite(metrics.AxisLocalizationErrorPixels) || metrics.AxisLocalizationErrorPixels < 0 ||
            !double.IsFinite(metrics.RuntimeMilliseconds) || metrics.RuntimeMilliseconds < 0 ||
            metrics.PeakMemoryBytes < 0)
        {
            throw new InvalidDataException("Benchmark timing, memory, and localization metrics must be finite and nonnegative.");
        }

        EnhancementHashing.NormalizeSha256(observation.EvidenceSha256);
    }

    private static void ValidateRate(double value, string name)
    {
        if (!double.IsFinite(value) || value is < 0 or > 1)
        {
            throw new InvalidDataException($"{name} must be a finite value in [0, 1].");
        }
    }
}
