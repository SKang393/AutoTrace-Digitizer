// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.IO;
using GraphReader.Inference;

namespace GraphReader.App.Integration;

public sealed record ProductionModelAvailabilitySnapshot
{
    public ProductionModelAvailabilitySnapshot(
        IEnumerable<string> approvedCpuTasks,
        string evidence,
        IReadOnlyDictionary<string, ResolvedProductionModel>? approvedCpuModels = null)
    {
        ArgumentNullException.ThrowIfNull(approvedCpuTasks);
        ArgumentException.ThrowIfNullOrWhiteSpace(evidence);
        ApprovedCpuTasks = new ReadOnlySet<string>(
            new HashSet<string>(approvedCpuTasks, StringComparer.Ordinal));
        ApprovedCpuModels = new ReadOnlyDictionary<string, ResolvedProductionModel>(
            new Dictionary<string, ResolvedProductionModel>(
                approvedCpuModels ?? new Dictionary<string, ResolvedProductionModel>(),
                StringComparer.Ordinal));
        Evidence = evidence;
    }

    public IReadOnlySet<string> ApprovedCpuTasks { get; }

    public IReadOnlyDictionary<string, ResolvedProductionModel> ApprovedCpuModels { get; }

    public string Evidence { get; }

    public static ProductionModelAvailabilitySnapshot Missing(string evidence) =>
        new([], evidence);
}

public static class ProductionModelAvailabilityProbe
{
    public static async Task<ProductionModelAvailabilitySnapshot> InspectAsync(
        string? modelRoot,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (string.IsNullOrWhiteSpace(modelRoot))
        {
            return ProductionModelAvailabilitySnapshot.Missing(
                "No application model root is configured.");
        }

        string root = Path.GetFullPath(modelRoot);
        string indexPath = Path.Combine(root, "production-model-index.json");
        if (!File.Exists(indexPath))
        {
            return ProductionModelAvailabilitySnapshot.Missing(
                "No production-model-index.json is installed in the application model root.");
        }

        try
        {
            var store = new ProductionModelStore(root);
            IReadOnlyList<ResolvedProductionModel> models = await store.ResolveAllAsync(
                    InferenceProvider.Cpu,
                    cancellationToken)
                .ConfigureAwait(false);
            string[] ambiguousTasks = models
                .GroupBy(static model => model.Task, StringComparer.Ordinal)
                .Where(static group => group.Count() != 1)
                .Select(static group => group.Key)
                .Order(StringComparer.Ordinal)
                .ToArray();
            if (ambiguousTasks.Length > 0)
            {
                return ProductionModelAvailabilitySnapshot.Missing(
                    $"Production model store validation failed closed because tasks resolve ambiguously: {string.Join(", ", ambiguousTasks)}.");
            }

            var modelsByTask = new ReadOnlyDictionary<string, ResolvedProductionModel>(
                models.ToDictionary(static model => model.Task, StringComparer.Ordinal));
            var tasks = new HashSet<string>(
                modelsByTask.Keys,
                StringComparer.Ordinal);
            string identities = models.Count == 0
                ? "none"
                : string.Join(
                    ", ",
                    models.Select(static model =>
                        $"{model.Identity.ModelId}@{model.Identity.Version}:{model.Identity.Sha256[..12].ToLowerInvariant()}"));
            return new ProductionModelAvailabilitySnapshot(
                tasks,
                $"Checksum-resolved CPU production models: {identities}.",
                modelsByTask);
        }
        catch (ProductionModelValidationException exception)
        {
            return ProductionModelAvailabilitySnapshot.Missing(
                $"Production model store validation failed closed ({exception.Code}): {exception.Message}");
        }
        catch (ArgumentException exception)
        {
            return ProductionModelAvailabilitySnapshot.Missing(
                $"Production model store path validation failed closed: {exception.Message}");
        }
        catch (NotSupportedException exception)
        {
            return ProductionModelAvailabilitySnapshot.Missing(
                $"Production model store path is unsupported: {exception.Message}");
        }
    }
}
