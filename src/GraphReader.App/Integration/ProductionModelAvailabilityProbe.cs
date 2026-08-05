// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.IO;
using GraphReader.Inference;

namespace GraphReader.App.Integration;

public sealed record ProductionModelAvailabilitySnapshot(
    IReadOnlySet<string> ApprovedCpuTasks,
    string Evidence)
{
    public static ProductionModelAvailabilitySnapshot Missing(string evidence) =>
        new(
            new ReadOnlySet<string>(new HashSet<string>(StringComparer.Ordinal)),
            evidence);
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
            var tasks = new HashSet<string>(
                models.Select(static model => model.Task),
                StringComparer.Ordinal);
            string identities = models.Count == 0
                ? "none"
                : string.Join(
                    ", ",
                    models.Select(static model =>
                        $"{model.Identity.ModelId}@{model.Identity.Version}:{model.Identity.Sha256[..12].ToLowerInvariant()}"));
            return new ProductionModelAvailabilitySnapshot(
                new ReadOnlySet<string>(tasks),
                $"Checksum-resolved CPU production models: {identities}.");
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
