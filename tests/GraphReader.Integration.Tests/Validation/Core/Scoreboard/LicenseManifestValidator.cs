// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;

namespace GraphReader.Validation.Scoreboard;

public static class LicenseManifestValidator
{
    private static readonly string[] ProhibitedLicenseFragments =
    [
        "AGPL",
        "GPL",
        "SSPL",
        "BUSL",
        "NON-COMMERCIAL",
        "NONCOMMERCIAL",
        "-NC-",
    ];

    public static IReadOnlyList<LicenseManifestValidation> ValidateRepository(
        string repositoryRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repositoryRoot);
        string root = Path.GetFullPath(repositoryRoot);
        string manifestRoot = Path.Combine(root, "models", "manifest");
        if (!Directory.Exists(manifestRoot))
        {
            return
            [
                new LicenseManifestValidation(
                    "models/manifest",
                    "model-manifests",
                    false,
                    ["Model manifest directory is missing."]),
            ];
        }

        string[] paths = Directory.GetFiles(manifestRoot, "*.json", SearchOption.AllDirectories)
            .Order(StringComparer.Ordinal)
            .ToArray();
        if (paths.Length == 0)
        {
            return
            [
                new LicenseManifestValidation(
                    "models/manifest",
                    "model-manifests",
                    false,
                    ["No model manifests were found."]),
            ];
        }

        return paths.Select(path => ValidateManifest(root, path)).ToArray();
    }

    private static LicenseManifestValidation ValidateManifest(
        string repositoryRoot,
        string manifestPath)
    {
        string relativePath = NormalizePath(Path.GetRelativePath(repositoryRoot, manifestPath));
        List<string> issues = [];
        string componentId = Path.GetFileNameWithoutExtension(manifestPath);

        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
            JsonElement root = document.RootElement;
            componentId = ReadRequiredString(root, "model_id", issues) ?? componentId;
            string? checksum = ReadRequiredString(root, "sha256", issues);
            if (checksum is not null &&
                (checksum.Length != 64 || checksum.Any(character => !char.IsAsciiHexDigit(character))))
            {
                issues.Add("sha256 must contain exactly 64 hexadecimal characters.");
            }

            bool commercialUse = ReadRequiredBoolean(root, "commercial_use", issues);
            if (!commercialUse)
            {
                issues.Add("commercial_use must be true for an eligible model.");
            }

            bool redistributable = ReadRequiredBoolean(root, "redistribution", issues);
            ValidateLicense(repositoryRoot, root, issues);
            if (!redistributable && HasBundledModelFile(repositoryRoot, manifestPath, root))
            {
                issues.Add("A bundled model file declares redistribution=false.");
            }
        }
        catch (JsonException exception)
        {
            issues.Add($"Manifest JSON is invalid: {exception.Message}");
        }
        catch (IOException exception)
        {
            issues.Add($"Manifest could not be read: {exception.Message}");
        }
        catch (UnauthorizedAccessException exception)
        {
            issues.Add($"Manifest could not be read: {exception.Message}");
        }

        return new LicenseManifestValidation(
            relativePath,
            componentId,
            issues.Count == 0,
            issues.Order(StringComparer.Ordinal).ToArray());
    }

    private static void ValidateLicense(
        string repositoryRoot,
        JsonElement manifest,
        List<string> issues)
    {
        if (!manifest.TryGetProperty("license", out JsonElement license) ||
            license.ValueKind != JsonValueKind.Object)
        {
            issues.Add("license object is missing.");
            return;
        }

        string? spdx = ReadRequiredString(license, "spdx", issues, "license.spdx");
        if (spdx is not null && ProhibitedLicenseFragments.Any(fragment =>
                spdx.Contains(fragment, StringComparison.OrdinalIgnoreCase)))
        {
            issues.Add($"license.spdx '{spdx}' is prohibited for distribution.");
        }

        if (!license.TryGetProperty("reviewed", out JsonElement reviewed) ||
            reviewed.ValueKind is not (JsonValueKind.True or JsonValueKind.False) ||
            !reviewed.GetBoolean())
        {
            issues.Add("license.reviewed must be true.");
        }

        string? noticePath = ReadRequiredString(
            license,
            "notice_path",
            issues,
            "license.notice_path");
        if (noticePath is null)
        {
            return;
        }

        string fullNoticePath = Path.GetFullPath(
            Path.Combine(repositoryRoot, noticePath.Replace('/', Path.DirectorySeparatorChar)));
        if (!IsWithin(repositoryRoot, fullNoticePath))
        {
            issues.Add("license.notice_path escapes the repository root.");
        }
        else if (!File.Exists(fullNoticePath))
        {
            issues.Add($"license.notice_path does not exist: {noticePath}");
        }
    }

    private static bool HasBundledModelFile(
        string repositoryRoot,
        string manifestPath,
        JsonElement manifest)
    {
        if (!manifest.TryGetProperty("files", out JsonElement files) ||
            files.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        foreach (JsonElement file in files.EnumerateArray())
        {
            if (file.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(file.GetString()))
            {
                continue;
            }

            string fileName = file.GetString()!;
            string besideManifest = Path.GetFullPath(Path.Combine(
                Path.GetDirectoryName(manifestPath)!,
                fileName));
            string underModels = Path.GetFullPath(Path.Combine(repositoryRoot, "models", fileName));
            if ((IsWithin(repositoryRoot, besideManifest) && File.Exists(besideManifest)) ||
                (IsWithin(repositoryRoot, underModels) && File.Exists(underModels)))
            {
                return true;
            }
        }

        return false;
    }

    private static string? ReadRequiredString(
        JsonElement element,
        string propertyName,
        List<string> issues,
        string? displayName = null)
    {
        string name = displayName ?? propertyName;
        if (!element.TryGetProperty(propertyName, out JsonElement property) ||
            property.ValueKind != JsonValueKind.String ||
            string.IsNullOrWhiteSpace(property.GetString()))
        {
            issues.Add($"{name} is missing or empty.");
            return null;
        }

        return property.GetString();
    }

    private static bool ReadRequiredBoolean(
        JsonElement element,
        string propertyName,
        List<string> issues)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property) ||
            property.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            issues.Add($"{propertyName} must be a Boolean.");
            return false;
        }

        return property.GetBoolean();
    }

    private static bool IsWithin(string root, string candidate)
    {
        string relative = Path.GetRelativePath(root, candidate);
        return !Path.IsPathRooted(relative) &&
               !string.Equals(relative, "..", StringComparison.Ordinal) &&
               !relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal);
    }

    private static string NormalizePath(string path) => path.Replace('\\', '/');
}
