// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security;
using System.Security.Cryptography;
using System.Text.Json;

namespace GraphReader.App.Integration;

public sealed record ProductionRuntimeAvailabilitySnapshot(
    bool AxisApproved,
    string Evidence,
    string? RuntimeSha256 = null)
{
    public static ProductionRuntimeAvailabilitySnapshot Missing(string evidence) =>
        new(false, evidence);
}

public static class ProductionRuntimeAvailabilityProbe
{
    private const string MetadataFileName = "reviewed-opencv-runtime.json";
    private const string RuntimeFileName = "OpenCvSharpExtern.dll";
    private const string RequiredSchema = "graphreader.reviewed-opencv-runtime.v1";
    private const string RequiredProfile = "graphreader-axis-minimal-win-x64";

    private static readonly HashSet<string> ExpectedProperties = new(StringComparer.Ordinal)
    {
        "schema",
        "runtimeId",
        "profileId",
        "evidenceRootName",
        "binarySha256",
        "replacedBinarySha256",
        "sourceRevisions",
        "provenanceValidated",
        "noticeReviewStatus",
        "maintainerAttestationStatus",
        "cleanMachineEvidence",
        "releaseApproved",
    };

    private static readonly HashSet<string> ExpectedSourceRevisionProperties = new(StringComparer.Ordinal)
    {
        "openCvSharp",
        "openCv",
        "vcpkg",
    };

    public static async Task<ProductionRuntimeAvailabilitySnapshot> InspectAsync(
        string? applicationRoot,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (string.IsNullOrWhiteSpace(applicationRoot))
        {
            return ProductionRuntimeAvailabilitySnapshot.Missing(
                "The application runtime root is unavailable.");
        }

        string root;
        try
        {
            root = Path.GetFullPath(applicationRoot);
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or
            PathTooLongException or SecurityException)
        {
            return ProductionRuntimeAvailabilitySnapshot.Missing(
                $"The application runtime root is invalid: {exception.Message}");
        }

        string metadataPath = Path.Combine(root, MetadataFileName);
        string runtimePath = Path.Combine(root, RuntimeFileName);
        if (!File.Exists(metadataPath) || !File.Exists(runtimePath))
        {
            return ProductionRuntimeAvailabilitySnapshot.Missing(
                $"The exact reviewed OpenCV runtime requires {MetadataFileName} and {RuntimeFileName} in the application root.");
        }

        try
        {
            await using FileStream metadataStream = new(
                metadataPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 4096,
                FileOptions.Asynchronous | FileOptions.SequentialScan);
            using JsonDocument document = await JsonDocument.ParseAsync(
                metadataStream,
                cancellationToken: cancellationToken).ConfigureAwait(false);
            JsonElement metadata = document.RootElement;
            RequireExactObject(metadata, ExpectedProperties, "reviewed OpenCV runtime metadata");

            string schema = RequireString(metadata, "schema");
            string runtimeId = RequireString(metadata, "runtimeId");
            string profileId = RequireString(metadata, "profileId");
            _ = RequireString(metadata, "evidenceRootName");
            string binarySha256 = RequireSha256(metadata, "binarySha256");
            _ = RequireSha256(metadata, "replacedBinarySha256");
            JsonElement revisions = RequireProperty(metadata, "sourceRevisions", JsonValueKind.Object);
            RequireExactObject(revisions, ExpectedSourceRevisionProperties, "reviewed OpenCV source revisions");
            foreach (string property in ExpectedSourceRevisionProperties)
            {
                _ = RequireString(revisions, property);
            }

            bool provenanceValidated = RequireBoolean(metadata, "provenanceValidated");
            string noticeReviewStatus = RequireString(metadata, "noticeReviewStatus");
            string attestationStatus = RequireString(metadata, "maintainerAttestationStatus");
            bool cleanMachineEvidence = RequireBoolean(metadata, "cleanMachineEvidence");
            bool releaseApproved = RequireBoolean(metadata, "releaseApproved");

            await using FileStream runtimeStream = new(
                runtimePath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 81920,
                FileOptions.Asynchronous | FileOptions.SequentialScan);
            string actualSha256 = Convert.ToHexStringLower(
                await SHA256.HashDataAsync(runtimeStream, cancellationToken).ConfigureAwait(false));
            if (!string.Equals(actualSha256, binarySha256, StringComparison.Ordinal))
            {
                return ProductionRuntimeAvailabilitySnapshot.Missing(
                    $"The installed OpenCV runtime checksum does not match {MetadataFileName}.");
            }

            bool approved = string.Equals(schema, RequiredSchema, StringComparison.Ordinal) &&
                string.Equals(runtimeId, "opencvsharpextern-source-audited", StringComparison.Ordinal) &&
                string.Equals(profileId, RequiredProfile, StringComparison.Ordinal) &&
                provenanceValidated &&
                string.Equals(noticeReviewStatus, "complete", StringComparison.Ordinal) &&
                string.Equals(attestationStatus, "recorded-private", StringComparison.Ordinal) &&
                cleanMachineEvidence &&
                releaseApproved;
            return approved
                ? new ProductionRuntimeAvailabilitySnapshot(
                    true,
                    $"Exact reviewed OpenCV runtime {actualSha256} has provenance, notice, clean-machine, and release approval evidence.",
                    actualSha256)
                : ProductionRuntimeAvailabilitySnapshot.Missing(
                    $"OpenCV runtime {actualSha256} is checksum-valid but lacks mandatory clean-machine or release approval evidence.");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or
            SecurityException or JsonException or InvalidDataException)
        {
            return ProductionRuntimeAvailabilitySnapshot.Missing(
                $"Reviewed OpenCV runtime evidence is invalid: {exception.Message}");
        }
    }

    private static void RequireExactObject(
        JsonElement element,
        HashSet<string> expected,
        string description)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"The {description} must be a JSON object.");
        }

        var actual = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonProperty property in element.EnumerateObject())
        {
            if (!actual.Add(property.Name))
            {
                throw new InvalidDataException(
                    $"The {description} contains duplicate property '{property.Name}'.");
            }
            if (!expected.Contains(property.Name))
            {
                throw new InvalidDataException(
                    $"The {description} contains unexpected property '{property.Name}'.");
            }
        }

        string[] missing = expected.Where(property => !actual.Contains(property)).Order().ToArray();
        if (missing.Length > 0)
        {
            throw new InvalidDataException(
                $"The {description} is missing properties: {string.Join(", ", missing)}.");
        }
    }

    private static JsonElement RequireProperty(
        JsonElement element,
        string propertyName,
        JsonValueKind kind)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement value) || value.ValueKind != kind)
        {
            throw new InvalidDataException(
                $"Reviewed OpenCV runtime property '{propertyName}' must be {kind}.");
        }

        return value;
    }

    private static string RequireString(JsonElement element, string propertyName)
    {
        string? value = RequireProperty(element, propertyName, JsonValueKind.String).GetString();
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidDataException(
                $"Reviewed OpenCV runtime property '{propertyName}' must not be empty.");
        }

        return value;
    }

    private static string RequireSha256(JsonElement element, string propertyName)
    {
        string value = RequireString(element, propertyName);
        if (value.Length != 64 || value.Any(character => !Uri.IsHexDigit(character)))
        {
            throw new InvalidDataException(
                $"Reviewed OpenCV runtime property '{propertyName}' must be a SHA-256 value.");
        }

        return value.ToLowerInvariant();
    }

    private static bool RequireBoolean(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw new InvalidDataException(
                $"Reviewed OpenCV runtime property '{propertyName}' must be Boolean.");
        }

        return value.GetBoolean();
    }
}
