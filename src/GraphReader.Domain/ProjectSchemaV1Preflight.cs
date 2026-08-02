// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json.Nodes;

namespace GraphReader.Domain;

internal static class ProjectSchemaV1Preflight
{
    private static readonly string[] AllowedRootProperties =
    {
        "app_version",
        "audit",
        "created_utc",
        "modified_utc",
        "panels",
        "project_id",
        "schema_version",
        "settings",
        "sources"
    };

    private static readonly string[] RequiredRootProperties =
    {
        "schema_version",
        "project_id",
        "app_version",
        "sources",
        "panels",
        "audit"
    };

    private static readonly string[] AllowedSettingsProperties =
    {
        "appearance",
        "default_enhancement_scale",
        "locale",
        "phase_overlay_visible",
        "require_first_session_one"
    };

    private static readonly string[] RequiredSourceProperties =
    {
        "source_id",
        "kind",
        "sha256",
        "display_name"
    };

    private static readonly string[] RequiredPanelProperties =
    {
        "panel_id",
        "source_id",
        "crop",
        "transforms",
        "series",
        "points",
        "phases"
    };

    private static readonly string[] AllowedCropProperties =
    {
        "height",
        "width",
        "x",
        "y"
    };

    private static readonly string[] RequiredCropProperties =
    {
        "x",
        "y",
        "width",
        "height"
    };

    private static readonly string[] RequiredAuditProperties = { "events" };

    public static DomainResult<JsonObject> Normalize(JsonObject source)
    {
        ArgumentNullException.ThrowIfNull(source);
        var errors = new List<DomainError>();
        RejectUnknownProperties(source, AllowedRootProperties, "project root", errors);
        RequireProperties(source, RequiredRootProperties, "project root", errors);

        JsonObject? settings = null;
        if (source.TryGetPropertyValue("settings", out JsonNode? settingsNode))
        {
            if (settingsNode is JsonObject settingsObject)
            {
                settings = settingsObject;
                RejectUnknownProperties(settings, AllowedSettingsProperties, "settings", errors);
            }
            else
            {
                errors.Add(SchemaError("settings must be an object when present."));
            }
        }

        JsonArray? sources = RequireArray(source, "sources", "project root", errors);
        if (sources is not null)
        {
            for (int index = 0; index < sources.Count; index++)
            {
                if (sources[index] is not JsonObject sourceObject)
                {
                    errors.Add(SchemaError($"sources[{index}] must be an object."));
                    continue;
                }

                RequireProperties(sourceObject, RequiredSourceProperties, $"sources[{index}]", errors);
            }
        }

        JsonArray? panels = RequireArray(source, "panels", "project root", errors);
        if (panels is not null)
        {
            for (int index = 0; index < panels.Count; index++)
            {
                if (panels[index] is not JsonObject panel)
                {
                    errors.Add(SchemaError($"panels[{index}] must be an object."));
                    continue;
                }

                RequireProperties(panel, RequiredPanelProperties, $"panels[{index}]", errors);
                RequireArray(panel, "transforms", $"panels[{index}]", errors);
                RequireArray(panel, "series", $"panels[{index}]", errors);
                RequireArray(panel, "points", $"panels[{index}]", errors);
                RequireArray(panel, "phases", $"panels[{index}]", errors);

                if (panel["crop"] is JsonObject crop)
                {
                    RejectUnknownProperties(crop, AllowedCropProperties, $"panels[{index}].crop", errors);
                    RequireProperties(crop, RequiredCropProperties, $"panels[{index}].crop", errors);
                }
                else if (panel.ContainsKey("crop"))
                {
                    errors.Add(SchemaError($"panels[{index}].crop must be an object."));
                }

                EnsureOptionalArray(panel, "ocr_regions", $"panels[{index}]", errors);
                EnsureOptionalArray(panel, "markers", $"panels[{index}]", errors);
            }
        }

        JsonObject? audit = null;
        if (source["audit"] is JsonObject auditObject)
        {
            audit = auditObject;
            RequireProperties(audit, RequiredAuditProperties, "audit", errors);
            RequireArray(audit, "events", "audit", errors);
        }
        else if (source.ContainsKey("audit"))
        {
            errors.Add(SchemaError("audit must be an object."));
        }

        if (errors.Count > 0)
        {
            return DomainResult<JsonObject>.Failure(errors);
        }

        var normalized = (JsonObject)source.DeepClone();
        NormalizeSettings(normalized, settings);
        NormalizeTimestamps(normalized);
        NormalizeSources(normalized);
        NormalizePanels(normalized);
        ((JsonObject)normalized["audit"]!)["last_autosave_utc"] ??= null;
        return DomainResult<JsonObject>.Success(normalized);
    }

    private static void NormalizeSettings(JsonObject normalized, JsonObject? originalSettings)
    {
        JsonObject normalizedSettings = originalSettings is null
            ? new JsonObject()
            : (JsonObject)originalSettings.DeepClone();
        normalizedSettings["require_first_session_one"] ??= true;
        normalizedSettings["default_enhancement_scale"] ??= 2;
        normalizedSettings["phase_overlay_visible"] ??= true;
        normalizedSettings["appearance"] ??= "system";
        normalizedSettings["locale"] ??= "en-US";
        normalized["settings"] = normalizedSettings;
    }

    private static void NormalizeTimestamps(JsonObject normalized)
    {
        const string epoch = "1970-01-01T00:00:00+00:00";
        if (normalized["created_utc"] is null)
        {
            normalized["created_utc"] = normalized["modified_utc"]?.DeepClone() ?? epoch;
        }

        if (normalized["modified_utc"] is null)
        {
            normalized["modified_utc"] = normalized["created_utc"]?.DeepClone() ?? epoch;
        }
    }

    private static void NormalizeSources(JsonObject normalized)
    {
        foreach (JsonNode? node in (JsonArray)normalized["sources"]!)
        {
            var source = (JsonObject)node!;
            source["local_path"] ??= null;
            source["article_metadata"] ??= null;
        }
    }

    private static void NormalizePanels(JsonObject normalized)
    {
        foreach (JsonNode? node in (JsonArray)normalized["panels"]!)
        {
            var panel = (JsonObject)node!;
            panel["page_number"] ??= null;
            panel["display_name"] ??= string.Empty;
            panel["participant"] ??= null;
            panel["enhancement"] ??= null;
            panel["calibration"] ??= null;
            panel["ocr_regions"] ??= new JsonArray();
            panel["markers"] ??= new JsonArray();
            panel["export_settings"] ??= null;
            panel["validation"] ??= null;
        }
    }

    private static void RejectUnknownProperties(
        JsonObject source,
        IReadOnlyCollection<string> allowed,
        string location,
        List<DomainError> errors)
    {
        foreach (string propertyName in source.Select(property => property.Key))
        {
            if (!allowed.Contains(propertyName, StringComparer.Ordinal))
            {
                errors.Add(SchemaError($"Unknown property '{propertyName}' at {location}."));
            }
        }
    }

    private static void RequireProperties(
        JsonObject source,
        IEnumerable<string> required,
        string location,
        List<DomainError> errors)
    {
        foreach (string propertyName in required)
        {
            if (!source.TryGetPropertyValue(propertyName, out JsonNode? value) || value is null)
            {
                errors.Add(SchemaError($"Missing required property '{propertyName}' at {location}."));
            }
        }
    }

    private static JsonArray? RequireArray(
        JsonObject source,
        string propertyName,
        string location,
        List<DomainError> errors)
    {
        if (source[propertyName] is JsonArray array)
        {
            return array;
        }

        if (source.ContainsKey(propertyName))
        {
            errors.Add(SchemaError($"{location}.{propertyName} must be an array."));
        }

        return null;
    }

    private static void EnsureOptionalArray(
        JsonObject source,
        string propertyName,
        string location,
        List<DomainError> errors)
    {
        if (source.TryGetPropertyValue(propertyName, out JsonNode? value) &&
            value is not null &&
            value is not JsonArray)
        {
            errors.Add(SchemaError($"{location}.{propertyName} must be an array when present."));
        }
    }

    private static DomainError SchemaError(string technicalMessage) =>
        DomainErrors.CorruptProject($"Schema version 1 validation failed: {technicalMessage}");
}
