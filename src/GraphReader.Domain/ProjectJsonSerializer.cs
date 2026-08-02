// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace GraphReader.Domain;

public sealed class ProjectJsonSerializer
{
    private static readonly JsonSerializerOptions SerializerOptions = CreateSerializerOptions();
    private readonly ProjectMigrationDispatcher _migrationDispatcher;
    private readonly ProjectValidator _validator;

    public ProjectJsonSerializer(
        ProjectMigrationDispatcher? migrationDispatcher = null,
        ProjectValidator? validator = null)
    {
        _migrationDispatcher = migrationDispatcher ?? new ProjectMigrationDispatcher();
        _validator = validator ?? new ProjectValidator();
    }

    public DomainResult<string> Serialize(ProjectDocument project)
    {
        ArgumentNullException.ThrowIfNull(project);

        try
        {
            ProjectDocument frozen = ProjectStateFreezer.Freeze(project);
            IReadOnlyList<DomainError> validationErrors = _validator.Validate(frozen);
            if (validationErrors.Count > 0)
            {
                return DomainResult<string>.Failure(validationErrors);
            }

            JsonNode? serialized = JsonSerializer.SerializeToNode(frozen, SerializerOptions);
            if (serialized is null)
            {
                return DomainResult<string>.Failure(DomainErrors.InvalidProject(
                    "The project serializer produced no JSON document."));
            }

            JsonNode canonical = Canonicalize(serialized);
            string deterministicJson = canonical
                .ToJsonString(SerializerOptions)
                .ReplaceLineEndings("\n");
            return DomainResult<string>.Success(deterministicJson + "\n");
        }
        catch (Exception exception) when (
            exception is JsonException or
            NotSupportedException or
            InvalidOperationException or
            NullReferenceException)
        {
            return DomainResult<string>.Failure(DomainErrors.InvalidProject(
                $"The project could not be serialized: {exception.Message}"));
        }
    }

    public DomainResult<ProjectDocument> Deserialize(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return DomainResult<ProjectDocument>.Failure(DomainErrors.CorruptProject(
                "The project file is empty."));
        }

        try
        {
            JsonNode? parsed = JsonNode.Parse(
                json,
                documentOptions: new JsonDocumentOptions
                {
                    AllowTrailingCommas = false,
                    CommentHandling = JsonCommentHandling.Disallow,
                    MaxDepth = 128
                });

            if (parsed is not JsonObject document)
            {
                return DomainResult<ProjectDocument>.Failure(DomainErrors.CorruptProject(
                    "The project root must be a JSON object."));
            }

            DomainResult<JsonObject> migrated = _migrationDispatcher.Dispatch(document);
            if (!migrated.IsSuccess || migrated.Value is null)
            {
                return DomainResult<ProjectDocument>.Failure(migrated.Errors);
            }

            DomainResult<JsonObject> preflight = ProjectSchemaV1Preflight.Normalize(migrated.Value);
            if (!preflight.IsSuccess || preflight.Value is null)
            {
                return DomainResult<ProjectDocument>.Failure(preflight.Errors);
            }

            ProjectDocument? project = preflight.Value.Deserialize<ProjectDocument>(SerializerOptions);
            if (project is null)
            {
                return DomainResult<ProjectDocument>.Failure(DomainErrors.CorruptProject(
                    "The project JSON did not produce a project document."));
            }

            IReadOnlyList<DomainError> validationErrors = _validator.Validate(project);
            return validationErrors.Count == 0
                ? DomainResult<ProjectDocument>.Success(ProjectStateFreezer.Freeze(project))
                : DomainResult<ProjectDocument>.Failure(validationErrors);
        }
        catch (Exception exception) when (
            exception is JsonException or
            NotSupportedException or
            InvalidOperationException or
            ArgumentException or
            NullReferenceException)
        {
            return DomainResult<ProjectDocument>.Failure(DomainErrors.CorruptProject(
                $"The project JSON is corrupt: {exception.Message}"));
        }
    }

    private static JsonSerializerOptions CreateSerializerOptions()
    {
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DictionaryKeyPolicy = null,
            WriteIndented = true,
            Encoder = JavaScriptEncoder.Default,
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
            NumberHandling = JsonNumberHandling.Strict
        };
        options.Converters.Add(new StableIdJsonConverterFactory());
        options.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower, allowIntegerValues: false));
        return options;
    }

    private static JsonNode Canonicalize(JsonNode node)
    {
        if (node is JsonObject jsonObject)
        {
            var canonicalObject = new JsonObject();
            foreach (KeyValuePair<string, JsonNode?> property in
                     jsonObject.OrderBy(property => property.Key, StringComparer.Ordinal))
            {
                canonicalObject[property.Key] = property.Value is null
                    ? null
                    : Canonicalize(property.Value);
            }

            return canonicalObject;
        }

        if (node is JsonArray jsonArray)
        {
            var canonicalArray = new JsonArray();
            foreach (JsonNode? item in jsonArray)
            {
                canonicalArray.Add(item is null ? null : Canonicalize(item));
            }

            return canonicalArray;
        }

        return node.DeepClone();
    }

    private sealed class StableIdJsonConverterFactory : JsonConverterFactory
    {
        public override bool CanConvert(Type typeToConvert) =>
            typeToConvert.GetInterfaces().Any(
                interfaceType => interfaceType.IsGenericType &&
                                 interfaceType.GetGenericTypeDefinition() == typeof(IStableId<>));

        public override JsonConverter CreateConverter(Type typeToConvert, JsonSerializerOptions options)
        {
            Type converterType = typeof(StableIdJsonConverter<>).MakeGenericType(typeToConvert);
            return (JsonConverter)(Activator.CreateInstance(converterType, nonPublic: true)
                ?? throw new InvalidOperationException($"Unable to create an ID converter for {typeToConvert.Name}."));
        }
    }

    private sealed class StableIdJsonConverter<TId> : JsonConverter<TId>
        where TId : struct, IStableId<TId>
    {
        public override TId Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.String || !reader.TryGetGuid(out Guid value))
            {
                throw new JsonException($"{typeof(TId).Name} must be a UUID string.");
            }

            return TId.FromGuid(value);
        }

        public override void Write(Utf8JsonWriter writer, TId value, JsonSerializerOptions options) =>
            writer.WriteStringValue(value.Value);
    }
}
