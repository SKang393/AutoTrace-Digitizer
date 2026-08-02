// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics.CodeAnalysis;

namespace GraphReader.Domain;

public enum DomainErrorSeverity
{
    Warning,
    Error
}

public sealed record DomainError(
    string Code,
    DomainErrorSeverity Severity,
    string UserMessageKey,
    string TechnicalMessage,
    bool Recoverable,
    string SuggestedAction);

public sealed record DomainResult<T>
{
    private DomainResult(T? value, IReadOnlyList<DomainError> errors)
    {
        Value = value;
        Errors = errors;
    }

    public T? Value { get; }

    public IReadOnlyList<DomainError> Errors { get; }

    public bool IsSuccess => Errors.Count == 0 && Value is not null;

    [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "The factory preserves the result's generic type at each call site.")]
    public static DomainResult<T> Success(T value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return new DomainResult<T>(value, Array.Empty<DomainError>());
    }

    [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "The factory preserves the result's generic type at each call site.")]
    public static DomainResult<T> Failure(params DomainError[] errors)
    {
        ArgumentNullException.ThrowIfNull(errors);
        if (errors.Length == 0)
        {
            throw new ArgumentException("At least one error is required.", nameof(errors));
        }

        return new DomainResult<T>(default, Array.AsReadOnly((DomainError[])errors.Clone()));
    }

    [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "The factory preserves the result's generic type at each call site.")]
    public static DomainResult<T> Failure(IEnumerable<DomainError> errors)
    {
        ArgumentNullException.ThrowIfNull(errors);
        DomainError[] materialized = errors.ToArray();
        return Failure(materialized);
    }
}

internal static class DomainErrors
{
    public static DomainError InvalidProject(string technicalMessage) =>
        new(
            "PROJECT_INVALID",
            DomainErrorSeverity.Error,
            "Errors.ProjectInvalid",
            technicalMessage,
            Recoverable: true,
            "select_recovery");

    public static DomainError CorruptProject(string technicalMessage) =>
        new(
            "PROJECT_CORRUPT",
            DomainErrorSeverity.Error,
            "Errors.ProjectCorrupt",
            technicalMessage,
            Recoverable: true,
            "select_recovery");

    public static DomainError IoFailure(string code, string userMessageKey, string technicalMessage, string suggestedAction) =>
        new(
            code,
            DomainErrorSeverity.Error,
            userMessageKey,
            technicalMessage,
            Recoverable: true,
            suggestedAction);
}
