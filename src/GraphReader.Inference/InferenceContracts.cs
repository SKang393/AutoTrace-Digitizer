// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;

namespace GraphReader.Inference;

public enum InferenceProvider
{
    DirectMl,
    Cpu,
    Fake
}

public sealed record ModelIdentity(
    string ModelId,
    string Version,
    string Sha256,
    string FilePath)
{
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentException.ThrowIfNullOrWhiteSpace(Version);
        ArgumentException.ThrowIfNullOrWhiteSpace(FilePath);
        if (Sha256.Length != 64 || !Sha256.All(Uri.IsHexDigit))
        {
            throw new ArgumentException("Model SHA-256 must contain exactly 64 hexadecimal characters.", nameof(Sha256));
        }
    }
}

public sealed record InferenceInput(
    ReadOnlyMemory<float> Values,
    IReadOnlyList<long> Shape,
    string? InputName = null,
    string? OutputName = null);

public sealed record InferenceError(
    string Code,
    string Severity,
    string UserMessageKey,
    string TechnicalMessage,
    bool Recoverable,
    string SuggestedAction);

public sealed record ProviderAttempt(
    InferenceProvider Provider,
    bool Succeeded,
    string? TechnicalMessage);

public sealed record StageTiming(
    double PreprocessMilliseconds,
    double InferenceMilliseconds,
    double PostprocessMilliseconds,
    double TotalMilliseconds,
    double SessionCreationMilliseconds,
    bool ColdSession,
    bool CacheHit);

public sealed record MemoryDiagnostics(
    long ManagedBytesBefore,
    long ManagedBytesAfter,
    long ProcessPrivateBytesBefore,
    long ProcessPrivateBytesAfter,
    int RentedBufferLength,
    bool ReusedTensorBuffer = false);

public sealed record InferenceExecution(
    IReadOnlyList<float> Output,
    InferenceProvider Provider,
    StageTiming Timing,
    MemoryDiagnostics Memory);

public sealed record SessionAcquisition(
    IInferenceSession? Session,
    IReadOnlyList<ProviderAttempt> Attempts,
    InferenceError? Error)
{
    public bool Succeeded => Session is not null && Error is null;
}

public sealed record InferenceRequest(
    ModelIdentity Model,
    InferenceInput Input,
    StageCacheMaterial CacheMaterial,
    TimeSpan Timeout);

public sealed record StageCacheMaterial(
    string InputSha256,
    string PanelCrop,
    string TransformChain,
    string StageName,
    string StageVersion,
    IReadOnlyDictionary<string, object?> Parameters,
    int ContractVersion);

public sealed record InferenceResponse(
    bool Succeeded,
    InferenceExecution? Execution,
    InferenceError? Error,
    IReadOnlyList<ProviderAttempt> ProviderAttempts)
{
    public static InferenceResponse Failure(InferenceError error, IReadOnlyList<ProviderAttempt>? attempts = null) =>
        new(false, null, error, attempts ?? Array.Empty<ProviderAttempt>());
}

public interface IInferenceSession : IAsyncDisposable
{
    InferenceProvider Provider { get; }

    ValueTask<InferenceExecution> RunAsync(InferenceInput input, CancellationToken cancellationToken);
}

public interface IInferenceSessionFactory
{
    ValueTask<IInferenceSession> CreateAsync(
        ModelIdentity model,
        InferenceProvider provider,
        CpuThreadConfiguration cpuConfiguration,
        CancellationToken cancellationToken);
}

public interface IExecutionProviderDiscovery
{
    IReadOnlyList<string> GetAvailableProviders();
}

public interface IUiThreadGuard
{
    void ThrowIfCurrentThreadIsUiThread();
}

public sealed class CapturedUiThreadGuard : IUiThreadGuard
{
    private readonly int _uiThreadId;

    public CapturedUiThreadGuard(int uiThreadId)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(uiThreadId);
        _uiThreadId = uiThreadId;
    }

    public static CapturedUiThreadGuard CaptureCurrentThread() => new(Environment.CurrentManagedThreadId);

    public void ThrowIfCurrentThreadIsUiThread()
    {
        if (Environment.CurrentManagedThreadId == _uiThreadId)
        {
            throw new UiThreadInferenceException("Inference execution is prohibited on the captured UI thread.");
        }
    }
}

public sealed class UiThreadInferenceException : InvalidOperationException
{
    public UiThreadInferenceException(string message)
        : base(message)
    {
    }
}

public sealed class NoUiThreadGuard : IUiThreadGuard
{
    public static NoUiThreadGuard Instance { get; } = new();

    private NoUiThreadGuard()
    {
    }

    public void ThrowIfCurrentThreadIsUiThread()
    {
    }
}

internal static class InferenceCollections
{
    public static IReadOnlyList<T> Freeze<T>(IEnumerable<T> source) =>
        new ReadOnlyCollection<T>(source.ToArray());
}
