// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;
using System.Diagnostics;

namespace GraphReader.Inference;

public sealed class FakeExecutionProviderDiscovery : IExecutionProviderDiscovery
{
    private readonly IReadOnlyList<string> _providers;

    public FakeExecutionProviderDiscovery(params string[] providers)
    {
        _providers = Array.AsReadOnly(providers.ToArray());
    }

    public IReadOnlyList<string> GetAvailableProviders() => _providers;
}

public sealed class FakeInferenceSessionFactory : IInferenceSessionFactory
{
    private readonly HashSet<InferenceProvider> _failingProviders;
    private readonly HashSet<InferenceProvider> _runFailingProviders;
    private readonly TimeSpan _runDelay;
    private readonly float _scale;
    private readonly ConcurrentBag<FakeInferenceSession> _sessions = new();

    public FakeInferenceSessionFactory(
        IEnumerable<InferenceProvider>? failingProviders = null,
        IEnumerable<InferenceProvider>? runFailingProviders = null,
        TimeSpan? runDelay = null,
        float scale = 2)
    {
        _failingProviders = failingProviders?.ToHashSet() ?? new HashSet<InferenceProvider>();
        _runFailingProviders = runFailingProviders?.ToHashSet() ?? new HashSet<InferenceProvider>();
        _runDelay = runDelay ?? TimeSpan.Zero;
        _scale = scale;
    }

    public int CreatedCount => _sessions.Count;

    public IReadOnlyList<FakeInferenceSession> Sessions => _sessions.ToArray();

    public ValueTask<IInferenceSession> CreateAsync(
        ModelIdentity model,
        InferenceProvider provider,
        CpuThreadConfiguration cpuConfiguration,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_failingProviders.Contains(provider))
        {
            throw new InvalidOperationException($"Simulated {provider} provider creation failure.");
        }

        var session = new FakeInferenceSession(
            provider,
            _runDelay,
            _scale,
            _runFailingProviders.Contains(provider));
        _sessions.Add(session);
        return ValueTask.FromResult<IInferenceSession>(session);
    }
}

public sealed class FakeInferenceSession : IInferenceSession
{
    private readonly TimeSpan _delay;
    private readonly float _scale;
    private readonly bool _failRuns;
    private int _disposed;
    private int _runCount;
    private int _running;
    private int _maximumConcurrentRuns;
    private float[] _lastInputValues = [];

    internal FakeInferenceSession(InferenceProvider provider, TimeSpan delay, float scale, bool failRuns)
    {
        Provider = provider;
        _delay = delay;
        _scale = scale;
        _failRuns = failRuns;
    }

    public InferenceProvider Provider { get; }

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public int RunCount => Volatile.Read(ref _runCount);

    public int MaximumConcurrentRuns => Volatile.Read(ref _maximumConcurrentRuns);

    public IReadOnlyList<float> LastInputValues => Array.AsReadOnly((float[])_lastInputValues.Clone());

    public async ValueTask<InferenceExecution> RunAsync(InferenceInput input, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var running = Interlocked.Increment(ref _running);
        UpdateMaximum(running);
        var stopwatch = Stopwatch.StartNew();
        try
        {
            if (_delay > TimeSpan.Zero)
            {
                await Task.Delay(_delay, cancellationToken).ConfigureAwait(false);
            }

            cancellationToken.ThrowIfCancellationRequested();
            _lastInputValues = input.Values.ToArray();
            if (_failRuns)
            {
                throw new InvalidOperationException($"Simulated {Provider} inference run failure.");
            }

            var output = input.Values.ToArray();
            for (var index = 0; index < output.Length; index++)
            {
                output[index] *= _scale;
            }

            stopwatch.Stop();
            var cold = Interlocked.Increment(ref _runCount) == 1;
            return new InferenceExecution(
                Array.AsReadOnly(output),
                Provider,
                new StageTiming(0, stopwatch.Elapsed.TotalMilliseconds, 0, stopwatch.Elapsed.TotalMilliseconds, 0, cold, false),
                new MemoryDiagnostics(0, 0, 0, 0, output.Length));
        }
        finally
        {
            Interlocked.Decrement(ref _running);
        }
    }

    public ValueTask DisposeAsync()
    {
        Interlocked.Exchange(ref _disposed, 1);
        return ValueTask.CompletedTask;
    }

    private void UpdateMaximum(int running)
    {
        int observed;
        do
        {
            observed = Volatile.Read(ref _maximumConcurrentRuns);
            if (observed >= running)
            {
                return;
            }
        }
        while (Interlocked.CompareExchange(ref _maximumConcurrentRuns, running, observed) != observed);
    }
}
