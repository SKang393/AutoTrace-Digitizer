// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Threading.Channels;

namespace GraphReader.Inference;

public sealed class BoundedInferenceScheduler : IAsyncDisposable
{
    private readonly Channel<WorkItem> _channel;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task[] _workers;
    private readonly TimeSpan _shutdownTimeout;
    private int _disposed;
    private int _queuedCount;
    private int _runningCount;
    private int _maximumObservedRunning;

    public BoundedInferenceScheduler(int capacity, int workerCount, TimeSpan? shutdownTimeout = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
        _shutdownTimeout = shutdownTimeout ?? TimeSpan.FromSeconds(2);
        if (_shutdownTimeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(shutdownTimeout));
        }
        _channel = Channel.CreateBounded<WorkItem>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = workerCount == 1,
            SingleWriter = false,
            AllowSynchronousContinuations = false
        });
        _workers = Enumerable.Range(0, workerCount)
            .Select(_ => Task.Run(ProcessAsync))
            .ToArray();
    }

    public int QueuedCount => Volatile.Read(ref _queuedCount);

    public int RunningCount => Volatile.Read(ref _runningCount);

    public int MaximumObservedRunning => Volatile.Read(ref _maximumObservedRunning);

    public async ValueTask<T> EnqueueAsync<T>(
        Func<CancellationToken, ValueTask<T>> operation,
        TimeSpan timeout,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(operation);
        if (timeout <= TimeSpan.Zero && timeout != Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout));
        }

        var timeoutSource = new CancellationTokenSource();
        if (timeout != Timeout.InfiniteTimeSpan)
        {
            timeoutSource.CancelAfter(timeout);
        }

        var item = new WorkItem(
            async token => await operation(token).ConfigureAwait(false),
            timeout,
            timeoutSource,
            cancellationToken);
        using var admission = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            timeoutSource.Token,
            _shutdown.Token);
        Interlocked.Increment(ref _queuedCount);
        try
        {
            await _channel.Writer.WriteAsync(item, admission.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (timeoutSource.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            Interlocked.Decrement(ref _queuedCount);
            item.Dispose();
            throw new TimeoutException($"Inference exceeded the {timeout} timeout while waiting for queue capacity.");
        }
        catch
        {
            Interlocked.Decrement(ref _queuedCount);
            item.Dispose();
            throw;
        }

        try
        {
            var result = await item.Completion.Task.WaitAsync(admission.Token).ConfigureAwait(false);
            return (T)result!;
        }
        catch (OperationCanceledException) when (timeoutSource.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Inference exceeded the {timeout} timeout.");
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _channel.Writer.TryComplete();
        _shutdown.Cancel();
        var workers = Task.WhenAll(_workers);
        try
        {
            await workers.WaitAsync(_shutdownTimeout).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
        }
        catch (TimeoutException)
        {
            _ = ObserveDetachedWorkersAsync(workers, _shutdown);
            return;
        }

        _shutdown.Dispose();
    }

    private async Task ProcessAsync()
    {
        try
        {
            await foreach (var item in _channel.Reader.ReadAllAsync(_shutdown.Token).ConfigureAwait(false))
            {
                Interlocked.Decrement(ref _queuedCount);
                if (item.CallerToken.IsCancellationRequested)
                {
                    item.Completion.TrySetCanceled(item.CallerToken);
                    item.Dispose();
                    continue;
                }

                if (item.TimeoutSource.IsCancellationRequested)
                {
                    item.Completion.TrySetException(new TimeoutException($"Inference exceeded the {item.Timeout} timeout."));
                    item.Dispose();
                    continue;
                }

                using var linked = CancellationTokenSource.CreateLinkedTokenSource(
                    _shutdown.Token,
                    item.CallerToken,
                    item.TimeoutSource.Token);
                var running = Interlocked.Increment(ref _runningCount);
                UpdateMaximum(running);
                Task<object?>? operationTask = null;
                try
                {
                    operationTask = item.Operation(linked.Token).AsTask();
                    var result = await operationTask.WaitAsync(linked.Token).ConfigureAwait(false);
                    item.Completion.TrySetResult(result);
                }
                catch (OperationCanceledException) when (item.TimeoutSource.IsCancellationRequested && !item.CallerToken.IsCancellationRequested)
                {
                    item.Completion.TrySetException(new TimeoutException($"Inference exceeded the {item.Timeout} timeout."));
                    await ObserveOperationAsync(operationTask).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (item.CallerToken.IsCancellationRequested)
                {
                    item.Completion.TrySetCanceled(item.CallerToken);
                    await ObserveOperationAsync(operationTask).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (_shutdown.IsCancellationRequested)
                {
                    item.Completion.TrySetCanceled(_shutdown.Token);
                    await ObserveOperationAsync(operationTask).ConfigureAwait(false);
                }
                catch (Exception exception)
                {
                    item.Completion.TrySetException(exception);
                }
                finally
                {
                    Interlocked.Decrement(ref _runningCount);
                    item.Dispose();
                }
            }
        }
        catch (OperationCanceledException) when (_shutdown.IsCancellationRequested)
        {
        }
        finally
        {
            while (_channel.Reader.TryRead(out var item))
            {
                Interlocked.Decrement(ref _queuedCount);
                item.Completion.TrySetCanceled(_shutdown.Token);
                item.Dispose();
            }
        }
    }

    private void UpdateMaximum(int running)
    {
        int observed;
        do
        {
            observed = Volatile.Read(ref _maximumObservedRunning);
            if (observed >= running)
            {
                return;
            }
        }
        while (Interlocked.CompareExchange(ref _maximumObservedRunning, running, observed) != observed);
    }

    private static async Task ObserveOperationAsync(Task<object?>? operationTask)
    {
        if (operationTask is null)
        {
            return;
        }

        try
        {
            _ = await operationTask.ConfigureAwait(false);
        }
        catch (Exception)
        {
            // The caller already received cancellation or timeout. Observation prevents an
            // unobserved exception while the worker remains accounted until native work exits.
        }
    }

    private static async Task ObserveDetachedWorkersAsync(Task workers, CancellationTokenSource shutdown)
    {
        try
        {
            await workers.ConfigureAwait(false);
        }
        catch (Exception)
        {
        }
        finally
        {
            shutdown.Dispose();
        }
    }

    private sealed class WorkItem : IDisposable
    {
        public WorkItem(
            Func<CancellationToken, ValueTask<object?>> operation,
            TimeSpan timeout,
            CancellationTokenSource timeoutSource,
            CancellationToken callerToken)
        {
            Operation = operation;
            Timeout = timeout;
            CallerToken = callerToken;
            TimeoutSource = timeoutSource;
        }

        public Func<CancellationToken, ValueTask<object?>> Operation { get; }

        public TimeSpan Timeout { get; }

        public CancellationToken CallerToken { get; }

        public CancellationTokenSource TimeoutSource { get; }

        public TaskCompletionSource<object?> Completion { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public void Dispose() => TimeoutSource.Dispose();
    }
}
