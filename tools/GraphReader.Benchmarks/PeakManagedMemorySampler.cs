// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Benchmarks;

internal sealed class PeakManagedMemorySampler : IDisposable
{
    private readonly ManualResetEventSlim _stop = new(false);
    private readonly Thread _samplingThread;
    private long _peakBytes;
    private bool _disposed;

    public PeakManagedMemorySampler()
    {
        Observe();
        _samplingThread = new Thread(SampleUntilStopped)
        {
            IsBackground = true,
            Name = "GraphReader benchmark managed-memory sampler",
        };
        _samplingThread.Start();
    }

    public long PeakBytes => Interlocked.Read(ref _peakBytes);

    public void Observe()
    {
        long currentBytes = GC.GetTotalMemory(forceFullCollection: false);
        long observedPeak = Interlocked.Read(ref _peakBytes);

        while (currentBytes > observedPeak)
        {
            long priorValue = Interlocked.CompareExchange(ref _peakBytes, currentBytes, observedPeak);
            if (priorValue == observedPeak)
            {
                break;
            }

            observedPeak = priorValue;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _stop.Set();
        _samplingThread.Join();
        Observe();
        _stop.Dispose();
        _disposed = true;
    }

    private void SampleUntilStopped()
    {
        while (!_stop.Wait(TimeSpan.FromMilliseconds(1)))
        {
            Observe();
        }
    }
}
