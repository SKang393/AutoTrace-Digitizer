// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows.Threading;
using GraphReader.Inference;

namespace GraphReader.App.Integration;

/// <summary>
/// Rejects inference on the actual WPF dispatcher thread without treating a
/// reusable test or worker-pool thread ID as a desktop UI thread.
/// </summary>
internal sealed class DispatcherUiThreadGuard(Dispatcher dispatcher) : IUiThreadGuard
{
    private readonly Dispatcher dispatcher =
        dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));

    public void ThrowIfCurrentThreadIsUiThread()
    {
        if (dispatcher.CheckAccess())
        {
            throw new UiThreadInferenceException(
                "Inference execution is prohibited on the WPF dispatcher thread.");
        }
    }
}
