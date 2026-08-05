// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Windows.Threading;
using GraphReader.App.Integration;
using GraphReader.Domain;
using GraphReader.Inference;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ApplicationCompositionLifecycleTests
{
    [TestMethod]
    public void DispatcherGuardRejectsOnlyTheActualWpfDispatcherThread()
    {
        var guard = new DispatcherUiThreadGuard(Dispatcher.CurrentDispatcher);

        Assert.ThrowsExactly<UiThreadInferenceException>(
            guard.ThrowIfCurrentThreadIsUiThread);
        Task.Run(guard.ThrowIfCurrentThreadIsUiThread).GetAwaiter().GetResult();
    }

    [TestMethod]
    public async Task CanceledCompositionDisposesInitializedOwnedInferenceRuntime()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.ApplicationComposition.Lifecycle",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        ProductionInferenceRuntimeHost? host = null;
        try
        {
            DomainResult<ProductionInferenceRuntimeHost> created =
                ProductionInferenceRuntimeFactory.Create(
                    new TestApplicationPaths(root),
                    CapturedUiThreadGuard.CaptureCurrentThread());
            host = created.Value ?? throw new AssertFailedException(
                string.Join(" | ", created.Errors.Select(static error => error.TechnicalMessage)));
            _ = host.Runtime;
            Assert.IsTrue(host.IsInitialized);

            using var cancellation = new CancellationTokenSource();
            cancellation.Cancel();
            await Assert.ThrowsExactlyAsync<TaskCanceledException>(() =>
                ApplicationComposition.CompleteWithOwnedInferenceAsync(
                    host,
                    () => Task.FromCanceled<ApplicationCompositionResult>(cancellation.Token)));

            Assert.IsTrue(host.IsDisposed);
            Assert.ThrowsExactly<ObjectDisposedException>(() => _ = host.Runtime);
        }
        finally
        {
            if (host is not null)
            {
                await host.DisposeAsync();
            }

            Directory.Delete(root, recursive: true);
        }
    }

    private sealed class TestApplicationPaths(string root) : IApplicationPaths
    {
        public DistributionMode Mode => DistributionMode.Portable;

        public string SettingsRoot { get; } = Path.Combine(root, "Data", "Settings");

        public string AutosaveRoot { get; } = Path.Combine(root, "Data", "Autosave");

        public string CacheRoot { get; } = Path.Combine(root, "Data", "Cache");

        public string LogsRoot { get; } = Path.Combine(root, "Data", "Logs");

        public string RecoveryRoot { get; } = Path.Combine(root, "Data", "Recovery");

        public string ModelRoot { get; } = Path.Combine(root, "Data", "Models");
    }
}
