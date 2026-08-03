// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Runtime.ExceptionServices;

namespace GraphReader.App.Tests;

internal static class StaTestHost
{
    public static void Run(Action action)
    {
        ArgumentNullException.ThrowIfNull(action);
        Run(
            () =>
            {
                action();
                return true;
            });
    }

    public static T Run<T>(Func<T> action)
    {
        ArgumentNullException.ThrowIfNull(action);

        T? result = default;
        ExceptionDispatchInfo? failure = null;
        using var completed = new ManualResetEventSlim();
        var thread = new Thread(
            () =>
            {
                try
                {
                    result = action();
                }
                catch (Exception exception)
                {
                    failure = ExceptionDispatchInfo.Capture(exception);
                }
                finally
                {
                    completed.Set();
                }
            });
        thread.SetApartmentState(ApartmentState.STA);
        thread.IsBackground = true;
        thread.Start();

        Assert.IsTrue(completed.Wait(TimeSpan.FromSeconds(20)), "The STA test operation timed out.");
        failure?.Throw();
        return result!;
    }
}
