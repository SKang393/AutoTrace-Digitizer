// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;

namespace GraphReader.Benchmarks;

internal static class BenchmarkCli
{
    private const int SuccessExitCode = 0;
    private const int GateFailureExitCode = 1;
    private const int UsageFailureExitCode = 2;

    public static async Task<int> RunAsync(string[] args, CancellationToken cancellationToken)
    {
        if (!BenchmarkOptions.TryParse(args, out BenchmarkOptions? options, out string? error))
        {
            Console.Error.WriteLine(error);
            WriteUsage(Console.Error);
            return UsageFailureExitCode;
        }

        BenchmarkOptions parsedOptions = options!;
        if (!string.Equals(parsedOptions.Suite, "public", StringComparison.Ordinal))
        {
            Console.Error.WriteLine(
                "Only '--suite public' is accepted. Private evaluation requires a separate local-only adapter and cannot be started by this executable.");
            return UsageFailureExitCode;
        }

        string repositoryRoot;
        try
        {
            repositoryRoot = RepositoryRoot.Find();
        }
        catch (InvalidOperationException exception)
        {
            Console.Error.WriteLine(exception.Message);
            return UsageFailureExitCode;
        }

        string outputDirectory = parsedOptions.OutputDirectory is null
            ? Path.Combine(repositoryRoot, "artifacts", "validation", "public-scoreboard")
            : Path.GetFullPath(parsedOptions.OutputDirectory);

        PublicScoreboardResult result;
        try
        {
            result = await PublicScoreboardRunner.RunAsync(
                repositoryRoot,
                outputDirectory,
                cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Console.Error.WriteLine("Public benchmark was cancelled.");
            return UsageFailureExitCode;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"Unable to write benchmark reports: {exception.Message}");
            return UsageFailureExitCode;
        }

        Console.WriteLine(
            string.Create(
                CultureInfo.InvariantCulture,
                $"Public scoreboard: {(result.Passed ? "PASS" : "FAIL")} ({result.PassedGateCount}/{result.TotalGateCount} gates)"));
        Console.WriteLine(
            "Evidence scope: synthetic metric-contract smoke only; not detector accuracy or end-to-end application performance.");
        Console.WriteLine(
            string.Create(
                CultureInfo.InvariantCulture,
                $"Elapsed: {result.ElapsedMilliseconds:F3} ms; peak managed memory: {result.PeakManagedMemoryBytes} bytes"));

        foreach (string reportPath in result.ReportPaths)
        {
            Console.WriteLine($"Report: {reportPath}");
        }

        foreach (PublicGateFailure failure in result.Failures)
        {
            Console.Error.WriteLine($"FAIL [{failure.Module}/{failure.CaseId}] {failure.Gate}: {failure.Detail}");
        }

        return result.Passed ? SuccessExitCode : GateFailureExitCode;
    }

    private static void WriteUsage(TextWriter writer)
    {
        writer.WriteLine(
            "Usage: dotnet run -c Release --project tools/GraphReader.Benchmarks -- --suite public [--output <directory>]");
    }
}

internal sealed record BenchmarkOptions(string Suite, string? OutputDirectory)
{
    public static bool TryParse(
        IReadOnlyList<string> args,
        out BenchmarkOptions? options,
        out string? error)
    {
        string? suite = null;
        string? outputDirectory = null;

        for (int index = 0; index < args.Count; index++)
        {
            string argument = args[index];
            if (string.Equals(argument, "--suite", StringComparison.Ordinal))
            {
                if (!TryReadValue(args, ref index, argument, out suite, out error))
                {
                    options = null;
                    return false;
                }
            }
            else if (string.Equals(argument, "--output", StringComparison.Ordinal))
            {
                if (!TryReadValue(args, ref index, argument, out outputDirectory, out error))
                {
                    options = null;
                    return false;
                }
            }
            else
            {
                options = null;
                error = $"Unknown argument '{argument}'.";
                return false;
            }
        }

        if (suite is null)
        {
            options = null;
            error = "The required '--suite public' argument is missing.";
            return false;
        }

        options = new BenchmarkOptions(suite, outputDirectory);
        error = null;
        return true;
    }

    private static bool TryReadValue(
        IReadOnlyList<string> args,
        ref int index,
        string argument,
        out string? value,
        out string? error)
    {
        if (index + 1 >= args.Count || args[index + 1].StartsWith("--", StringComparison.Ordinal))
        {
            value = null;
            error = $"Argument '{argument}' requires a value.";
            return false;
        }

        index++;
        value = args[index];
        error = null;
        return true;
    }
}

internal static class RepositoryRoot
{
    public static string Find()
    {
        foreach (string startingPath in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            DirectoryInfo? directory = new(startingPath);
            while (directory is not null)
            {
                if (File.Exists(Path.Combine(directory.FullName, "contracts", "model-manifest.schema.json")) &&
                    File.Exists(Path.Combine(directory.FullName, "GraphAutoReader.slnx")))
                {
                    return directory.FullName;
                }

                directory = directory.Parent;
            }
        }

        throw new InvalidOperationException(
            "Unable to locate the Graph Auto Reader repository root from the current directory or executable path.");
    }
}
