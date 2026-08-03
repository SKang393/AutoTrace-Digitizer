// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;
using System.Text;

namespace GraphReader.SuperResolution.Tests;

internal sealed class AdapterTestEnvironment : IDisposable
{
    private static readonly byte[] DefaultRuntimeBytes = Encoding.UTF8.GetBytes("verified test runtime");
    private static readonly byte[] DefaultModelBytes = Encoding.UTF8.GetBytes("verified test model");
    private static readonly byte[] DefaultInputBytes = Encoding.UTF8.GetBytes("immutable original image");

    public AdapterTestEnvironment()
    {
        Root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader SuperResolution Tests",
            Guid.NewGuid().ToString("N"),
            "spaces & meta (한글);^");
        RuntimeDirectory = Path.Combine(Root, "runtime & tools");
        ModelsDirectory = Path.Combine(Root, "models & weights");
        CacheDirectory = Path.Combine(Root, "cache & work");
        OutputDirectory = Path.Combine(Root, "review outputs");
        Directory.CreateDirectory(RuntimeDirectory);
        Directory.CreateDirectory(ModelsDirectory);
        Directory.CreateDirectory(CacheDirectory);
        Directory.CreateDirectory(OutputDirectory);

        ExecutablePath = Path.Combine(RuntimeDirectory, "realesrgan runner.exe");
        InputPath = Path.Combine(Root, "input & phase (한글);^.png");
        ModelParameterRelativePath = "realesr-animevideov3-x2.param";
        ModelWeightsRelativePath = "realesr-animevideov3-x2.bin";
        ModelParameterPath = Path.Combine(ModelsDirectory, ModelParameterRelativePath);
        ModelPath = Path.Combine(ModelsDirectory, ModelWeightsRelativePath);
        File.WriteAllBytes(ExecutablePath, DefaultRuntimeBytes);
        File.WriteAllBytes(InputPath, DefaultInputBytes);
        File.WriteAllBytes(ModelParameterPath, DefaultModelBytes);
        File.WriteAllBytes(ModelPath, DefaultModelBytes);

        Model = CreateModel(DefaultModelBytes);
        Inspector = new FakeImageInspector { Dimensions = ExpectedEnhancedDimensions };
        Runner = new FakeProcessRunner();
        Runner.Handler = CompleteSuccessfullyAsync;
    }

    public string Root { get; }
    public string RuntimeDirectory { get; }
    public string ModelsDirectory { get; }
    public string CacheDirectory { get; }
    public string OutputDirectory { get; }
    public string ExecutablePath { get; }
    public string InputPath { get; }
    public string ModelParameterRelativePath { get; }
    public string ModelWeightsRelativePath { get; }
    public string ModelParameterPath { get; }
    public string ModelPath { get; }
    public PixelDimensions SourceDimensions { get; } = new(31, 19);
    public PixelDimensions ExpectedEnhancedDimensions =>
        new(SourceDimensions.Width * 2, SourceDimensions.Height * 2);
    public byte[] OutputBytes { get; set; } = Encoding.UTF8.GetBytes("verified enhanced derivative");
    public EnhancementModel Model { get; private set; }
    public FakeImageInspector Inspector { get; }
    public FakeProcessRunner Runner { get; }

    public RealEsrganAdapter CreateAdapter(
        string? expectedRuntimeSha256 = null,
        int maxDiagnosticCharacters = 32_768) =>
        new(
            new RealEsrganConfiguration(
                ExecutablePath,
                ModelsDirectory,
                CacheDirectory,
                expectedRuntimeSha256,
                TimeSpan.FromSeconds(17),
                maxDiagnosticCharacters),
            Runner,
            Inspector);

    public EnhancementRequest CreateRequest(
        string outputName = "enhanced.png",
        EnhancementOptions? options = null,
        string? inputPath = null,
        EnhancementModel? model = null) =>
        new(
            Guid.Parse("110e8400-e29b-41d4-a716-446655440000"),
            Guid.Parse("220e8400-e29b-41d4-a716-446655440000"),
            inputPath ?? InputPath,
            Path.Combine(OutputDirectory, outputName),
            SourceDimensions,
            model ?? Model,
            options);

    public async Task<ProcessExecutionResult> CompleteSuccessfullyAsync(
        ProcessInvocation invocation,
        CancellationToken cancellationToken)
    {
        string outputPath = FakeProcessRunner.ArgumentValue(invocation, "-o");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        await File.WriteAllBytesAsync(outputPath, OutputBytes, cancellationToken).ConfigureAwait(false);
        return new ProcessExecutionResult(
            ProcessCompletion.Completed,
            0,
            "fake stdout",
            "fake stderr",
            TimeSpan.FromMilliseconds(12));
    }

    public void ReplaceModelArtifact(byte[] bytes)
    {
        File.WriteAllBytes(ModelParameterPath, bytes);
        File.WriteAllBytes(ModelPath, bytes);
        Model = CreateModel(bytes);
    }

    public bool WorkDirectoryIsEmpty()
    {
        string workRoot = Path.Combine(CacheDirectory, "work");
        return !Directory.Exists(workRoot) || !Directory.EnumerateFileSystemEntries(workRoot).Any();
    }

    public void Dispose()
    {
        if (Directory.Exists(Root))
        {
            Directory.Delete(Root, recursive: true);
        }
    }

    private EnhancementModel CreateModel(byte[] artifactBytes)
    {
        string artifactSha256 = Convert.ToHexStringLower(
            System.Security.Cryptography.SHA256.HashData(artifactBytes));
        return new EnhancementModel(
            "realesr-animevideov3",
            "v0.2.5.0-ncnn-x2",
            artifactSha256,
            "official test source",
            "test revision",
            "BSD-3-Clause",
            "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
            [
                new ModelArtifact(ModelParameterRelativePath, artifactSha256),
                new ModelArtifact(ModelWeightsRelativePath, artifactSha256)
            ]);
    }
}

internal sealed class FakeProcessRunner : IProcessRunner
{
    private readonly ConcurrentQueue<ProcessInvocation> _invocations = new();
    private readonly ConcurrentQueue<byte[]> _inputSnapshots = new();

    public Func<ProcessInvocation, CancellationToken, Task<ProcessExecutionResult>> Handler { get; set; } =
        static (_, _) => Task.FromResult(
            new ProcessExecutionResult(
                ProcessCompletion.Completed,
                0,
                string.Empty,
                string.Empty,
                TimeSpan.Zero));

    public IReadOnlyList<ProcessInvocation> Invocations => _invocations.ToArray();

    public IReadOnlyList<byte[]> InputSnapshots => _inputSnapshots.ToArray();

    public int InvocationCount => _invocations.Count;

    public Task<ProcessExecutionResult> RunAsync(
        ProcessInvocation invocation,
        CancellationToken cancellationToken)
    {
        _invocations.Enqueue(invocation);
        string inputPath = ArgumentValue(invocation, "-i");
        if (File.Exists(inputPath))
        {
            _inputSnapshots.Enqueue(File.ReadAllBytes(inputPath));
        }

        return Handler(invocation, cancellationToken);
    }

    public static string ArgumentValue(ProcessInvocation invocation, string option)
    {
        int index = invocation.Arguments.ToList().IndexOf(option);
        if (index < 0 || index + 1 >= invocation.Arguments.Count)
        {
            throw new InvalidOperationException($"Invocation is missing required option '{option}'.");
        }

        return invocation.Arguments[index + 1];
    }
}

internal sealed class FakeImageInspector : IOutputImageInspector
{
    public PixelDimensions Dimensions { get; set; }
    public Exception? ExceptionToThrow { get; set; }
    public Action<string>? OnRead { get; set; }

    public PixelDimensions ReadDimensions(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The fake inspector requires an existing output.", path);
        }

        if (ExceptionToThrow is not null)
        {
            throw ExceptionToThrow;
        }

        OnRead?.Invoke(path);
        return Dimensions;
    }
}
