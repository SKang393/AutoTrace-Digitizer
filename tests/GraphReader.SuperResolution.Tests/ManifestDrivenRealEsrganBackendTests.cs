// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class ManifestDrivenRealEsrganBackendTests
{
    [TestMethod]
    public async Task LocalEvaluationResolvesManifestInvokesExactModelAndPreservesOriginal()
    {
        using var environment = new ManifestBackendTestEnvironment();
        environment.Runner.Handler = async (invocation, cancellationToken) =>
        {
            await File.WriteAllBytesAsync(
                FakeProcessRunner.ArgumentValue(invocation, "-o"),
                environment.OutputBytes,
                cancellationToken);
            return new ProcessExecutionResult(
                ProcessCompletion.Completed,
                0,
                "ok",
                string.Empty,
                TimeSpan.Zero);
        };

        RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            environment.ManifestPath,
            environment.RuntimeRoot,
            environment.CacheRoot,
            RealEsrganBackendPurpose.LocalEvaluation,
            CancellationToken.None,
            environment.Runner,
            environment.Inspector);

        Assert.IsTrue(resolution.IsAvailable);
        Assert.AreEqual(RealEsrganBackendAvailability.AvailableForLocalEvaluationOnly, resolution.Availability);
        Assert.IsFalse(resolution.ReleaseEligible);
        Assert.IsNotNull(resolution.Service);
        Assert.IsNotNull(resolution.Model);
        Assert.AreEqual("official-runtime-name", resolution.Model.RuntimeModelName);
        Assert.AreEqual("ENHANCEMENT_LOCAL_EVALUATION_ONLY", resolution.Diagnostic?.Code);

        string sourceBefore = await EnhancementHashing.ComputeFileSha256Async(
            environment.SourcePath,
            CancellationToken.None);
        EnhancementResult result = await resolution.Service.EnhanceAsync(
            new EnhancementRequest(
                Guid.NewGuid(),
                Guid.NewGuid(),
                environment.SourcePath,
                environment.OutputPath,
                environment.SourceDimensions,
                resolution.Model,
                new EnhancementOptions(Scale: 2)),
            CancellationToken.None);
        string sourceAfter = await EnhancementHashing.ComputeFileSha256Async(
            environment.SourcePath,
            CancellationToken.None);

        Assert.IsTrue(result.IsSuccess, result.Diagnostic.Message);
        Assert.AreEqual(sourceBefore, sourceAfter);
        Assert.AreEqual(1, environment.Runner.InvocationCount);
        ProcessInvocation invocation = environment.Runner.Invocations.Single();
        Assert.AreEqual("official-runtime-name", FakeProcessRunner.ArgumentValue(invocation, "-n"));
        Assert.AreEqual("2", FakeProcessRunner.ArgumentValue(invocation, "-s"));
        Assert.AreEqual(environment.ModelsRoot, FakeProcessRunner.ArgumentValue(invocation, "-m"));
    }

    [TestMethod]
    public async Task DistributionPurposeFailsClosedOnUnapprovedRuntimeAndBenchmark()
    {
        using var environment = new ManifestBackendTestEnvironment();

        RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            environment.ManifestPath,
            environment.RuntimeRoot,
            environment.CacheRoot,
            RealEsrganBackendPurpose.Distribution,
            CancellationToken.None,
            environment.Runner,
            environment.Inspector);

        Assert.AreEqual(RealEsrganBackendAvailability.RedistributionBlocked, resolution.Availability);
        Assert.IsFalse(resolution.IsAvailable);
        Assert.IsFalse(resolution.ReleaseEligible);
        Assert.IsNull(resolution.Service);
        Assert.IsNotNull(resolution.Model);
        Assert.AreEqual("ENHANCEMENT_REDISTRIBUTION_BLOCKED", resolution.Diagnostic?.Code);
        StringAssert.Contains(resolution.Diagnostic?.TechnicalMessage, "Production benchmark approval is false");
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task LocalEvaluationFailsClosedWhenScientificFidelityApprovalFailed()
    {
        using var environment = new ManifestBackendTestEnvironment(localAdapterApproved: false);

        RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            environment.ManifestPath,
            environment.RuntimeRoot,
            environment.CacheRoot,
            RealEsrganBackendPurpose.LocalEvaluation,
            CancellationToken.None,
            environment.Runner,
            environment.Inspector);

        Assert.AreEqual(RealEsrganBackendAvailability.ModelIncompatible, resolution.Availability);
        Assert.IsFalse(resolution.IsAvailable);
        Assert.IsNull(resolution.Service);
        Assert.AreEqual("MODEL_RUNTIME_INCOMPATIBLE", resolution.Diagnostic?.Code);
        StringAssert.Contains(resolution.Diagnostic?.TechnicalMessage, "scientific content");
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task EmptyOutputContractReturnsConfigurationInvalidInsteadOfThrowing()
    {
        using var environment = new ManifestBackendTestEnvironment();
        string manifest = await File.ReadAllTextAsync(environment.ManifestPath);
        int outputStart = manifest.IndexOf("\"outputs\": [{", StringComparison.Ordinal);
        int preprocessingStart = manifest.IndexOf("\"preprocessing\":", outputStart, StringComparison.Ordinal);
        Assert.IsGreaterThanOrEqualTo(0, outputStart);
        Assert.IsGreaterThan(outputStart, preprocessingStart);
        string malformed = string.Concat(
            manifest.AsSpan(0, outputStart),
            "\"outputs\": [],\n          ",
            manifest.AsSpan(preprocessingStart));
        await File.WriteAllTextAsync(environment.ManifestPath, malformed);

        RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            environment.ManifestPath,
            environment.RuntimeRoot,
            environment.CacheRoot,
            RealEsrganBackendPurpose.LocalEvaluation,
            CancellationToken.None,
            environment.Runner,
            environment.Inspector);

        Assert.AreEqual(RealEsrganBackendAvailability.ManifestInvalid, resolution.Availability);
        Assert.IsFalse(resolution.IsAvailable);
        Assert.IsNull(resolution.Service);
        Assert.AreEqual("MODEL_MANIFEST_INVALID", resolution.Diagnostic?.Code);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task TrackedSecondaryManifestFailsClosedBeforeRuntimeDiscovery()
    {
        string manifestPath = RepositoryPaths.FromRoot(
            "models",
            "manifest",
            "super-resolution",
            "RealESRGAN_x4plus_anime_6B-ncnn-outscale2.json");

        RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            manifestPath,
            Path.Combine(Path.GetTempPath(), $"missing-realesrgan-{Guid.NewGuid():N}"),
            Path.Combine(Path.GetTempPath(), $"unused-realesrgan-cache-{Guid.NewGuid():N}"),
            RealEsrganBackendPurpose.LocalEvaluation,
            CancellationToken.None);

        Assert.AreEqual(RealEsrganBackendAvailability.ModelIncompatible, resolution.Availability);
        Assert.IsNull(resolution.Service);
        Assert.AreEqual("MODEL_RUNTIME_INCOMPATIBLE", resolution.Diagnostic?.Code);
        StringAssert.Contains(resolution.Diagnostic?.TechnicalMessage, "cropped and zoomed");
    }

    [TestMethod]
    [DataRow("runtime_missing", RealEsrganBackendAvailability.RuntimeMissing, "RUNTIME_NOT_FOUND")]
    [DataRow("runtime_dependency_missing", RealEsrganBackendAvailability.RuntimeMissing, "RUNTIME_NOT_FOUND")]
    [DataRow("runtime_checksum", RealEsrganBackendAvailability.RuntimeChecksumMismatch, "RUNTIME_CHECKSUM_MISMATCH")]
    [DataRow("runtime_dependency_checksum", RealEsrganBackendAvailability.RuntimeChecksumMismatch, "RUNTIME_CHECKSUM_MISMATCH")]
    [DataRow("model_missing", RealEsrganBackendAvailability.ModelMissing, "MODEL_NOT_FOUND")]
    [DataRow("model_checksum", RealEsrganBackendAvailability.ModelChecksumMismatch, "MODEL_CHECKSUM_MISMATCH")]
    public async Task AvailabilityProbeReportsExactFailClosedUnavailableState(
        string scenario,
        RealEsrganBackendAvailability expectedAvailability,
        string expectedCode)
    {
        using var environment = new ManifestBackendTestEnvironment();
        switch (scenario)
        {
            case "runtime_missing":
                File.Delete(environment.ExecutablePath);
                break;
            case "runtime_checksum":
                await File.WriteAllTextAsync(environment.ExecutablePath, "changed runtime");
                break;
            case "runtime_dependency_missing":
                File.Delete(environment.RuntimeDependencyPath);
                break;
            case "runtime_dependency_checksum":
                await File.WriteAllTextAsync(environment.RuntimeDependencyPath, "changed dependency");
                break;
            case "model_missing":
                File.Delete(environment.ModelBinPath);
                break;
            case "model_checksum":
                await File.WriteAllTextAsync(environment.ModelBinPath, "changed model");
                break;
            default:
                Assert.Fail($"Unexpected scenario '{scenario}'.");
                break;
        }

        RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            environment.ManifestPath,
            environment.RuntimeRoot,
            environment.CacheRoot,
            RealEsrganBackendPurpose.LocalEvaluation,
            CancellationToken.None,
            environment.Runner,
            environment.Inspector);

        Assert.AreEqual(expectedAvailability, resolution.Availability);
        Assert.AreEqual(expectedCode, resolution.Diagnostic?.Code);
        Assert.IsFalse(resolution.IsAvailable);
        Assert.IsNull(resolution.Service);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task AuthorizedOfficialLocalRuntimeRunsPrimaryAndRejectsSecondary()
    {
        string? runtimeRoot = Environment.GetEnvironmentVariable("GRAPHREADER_REALESRGAN_RUNTIME_ROOT");
        string? datasetRoot = Environment.GetEnvironmentVariable("GRAPHREADER_REALESRGAN_DATASET_ROOT");
        if (string.IsNullOrWhiteSpace(runtimeRoot) || string.IsNullOrWhiteSpace(datasetRoot))
        {
            Assert.Inconclusive(
                "Set GRAPHREADER_REALESRGAN_RUNTIME_ROOT and GRAPHREADER_REALESRGAN_DATASET_ROOT to run the authorized local runtime test.");
        }

        string sourcePath = Path.Combine(
            datasetRoot,
            "images",
            "a1b41e74-1808-5dec-99c9-59f4c88f4004.png");
        var inspector = new PngOutputImageInspector();
        PixelDimensions sourceDimensions = inspector.ReadDimensions(sourcePath);
        string sourceBefore = await EnhancementHashing.ComputeFileSha256Async(sourcePath, CancellationToken.None);
        string testRoot = Path.Combine(Path.GetTempPath(), $"graphreader-realesrgan-{Guid.NewGuid():N}");
        Directory.CreateDirectory(testRoot);
        try
        {
            string primaryManifest = RepositoryPaths.FromRoot(
                "models",
                "manifest",
                "super-resolution",
                ManifestDrivenRealEsrganBackend.DefaultManifestFileName);
            string modelRoot = Path.Combine(testRoot, "primary");
            RealEsrganBackendResolution resolution = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
                primaryManifest,
                runtimeRoot,
                Path.Combine(modelRoot, "cache"),
                RealEsrganBackendPurpose.LocalEvaluation,
                CancellationToken.None);
            Assert.IsTrue(resolution.IsAvailable, resolution.Diagnostic?.TechnicalMessage);
            Assert.IsNotNull(resolution.Service);
            Assert.IsNotNull(resolution.Model);

            string outputPath = Path.Combine(modelRoot, "enhanced.png");
            Directory.CreateDirectory(modelRoot);
            EnhancementResult result = await resolution.Service.EnhanceAsync(
                new EnhancementRequest(
                    Guid.NewGuid(),
                    Guid.NewGuid(),
                    sourcePath,
                    outputPath,
                    sourceDimensions,
                    resolution.Model,
                    new EnhancementOptions(Scale: 2, Timeout: TimeSpan.FromMinutes(2))),
                CancellationToken.None);

            Assert.IsTrue(result.IsSuccess, result.Diagnostic.Message);
            Assert.AreEqual(
                new PixelDimensions(sourceDimensions.Width * 2, sourceDimensions.Height * 2),
                inspector.ReadDimensions(outputPath));
            Console.WriteLine(
                $"{resolution.Model.ModelId}: status={result.Status}; total_ms={result.Envelope?.TimingMs.Total:F4}; inference_ms={result.Envelope?.TimingMs.Inference:F4}; output_sha256={result.Envelope?.Payload.OutputSha256}");

            string secondaryManifest = RepositoryPaths.FromRoot(
                "models",
                "manifest",
                "super-resolution",
                "RealESRGAN_x4plus_anime_6B-ncnn-outscale2.json");
            RealEsrganBackendResolution secondary = await ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
                secondaryManifest,
                runtimeRoot,
                Path.Combine(testRoot, "secondary-cache"),
                RealEsrganBackendPurpose.LocalEvaluation,
                CancellationToken.None);
            Assert.AreEqual(RealEsrganBackendAvailability.ModelIncompatible, secondary.Availability);
            Assert.IsNull(secondary.Service);
            Assert.AreEqual("MODEL_RUNTIME_INCOMPATIBLE", secondary.Diagnostic?.Code);

            string sourceAfter = await EnhancementHashing.ComputeFileSha256Async(sourcePath, CancellationToken.None);
            Assert.AreEqual(sourceBefore, sourceAfter);
        }
        finally
        {
            Directory.Delete(testRoot, recursive: true);
        }
    }
}

internal sealed class ManifestBackendTestEnvironment : IDisposable
{
    private static readonly byte[] RuntimeBytes = Encoding.UTF8.GetBytes("official test runtime");
    private static readonly byte[] RuntimeDependencyBytes = Encoding.UTF8.GetBytes("official test dependency");
    private static readonly byte[] ParamBytes = Encoding.UTF8.GetBytes("official test parameter");
    private static readonly byte[] ModelBytes = Encoding.UTF8.GetBytes("official test weights");

    public ManifestBackendTestEnvironment(bool localAdapterApproved = true)
    {
        Root = Path.Combine(Path.GetTempPath(), $"graphreader-manifest-backend-{Guid.NewGuid():N}");
        RepositoryRoot = Path.Combine(Root, "distribution");
        RuntimeRoot = Path.Combine(Root, "runtime");
        ModelsRoot = Path.Combine(RuntimeRoot, "models");
        CacheRoot = Path.Combine(Root, "cache");
        SourcePath = Path.Combine(Root, "immutable-source.png");
        OutputPath = Path.Combine(Root, "output", "enhanced.png");
        ExecutablePath = Path.Combine(RuntimeRoot, "realesrgan-ncnn-vulkan.exe");
        RuntimeDependencyPath = Path.Combine(RuntimeRoot, "vcomp140.dll");
        ModelParamPath = Path.Combine(ModelsRoot, "official-runtime-name.param");
        ModelBinPath = Path.Combine(ModelsRoot, "official-runtime-name.bin");
        ManifestPath = Path.Combine(
            RepositoryRoot,
            "models",
            "manifest",
            "super-resolution",
            "official-model.json");
        string noticePath = Path.Combine(RepositoryRoot, "LICENSES", "Real-ESRGAN-BSD-3-Clause.txt");

        Directory.CreateDirectory(Path.GetDirectoryName(ManifestPath)!);
        Directory.CreateDirectory(Path.GetDirectoryName(noticePath)!);
        Directory.CreateDirectory(ModelsRoot);
        Directory.CreateDirectory(CacheRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(OutputPath)!);
        File.WriteAllBytes(ExecutablePath, RuntimeBytes);
        File.WriteAllBytes(RuntimeDependencyPath, RuntimeDependencyBytes);
        File.WriteAllBytes(ModelParamPath, ParamBytes);
        File.WriteAllBytes(ModelBinPath, ModelBytes);
        File.WriteAllText(SourcePath, "immutable original image");
        File.WriteAllText(noticePath, "BSD-3-Clause test notice");
        File.WriteAllText(ManifestPath, CreateManifest(localAdapterApproved));

        Inspector = new FakeImageInspector
        {
            Dimensions = new PixelDimensions(SourceDimensions.Width * 2, SourceDimensions.Height * 2)
        };
    }

    public string Root { get; }
    public string RepositoryRoot { get; }
    public string RuntimeRoot { get; }
    public string ModelsRoot { get; }
    public string CacheRoot { get; }
    public string SourcePath { get; }
    public string OutputPath { get; }
    public string ExecutablePath { get; }
    public string RuntimeDependencyPath { get; }
    public string ModelParamPath { get; }
    public string ModelBinPath { get; }
    public string ManifestPath { get; }
    public PixelDimensions SourceDimensions { get; } = new(31, 19);
    public byte[] OutputBytes { get; } = Encoding.UTF8.GetBytes("verified derived output");
    public FakeProcessRunner Runner { get; } = new();
    public FakeImageInspector Inspector { get; }

    public void Dispose()
    {
        if (Directory.Exists(Root))
        {
            Directory.Delete(Root, recursive: true);
        }
    }

    private static string CreateManifest(bool localAdapterApproved)
    {
        string runtimeSha256 = Convert.ToHexStringLower(
            System.Security.Cryptography.SHA256.HashData(RuntimeBytes));
        string paramSha256 = Convert.ToHexStringLower(
            System.Security.Cryptography.SHA256.HashData(ParamBytes));
        string modelSha256 = Convert.ToHexStringLower(
            System.Security.Cryptography.SHA256.HashData(ModelBytes));
        string runtimeDependencySha256 = Convert.ToHexStringLower(
            System.Security.Cryptography.SHA256.HashData(RuntimeDependencyBytes));
        string localAdapterApproval = localAdapterApproved ? "true" : "false";
        string localAdapterBlocker = localAdapterApproved
            ? string.Empty
            : "\"local_adapter_blocker\": \"The runtime lost scientific content.\",";
        return $$"""
        {
          "manifest_version": 1,
          "model_id": "official-model-id",
          "model_version": "1.0.0",
          "task": "super_resolution",
          "source": {
            "name": "official source",
            "url": "https://example.invalid/official.zip",
            "revision": "revision-1"
          },
          "license": {
            "spdx": "BSD-3-Clause",
            "notice_path": "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
            "reviewed": true
          },
          "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
          "files": [
            "models/official-runtime-name.param",
            "models/official-runtime-name.bin"
          ],
          "inputs": [{}],
          "outputs": [{
            "configured_output_scale": 2,
            "coordinate_space": "enhanced_pixels"
          }],
          "preprocessing": {
            "runtime": "realesrgan-ncnn-vulkan",
            "runtime_model_name": "official-runtime-name",
            "local_adapter_approval": {{localAdapterApproval}},
            {{localAdapterBlocker}}
            "runtime_scale_argument": 2,
            "runtime_executable_sha256": "{{runtimeSha256}}",
            "runtime_files_sha256": {
              "realesrgan-ncnn-vulkan.exe": "{{runtimeSha256}}",
              "vcomp140.dll": "{{runtimeDependencySha256}}"
            },
            "runtime_redistribution": {
              "approved": false,
              "blocker": "Runtime redistribution is not approved."
            },
            "model_payload_sha256": {
              "models/official-runtime-name.param": "{{paramSha256}}",
              "models/official-runtime-name.bin": "{{modelSha256}}"
            }
          },
          "commercial_use": true,
          "redistribution": true,
          "providers": ["vulkan"],
          "benchmarks": [{
            "production_approval": false
          }]
        }
        """;
    }
}
