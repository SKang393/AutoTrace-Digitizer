// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;
using System.Text.Json;
using Json.Schema;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class RealEsrganAdapterTests
{
    private static readonly string[] ExpectedOptions =
        ["-i", "-o", "-n", "-s", "-t", "-m", "-g", "-f"];
    private static readonly string[] ExpectedTransformProperties =
    [
        "transform_id", "kind", "source_space", "target_space", "matrix_3x3",
        "inverse_matrix_3x3", "parameters", "lossy"
    ];

    [TestMethod]
    public async Task SuccessUsesArgumentListPreservesSourceAndReturnsProvenance()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter(maxDiagnosticCharacters: 4096);
        var options = new EnhancementOptions(TileSize: 64, GpuIndex: 3);
        EnhancementRequest request = environment.CreateRequest("safe enhanced.png", options);
        string sourceHashBefore = await EnhancementHashing.ComputeFileSha256Async(
            environment.InputPath,
            CancellationToken.None);

        EnhancementResult result = await adapter.EnhanceAsync(request, CancellationToken.None);

        Assert.AreEqual(EnhancementStatus.Succeeded, result.Status);
        Assert.IsTrue(result.IsSuccess);
        Assert.AreEqual(request.OutputPath, result.OutputPath);
        Assert.IsTrue(File.Exists(request.OutputPath));
        CollectionAssert.AreEqual(environment.OutputBytes, File.ReadAllBytes(request.OutputPath));
        Assert.AreEqual(
            sourceHashBefore,
            await EnhancementHashing.ComputeFileSha256Async(environment.InputPath, CancellationToken.None));

        Assert.AreEqual(1, environment.Runner.InvocationCount);
        ProcessInvocation invocation = environment.Runner.Invocations[0];
        Assert.AreEqual(environment.ExecutablePath, invocation.FileName);
        Assert.AreEqual(environment.RuntimeDirectory, invocation.WorkingDirectory);
        Assert.AreEqual(TimeSpan.FromSeconds(17), invocation.Timeout);
        Assert.AreEqual(4096, invocation.MaxDiagnosticCharacters);
        CollectionAssert.AreEqual(ExpectedOptions, invocation.Arguments.Where(IsOption).ToArray());
        string stagedInput = FakeProcessRunner.ArgumentValue(invocation, "-i");
        Assert.AreNotEqual(environment.InputPath, stagedInput);
        Assert.IsFalse(File.Exists(stagedInput), "The private process input should be removed after use.");
        CollectionAssert.AreEqual(File.ReadAllBytes(environment.InputPath), environment.Runner.InputSnapshots[0]);
        Assert.AreEqual(environment.ModelsDirectory, FakeProcessRunner.ArgumentValue(invocation, "-m"));
        Assert.AreEqual(environment.Model.ModelId, FakeProcessRunner.ArgumentValue(invocation, "-n"));
        Assert.AreEqual("2", FakeProcessRunner.ArgumentValue(invocation, "-s"));
        Assert.AreEqual("64", FakeProcessRunner.ArgumentValue(invocation, "-t"));
        Assert.AreEqual("3", FakeProcessRunner.ArgumentValue(invocation, "-g"));
        Assert.AreEqual("png", FakeProcessRunner.ArgumentValue(invocation, "-f"));

        string stagedOutput = FakeProcessRunner.ArgumentValue(invocation, "-o");
        Assert.AreNotEqual(request.OutputPath, stagedOutput);
        Assert.AreNotEqual(environment.InputPath, stagedOutput);
        Assert.IsFalse(File.Exists(stagedOutput), "The isolated work file should be removed after promotion.");
        StringAssert.Contains(environment.InputPath, " & ");
        StringAssert.Contains(environment.InputPath, "한글");
        Assert.IsTrue(environment.WorkDirectoryIsEmpty());

        EnhancementEnvelope envelope = AssertEnvelope(result);
        Assert.AreEqual(sourceHashBefore, envelope.InputSha256);
        Assert.AreEqual(
            await EnhancementHashing.ComputeFileSha256Async(request.OutputPath, CancellationToken.None),
            envelope.Payload.OutputSha256);
        Assert.AreEqual(environment.SourceDimensions, envelope.Payload.OriginalDimensions);
        Assert.AreEqual(environment.ExpectedEnhancedDimensions, envelope.Payload.EnhancedDimensions);
        Assert.AreEqual(64, envelope.Payload.TileSize);
        Assert.AreEqual(3, envelope.Payload.GpuIndex);
        Assert.IsFalse(envelope.Payload.CacheHit);
        Assert.AreEqual("vulkan", envelope.Model.Provider);
        Assert.AreEqual(environment.Model.Sha256, envelope.Model.Sha256);
        Assert.AreEqual(
            EnhancementHashing.ComputeModelSha256(
                environment.Model.Artifacts.Select(static artifact => (artifact.RelativePath, artifact.Sha256))),
            envelope.Payload.ModelProvenance.VerifiedArtifactSetSha256);
        Assert.IsTrue(envelope.Warnings.Any(static warning => warning.Contains("derivative evidence", StringComparison.Ordinal)));

        string envelopeJson = JsonSerializer.Serialize(envelope);
        using JsonDocument envelopeDocument = JsonDocument.Parse(envelopeJson);
        JsonSchema schema = JsonSchema.FromText(File.ReadAllText(
            RepositoryPaths.FromRoot("contracts", "vision-result.schema.json")));
        Assert.IsTrue(schema.Evaluate(envelopeDocument.RootElement).IsValid, envelopeJson);
        JsonElement transform = envelopeDocument.RootElement
            .GetProperty("payload")
            .GetProperty("transform");
        CollectionAssert.AreEquivalent(
            ExpectedTransformProperties,
            transform.EnumerateObject().Select(static property => property.Name).ToArray());
    }

    [TestMethod]
    public async Task SameInputAndOutputIsRejectedWithoutChangingOriginal()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        EnhancementRequest request = environment.CreateRequest() with
        {
            OutputPath = environment.InputPath
        };
        byte[] original = File.ReadAllBytes(environment.InputPath);

        EnhancementResult result = await adapter.EnhanceAsync(request, CancellationToken.None);

        AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.InvalidRequest, true);
        CollectionAssert.AreEqual(original, File.ReadAllBytes(environment.InputPath));
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task ExistingOutputIsNeverOverwritten()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        EnhancementRequest request = environment.CreateRequest();
        byte[] sentinel = Encoding.UTF8.GetBytes("reviewed output owned by caller");
        File.WriteAllBytes(request.OutputPath, sentinel);

        EnhancementResult result = await adapter.EnhanceAsync(request, CancellationToken.None);

        AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.OutputAlreadyExists, true);
        CollectionAssert.AreEqual(sentinel, File.ReadAllBytes(request.OutputPath));
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task MissingRuntimeAndModelReturnStructuredUnenhancedContinuation()
    {
        using (var environment = new AdapterTestEnvironment())
        {
            File.Delete(environment.ExecutablePath);
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                environment.CreateRequest(),
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.RuntimeMissing,
                true);
        }

        using (var environment = new AdapterTestEnvironment())
        {
            File.Delete(environment.ModelPath);
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                environment.CreateRequest(),
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.ModelMissing,
                true);
        }
    }

    [TestMethod]
    public async Task DisabledContinuationConvertsRecoverableRuntimeFailureToFailure()
    {
        using var environment = new AdapterTestEnvironment();
        File.Delete(environment.ExecutablePath);
        EnhancementRequest request = environment.CreateRequest(
            options: new EnhancementOptions(ContinueWithoutEnhancement: false));

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            request,
            CancellationToken.None);

        AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.RuntimeMissing, false);
    }

    [TestMethod]
    public async Task RuntimeAndModelChecksumMismatchesStopBeforeProcessLaunch()
    {
        using (var environment = new AdapterTestEnvironment())
        {
            EnhancementResult result = await environment.CreateAdapter(new string('0', 64)).EnhanceAsync(
                environment.CreateRequest(),
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.RuntimeChecksumMismatch,
                true);
            Assert.AreEqual(0, environment.Runner.InvocationCount);
        }

        using (var environment = new AdapterTestEnvironment())
        {
            string wrongHash = new('0', 64);
            var mismatchedModel = new EnhancementModel(
                environment.Model.ModelId,
                environment.Model.Version,
                wrongHash,
                environment.Model.Source,
                environment.Model.Revision,
                environment.Model.LicenseSpdx,
                environment.Model.NoticePath,
                environment.Model.Artifacts.Select(artifact =>
                    new ModelArtifact(artifact.RelativePath, wrongHash)));
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                environment.CreateRequest(model: mismatchedModel),
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.ModelChecksumMismatch,
                true);
            Assert.AreEqual(0, environment.Runner.InvocationCount);
        }
    }

    [TestMethod]
    public async Task CpuFallbackRequestIsTruthfullyRejectedBeforeProcessLaunch()
    {
        using var environment = new AdapterTestEnvironment();
        EnhancementRequest request = environment.CreateRequest(
            options: new EnhancementOptions(RequestCpuFallback: true));

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            request,
            CancellationToken.None);

        AssertFailure(
            result,
            EnhancementStatus.ContinuedWithoutEnhancement,
            EnhancementFailureCode.CpuFallbackUnsupported,
            true);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
        StringAssert.Contains(result.Diagnostic.Message, "no supported CPU execution provider");
    }

    [TestMethod]
    public async Task UnsafeModelIdAndEscapingArtifactAreRejectedWithoutLaunch()
    {
        using (var environment = new AdapterTestEnvironment())
        {
            var unsafeId = new EnhancementModel(
                "model & calc.exe",
                environment.Model.Version,
                environment.Model.Sha256,
                environment.Model.Source,
                environment.Model.Revision,
                environment.Model.LicenseSpdx,
                environment.Model.NoticePath,
                environment.Model.Artifacts);
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                environment.CreateRequest(model: unsafeId),
                CancellationToken.None);
            AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.InvalidRequest, true);
            Assert.AreEqual(0, environment.Runner.InvocationCount);
        }

        using (var environment = new AdapterTestEnvironment())
        {
            var escapingArtifact = new EnhancementModel(
                environment.Model.ModelId,
                environment.Model.Version,
                environment.Model.Sha256,
                environment.Model.Source,
                environment.Model.Revision,
                environment.Model.LicenseSpdx,
                environment.Model.NoticePath,
                [new ModelArtifact(Path.Combine("..", "outside.bin"), environment.Model.Sha256)]);
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                environment.CreateRequest(model: escapingArtifact),
                CancellationToken.None);
            AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.InvalidRequest, true);
            Assert.AreEqual(0, environment.Runner.InvocationCount);
        }
    }

    [TestMethod]
    public async Task VerifiedArtifactsMustBeTheExactFilesSelectedByRuntimeArguments()
    {
        using var environment = new AdapterTestEnvironment();
        string artifactHash = environment.Model.Artifacts[0].Sha256;
        var unrelatedArtifacts = new EnhancementModel(
            environment.Model.ModelId,
            environment.Model.Version,
            environment.Model.Sha256,
            environment.Model.Source,
            environment.Model.Revision,
            environment.Model.LicenseSpdx,
            environment.Model.NoticePath,
            [
                new ModelArtifact("unrelated-x2.param", artifactHash),
                new ModelArtifact("unrelated-x2.bin", artifactHash)
            ]);

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            environment.CreateRequest(model: unrelatedArtifacts),
            CancellationToken.None);

        AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.InvalidRequest, true);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    [DataRow("scale")]
    [DataRow("tile-1")]
    [DataRow("tile-31")]
    [DataRow("gpu")]
    [DataRow("timeout")]
    [DataRow("relative-config")]
    [DataRow("empty-project")]
    [DataRow("empty-panel")]
    [DataRow("missing-model")]
    [DataRow("incomplete-provenance")]
    public async Task InvalidRequestMatrixFailsBeforeProcessLaunch(string scenario)
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        EnhancementRequest request = environment.CreateRequest();
        switch (scenario)
        {
            case "scale":
                request = request with { Options = new EnhancementOptions(Scale: 4) };
                break;
            case "tile-1":
                request = request with { Options = new EnhancementOptions(TileSize: 1) };
                break;
            case "tile-31":
                request = request with { Options = new EnhancementOptions(TileSize: 31) };
                break;
            case "gpu":
                request = request with { Options = new EnhancementOptions(GpuIndex: -1) };
                break;
            case "timeout":
                request = request with { Options = new EnhancementOptions(Timeout: TimeSpan.Zero) };
                break;
            case "relative-config":
                adapter = new RealEsrganAdapter(
                    new RealEsrganConfiguration("runtime.exe", "models", "cache"),
                    environment.Runner,
                    environment.Inspector);
                break;
            case "empty-project":
                request = request with { ProjectId = Guid.Empty };
                break;
            case "empty-panel":
                request = request with { PanelId = Guid.Empty };
                break;
            case "missing-model":
                request = request with { Model = null! };
                break;
            case "incomplete-provenance":
                var incomplete = new EnhancementModel(
                    environment.Model.ModelId,
                    environment.Model.Version,
                    environment.Model.Sha256,
                    string.Empty,
                    environment.Model.Revision,
                    environment.Model.LicenseSpdx,
                    environment.Model.NoticePath,
                    environment.Model.Artifacts);
                request = request with { Model = incomplete };
                break;
            default:
                Assert.Fail($"Unknown validation scenario '{scenario}'.");
                break;
        }

        EnhancementResult result = await adapter.EnhanceAsync(request, CancellationToken.None);

        AssertFailure(result, EnhancementStatus.Failed, EnhancementFailureCode.InvalidRequest, true);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task ProcessFailureTimeoutCancellationAndStartFailureRemainDistinct()
    {
        await AssertProcessFailureAsync(
            new ProcessExecutionResult(
                ProcessCompletion.Completed,
                17,
                "bounded stdout",
                "bounded stderr",
                TimeSpan.FromMilliseconds(3)),
            EnhancementStatus.ContinuedWithoutEnhancement,
            EnhancementFailureCode.ProcessFailed,
            expectedExitCode: 17);
        await AssertProcessFailureAsync(
            new ProcessExecutionResult(
                ProcessCompletion.TimedOut,
                -1,
                string.Empty,
                "timeout",
                TimeSpan.FromSeconds(17)),
            EnhancementStatus.TimedOut,
            EnhancementFailureCode.ProcessTimedOut,
            expectedExitCode: -1);
        await AssertProcessFailureAsync(
            new ProcessExecutionResult(
                ProcessCompletion.Cancelled,
                -1,
                string.Empty,
                "cancelled",
                TimeSpan.FromMilliseconds(1)),
            EnhancementStatus.Cancelled,
            EnhancementFailureCode.ProcessCancelled,
            expectedExitCode: -1);
        await AssertProcessFailureAsync(
            new ProcessExecutionResult(
                ProcessCompletion.StartFailed,
                null,
                string.Empty,
                string.Empty,
                TimeSpan.Zero,
                "start error"),
            EnhancementStatus.ContinuedWithoutEnhancement,
            EnhancementFailureCode.ProcessStartFailed,
            expectedExitCode: null);
    }

    [TestMethod]
    public async Task CallerCancellationReturnsCancelledAndCleansIsolatedWork()
    {
        using var environment = new AdapterTestEnvironment();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        environment.Runner.Handler = async (_, cancellationToken) =>
        {
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            throw new InvalidOperationException("Cancellation should interrupt the fake process.");
        };
        using var cancellation = new CancellationTokenSource();
        Task<EnhancementResult> pending = environment.CreateAdapter().EnhanceAsync(
            environment.CreateRequest(),
            cancellation.Token);
        await entered.Task;

        cancellation.Cancel();
        EnhancementResult result = await pending;

        AssertFailure(result, EnhancementStatus.Cancelled, EnhancementFailureCode.ProcessCancelled, true);
        Assert.IsFalse(File.Exists(environment.CreateRequest().OutputPath));
        Assert.IsTrue(environment.WorkDirectoryIsEmpty());
    }

    [TestMethod]
    public async Task PreCancelledRequestNeverInvokesProcess()
    {
        using var environment = new AdapterTestEnvironment();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            environment.CreateRequest(),
            cancellation.Token);

        AssertFailure(result, EnhancementStatus.Cancelled, EnhancementFailureCode.ProcessCancelled, true);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
    }

    [TestMethod]
    public async Task MissingCorruptAndWrongSizedOutputsAreRejectedWithoutPromotion()
    {
        using (var environment = new AdapterTestEnvironment())
        {
            environment.Runner.Handler = static (_, _) => Task.FromResult(
                new ProcessExecutionResult(
                    ProcessCompletion.Completed,
                    0,
                    string.Empty,
                    string.Empty,
                    TimeSpan.Zero));
            EnhancementRequest request = environment.CreateRequest();
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                request,
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.OutputMissing,
                true);
            Assert.IsFalse(File.Exists(request.OutputPath));
        }

        using (var environment = new AdapterTestEnvironment())
        {
            environment.Inspector.ExceptionToThrow = new InvalidDataException("corrupt image");
            EnhancementRequest request = environment.CreateRequest();
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                request,
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.OutputCorrupt,
                true);
            Assert.IsFalse(File.Exists(request.OutputPath));
        }

        using (var environment = new AdapterTestEnvironment())
        {
            environment.Inspector.Dimensions = new PixelDimensions(61, 38);
            EnhancementRequest request = environment.CreateRequest();
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                request,
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.DimensionMismatch,
                true);
            Assert.IsFalse(File.Exists(request.OutputPath));
        }
    }

    [TestMethod]
    public async Task SourceMutationDuringEnhancementDiscardsDerivative()
    {
        using var environment = new AdapterTestEnvironment();
        environment.Runner.Handler = async (invocation, cancellationToken) =>
        {
            ProcessExecutionResult success = await environment.CompleteSuccessfullyAsync(
                invocation,
                cancellationToken);
            await File.AppendAllTextAsync(
                environment.InputPath,
                "changed during process",
                cancellationToken);
            return success;
        };
        EnhancementRequest request = environment.CreateRequest();

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            request,
            CancellationToken.None);

        AssertFailure(
            result,
            EnhancementStatus.ContinuedWithoutEnhancement,
            EnhancementFailureCode.SourceChanged,
            true);
        Assert.IsFalse(File.Exists(request.OutputPath));
        Assert.IsTrue(environment.WorkDirectoryIsEmpty());
    }

    [TestMethod]
    public async Task RuntimeAndModelMutationDuringEnhancementDiscardDerivative()
    {
        using (var environment = new AdapterTestEnvironment())
        {
            environment.Runner.Handler = async (invocation, cancellationToken) =>
            {
                ProcessExecutionResult success = await environment.CompleteSuccessfullyAsync(
                    invocation,
                    cancellationToken);
                await File.AppendAllTextAsync(
                    environment.ExecutablePath,
                    "runtime changed during process",
                    cancellationToken);
                return success;
            };
            EnhancementRequest request = environment.CreateRequest();
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                request,
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.RuntimeChecksumMismatch,
                true);
            Assert.IsFalse(File.Exists(request.OutputPath));
        }

        using (var environment = new AdapterTestEnvironment())
        {
            environment.Runner.Handler = async (invocation, cancellationToken) =>
            {
                ProcessExecutionResult success = await environment.CompleteSuccessfullyAsync(
                    invocation,
                    cancellationToken);
                await File.AppendAllTextAsync(
                    environment.ModelPath,
                    "model changed during process",
                    cancellationToken);
                return success;
            };
            EnhancementRequest request = environment.CreateRequest();
            EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
                request,
                CancellationToken.None);
            AssertFailure(
                result,
                EnhancementStatus.ContinuedWithoutEnhancement,
                EnhancementFailureCode.ModelChecksumMismatch,
                true);
            Assert.IsFalse(File.Exists(request.OutputPath));
        }
    }

    [TestMethod]
    public async Task RepeatedRequestUsesVerifiedCacheWithoutSecondProcessRun()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();

        EnhancementResult first = await adapter.EnhanceAsync(
            environment.CreateRequest("first.png"),
            CancellationToken.None);
        EnhancementResult second = await adapter.EnhanceAsync(
            environment.CreateRequest("second.png"),
            CancellationToken.None);

        Assert.AreEqual(EnhancementStatus.Succeeded, first.Status);
        Assert.AreEqual(EnhancementStatus.CacheHit, second.Status);
        Assert.AreEqual(1, environment.Runner.InvocationCount);
        Assert.IsFalse(AssertEnvelope(first).Payload.CacheHit);
        Assert.IsTrue(AssertEnvelope(second).Payload.CacheHit);
        Assert.AreEqual(AssertEnvelope(first).Payload.CacheKey, AssertEnvelope(second).Payload.CacheKey);
        Assert.AreEqual(AssertEnvelope(first).Payload.OutputSha256, AssertEnvelope(second).Payload.OutputSha256);
        CollectionAssert.AreEqual(
            File.ReadAllBytes(first.OutputPath!),
            File.ReadAllBytes(second.OutputPath!));
    }

    [TestMethod]
    public async Task SourceMutationDuringCacheRestoreDiscardsDerivativeBeforePromotion()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        EnhancementResult first = await adapter.EnhanceAsync(
            environment.CreateRequest("first.png"),
            CancellationToken.None);
        Assert.AreEqual(EnhancementStatus.Succeeded, first.Status);

        environment.Inspector.OnRead = _ => File.AppendAllText(
            environment.InputPath,
            "changed during cache restore");
        EnhancementRequest cachedRequest = environment.CreateRequest("cache-race.png");

        EnhancementResult result = await adapter.EnhanceAsync(cachedRequest, CancellationToken.None);

        AssertFailure(
            result,
            EnhancementStatus.ContinuedWithoutEnhancement,
            EnhancementFailureCode.SourceChanged,
            true);
        Assert.IsFalse(File.Exists(cachedRequest.OutputPath));
        Assert.AreEqual(1, environment.Runner.InvocationCount);
        Assert.IsTrue(environment.WorkDirectoryIsEmpty());
    }

    [TestMethod]
    public async Task CacheInvalidatesForParametersRuntimeModelAndSourceIdentity()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        var results = new List<EnhancementResult>
        {
            await adapter.EnhanceAsync(environment.CreateRequest("base.png"), CancellationToken.None),
            await adapter.EnhanceAsync(environment.CreateRequest("base-hit.png"), CancellationToken.None),
            await adapter.EnhanceAsync(
                environment.CreateRequest("tile.png", new EnhancementOptions(TileSize: 64)),
                CancellationToken.None),
            await adapter.EnhanceAsync(
                environment.CreateRequest("gpu.png", new EnhancementOptions(GpuIndex: 1)),
                CancellationToken.None)
        };

        await File.AppendAllTextAsync(environment.ExecutablePath, "runtime revision");
        results.Add(await adapter.EnhanceAsync(
            environment.CreateRequest("runtime.png"),
            CancellationToken.None));

        environment.ReplaceModelArtifact(Encoding.UTF8.GetBytes("updated model revision"));
        results.Add(await adapter.EnhanceAsync(
            environment.CreateRequest("model.png"),
            CancellationToken.None));

        await File.AppendAllTextAsync(environment.InputPath, "new source revision");
        results.Add(await adapter.EnhanceAsync(
            environment.CreateRequest("source.png"),
            CancellationToken.None));

        Assert.AreEqual(EnhancementStatus.CacheHit, results[1].Status);
        Assert.IsTrue(results.Where((_, index) => index != 1).All(static result => result.Status == EnhancementStatus.Succeeded));
        Assert.AreEqual(6, environment.Runner.InvocationCount);
        string[] uniqueKeys = results
            .Where((_, index) => index != 1)
            .Select(static result => AssertEnvelope(result).Payload.CacheKey)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        Assert.AreEqual(6, uniqueKeys.Length);
    }

    [TestMethod]
    public async Task CorruptCachedBytesAreRejectedAndRecomputed()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        EnhancementResult first = await adapter.EnhanceAsync(
            environment.CreateRequest("first.png"),
            CancellationToken.None);
        Assert.AreEqual(EnhancementStatus.Succeeded, first.Status);
        string cachedImage = Directory.GetFiles(
            environment.CacheDirectory,
            "enhanced.png",
            SearchOption.AllDirectories).Single();
        await File.WriteAllTextAsync(cachedImage, "tampered cache bytes");

        EnhancementResult second = await adapter.EnhanceAsync(
            environment.CreateRequest("second.png"),
            CancellationToken.None);

        Assert.AreEqual(EnhancementStatus.Succeeded, second.Status);
        Assert.AreEqual(2, environment.Runner.InvocationCount);
        CollectionAssert.AreEqual(environment.OutputBytes, File.ReadAllBytes(second.OutputPath!));
    }

    [TestMethod]
    public async Task CacheMutationDuringRestoreCannotPromoteBytesWithAStaleHash()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter adapter = environment.CreateAdapter();
        EnhancementResult first = await adapter.EnhanceAsync(
            environment.CreateRequest("first.png"),
            CancellationToken.None);
        Assert.AreEqual(EnhancementStatus.Succeeded, first.Status);
        string cachedImage = Directory.GetFiles(
            environment.CacheDirectory,
            "enhanced.png",
            SearchOption.AllDirectories).Single();
        int mutated = 0;
        environment.Inspector.OnRead = path =>
        {
            if (string.Equals(path, cachedImage, StringComparison.OrdinalIgnoreCase) &&
                Interlocked.Exchange(ref mutated, 1) == 0)
            {
                File.WriteAllText(path, "mutated after cache verification");
            }
        };

        EnhancementResult second = await adapter.EnhanceAsync(
            environment.CreateRequest("second.png"),
            CancellationToken.None);

        Assert.AreEqual(EnhancementStatus.Succeeded, second.Status);
        Assert.AreEqual(2, environment.Runner.InvocationCount);
        CollectionAssert.AreEqual(environment.OutputBytes, File.ReadAllBytes(second.OutputPath!));
        Assert.AreEqual(
            await EnhancementHashing.ComputeFileSha256Async(second.OutputPath!, CancellationToken.None),
            AssertEnvelope(second).Payload.OutputSha256);
    }

    [TestMethod]
    public async Task IndependentAdapterInstancesCoordinateTheSameCacheEntry()
    {
        using var environment = new AdapterTestEnvironment();
        RealEsrganAdapter firstAdapter = environment.CreateAdapter();
        RealEsrganAdapter secondAdapter = environment.CreateAdapter();

        EnhancementResult[] results = await Task.WhenAll(
            firstAdapter.EnhanceAsync(environment.CreateRequest("instance-one.png"), CancellationToken.None),
            secondAdapter.EnhanceAsync(environment.CreateRequest("instance-two.png"), CancellationToken.None));

        CollectionAssert.AreEquivalent(
            new[] { EnhancementStatus.Succeeded, EnhancementStatus.CacheHit },
            results.Select(static result => result.Status).ToArray());
        Assert.AreEqual(1, environment.Runner.InvocationCount);
        Assert.IsTrue(results.All(static result => File.Exists(result.OutputPath)));
    }

    [TestMethod]
    public async Task ConcurrentIdenticalRequestsRunProcessOnceAndPromoteBothOutputs()
    {
        using var environment = new AdapterTestEnvironment();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        environment.Runner.Handler = async (invocation, cancellationToken) =>
        {
            entered.SetResult();
            await release.Task.WaitAsync(cancellationToken);
            return await environment.CompleteSuccessfullyAsync(invocation, cancellationToken);
        };
        RealEsrganAdapter adapter = environment.CreateAdapter();
        Task<EnhancementResult> firstTask = adapter.EnhanceAsync(
            environment.CreateRequest("concurrent-one.png"),
            CancellationToken.None);
        await entered.Task;
        Task<EnhancementResult> secondTask = adapter.EnhanceAsync(
            environment.CreateRequest("concurrent-two.png"),
            CancellationToken.None);
        release.SetResult();

        EnhancementResult[] results = await Task.WhenAll(firstTask, secondTask);

        CollectionAssert.AreEquivalent(
            new[] { EnhancementStatus.Succeeded, EnhancementStatus.CacheHit },
            results.Select(static result => result.Status).ToArray());
        Assert.AreEqual(1, environment.Runner.InvocationCount);
        Assert.IsTrue(results.All(static result => File.Exists(result.OutputPath)));
        Assert.IsTrue(environment.WorkDirectoryIsEmpty());
    }

    private static bool IsOption(string value) => value.Length == 2 && value[0] == '-';

    private static EnhancementEnvelope AssertEnvelope(EnhancementResult result)
    {
        Assert.IsNotNull(result.Envelope);
        EnhancementEnvelope envelope = result.Envelope;
        Assert.AreEqual(1, envelope.ContractVersion);
        Assert.AreNotEqual(Guid.Empty, envelope.RunId);
        Assert.AreEqual("enhancement", envelope.Stage);
        Assert.AreEqual("original_pixels", envelope.CoordinateSpace);
        Assert.AreEqual(1d, envelope.Confidence);
        Assert.IsTrue(envelope.TimingMs.Total >= 0);
        Assert.AreEqual(2, envelope.Payload.Transform.Scale);
        return envelope;
    }

    private static void AssertFailure(
        EnhancementResult result,
        EnhancementStatus status,
        EnhancementFailureCode code,
        bool mayContinue)
    {
        Assert.AreEqual(status, result.Status);
        Assert.AreEqual(code, result.Diagnostic.Code);
        Assert.AreEqual(mayContinue, result.MayContinueUnenhanced);
        Assert.IsFalse(result.IsSuccess);
        Assert.IsNull(result.OutputPath);
        Assert.IsNull(result.Envelope);
        Assert.IsFalse(string.IsNullOrWhiteSpace(result.Diagnostic.Message));
    }

    private static async Task AssertProcessFailureAsync(
        ProcessExecutionResult processResult,
        EnhancementStatus expectedStatus,
        EnhancementFailureCode expectedCode,
        int? expectedExitCode)
    {
        using var environment = new AdapterTestEnvironment();
        environment.Runner.Handler = (_, _) => Task.FromResult(processResult);

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            environment.CreateRequest(),
            CancellationToken.None);

        AssertFailure(result, expectedStatus, expectedCode, true);
        Assert.AreEqual(expectedExitCode, result.Diagnostic.ExitCode);
        Assert.AreEqual(processResult.StandardOutput, result.Diagnostic.StandardOutput);
        string expectedError = string.IsNullOrEmpty(processResult.StandardError)
            ? processResult.StartError ?? string.Empty
            : processResult.StandardError;
        Assert.AreEqual(expectedError, result.Diagnostic.StandardError);
        Assert.AreEqual(1, environment.Runner.InvocationCount);
        Assert.IsTrue(environment.WorkDirectoryIsEmpty());
    }
}
