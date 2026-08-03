// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class ConsensusAndBenchmarkTests
{
    private static readonly string[] CpuCudaProviders = ["cpu", "cuda"];
    private static readonly string[] VulkanProviders = ["vulkan"];

    [TestMethod]
    public void ConsensusMapsEnhancedEvidenceBackAndFlagsEveryDisagreementKind()
    {
        EnhancementEvidencePoint[] original =
        [
            new("agree", new EnhancementPoint(10, 20), 0.9),
            new("moves", new EnhancementPoint(5, 5), 0.8),
            new("original-only", new EnhancementPoint(7, 8), 0.7)
        ];
        EnhancementEvidencePoint[] enhanced =
        [
            new("agree", new EnhancementPoint(20, 40), 0.9),
            new("moves", new EnhancementPoint(20, 20), 0.8),
            new("enhanced-only", new EnhancementPoint(12, 14), 0.7)
        ];

        EnhancementConsensusResult result = EnhancementConsensus.Compare(
            original,
            enhanced,
            EnhancementTransform.CreateScale2(),
            maximumDisplacementPixels: 1);

        Assert.IsTrue(result.RequiresReview);
        Assert.AreEqual(0.25, result.ConfidenceMultiplier);
        Assert.AreEqual(4, result.Items.Count);

        EnhancementConsensusItem agree = Find(result, "agree");
        Assert.IsFalse(agree.RequiresReview);
        Assert.AreEqual(new EnhancementPoint(10, 20), agree.EnhancedLocationInOriginalPixels);
        Assert.AreEqual(0d, agree.DisplacementPixels);

        EnhancementConsensusItem moves = Find(result, "moves");
        Assert.IsTrue(moves.RequiresReview);
        Assert.IsTrue(moves.DisplacementPixels > 1);
        StringAssert.Contains(moves.Reason, "disagree beyond tolerance");

        EnhancementConsensusItem enhancedOnly = Find(result, "enhanced-only");
        Assert.IsTrue(enhancedOnly.RequiresReview);
        Assert.IsNull(enhancedOnly.OriginalLocation);
        Assert.AreEqual(new EnhancementPoint(6, 7), enhancedOnly.EnhancedLocationInOriginalPixels);

        EnhancementConsensusItem originalOnly = Find(result, "original-only");
        Assert.IsTrue(originalOnly.RequiresReview);
        Assert.IsNull(originalOnly.EnhancedLocationInOriginalPixels);
    }

    [TestMethod]
    public void ConsensusRejectsDuplicateIdsInvalidConfidenceAndInvalidTolerance()
    {
        EnhancementEvidencePoint duplicate = new("same", new EnhancementPoint(1, 2), 0.5);
        Assert.ThrowsExactly<ArgumentException>(() => EnhancementConsensus.Compare(
            [duplicate, duplicate],
            [],
            EnhancementTransform.CreateScale2(),
            1));
        Assert.ThrowsExactly<ArgumentException>(() => EnhancementConsensus.Compare(
            [new EnhancementEvidencePoint("bad", new EnhancementPoint(1, 2), 1.1)],
            [],
            EnhancementTransform.CreateScale2(),
            1));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => EnhancementConsensus.Compare(
            [],
            [],
            EnhancementTransform.CreateScale2(),
            double.NaN));
    }

    [TestMethod]
    public void BenchmarkPlanContainsThreeTruthfulTwoXModelConfigurationsPerCase()
    {
        EnhancementBenchmarkCase[] cases =
        [
            new("case-a", "input-a.png", "truth-a.json"),
            new("case-b", "input-b.png", "truth-b.json")
        ];

        IReadOnlyList<EnhancementBenchmarkWorkItem> plan =
            EnhancementBenchmarkHarness.CreatePlan(cases);

        Assert.AreEqual(6, plan.Count);
        Assert.IsTrue(plan.All(static item => item.Model.OutputScale == 2));
        Assert.AreEqual(2, plan.Count(static item => item.Model.ModelId == "RealESRGAN_x2plus"));
        Assert.AreEqual(2, plan.Count(static item => item.Model.ModelId == "realesr-general-x4v3"));
        Assert.AreEqual(2, plan.Count(static item => item.Model.ModelId == "realesr-animevideov3"));

        EnhancementBenchmarkModel x2Plus = EnhancementBenchmarkHarness.RequiredModels[0];
        Assert.AreEqual(BenchmarkRuntimeCompatibility.ReferenceWeightsOnly, x2Plus.RuntimeCompatibility);
        CollectionAssert.AreEqual(CpuCudaProviders, x2Plus.Providers.ToArray());

        EnhancementBenchmarkModel general = EnhancementBenchmarkHarness.RequiredModels[1];
        Assert.AreEqual(4, general.NativeScale);
        Assert.AreEqual(BenchmarkRuntimeCompatibility.ReferenceWeightsOnly, general.RuntimeCompatibility);
        CollectionAssert.AreEqual(CpuCudaProviders, general.Providers.ToArray());

        EnhancementBenchmarkModel anime = EnhancementBenchmarkHarness.RequiredModels[2];
        Assert.AreEqual(BenchmarkRuntimeCompatibility.NcnnVulkan, anime.RuntimeCompatibility);
        CollectionAssert.AreEqual(VulkanProviders, anime.Providers.ToArray());
        Assert.AreEqual("ncnn_param_bin", anime.ArtifactFormat);
    }

    [TestMethod]
    public async Task BenchmarkHarnessAcceptsCompleteValidatedObservationsWithoutInventingValues()
    {
        EnhancementBenchmarkCase[] cases = [new("fixed-case", "input.png", "truth.json")];
        var runner = new FakeBenchmarkRunner(static workItem =>
            new EnhancementBenchmarkObservation(
                workItem,
                new EnhancementBenchmarkMetrics(
                    MarkerCenterF1: 0.8,
                    ShapeFillClassificationF1: 0.7,
                    NumericOcrExactMatch: 0.6,
                    AxisLocalizationErrorPixels: 1.5,
                    HallucinatedStructureRate: 0.1,
                    RuntimeMilliseconds: 12.3,
                    PeakMemoryBytes: 456),
                new string('a', 64)));

        IReadOnlyList<EnhancementBenchmarkObservation> observations =
            await EnhancementBenchmarkHarness.RunAsync(cases, runner, CancellationToken.None);

        Assert.AreEqual(3, observations.Count);
        Assert.AreEqual(3, runner.RunCount);
        Assert.IsTrue(observations.All(static item => item.Metrics.RuntimeMilliseconds == 12.3));
    }

    [TestMethod]
    public async Task BenchmarkHarnessRejectsInvalidMetricsAndMismatchedEvidence()
    {
        EnhancementBenchmarkCase[] cases = [new("fixed-case", "input.png", "truth.json")];
        var invalidMetrics = new FakeBenchmarkRunner(static workItem =>
            new EnhancementBenchmarkObservation(
                workItem,
                new EnhancementBenchmarkMetrics(1.1, 0, 0, 0, 0, 0, 0),
                new string('a', 64)));
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() =>
            EnhancementBenchmarkHarness.RunAsync(cases, invalidMetrics, CancellationToken.None));

        var mismatched = new FakeBenchmarkRunner(static workItem =>
            new EnhancementBenchmarkObservation(
                workItem with { TestCase = workItem.TestCase with { CaseId = "different" } },
                new EnhancementBenchmarkMetrics(0, 0, 0, 0, 0, 0, 0),
                new string('a', 64)));
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() =>
            EnhancementBenchmarkHarness.RunAsync(cases, mismatched, CancellationToken.None));
    }

    [TestMethod]
    public void BenchmarkPlanRequiresANonemptyFixedTestSet()
    {
        Assert.ThrowsExactly<ArgumentException>(() =>
            EnhancementBenchmarkHarness.CreatePlan([]));
        Assert.ThrowsExactly<ArgumentException>(() =>
            EnhancementBenchmarkHarness.CreatePlan(
                [new EnhancementBenchmarkCase("", "input.png", "truth.json")]));
    }

    private static EnhancementConsensusItem Find(
        EnhancementConsensusResult result,
        string evidenceId) =>
        result.Items.Single(item => item.EvidenceId == evidenceId);
}

internal sealed class FakeBenchmarkRunner : IEnhancementBenchmarkRunner
{
    private readonly Func<EnhancementBenchmarkWorkItem, EnhancementBenchmarkObservation> _factory;
    private int _runCount;

    public FakeBenchmarkRunner(
        Func<EnhancementBenchmarkWorkItem, EnhancementBenchmarkObservation> factory)
    {
        _factory = factory;
    }

    public int RunCount => Volatile.Read(ref _runCount);

    public Task<EnhancementBenchmarkObservation> RunAsync(
        EnhancementBenchmarkWorkItem workItem,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _runCount);
        return Task.FromResult(_factory(workItem));
    }
}
