// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV9P3DirectCorpusTests
{
    private const string RunSelectionVariable = "GRAPHREADER_RUN_OCR_V9_P3_SELECTION";
    private const string RunPublicVariable = "GRAPHREADER_RUN_OCR_V9_P3_PUBLIC";
    private const string PrimaryVariable = "GRAPHREADER_OCR_V9_P3_PRIMARY";
    private const string RoleVariable = "GRAPHREADER_OCR_V9_P3_ROLE";
    private const string OfficialVariable = "GRAPHREADER_OCR_V9_P3_OFFICIAL";
    private const string NumericVariable = "GRAPHREADER_OCR_V9_P3_NUMERIC";
    private const string AmbiguityVariable = "GRAPHREADER_OCR_V9_P3_AMBIGUITY";
    private const string YamlVariable = "GRAPHREADER_OCR_V9_P3_OFFICIAL_YAML";
    private const string SelectionReportVariable = "GRAPHREADER_OCR_V9_P3_SELECTION_REPORT";
    private const string PublicReportVariable = "GRAPHREADER_OCR_V9_P3_PUBLIC_REPORT";
    private const string SourceCommitVariable = "GRAPHREADER_OCR_V9_P3_SOURCE_COMMIT";
    private const string InferenceYamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";
    private const string SelectionExactTest =
        "OcrV9P3DirectCorpusTests.FreshSelectionBytesExecuteOnceThroughCSharpCpuConsensus";
    private const string PublicExactTest =
        "OcrV9P3DirectCorpusTests.FreshPublicBytesExecuteOnceThroughCSharpCpuConsensus";
    private static readonly OcrRectangle PlotBounds = new(104, 48, 406, 208);
    private static readonly string[] RequiredRoles =
    [
        "y_tick", "x_tick", "axis_title", "phase_heading",
        "legend_text", "participant", "annotation", "other",
    ];
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    [TestMethod]
    public Task FreshSelectionBytesExecuteOnceThroughCSharpCpuConsensus()
    {
        if (!Enabled(RunSelectionVariable))
        {
            Assert.Inconclusive($"Set {RunSelectionVariable}=1 only after tracked selection authorization exists.");
        }
        return ExecuteAsync("selection", SelectionExactTest, SelectionReportVariable, CancellationToken.None);
    }

    [TestMethod]
    public Task FreshPublicBytesExecuteOnceThroughCSharpCpuConsensus()
    {
        if (!Enabled(RunPublicVariable))
        {
            Assert.Inconclusive($"Set {RunPublicVariable}=1 only after a passing selection result and public authorization exist.");
        }
        return ExecuteAsync("sealed_public", PublicExactTest, PublicReportVariable, CancellationToken.None);
    }

    private static async Task ExecuteAsync(
        string split,
        string exactTest,
        string reportVariable,
        CancellationToken cancellationToken)
    {
        string root = FindRepositoryRoot();
        string contractRoot = Path.Combine(root, "ml", "ocr", "cross_model_consensus_v9_p3");
        string sealPath = Path.Combine(contractRoot, "SPLIT_SEAL.json");
        string authorizationPath = Path.Combine(
            contractRoot,
            split == "selection" ? "SELECTION_AUTHORIZATION.json" : "PUBLIC_GATE_AUTHORIZATION.json");
        Assert.IsTrue(File.Exists(sealPath), "P3 split identity is not frozen.");
        Assert.IsTrue(File.Exists(authorizationPath), $"P3 {split} execution is not authorized.");
        SplitSeal seal = Deserialize<SplitSeal>(sealPath);
        CandidateAuthorization authorization = Deserialize<CandidateAuthorization>(authorizationPath);
        ValidateAuthorization(root, sealPath, seal, authorizationPath, authorization, split, exactTest);

        string archiveRelative = split == "selection" ? seal.SelectionArchivePath : seal.PublicArchivePath;
        string archiveSha256 = split == "selection" ? seal.SelectionArchiveSha256 : seal.PublicArchiveSha256;
        string manifestSha256 = split == "selection" ? seal.SelectionManifestSha256 : seal.PublicManifestSha256;
        int expectedScenes = split == "selection" ? 192 : 256;
        string archivePath = Path.GetFullPath(Path.Combine(root, archiveRelative));
        AssertHash(archivePath, archiveSha256, $"P3 {split} archive");
        string reportPath = RequiredOutputPath(reportVariable);
        Assert.AreEqual(
            Path.GetFullPath(Path.Combine(root, authorization.ResultPath)),
            reportPath,
            true,
            "P3 report path is not authorization-bound.");
        string sourceCommit = RequiredSourceCommit();
        string yamlPath = RequiredPath(YamlVariable);
        AssertHash(yamlPath, InferenceYamlSha256, "official inference YAML");
        OcrV9P3PayloadSet payloads = Payloads(
            RequiredPath(PrimaryVariable),
            RequiredPath(RoleVariable),
            RequiredPath(OfficialVariable),
            RequiredPath(NumericVariable),
            RequiredPath(AmbiguityVariable),
            OcrV8DirectPublicCorpusTests.ReadOfficialAlphabet(yamlPath));
        ValidateCandidateHashes(authorization, payloads);

        var evidenceFactory = new OcrV8DirectPublicCorpusTests.EvidenceInferenceSessionFactory(
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance));
        var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            evidenceFactory,
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        string cacheRoot = Path.Combine(Path.GetTempPath(), "GraphReaderOcrV9P3", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(cacheRoot);
        try
        {
            await using var runtime = new InferenceRuntime(
                registry,
                new BoundedInferenceScheduler(capacity: 2, workerCount: 1),
                new ContentAddressedStageCache(cacheRoot));
            OcrV9P3CrossModelConsensusPipeline pipeline =
                OcrV9P3CrossModelConsensusFactory.Create(
                    runtime, payloads, [InferenceProvider.Cpu], bypassCache: true);
            DirectReport report = await EvaluateAsync(
                archivePath,
                archiveSha256,
                manifestSha256,
                split,
                expectedScenes,
                authorizationPath,
                pipeline,
                evidenceFactory,
                sourceCommit,
                cancellationToken);
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            await File.WriteAllBytesAsync(
                reportPath,
                JsonSerializer.SerializeToUtf8Bytes(report, JsonOptions),
                cancellationToken);

            Assert.IsTrue(report.GatePassed, JsonSerializer.Serialize(report.Metrics));
            Assert.IsTrue(report.FullEightRoleCoverageProven);
            Assert.IsFalse(report.MarkerCreationEvaluated);
            Assert.IsFalse(report.ArtifactMaskProductionApproval);
            Assert.IsFalse(report.ProductionApproval);
            Assert.IsFalse(report.ReleaseEligible);
        }
        finally
        {
            if (Directory.Exists(cacheRoot))
            {
                Directory.Delete(cacheRoot, recursive: true);
            }
        }
    }

    private static async Task<DirectReport> EvaluateAsync(
        string archivePath,
        string archiveSha256,
        string manifestSha256,
        string split,
        int expectedScenes,
        string authorizationPath,
        OcrV9P3CrossModelConsensusPipeline pipeline,
        OcrV8DirectPublicCorpusTests.EvidenceInferenceSessionFactory evidenceFactory,
        string sourceCommit,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(archivePath);
        ZipArchiveEntry manifestEntry = archive.GetEntry("manifest.json") ??
            throw new InvalidDataException("P3 archive has no manifest.json.");
        byte[] manifestBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(manifestEntry, cancellationToken);
        Assert.AreEqual(manifestSha256, Sha256(manifestBytes), "P3 manifest bytes changed.");
        FixtureManifest manifest = JsonSerializer.Deserialize<FixtureManifest>(manifestBytes, JsonOptions) ??
            throw new InvalidDataException("P3 fixture manifest is invalid.");
        Assert.AreEqual("graphreader.ocr-cross-model-consensus-fixtures.v9-p3", manifest.Schema);
        Assert.AreEqual(split, manifest.Split);
        Assert.AreEqual(expectedScenes, manifest.SceneCount);
        Assert.AreEqual(expectedScenes * 8, manifest.TruthRegionCount);
        Assert.IsTrue(manifest.SyntheticOnly);
        Assert.IsFalse(manifest.PrivateOrArticleImages);
        Assert.IsFalse(manifest.ChandlerIncluded);
        Assert.IsFalse(manifest.GeneralizationLabelIncluded);
        CollectionAssert.AreEquivalent(RequiredRoles, manifest.RequiredRoles.ToArray());

        var familyCounts = new Dictionary<string, (int Correct, int Total)>(StringComparer.Ordinal);
        var roleCounts = new Dictionary<string, (int Correct, int Total)>(StringComparer.Ordinal);
        var roleConfusions = new Dictionary<string, int>(StringComparer.Ordinal);
        var correctText = 0;
        var correctRole = 0;
        var characterCount = 0;
        var editErrors = 0;
        var truePositives = 0;
        var falsePositives = 0;
        var falseNegatives = 0;
        var duplicates = 0;
        var exactScenes = 0;

        foreach (FixtureCase fixture in manifest.Cases)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.AreEqual(8, fixture.TextTruths.Count);
            Assert.AreEqual(4, fixture.StructureCollisionCount);
            ZipArchiveEntry imageEntry = archive.GetEntry(fixture.ImagePath) ??
                throw new InvalidDataException($"P3 fixture image is missing: {fixture.ImagePath}");
            byte[] sourceBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(imageEntry, cancellationToken);
            Assert.AreEqual(fixture.ImageSha256, Sha256(sourceBytes), $"P3 PNG changed: {fixture.SceneId}");
            (byte[] pixels, int width, int height) = OcrV8DirectPublicCorpusTests.DecodeGray8(sourceBytes);
            Assert.AreEqual(640, width);
            Assert.AreEqual(320, height);
            Assert.AreEqual(fixture.RasterSha256, Sha256(pixels), $"P3 raster changed: {fixture.SceneId}");

            var image = new OcrImage(
                width, height, width, pixels, OcrSourceImage.Original, OcrFrameTransform.Identity,
                CanonicalOriginalWidth: width, CanonicalOriginalHeight: height);
            OcrResult result = await pipeline.RecognizeAsync(
                new OcrRequest(
                    "ocr-v9-p3-cross-model-consensus",
                    fixture.SceneId,
                    fixture.ImageSha256,
                    image,
                    PlotBounds),
                cancellationToken);
            Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
            Assert.AreEqual(result.Regions.Count, result.Masks.Count);

            HashSet<int> matchedTruths = [];
            var sceneFalsePositives = 0;
            var sceneDuplicates = 0;
            foreach (OcrRegion region in result.Regions)
            {
                (int Index, double Iou)[] matches = fixture.TextTruths
                    .Select((truth, index) => (
                        Index: index,
                        Iou: OcrV8DirectPublicCorpusTests.IntersectionOverUnion(region.Polygon.Bounds, truth.Bbox)))
                    .Where(static item => item.Iou >= 0.5)
                    .OrderByDescending(static item => item.Iou)
                    .ThenBy(static item => item.Index)
                    .ToArray();
                if (matches.Length == 0)
                {
                    sceneFalsePositives++;
                    continue;
                }
                int truthIndex = matches[0].Index;
                if (!matchedTruths.Add(truthIndex))
                {
                    sceneDuplicates++;
                    continue;
                }

                FixtureTruth truth = fixture.TextTruths[truthIndex];
                bool textMatch = string.Equals(region.Text, truth.TruthText, StringComparison.Ordinal);
                string predictedRole = OcrV8DirectPublicCorpusTests.RoleName(region.Role);
                bool roleMatch = string.Equals(predictedRole, truth.Role, StringComparison.Ordinal);
                if (!roleMatch)
                {
                    string key = $"{truth.Role}->{predictedRole}";
                    roleConfusions[key] = roleConfusions.GetValueOrDefault(key) + 1;
                }
                correctText += textMatch ? 1 : 0;
                correctRole += roleMatch ? 1 : 0;
                characterCount += truth.TruthText.EnumerateRunes().Count();
                editErrors += OcrV8DirectPublicCorpusTests.LevenshteinDistance(truth.TruthText, region.Text);
                (int familyCorrect, int familyTotal) = familyCounts.GetValueOrDefault(truth.Family);
                familyCounts[truth.Family] = (familyCorrect + (textMatch ? 1 : 0), familyTotal + 1);
                (int roleCorrect, int roleTotal) = roleCounts.GetValueOrDefault(truth.Role);
                roleCounts[truth.Role] = (roleCorrect + (roleMatch ? 1 : 0), roleTotal + 1);
            }

            int sceneFalseNegatives = fixture.TextTruths.Count - matchedTruths.Count;
            if (sceneFalsePositives == 0 && sceneFalseNegatives == 0 && sceneDuplicates == 0 &&
                result.Regions.Count == fixture.TextTruths.Count)
            {
                exactScenes++;
            }
            truePositives += matchedTruths.Count;
            falsePositives += sceneFalsePositives;
            falseNegatives += sceneFalseNegatives;
            duplicates += sceneDuplicates;
        }

        IReadOnlyDictionary<string, double> perRoleAccuracy = RequiredRoles.ToDictionary(
            static role => role,
            role => Ratio(roleCounts, role),
            StringComparer.Ordinal);
        bool everyRoleObserved = RequiredRoles.All(role =>
            roleCounts.TryGetValue(role, out (int Correct, int Total) value) && value.Total > 0);
        var metrics = new DirectMetrics(
            manifest.SceneCount,
            manifest.TruthRegionCount,
            exactScenes,
            truePositives,
            falsePositives,
            falseNegatives,
            duplicates,
            correctText / (double)manifest.TruthRegionCount,
            editErrors / (double)characterCount,
            correctRole / (double)manifest.TruthRegionCount,
            perRoleAccuracy,
            everyRoleObserved,
            Ratio(familyCounts, "numeric"),
            Ratio(familyCounts, "word"),
            Ratio(familyCounts, "ambiguity"),
            roleConfusions,
            falsePositives);
        IReadOnlyDictionary<string, OcrV8DirectPublicCorpusTests.ModelExecutionEvidence> executions =
            evidenceFactory.Snapshot();
        bool runtimeEvidencePassed = executions.Count == 5 &&
            executions.Keys.Order(StringComparer.Ordinal).SequenceEqual(
                ExpectedModelHashes().Order(StringComparer.Ordinal), StringComparer.Ordinal) &&
            executions.Values.All(static item =>
                item.CallCount > 0 &&
                item.InputTensorSha256.Count == item.CallCount &&
                item.OutputTensorSha256.Count == item.CallCount &&
                item.Providers.SequenceEqual(["Cpu"], StringComparer.Ordinal));
        bool fullEightRoleCoverage = everyRoleObserved &&
            perRoleAccuracy.Count == RequiredRoles.Length &&
            perRoleAccuracy.Values.All(static value => value >= 0.90);
        bool gatesPassed = exactScenes == manifest.SceneCount &&
            truePositives == manifest.TruthRegionCount &&
            falsePositives == 0 && falseNegatives == 0 && duplicates == 0 &&
            metrics.ProhibitedStructureHits == 0 &&
            metrics.RecognitionExactMatch >= 0.90 &&
            metrics.CharacterErrorRate <= 0.05 &&
            metrics.RoleAccuracy >= 0.90 &&
            fullEightRoleCoverage &&
            metrics.NumericExactMatch >= 0.90 &&
            metrics.WordExactMatch >= 0.90 &&
            metrics.AmbiguityExactMatch >= 0.90 &&
            runtimeEvidencePassed;
        return new DirectReport(
            "graphreader.ocr-cross-model-consensus-direct-report.v1",
            "P3",
            OcrV9P3CrossModelConsensusPipeline.CandidateCompositionId,
            split,
            sourceCommit,
            archiveSha256,
            manifestSha256,
            Sha256(File.ReadAllBytes(authorizationPath)),
            pipeline.ConfigurationFingerprint,
            "CPUExecutionProvider",
            executions,
            metrics,
            runtimeEvidencePassed,
            gatesPassed,
            fullEightRoleCoverage,
            MarkerCreationEvaluated: false,
            ArtifactMaskProductionApproval: false,
            ProductionApproval: false,
            ReleaseEligible: false,
            BlockingGates: gatesPassed
                ?
                [
                    "marker_stage_direct_composition_evidence",
                    "approved_artifact_mask_provider",
                    "approved_production_model_store",
                    "packaging_discovery_and_clean_machine_evidence",
                    "private_chandler_automatic_validation",
                ]
                : ["p3_direct_gate_failed"]);
    }

    private static void ValidateAuthorization(
        string root,
        string sealPath,
        SplitSeal seal,
        string authorizationPath,
        CandidateAuthorization authorization,
        string split,
        string exactTest)
    {
        Assert.AreEqual("graphreader.ocr-cross-model-consensus-split-seal.v1", seal.Schema);
        Assert.AreEqual("P3", seal.CandidateId);
        Assert.AreEqual(0, seal.ModelExecutionCountAtFreeze);
        Assert.AreEqual(0, seal.SelectionEvaluations);
        Assert.AreEqual(0, seal.PublicEvaluations);
        Assert.IsFalse(seal.SelectionExecutionAuthorized);
        Assert.IsFalse(seal.PublicExecutionAuthorized);
        Assert.IsFalse(seal.MarkerCreationEvaluated);
        Assert.IsFalse(seal.ProductionApproval);
        Assert.IsFalse(seal.ReleaseEligible);
        foreach ((string relativePath, string expectedHash) in seal.SourceSha256)
        {
            AssertHash(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)), expectedHash, relativePath);
        }

        Assert.AreEqual("graphreader.ocr-cross-model-consensus-authorization.v1", authorization.Schema);
        Assert.AreEqual("P3", authorization.CandidateId);
        Assert.AreEqual(split, authorization.Split);
        Assert.IsTrue(authorization.ExecutionAuthorized);
        Assert.AreEqual(1, authorization.ExecutionCountAuthorized);
        Assert.AreEqual("CPUExecutionProvider", authorization.Provider);
        Assert.IsTrue(
            authorization.SealedIdentityCommit.Length == 40
            && authorization.SealedIdentityCommit.All(Uri.IsHexDigit));
        Assert.AreEqual(Sha256(File.ReadAllBytes(sealPath)), authorization.SplitSealSha256);
        string expectedArchive = split == "selection" ? seal.SelectionArchiveSha256 : seal.PublicArchiveSha256;
        string expectedManifest = split == "selection" ? seal.SelectionManifestSha256 : seal.PublicManifestSha256;
        Assert.AreEqual(expectedArchive, authorization.FixtureArchiveSha256);
        Assert.AreEqual(expectedManifest, authorization.FixtureManifestSha256);
        CollectionAssert.AreEqual(
            ExpectedModelHashes().Order(StringComparer.Ordinal).ToArray(),
            authorization.CandidateSha256.Order(StringComparer.Ordinal).ToArray());
        Assert.AreEqual(exactTest, authorization.ExactTest);
        Assert.IsFalse(authorization.RerunOrRepairAuthorized);
        Assert.IsFalse(authorization.MarkerStageAuthorized);
        Assert.IsFalse(authorization.ArtifactMaskProductionApproval);
        Assert.IsFalse(authorization.ManifestCreationAuthorized);
        Assert.IsFalse(authorization.ModelStorePromotionAuthorized);
        Assert.IsFalse(authorization.PrivateValidationAuthorized);
        Assert.IsFalse(authorization.ProductionApproval);
        Assert.IsFalse(authorization.ReleaseEligible);
        if (split == "sealed_public")
        {
            Assert.IsFalse(string.IsNullOrWhiteSpace(authorization.SelectionResultSha256));
            string selectionResultPath = Path.Combine(
                root, "ml", "ocr", "cross_model_consensus_v9_p3", "P3_SELECTION_RESULT.json");
            AssertHash(selectionResultPath, authorization.SelectionResultSha256!, "P3 selection result");
            SelectionResult selection = Deserialize<SelectionResult>(selectionResultPath);
            Assert.IsTrue(selection.SelectionGatePassed);
            Assert.IsTrue(selection.ExecutionConsumed);
            Assert.IsFalse(selection.RerunOrRepairAuthorized);
        }
    }

    private static OcrV9P3PayloadSet Payloads(
        string primary,
        string role,
        string official,
        string numeric,
        string ambiguity,
        string alphabet) =>
        new(
            new OcrV8ProductionPayloadSet(
                new ModelIdentity(
                    "graph-text-spaced-component-recall-v10-p2", "0.0.21-p2",
                    OcrV8ProductionCompositionFactory.DetectorSha256, primary),
                new ModelIdentity(
                    "en_PP-OCRv5_mobile_rec", "0.0.21-converted",
                    OcrV8ProductionCompositionFactory.OfficialRecognizerSha256, official),
                new ModelIdentity(
                    "graph-numeric-component-ensemble-v5", "0.0.21-p1",
                    OcrV8ProductionCompositionFactory.NumericRecognizerSha256, numeric),
                new ModelIdentity(
                    "graph-ambiguity-source-group-v3-p2", "0.0.21-p2",
                    OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256, ambiguity),
                alphabet),
            new ModelIdentity(
                OcrV11CandidateCompositionFactory.DetectorModelId, "0.0.21-p2",
                OcrV11CandidateCompositionFactory.DetectorSha256, role));

    private static void ValidateCandidateHashes(
        CandidateAuthorization authorization,
        OcrV9P3PayloadSet payloads)
    {
        string[] actual =
        [
            payloads.Primary.Detector.Sha256,
            payloads.RoleDetector.Sha256,
            payloads.Primary.OfficialRecognizer.Sha256,
            payloads.Primary.NumericRecognizer.Sha256,
            payloads.Primary.AmbiguityRecognizer.Sha256,
        ];
        CollectionAssert.AreEqual(
            authorization.CandidateSha256.Order(StringComparer.Ordinal).ToArray(),
            actual.Order(StringComparer.Ordinal).ToArray());
    }

    private static string[] ExpectedModelHashes() =>
    [
        OcrV8ProductionCompositionFactory.DetectorSha256,
        OcrV11CandidateCompositionFactory.DetectorSha256,
        OcrV8ProductionCompositionFactory.OfficialRecognizerSha256,
        OcrV8ProductionCompositionFactory.NumericRecognizerSha256,
        OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256,
    ];

    private static double Ratio(
        IReadOnlyDictionary<string, (int Correct, int Total)> values,
        string key)
    {
        (int correct, int total) = values.GetValueOrDefault(key);
        return total == 0 ? 0 : correct / (double)total;
    }

    private static bool Enabled(string variable) =>
        string.Equals(Environment.GetEnvironmentVariable(variable), "1", StringComparison.Ordinal);

    private static T Deserialize<T>(string path) =>
        JsonSerializer.Deserialize<T>(File.ReadAllBytes(path), JsonOptions) ??
        throw new InvalidDataException($"Invalid JSON contract: {path}");

    private static string RequiredPath(string variable)
    {
        string? value = Environment.GetEnvironmentVariable(variable);
        Assert.IsFalse(string.IsNullOrWhiteSpace(value), $"Missing {variable}.");
        string path = Path.GetFullPath(value!);
        Assert.IsTrue(File.Exists(path), $"Missing required file for {variable}: {path}");
        return path;
    }

    private static string RequiredOutputPath(string variable)
    {
        string? value = Environment.GetEnvironmentVariable(variable);
        Assert.IsFalse(string.IsNullOrWhiteSpace(value), $"Missing {variable}.");
        string path = Path.GetFullPath(value!);
        Assert.IsFalse(File.Exists(path), $"Refusing to replace consumed P3 report: {path}");
        return path;
    }

    private static string RequiredSourceCommit()
    {
        string? value = Environment.GetEnvironmentVariable(SourceCommitVariable);
        Assert.IsTrue(value is { Length: 40 } && value.All(Uri.IsHexDigit),
            $"{SourceCommitVariable} must be an exact commit.");
        return value!;
    }

    private static void AssertHash(string path, string expected, string label) =>
        Assert.AreEqual(expected, Sha256(File.ReadAllBytes(path)), true, $"Hash changed: {label}");

    private static string Sha256(ReadOnlySpan<byte> payload) =>
        Convert.ToHexString(SHA256.HashData(payload)).ToLowerInvariant();

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "GraphAutoReader.slnx")))
            {
                return directory.FullName;
            }
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate the repository root.");
    }

    private sealed record SplitSeal(
        string Schema,
        string CandidateId,
        string SelectionArchivePath,
        string SelectionArchiveSha256,
        string SelectionManifestSha256,
        int SelectionSceneCount,
        int SelectionTruthRegionCount,
        string PublicArchivePath,
        string PublicArchiveSha256,
        string PublicManifestSha256,
        int PublicSceneCount,
        int PublicTruthRegionCount,
        IReadOnlyDictionary<string, string> SourceSha256,
        bool SecretSeedsSerialized,
        int ModelExecutionCountAtFreeze,
        int SelectionEvaluations,
        int PublicEvaluations,
        bool SelectionExecutionAuthorized,
        bool PublicExecutionAuthorized,
        bool MarkerCreationEvaluated,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record CandidateAuthorization(
        string Schema,
        string CandidateId,
        string Split,
        bool ExecutionAuthorized,
        int ExecutionCountAuthorized,
        string Provider,
        string SealedIdentityCommit,
        string SplitSealSha256,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        IReadOnlyList<string> CandidateSha256,
        string ExactTest,
        string ResultPath,
        string? SelectionResultSha256,
        bool RerunOrRepairAuthorized,
        bool MarkerStageAuthorized,
        bool ArtifactMaskProductionApproval,
        bool ManifestCreationAuthorized,
        bool ModelStorePromotionAuthorized,
        bool PrivateValidationAuthorized,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record SelectionResult(
        bool ExecutionConsumed,
        bool SelectionGatePassed,
        bool RerunOrRepairAuthorized);

    private sealed record FixtureManifest(
        string Schema,
        string Revision,
        string Split,
        int SceneCount,
        int TruthRegionCount,
        IReadOnlyList<string> RequiredRoles,
        IReadOnlyList<FixtureCase> Cases,
        bool SyntheticOnly,
        bool PrivateOrArticleImages,
        bool ChandlerIncluded,
        bool GeneralizationLabelIncluded);

    private sealed record FixtureCase(
        string SceneId,
        string ImagePath,
        string ImageSha256,
        string RasterSha256,
        string RendererFamily,
        string DegradationFamily,
        IReadOnlyList<FixtureTruth> TextTruths,
        int StructureCollisionCount);

    private sealed record FixtureTruth(
        string DisplayText,
        string TruthText,
        string Role,
        string Family,
        int[] Bbox);

    private sealed record DirectReport(
        string Schema,
        string CandidateId,
        string CompositionId,
        string Split,
        string SourceCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string AuthorizationSha256,
        string ConfigurationFingerprint,
        string Provider,
        IReadOnlyDictionary<string, OcrV8DirectPublicCorpusTests.ModelExecutionEvidence> ModelExecutions,
        DirectMetrics Metrics,
        bool DirectRuntimeEvidencePassed,
        bool GatePassed,
        bool FullEightRoleCoverageProven,
        bool MarkerCreationEvaluated,
        bool ArtifactMaskProductionApproval,
        bool ProductionApproval,
        bool ReleaseEligible,
        IReadOnlyList<string> BlockingGates);

    private sealed record DirectMetrics(
        int SceneCount,
        int TruthRegionCount,
        int ExactDetectionSceneCount,
        int TruePositives,
        int FalsePositives,
        int FalseNegatives,
        int DuplicateRegionCount,
        double RecognitionExactMatch,
        double CharacterErrorRate,
        double RoleAccuracy,
        IReadOnlyDictionary<string, double> PerRoleAccuracy,
        bool EveryRequiredRoleObserved,
        double NumericExactMatch,
        double WordExactMatch,
        double AmbiguityExactMatch,
        IReadOnlyDictionary<string, int> RoleConfusions,
        int ProhibitedStructureHits);

    private sealed class SingleCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }
}
