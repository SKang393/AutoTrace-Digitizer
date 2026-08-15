// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV9CandidateSelectionTests
{
    private const string RunVariable = "GRAPHREADER_RUN_OCR_V9_P1_SELECTION";
    private const string DetectorVariable = "GRAPHREADER_OCR_V9_DETECTOR";
    private const string OfficialVariable = "GRAPHREADER_OCR_V9_OFFICIAL";
    private const string NumericVariable = "GRAPHREADER_OCR_V9_NUMERIC";
    private const string AmbiguityVariable = "GRAPHREADER_OCR_V9_AMBIGUITY";
    private const string YamlVariable = "GRAPHREADER_OCR_V9_OFFICIAL_YAML";
    private const string ReportVariable = "GRAPHREADER_OCR_V9_P1_SELECTION_REPORT";
    private const string SourceCommitVariable = "GRAPHREADER_OCR_V9_P1_SOURCE_COMMIT";
    private const string InferenceYamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";
    private const string ExactTest =
        "OcrV9CandidateSelectionTests.FreshVisibleSelectionExecutesOnceThroughCSharpCpuCandidate";
    private static readonly OcrRectangle PlotBounds = new(104, 48, 406, 208);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    [TestMethod]
    public async Task FreshVisibleSelectionExecutesOnceThroughCSharpCpuCandidate()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable(RunVariable), "1", StringComparison.Ordinal))
        {
            Assert.Inconclusive(
                $"Set {RunVariable}=1 only after the tracked one-run selection authorization exists.");
        }

        string root = FindRepositoryRoot();
        string sealPath = Path.Combine(
            root,
            "ml",
            "ocr",
            "recognizer_confirmed_acceptance_v9",
            "SELECTION_SEAL.json");
        string authorizationPath = Path.Combine(
            root,
            "ml",
            "ocr",
            "recognizer_confirmed_acceptance_v9",
            "SELECTION_AUTHORIZATION.json");
        Assert.IsTrue(File.Exists(sealPath), "The V9 selection identity has not been frozen.");
        Assert.IsTrue(File.Exists(authorizationPath), "The V9 selection execution is not authorized.");
        SelectionSeal seal = Deserialize<SelectionSeal>(sealPath);
        SelectionAuthorization authorization = Deserialize<SelectionAuthorization>(authorizationPath);
        ValidateAuthorization(sealPath, seal, authorization);

        string archivePath = Path.GetFullPath(Path.Combine(root, seal.FixtureArchivePath));
        Assert.IsTrue(File.Exists(archivePath), $"The frozen selection archive is missing: {archivePath}");
        AssertHash(archivePath, authorization.FixtureArchiveSha256, "selection archive");
        string reportPath = RequiredOutputPath(ReportVariable);
        string expectedReportPath = Path.GetFullPath(Path.Combine(root, authorization.ResultPath));
        Assert.AreEqual(expectedReportPath, reportPath, true, "The report path is not authorization-bound.");
        string sourceCommit = RequiredSourceCommit();
        string yamlPath = RequiredPath(YamlVariable);
        AssertHash(yamlPath, InferenceYamlSha256, "official inference YAML");

        OcrV8ProductionPayloadSet payloads = Payloads(
            RequiredPath(DetectorVariable),
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
        string cacheRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderOcrV9P1Selection",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(cacheRoot);
        try
        {
            await using var runtime = new InferenceRuntime(
                registry,
                new BoundedInferenceScheduler(capacity: 2, workerCount: 1),
                new ContentAddressedStageCache(cacheRoot));
            OcrV8ProductionCompositionPipeline pipeline = OcrV9CandidateCompositionFactory.Create(
                runtime,
                payloads,
                [InferenceProvider.Cpu],
                bypassCache: true);

            SelectionReport report = await EvaluateAsync(
                archivePath,
                seal,
                authorizationPath,
                pipeline,
                evidenceFactory,
                sourceCommit,
                CancellationToken.None);
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            await File.WriteAllBytesAsync(
                reportPath,
                JsonSerializer.SerializeToUtf8Bytes(report, JsonOptions));

            Assert.IsTrue(report.SelectionGatesPassed, JsonSerializer.Serialize(report.Metrics));
            Assert.IsFalse(report.FullEightRoleCoverageProven);
            Assert.IsFalse(report.MarkerCreationEvaluated);
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

    private static async Task<SelectionReport> EvaluateAsync(
        string archivePath,
        SelectionSeal seal,
        string authorizationPath,
        OcrV8ProductionCompositionPipeline pipeline,
        OcrV8DirectPublicCorpusTests.EvidenceInferenceSessionFactory evidenceFactory,
        string sourceCommit,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(archivePath);
        ZipArchiveEntry manifestEntry = archive.GetEntry("manifest.json") ??
            throw new InvalidDataException("V9 selection archive has no manifest.json.");
        byte[] manifestBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(
            manifestEntry,
            cancellationToken);
        Assert.AreEqual(seal.FixtureManifestSha256, Sha256(manifestBytes), "Selection manifest changed.");
        FixtureManifest manifest = JsonSerializer.Deserialize<FixtureManifest>(manifestBytes, JsonOptions) ??
            throw new InvalidDataException("V9 selection manifest is invalid.");
        Assert.AreEqual("graphreader.ocr-recognizer-confirmed-selection-fixtures.v9", manifest.Schema);
        Assert.AreEqual("graphreader-v10-recognizer-confirmed-acceptance-v9", manifest.Revision);
        Assert.AreEqual("visible_selection", manifest.Split);
        Assert.AreEqual(96, manifest.SceneCount);
        Assert.AreEqual(480, manifest.TruthRegionCount);
        Assert.HasCount(manifest.SceneCount, manifest.Cases);
        Assert.IsTrue(manifest.SyntheticOnly);
        Assert.IsFalse(manifest.PrivateOrArticleImages);
        Assert.IsFalse(manifest.ChandlerIncluded);
        Assert.IsFalse(manifest.GeneralizationLabelIncluded);
        Assert.IsFalse(manifest.PredecessorFixtureBytesReused);
        Assert.IsFalse(manifest.PredecessorTruthOrSceneIdsReused);

        var exactScenes = 0;
        var truePositives = 0;
        var falsePositives = 0;
        var falseNegatives = 0;
        var duplicates = 0;
        var correctText = 0;
        var correctRole = 0;
        var characterCount = 0;
        var editErrors = 0;
        var familyCounts = new Dictionary<string, (int Correct, int Total)>(StringComparer.Ordinal);
        var roleConfusions = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (FixtureCase fixture in manifest.Cases)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.AreEqual(3, fixture.StructureCollisionCount);
            ZipArchiveEntry imageEntry = archive.GetEntry(fixture.ImagePath) ??
                throw new InvalidDataException($"V9 selection image is missing: {fixture.ImagePath}");
            byte[] sourceBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(
                imageEntry,
                cancellationToken);
            Assert.AreEqual(fixture.ImageSha256, Sha256(sourceBytes), $"PNG changed: {fixture.SceneId}");
            (byte[] pixels, int width, int height) =
                OcrV8DirectPublicCorpusTests.DecodeGray8(sourceBytes);
            Assert.AreEqual(640, width);
            Assert.AreEqual(320, height);
            Assert.AreEqual(fixture.RasterSha256, Sha256(pixels), $"Raster changed: {fixture.SceneId}");
            var image = new OcrImage(
                width,
                height,
                width,
                pixels,
                OcrSourceImage.Original,
                OcrFrameTransform.Identity,
                CanonicalOriginalWidth: width,
                CanonicalOriginalHeight: height);
            OcrResult result = await pipeline.RecognizeAsync(
                new OcrRequest(
                    "ocr-v9-p1-visible-selection",
                    fixture.SceneId,
                    fixture.ImageSha256,
                    image,
                    PlotBounds),
                cancellationToken);
            Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);

            HashSet<int> matchedTruths = [];
            var sceneFalsePositives = 0;
            var sceneDuplicates = 0;
            foreach (OcrRegion region in result.Regions)
            {
                (int Index, double Iou)[] matches = fixture.TextTruths
                    .Select((truth, index) => (
                        Index: index,
                        Iou: OcrV8DirectPublicCorpusTests.IntersectionOverUnion(
                            region.Polygon.Bounds,
                            truth.Bbox)))
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
                editErrors += OcrV8DirectPublicCorpusTests.LevenshteinDistance(
                    truth.TruthText,
                    region.Text);
                (int correct, int total) = familyCounts.GetValueOrDefault(truth.Family);
                familyCounts[truth.Family] = (correct + (textMatch ? 1 : 0), total + 1);
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

        var metrics = new SelectionMetrics(
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
            Ratio(familyCounts, "numeric"),
            Ratio(familyCounts, "word"),
            Ratio(familyCounts, "ambiguity"),
            roleConfusions,
            falsePositives);
        IReadOnlyDictionary<string, OcrV8DirectPublicCorpusTests.ModelExecutionEvidence> executions =
            evidenceFactory.Snapshot();
        bool runtimeEvidencePassed = executions.Count == 4 &&
            executions.Keys.Order(StringComparer.Ordinal).SequenceEqual(
                ExpectedModelHashes().Order(StringComparer.Ordinal),
                StringComparer.Ordinal) &&
            executions.Values.All(static item =>
                item.CallCount > 0 &&
                item.InputTensorSha256.Count == item.CallCount &&
                item.OutputTensorSha256.Count == item.CallCount &&
                item.Providers.SequenceEqual(["Cpu"], StringComparer.Ordinal));
        bool gatesPassed = exactScenes == manifest.SceneCount &&
            truePositives == manifest.TruthRegionCount &&
            falsePositives == 0 && falseNegatives == 0 && duplicates == 0 &&
            metrics.RecognitionExactMatch >= 0.90 &&
            metrics.CharacterErrorRate <= 0.05 &&
            metrics.RoleAccuracy >= 0.90 &&
            metrics.NumericExactMatch >= 0.90 &&
            metrics.WordExactMatch >= 0.90 &&
            metrics.AmbiguityExactMatch >= 0.90 &&
            runtimeEvidencePassed;
        return new SelectionReport(
            "graphreader.ocr-recognizer-confirmed-selection-report.v1",
            "P1",
            OcrV9CandidateCompositionFactory.CandidateCompositionId,
            sourceCommit,
            seal.FixtureArchiveSha256,
            seal.FixtureManifestSha256,
            Sha256(File.ReadAllBytes(authorizationPath)),
            pipeline.ConfigurationFingerprint,
            "CPUExecutionProvider",
            executions,
            metrics,
            runtimeEvidencePassed,
            gatesPassed,
            FullEightRoleCoverageProven: false,
            MarkerCreationEvaluated: false,
            ProductionApproval: false,
            ReleaseEligible: false,
            BlockingGates:
            [
                "fresh_truth_hidden_public_eight_role_gate",
                "marker_stage_direct_composition_evidence",
                "approved_artifact_mask_provider",
                "approved_production_model_store",
                "packaging_discovery_and_clean_machine_evidence",
                "private_chandler_automatic_validation",
            ]);
    }

    private static void ValidateAuthorization(
        string sealPath,
        SelectionSeal seal,
        SelectionAuthorization authorization)
    {
        Assert.AreEqual("graphreader.ocr-recognizer-confirmed-selection-seal.v1", seal.Schema);
        Assert.AreEqual("P1", seal.CandidateId);
        Assert.AreEqual(96, seal.SceneCount);
        Assert.AreEqual(480, seal.TruthRegionCount);
        Assert.AreEqual(0, seal.ModelExecutionCountAtFreeze);
        Assert.IsFalse(seal.SelectionExecutionAuthorized);
        Assert.IsFalse(seal.PublicGateAuthorized);
        Assert.IsFalse(seal.ProductionApproval);
        Assert.IsFalse(seal.ReleaseEligible);
        Assert.AreEqual("graphreader.ocr-recognizer-confirmed-selection-authorization.v1", authorization.Schema);
        Assert.AreEqual("P1", authorization.CandidateId);
        Assert.IsTrue(authorization.ExecutionAuthorized);
        Assert.AreEqual(1, authorization.ExecutionCountAuthorized);
        Assert.AreEqual("CPUExecutionProvider", authorization.Provider);
        Assert.IsTrue(
            authorization.SealedIdentityCommit.Length == 40 &&
            authorization.SealedIdentityCommit.All(Uri.IsHexDigit),
            "The authorization must bind the exact committed selection identity.");
        Assert.AreEqual(seal.FixtureArchiveSha256, authorization.FixtureArchiveSha256);
        Assert.AreEqual(seal.FixtureManifestSha256, authorization.FixtureManifestSha256);
        Assert.AreEqual(Sha256(File.ReadAllBytes(sealPath)), authorization.SplitSealSha256);
        CollectionAssert.AreEqual(
            ExpectedModelHashes().Order(StringComparer.Ordinal).ToArray(),
            authorization.CandidateSha256.Order(StringComparer.Ordinal).ToArray());
        Assert.AreEqual(ExactTest, authorization.ExactTest);
        Assert.IsFalse(authorization.RerunOrRepairAuthorized);
        Assert.IsFalse(authorization.PublicGateAuthorized);
        Assert.IsFalse(authorization.ManifestCreationAuthorized);
        Assert.IsFalse(authorization.ModelStorePromotionAuthorized);
        Assert.IsFalse(authorization.PrivateValidationAuthorized);
        Assert.IsFalse(authorization.ProductionApproval);
        Assert.IsFalse(authorization.ReleaseEligible);
    }

    private static void ValidateCandidateHashes(
        SelectionAuthorization authorization,
        OcrV8ProductionPayloadSet payloads)
    {
        string[] actual =
        [
            payloads.Detector.Sha256,
            payloads.OfficialRecognizer.Sha256,
            payloads.NumericRecognizer.Sha256,
            payloads.AmbiguityRecognizer.Sha256,
        ];
        CollectionAssert.AreEqual(
            authorization.CandidateSha256.Order(StringComparer.Ordinal).ToArray(),
            actual.Order(StringComparer.Ordinal).ToArray());
    }

    private static OcrV8ProductionPayloadSet Payloads(
        string detector,
        string official,
        string numeric,
        string ambiguity,
        string alphabet) =>
        new(
            new ModelIdentity(
                "graph-text-spaced-component-recall-v10-p2",
                "0.0.21-p2",
                OcrV8ProductionCompositionFactory.DetectorSha256,
                detector),
            new ModelIdentity(
                "en_PP-OCRv5_mobile_rec",
                "0.0.21-converted",
                OcrV8ProductionCompositionFactory.OfficialRecognizerSha256,
                official),
            new ModelIdentity(
                "graph-numeric-component-ensemble-v5",
                "0.0.21-p1",
                OcrV8ProductionCompositionFactory.NumericRecognizerSha256,
                numeric),
            new ModelIdentity(
                "graph-ambiguity-source-group-v3-p2",
                "0.0.21-p2",
                OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256,
                ambiguity),
            alphabet);

    private static string[] ExpectedModelHashes() =>
    [
        OcrV8ProductionCompositionFactory.DetectorSha256,
        OcrV8ProductionCompositionFactory.OfficialRecognizerSha256,
        OcrV8ProductionCompositionFactory.NumericRecognizerSha256,
        OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256,
    ];

    private static double Ratio(
        IReadOnlyDictionary<string, (int Correct, int Total)> values,
        string family)
    {
        (int correct, int total) = values.GetValueOrDefault(family);
        return total == 0 ? 0 : correct / (double)total;
    }

    private static T Deserialize<T>(string path) =>
        JsonSerializer.Deserialize<T>(File.ReadAllBytes(path), JsonOptions) ??
        throw new InvalidDataException($"Invalid JSON contract: {path}");

    private static string RequiredPath(string variable)
    {
        string value = Environment.GetEnvironmentVariable(variable) ?? string.Empty;
        Assert.IsFalse(string.IsNullOrWhiteSpace(value), $"{variable} is required.");
        string path = Path.GetFullPath(value);
        Assert.IsTrue(File.Exists(path), $"{variable} does not exist: {path}");
        return path;
    }

    private static string RequiredOutputPath(string variable)
    {
        string value = Environment.GetEnvironmentVariable(variable) ?? string.Empty;
        Assert.IsFalse(string.IsNullOrWhiteSpace(value), $"{variable} is required.");
        string path = Path.GetFullPath(value);
        Assert.IsFalse(File.Exists(path), $"The one-time V9 P1 report already exists: {path}");
        return path;
    }

    private static string RequiredSourceCommit()
    {
        string value = Environment.GetEnvironmentVariable(SourceCommitVariable) ?? string.Empty;
        Assert.IsTrue(
            value.Length == 40 && value.All(Uri.IsHexDigit),
            $"{SourceCommitVariable} must be an exact committed source identity.");
        return value.ToLowerInvariant();
    }

    private static void AssertHash(string path, string expected, string label) =>
        Assert.AreEqual(expected, Sha256(File.ReadAllBytes(path)), $"{label} checksum changed.");

    private static string Sha256(ReadOnlySpan<byte> bytes) =>
        Convert.ToHexStringLower(SHA256.HashData(bytes));

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

        throw new DirectoryNotFoundException("Could not locate the Graph Auto Reader repository root.");
    }

    private sealed record SelectionSeal(
        string Schema,
        string CandidateId,
        string FixtureArchivePath,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        int SceneCount,
        int TruthRegionCount,
        int ModelExecutionCountAtFreeze,
        bool SelectionExecutionAuthorized,
        bool PublicGateAuthorized,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record SelectionAuthorization(
        string Schema,
        string CandidateId,
        bool ExecutionAuthorized,
        int ExecutionCountAuthorized,
        string Provider,
        string SealedIdentityCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string SplitSealSha256,
        IReadOnlyList<string> CandidateSha256,
        string ExactTest,
        string ResultPath,
        bool RerunOrRepairAuthorized,
        bool PublicGateAuthorized,
        bool ManifestCreationAuthorized,
        bool ModelStorePromotionAuthorized,
        bool PrivateValidationAuthorized,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record FixtureManifest(
        string Schema,
        string Revision,
        string Split,
        int SceneCount,
        int TruthRegionCount,
        bool SyntheticOnly,
        bool PrivateOrArticleImages,
        bool ChandlerIncluded,
        bool GeneralizationLabelIncluded,
        bool PredecessorFixtureBytesReused,
        bool PredecessorTruthOrSceneIdsReused,
        IReadOnlyList<FixtureCase> Cases);

    private sealed record FixtureCase(
        string SceneId,
        string ImagePath,
        string ImageSha256,
        string RasterSha256,
        int StructureCollisionCount,
        IReadOnlyList<FixtureTruth> TextTruths);

    private sealed record FixtureTruth(
        int[] Bbox,
        string TruthText,
        string DisplayText,
        string Role,
        string Family);

    private sealed record SelectionReport(
        string Schema,
        string CandidateId,
        string CompositionId,
        string SourceCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string AuthorizationSha256,
        string ConfigurationFingerprint,
        string Provider,
        IReadOnlyDictionary<string, OcrV8DirectPublicCorpusTests.ModelExecutionEvidence> ModelExecutions,
        SelectionMetrics Metrics,
        bool DirectRuntimeEvidencePassed,
        bool SelectionGatesPassed,
        bool FullEightRoleCoverageProven,
        bool MarkerCreationEvaluated,
        bool ProductionApproval,
        bool ReleaseEligible,
        IReadOnlyList<string> BlockingGates);

    private sealed record SelectionMetrics(
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
