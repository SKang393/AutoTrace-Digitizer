// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV9P2DirectPublicCorpusTests
{
    private const string RunVariable = "GRAPHREADER_RUN_OCR_V9_P2_PUBLIC";
    private const string DetectorVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_DETECTOR";
    private const string OfficialVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_OFFICIAL";
    private const string NumericVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_NUMERIC";
    private const string AmbiguityVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_AMBIGUITY";
    private const string YamlVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_OFFICIAL_YAML";
    private const string ReportVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_REPORT";
    private const string SourceCommitVariable = "GRAPHREADER_OCR_V9_P2_PUBLIC_SOURCE_COMMIT";
    private const string InferenceYamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";
    private const string ExactTest =
        "OcrV9P2DirectPublicCorpusTests.FreshEightRolePublicBytesExecuteOnceThroughCSharpCpuCandidate";
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
    public async Task FreshEightRolePublicBytesExecuteOnceThroughCSharpCpuCandidate()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable(RunVariable), "1", StringComparison.Ordinal))
        {
            Assert.Inconclusive(
                $"Set {RunVariable}=1 only after the tracked one-run public authorization exists.");
        }

        string root = FindRepositoryRoot();
        string contractRoot = Path.Combine(
            root,
            "ml",
            "ocr",
            "selected_confidence_public_gate_v9_p2");
        string sealPath = Path.Combine(contractRoot, "SEALED_PUBLIC_TEST_SEAL.json");
        string authorizationPath = Path.Combine(contractRoot, "PUBLIC_GATE_AUTHORIZATION.json");
        Assert.IsTrue(File.Exists(sealPath), "The P2 public identity has not been frozen.");
        Assert.IsTrue(File.Exists(authorizationPath), "The P2 public execution is not authorized.");
        PublicSeal seal = Deserialize<PublicSeal>(sealPath);
        PublicAuthorization authorization = Deserialize<PublicAuthorization>(authorizationPath);
        ValidateAuthorization(root, sealPath, seal, authorization);

        string archivePath = Path.GetFullPath(Path.Combine(root, seal.FixtureArchivePath));
        Assert.IsTrue(File.Exists(archivePath), $"The frozen public archive is missing: {archivePath}");
        AssertHash(archivePath, authorization.FixtureArchiveSha256, "P2 public archive");
        string reportPath = RequiredOutputPath(ReportVariable);
        Assert.AreEqual(
            Path.GetFullPath(Path.Combine(root, authorization.ResultPath)),
            reportPath,
            true,
            "The P2 public report path is not authorization-bound.");
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
            "GraphReaderOcrV9P2Public",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(cacheRoot);
        try
        {
            await using var runtime = new InferenceRuntime(
                registry,
                new BoundedInferenceScheduler(capacity: 2, workerCount: 1),
                new ContentAddressedStageCache(cacheRoot));
            OcrV9P2CandidateCompositionPipeline pipeline = OcrV9P2CandidateCompositionFactory.Create(
                runtime,
                payloads,
                [InferenceProvider.Cpu],
                bypassCache: true);
            PublicReport report = await EvaluateAsync(
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

            Assert.IsTrue(report.PublicGatePassed, JsonSerializer.Serialize(report.Metrics));
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

    private static async Task<PublicReport> EvaluateAsync(
        string archivePath,
        PublicSeal seal,
        string authorizationPath,
        OcrV9P2CandidateCompositionPipeline pipeline,
        OcrV8DirectPublicCorpusTests.EvidenceInferenceSessionFactory evidenceFactory,
        string sourceCommit,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(archivePath);
        ZipArchiveEntry manifestEntry = archive.GetEntry("manifest.json") ??
            throw new InvalidDataException("P2 public archive has no manifest.json.");
        byte[] manifestBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(
            manifestEntry,
            cancellationToken);
        Assert.AreEqual(seal.FixtureManifestSha256, Sha256(manifestBytes), "Manifest bytes changed.");
        FixtureManifest manifest = JsonSerializer.Deserialize<FixtureManifest>(
            manifestBytes,
            JsonOptions) ??
            throw new InvalidDataException("P2 public manifest is invalid.");
        Assert.AreEqual("graphreader.ocr-selected-confidence-public-fixtures.v9-p2", manifest.Schema);
        Assert.AreEqual("sealed_public", manifest.Split);
        Assert.AreEqual(160, manifest.SceneCount);
        Assert.AreEqual(1280, manifest.TruthRegionCount);
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
                throw new InvalidDataException($"Fixture image is missing: {fixture.ImagePath}");
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
                    "ocr-v9-p2-eight-role-public",
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
                (int familyCorrect, int familyTotal) = familyCounts.GetValueOrDefault(truth.Family);
                familyCounts[truth.Family] = (
                    familyCorrect + (textMatch ? 1 : 0),
                    familyTotal + 1);
                (int roleCorrect, int roleTotal) = roleCounts.GetValueOrDefault(truth.Role);
                roleCounts[truth.Role] = (
                    roleCorrect + (roleMatch ? 1 : 0),
                    roleTotal + 1);
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
        var metrics = new PublicMetrics(
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
        bool runtimeEvidencePassed = executions.Count == 4 &&
            executions.Keys.Order(StringComparer.Ordinal).SequenceEqual(
                ExpectedModelHashes().Order(StringComparer.Ordinal),
                StringComparer.Ordinal) &&
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
        return new PublicReport(
            "graphreader.ocr-selected-confidence-public-report.v1",
            "P2",
            OcrV9P2CandidateCompositionPipeline.CandidateCompositionId,
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
            fullEightRoleCoverage,
            MarkerCreationEvaluated: false,
            ArtifactMaskProductionApproval: false,
            ProductionApproval: false,
            ReleaseEligible: false,
            BlockingGates:
            [
                "marker_stage_direct_composition_evidence",
                "approved_artifact_mask_provider",
                "approved_production_model_store",
                "packaging_discovery_and_clean_machine_evidence",
                "private_chandler_automatic_validation",
            ]);
    }

    private static void ValidateAuthorization(
        string root,
        string sealPath,
        PublicSeal seal,
        PublicAuthorization authorization)
    {
        Assert.AreEqual("graphreader.ocr-selected-confidence-public-seal.v1", seal.Schema);
        Assert.AreEqual("P2", seal.CandidateId);
        Assert.AreEqual(160, seal.SceneCount);
        Assert.AreEqual(1280, seal.TruthRegionCount);
        Assert.AreEqual(0, seal.ModelExecutionCountAtFreeze);
        Assert.IsFalse(seal.PublicExecutionAuthorized);
        Assert.AreEqual(0, seal.PublicGateEvaluations);
        Assert.IsFalse(seal.MarkerCreationEvaluated);
        Assert.IsFalse(seal.ProductionApproval);
        Assert.IsFalse(seal.ReleaseEligible);
        foreach ((string relativePath, string expectedHash) in seal.SourceSha256)
        {
            AssertHash(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)), expectedHash, relativePath);
        }

        Assert.AreEqual("graphreader.ocr-selected-confidence-public-authorization.v1", authorization.Schema);
        Assert.AreEqual("P2", authorization.CandidateId);
        Assert.IsTrue(authorization.ExecutionAuthorized);
        Assert.IsTrue(authorization.PublicGateAuthorized);
        Assert.AreEqual(1, authorization.ExecutionCountAuthorized);
        Assert.AreEqual("CPUExecutionProvider", authorization.Provider);
        Assert.IsTrue(
            authorization.SealedIdentityCommit.Length == 40 &&
            authorization.SealedIdentityCommit.All(Uri.IsHexDigit));
        Assert.AreEqual(seal.FixtureArchiveSha256, authorization.FixtureArchiveSha256);
        Assert.AreEqual(seal.FixtureManifestSha256, authorization.FixtureManifestSha256);
        Assert.AreEqual(Sha256(File.ReadAllBytes(sealPath)), authorization.PublicSealSha256);
        CollectionAssert.AreEqual(
            ExpectedModelHashes().Order(StringComparer.Ordinal).ToArray(),
            authorization.CandidateSha256.Order(StringComparer.Ordinal).ToArray());
        Assert.AreEqual(ExactTest, authorization.ExactTest);
        Assert.IsFalse(authorization.RerunOrRepairAuthorized);
        Assert.IsFalse(authorization.MarkerStageAuthorized);
        Assert.IsFalse(authorization.ArtifactMaskProductionApproval);
        Assert.IsFalse(authorization.ManifestCreationAuthorized);
        Assert.IsFalse(authorization.ModelStorePromotionAuthorized);
        Assert.IsFalse(authorization.PrivateValidationAuthorized);
        Assert.IsFalse(authorization.ProductionApproval);
        Assert.IsFalse(authorization.ReleaseEligible);
    }

    private static void ValidateCandidateHashes(
        PublicAuthorization authorization,
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
        string key)
    {
        (int correct, int total) = values.GetValueOrDefault(key);
        return total == 0 ? 0 : correct / (double)total;
    }

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
        Assert.IsFalse(File.Exists(path), $"Refusing to replace consumed public report: {path}");
        return path;
    }

    private static string RequiredSourceCommit()
    {
        string? value = Environment.GetEnvironmentVariable(SourceCommitVariable);
        Assert.IsTrue(
            value is { Length: 40 } && value.All(Uri.IsHexDigit),
            $"{SourceCommitVariable} must be an exact 40-character commit.");
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

    private sealed record PublicSeal(
        string Schema,
        string CandidateId,
        string FixtureArchivePath,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        int SceneCount,
        int TruthRegionCount,
        IReadOnlyDictionary<string, string> SourceSha256,
        string P2SelectionResultSha256,
        int ModelExecutionCountAtFreeze,
        bool PublicExecutionAuthorized,
        int PublicGateEvaluations,
        bool MarkerCreationEvaluated,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record PublicAuthorization(
        string Schema,
        string CandidateId,
        bool ExecutionAuthorized,
        bool PublicGateAuthorized,
        int ExecutionCountAuthorized,
        string Provider,
        string SealedIdentityCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string PublicSealSha256,
        IReadOnlyList<string> CandidateSha256,
        string ExactTest,
        string ResultPath,
        bool RerunOrRepairAuthorized,
        bool MarkerStageAuthorized,
        bool ArtifactMaskProductionApproval,
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

    private sealed record PublicReport(
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
        PublicMetrics Metrics,
        bool DirectRuntimeEvidencePassed,
        bool PublicGatePassed,
        bool FullEightRoleCoverageProven,
        bool MarkerCreationEvaluated,
        bool ArtifactMaskProductionApproval,
        bool ProductionApproval,
        bool ReleaseEligible,
        IReadOnlyList<string> BlockingGates);

    private sealed record PublicMetrics(
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
