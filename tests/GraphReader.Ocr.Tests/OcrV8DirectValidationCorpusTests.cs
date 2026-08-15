// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV8DirectValidationCorpusTests
{
    private const string RunVariable = "GRAPHREADER_RUN_OCR_V8_CSHARP_VALIDATION";
    private const string ArchiveVariable = "GRAPHREADER_OCR_V8_VALIDATION_FIXTURES";
    private const string DetectorVariable = "GRAPHREADER_OCR_V8_DETECTOR";
    private const string OfficialVariable = "GRAPHREADER_OCR_V8_OFFICIAL";
    private const string NumericVariable = "GRAPHREADER_OCR_V8_NUMERIC";
    private const string AmbiguityVariable = "GRAPHREADER_OCR_V8_AMBIGUITY";
    private const string YamlVariable = "GRAPHREADER_OCR_V8_OFFICIAL_YAML";
    private const string ReportVariable = "GRAPHREADER_OCR_V8_CSHARP_VALIDATION_REPORT";
    private const string SourceCommitVariable = "GRAPHREADER_OCR_V8_VALIDATION_SOURCE_COMMIT";
    private const string ArchiveSha256 =
        "fe2807579994d97063342e130bc9672fdf8dc6df6efc1e0a9e1e1a5f64bcf40f";
    private const string ManifestSha256 =
        "221659fa80abc1b6ba27de51cf2c8a55ce5b4ca575a2dc8e35c04c9e69c5ea2b";
    private const string InferenceYamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";
    private static readonly OcrRectangle PlotBounds = new(104, 48, 406, 208);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    [TestMethod]
    public async Task ExactVisibleValidationBytesExecuteThroughCSharpCpuComposition()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable(RunVariable), "1", StringComparison.Ordinal))
        {
            Assert.Inconclusive(
                $"Set {RunVariable}=1 plus the exact validation, payload, YAML, report, and source paths.");
        }

        string archivePath = RequiredPath(ArchiveVariable);
        Assert.AreEqual(
            ArchiveSha256,
            OcrV8DirectPublicCorpusTests.Sha256(File.ReadAllBytes(archivePath)),
            "Visible validation archive changed.");
        string reportPath = RequiredValue(ReportVariable);
        reportPath = Path.GetFullPath(reportPath);
        string sourceCommit = RequiredSourceCommit();
        string yamlPath = RequiredPath(YamlVariable);
        Assert.AreEqual(
            InferenceYamlSha256,
            OcrV8DirectPublicCorpusTests.Sha256(File.ReadAllBytes(yamlPath)),
            "Official inference YAML changed.");
        OcrV8ProductionPayloadSet payloads = Payloads(
            RequiredPath(DetectorVariable),
            RequiredPath(OfficialVariable),
            RequiredPath(NumericVariable),
            RequiredPath(AmbiguityVariable),
            OcrV8DirectPublicCorpusTests.ReadOfficialAlphabet(yamlPath));

        var evidenceFactory = new OcrV8DirectPublicCorpusTests.EvidenceInferenceSessionFactory(
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance));
        var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            evidenceFactory,
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        string cacheRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderOcrV8CSharpValidation",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(cacheRoot);
        try
        {
            await using var runtime = new InferenceRuntime(
                registry,
                new BoundedInferenceScheduler(capacity: 2, workerCount: 1),
                new ContentAddressedStageCache(cacheRoot));
            OcrV8ProductionCompositionPipeline pipeline =
                OcrV8ProductionCompositionFactory.Create(
                    runtime,
                    payloads,
                    [InferenceProvider.Cpu],
                    bypassCache: true);

            ValidationReport report = await EvaluateAsync(
                archivePath,
                pipeline,
                evidenceFactory,
                sourceCommit,
                CancellationToken.None);
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            await File.WriteAllBytesAsync(
                reportPath,
                JsonSerializer.SerializeToUtf8Bytes(report, JsonOptions));

            Assert.IsTrue(report.ValidationGatesPassed, JsonSerializer.Serialize(report.Metrics));
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

    private static async Task<ValidationReport> EvaluateAsync(
        string archivePath,
        OcrV8ProductionCompositionPipeline pipeline,
        OcrV8DirectPublicCorpusTests.EvidenceInferenceSessionFactory evidenceFactory,
        string sourceCommit,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(archivePath);
        ZipArchiveEntry manifestEntry = archive.GetEntry("manifest.json") ??
            throw new InvalidDataException("Validation archive has no manifest.json.");
        byte[] manifestBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(
            manifestEntry,
            cancellationToken);
        Assert.AreEqual(
            ManifestSha256,
            OcrV8DirectPublicCorpusTests.Sha256(manifestBytes),
            "Visible validation manifest changed.");
        FixtureManifest manifest = JsonSerializer.Deserialize<FixtureManifest>(manifestBytes, JsonOptions) ??
            throw new InvalidDataException("Validation manifest is invalid.");
        Assert.AreEqual("graphreader.ocr-production-composition-fixtures.v8", manifest.Schema);
        Assert.AreEqual("validation", manifest.Split);
        Assert.AreEqual(128, manifest.SceneCount);
        Assert.AreEqual(640, manifest.TruthRegionCount);
        Assert.IsTrue(manifest.SyntheticOnly);
        Assert.IsFalse(manifest.PrivateOrArticleImages);
        Assert.IsFalse(manifest.ChandlerIncluded);
        Assert.IsFalse(manifest.GeneralizationLabelIncluded);

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
        var roleConfusionDiagnostics = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (FixtureCase fixture in manifest.Cases)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ZipArchiveEntry imageEntry = archive.GetEntry(fixture.ImagePath) ??
                throw new InvalidDataException($"Validation image is missing: {fixture.ImagePath}");
            byte[] sourceBytes = await OcrV8DirectPublicCorpusTests.ReadEntryAsync(
                imageEntry,
                cancellationToken);
            Assert.AreEqual(
                fixture.ImageSha256,
                OcrV8DirectPublicCorpusTests.Sha256(sourceBytes),
                $"PNG changed: {fixture.SceneId}");
            (byte[] pixels, int width, int height) =
                OcrV8DirectPublicCorpusTests.DecodeGray8(sourceBytes);
            Assert.AreEqual(
                fixture.RasterSha256,
                OcrV8DirectPublicCorpusTests.Sha256(pixels),
                $"Raster changed: {fixture.SceneId}");
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
                    "ocr-v8-direct-validation",
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
                (int Index, double Iou)[] matches = fixture.Truths
                    .Select((truth, index) => (
                        Index: index,
                        Iou: OcrV8DirectPublicCorpusTests.IntersectionOverUnion(
                            region.Polygon.Bounds,
                            truth.Box)))
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

                FixtureTruth truth = fixture.Truths[truthIndex];
                bool textMatch = string.Equals(region.Text, truth.TruthText, StringComparison.Ordinal);
                bool roleMatch = string.Equals(
                    OcrV8DirectPublicCorpusTests.RoleName(region.Role),
                    truth.Role,
                    StringComparison.Ordinal);
                if (!roleMatch)
                {
                    string predictedRole = OcrV8DirectPublicCorpusTests.RoleName(region.Role);
                    string key = $"{truth.Role}->{predictedRole}";
                    roleConfusions[key] = roleConfusions.GetValueOrDefault(key) + 1;
                    OcrPoint center = region.Polygon.Bounds.Center;
                    bool insidePlot = center.X >= PlotBounds.Left && center.X <= PlotBounds.Right &&
                        center.Y >= PlotBounds.Top && center.Y <= PlotBounds.Bottom;
                    bool numeric = GraphNumericParser.Parse(region.Text).IsSuccess;
                    string diagnostic = $"{key}:inside_plot={insidePlot}:numeric={numeric}";
                    roleConfusionDiagnostics[diagnostic] =
                        roleConfusionDiagnostics.GetValueOrDefault(diagnostic) + 1;
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

            int sceneFalseNegatives = fixture.Truths.Count - matchedTruths.Count;
            if (sceneFalsePositives == 0 && sceneFalseNegatives == 0 && sceneDuplicates == 0 &&
                result.Regions.Count == fixture.Truths.Count)
            {
                exactScenes++;
            }

            truePositives += matchedTruths.Count;
            falsePositives += sceneFalsePositives;
            falseNegatives += sceneFalseNegatives;
            duplicates += sceneDuplicates;
        }

        var metrics = new ValidationMetrics(
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
            roleConfusionDiagnostics,
            falsePositives);
        bool passed = exactScenes == manifest.SceneCount &&
            truePositives == manifest.TruthRegionCount &&
            falsePositives == 0 && falseNegatives == 0 && duplicates == 0 &&
            metrics.RecognitionExactMatch >= 0.90 &&
            metrics.CharacterErrorRate <= 0.05 &&
            metrics.RoleAccuracy >= 0.90 &&
            metrics.NumericExactMatch >= 0.90 &&
            metrics.WordExactMatch >= 0.90 &&
            metrics.AmbiguityExactMatch >= 0.90;
        Dictionary<string, OcrV8DirectPublicCorpusTests.ModelExecutionEvidence> modelExecutions =
            evidenceFactory.Snapshot();
        bool directRuntimeEvidencePassed = modelExecutions.Count == 4 &&
            modelExecutions.Keys.Order(StringComparer.Ordinal).SequenceEqual(
                new[]
                {
                    OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256,
                    OcrV8ProductionCompositionFactory.DetectorSha256,
                    OcrV8ProductionCompositionFactory.NumericRecognizerSha256,
                    OcrV8ProductionCompositionFactory.OfficialRecognizerSha256,
                }.Order(StringComparer.Ordinal),
                StringComparer.Ordinal) &&
            modelExecutions.Values.All(static item =>
                item.CallCount > 0 &&
                item.InputTensorSha256.Count == item.CallCount &&
                item.OutputTensorSha256.Count == item.CallCount &&
                item.Providers.SequenceEqual(["Cpu"], StringComparer.Ordinal));
        passed &= directRuntimeEvidencePassed;
        return new ValidationReport(
            "graphreader.ocr-production-composition-csharp-validation.v1",
            OcrV8ProductionCompositionOptions.ReviewedCompositionId,
            sourceCommit,
            ArchiveSha256,
            ManifestSha256,
            pipeline.ConfigurationFingerprint,
            "CPUExecutionProvider",
            modelExecutions,
            metrics,
            passed,
            MarkerCreationEvaluated: false,
            ProductionApproval: false,
            ReleaseEligible: false,
            BlockingGates:
            [
                "fresh_direct_csharp_public_composition_evidence",
                "marker_stage_direct_composition_evidence",
                "approved_production_model_store",
                "packaging_discovery_and_clean_machine_evidence",
                "private_chandler_automatic_validation",
            ]);
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

    private static double Ratio(
        IReadOnlyDictionary<string, (int Correct, int Total)> values,
        string family)
    {
        (int correct, int total) = values.GetValueOrDefault(family);
        return total == 0 ? 0 : correct / (double)total;
    }

    private static string RequiredPath(string variable)
    {
        string path = Path.GetFullPath(RequiredValue(variable));
        Assert.IsTrue(File.Exists(path), $"{variable} does not exist: {path}");
        return path;
    }

    private static string RequiredValue(string variable)
    {
        string value = Environment.GetEnvironmentVariable(variable) ?? string.Empty;
        Assert.IsFalse(string.IsNullOrWhiteSpace(value), $"{variable} is required.");
        return value;
    }

    private static string RequiredSourceCommit()
    {
        string value = RequiredValue(SourceCommitVariable);
        Assert.IsTrue(
            value.Length == 40 && value.All(Uri.IsHexDigit),
            $"{SourceCommitVariable} must be the exact committed source identity.");
        return value.ToLowerInvariant();
    }

    private sealed record FixtureManifest(
        string Schema,
        string Split,
        int SceneCount,
        int TruthRegionCount,
        bool SyntheticOnly,
        bool PrivateOrArticleImages,
        bool ChandlerIncluded,
        bool GeneralizationLabelIncluded,
        IReadOnlyList<FixtureCase> Cases);

    private sealed record FixtureCase(
        string SceneId,
        string ImagePath,
        string ImageSha256,
        string RasterSha256,
        IReadOnlyList<FixtureTruth> Truths);

    private sealed record FixtureTruth(
        int[] Bbox,
        string TruthText,
        string Role,
        string Family)
    {
        [JsonIgnore]
        public int[] Box => Bbox;
    }

    private sealed record ValidationReport(
        string Schema,
        string CompositionId,
        string SourceCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string ConfigurationFingerprint,
        string Provider,
        IReadOnlyDictionary<string, OcrV8DirectPublicCorpusTests.ModelExecutionEvidence> ModelExecutions,
        ValidationMetrics Metrics,
        bool ValidationGatesPassed,
        bool MarkerCreationEvaluated,
        bool ProductionApproval,
        bool ReleaseEligible,
        IReadOnlyList<string> BlockingGates);

    private sealed record ValidationMetrics(
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
        IReadOnlyDictionary<string, int> RoleConfusionDiagnostics,
        int ProhibitedStructureHits);

    private sealed class SingleCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }
}
