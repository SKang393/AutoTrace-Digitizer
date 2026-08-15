// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using GraphReader.Inference;
using GraphReader.Markers.Detection;
using GraphReader.Ocr;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

/// <summary>
/// One-time direct gate over new sealed public bytes. This is candidate
/// composition evidence only. It cannot approve manifests, the production
/// model store, the fixture-authored artifact mask, packaging, or release.
/// </summary>
[TestClass]
public sealed class OcrMarkerDirectPublicGateV4Tests
{
    private const string RunVariable = "GRAPHREADER_RUN_OCR_MARKER_CSHARP_PUBLIC_V4";
    private const string ArchiveVariable = "GRAPHREADER_OCR_MARKER_PUBLIC_FIXTURES_V4";
    private const string DetectorVariable = "GRAPHREADER_OCR_MARKER_DETECTOR_V4";
    private const string OfficialVariable = "GRAPHREADER_OCR_MARKER_OFFICIAL_V4";
    private const string NumericVariable = "GRAPHREADER_OCR_MARKER_NUMERIC_V4";
    private const string AmbiguityVariable = "GRAPHREADER_OCR_MARKER_AMBIGUITY_V4";
    private const string MarkerVariable = "GRAPHREADER_OCR_MARKER_CENTER_V4";
    private const string YamlVariable = "GRAPHREADER_OCR_MARKER_OFFICIAL_YAML_V4";
    private const string ReportVariable = "GRAPHREADER_OCR_MARKER_CSHARP_REPORT_V4";
    private const string SourceCommitVariable = "GRAPHREADER_OCR_MARKER_SOURCE_COMMIT_V4";
    private const string CompositionId = "production-v8-ocr-to-normalized-marker-composed-v4";
    private const string MarkerSha256 =
        "017fca04fa3817596ce3088d73f51003dd3658bc56ec3130e25c92252e6bf739";
    private const string InferenceYamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";
    private const double TruthMatchIouMinimum = 0.5;
    private const double MarkerMatchTolerance = 5;
    private const double ProhibitedHitTolerance = 6;
    private const int ExpectedSceneCount = 64;
    private static readonly string[] RequiredRoles =
    [
        "annotation",
        "legend_text",
        "phase_heading",
        "x_tick",
        "y_tick",
    ];
    private static readonly string[] AllowedDirtyUserPaths =
    [
        "README.md",
        "GOAL_SEQUENCE.md",
        "MANIFEST.json",
        "PORTABLE_PREVIEW_TESTING_GUIDE.md",
        "PORTABLE_SIZE_AUDIT_AND_CLEANUP.md",
        "README_NEXT_STEPS.md",
        "REAL_ESRGAN_MODEL_DECISION.md",
        "UI_REFINEMENT_SPEC.md",
    ];
    private static readonly string[] MandatoryProhibitedKinds =
    [
        "arrow_shaft",
        "arrowhead",
        "axis",
        "bracket",
        "divider",
        "legend",
        "line_intersection",
        "text",
        "tick",
    ];
    private static readonly OcrRectangle PlotBounds = new(104, 48, 406, 208);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    [TestMethod]
    public async Task NewSealedBytesExecuteOnceThroughCSharpOcrAndMarkerCpuComposition()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable(RunVariable), "1", StringComparison.Ordinal))
        {
            Assert.Inconclusive(
                $"Set {RunVariable}=1 plus the exact payload, archive, report, and source paths to run the one-time composed gate.");
        }

        string repoRoot = FindRepositoryRoot();
        string sealPath = Path.Combine(
            repoRoot,
            "ml",
            "ocr",
            "production_csharp_marker_gate_v4",
            "SEALED_PUBLIC_TEST_SEAL.json");
        Assert.IsTrue(File.Exists(sealPath), "The tracked sealed-public split identity is absent.");
        SplitSeal seal = ReadSplitSeal(sealPath, repoRoot);
        string authorizationPath = Path.Combine(
            repoRoot,
            "ml",
            "ocr",
            "production_csharp_marker_gate_v4",
            "PUBLIC_GATE_AUTHORIZATION.json");
        string archivePath = RequiredPath(ArchiveVariable);
        Assert.AreEqual(seal.FixtureArchiveSha256, Sha256(File.ReadAllBytes(archivePath)));
        string detectorPath = RequiredPath(DetectorVariable);
        string officialPath = RequiredPath(OfficialVariable);
        string numericPath = RequiredPath(NumericVariable);
        string ambiguityPath = RequiredPath(AmbiguityVariable);
        string markerPath = RequiredPath(MarkerVariable);
        string yamlPath = RequiredPath(YamlVariable);
        string reportPath = RequiredOutputPath(ReportVariable);
        string sourceCommit = RequiredSourceCommit();
        Assert.AreEqual(sourceCommit, ReadGitHead(repoRoot), "The gate must execute from its exact source commit.");
        GateAuthorization authorization = ReadAuthorization(
            authorizationPath,
            sealPath,
            seal,
            repoRoot,
            reportPath);
        Assert.AreEqual(
            authorization.SealedIdentityCommit,
            ReadGitParent(repoRoot),
            "The authorization commit must be the single child of the committed split identity.");
        RequireGateWorkingTreeScope(repoRoot);
        AssertHash(markerPath, MarkerSha256, "normalized marker-center payload");
        AssertHash(yamlPath, InferenceYamlSha256, "official inference YAML");
        string alphabet = ReadOfficialAlphabet(yamlPath);
        OcrV8ProductionPayloadSet ocrPayloads = OcrPayloads(
            detectorPath,
            officialPath,
            numericPath,
            ambiguityPath,
            alphabet);
        string[] authorizedCandidates =
        [
            OcrV8ProductionCompositionFactory.DetectorSha256,
            OcrV8ProductionCompositionFactory.OfficialRecognizerSha256,
            OcrV8ProductionCompositionFactory.NumericRecognizerSha256,
            OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256,
            MarkerSha256,
        ];
        Assert.IsTrue(
            authorization.CandidateSha256.SequenceEqual(
                authorizedCandidates.Order(StringComparer.Ordinal),
                StringComparer.Ordinal),
            "The authorization does not bind the exact five candidates.");
        var markerModel = new ModelIdentity(
            "graph-marker-center-normalized-v4-p1",
            "0.1.0-candidate",
            MarkerSha256,
            markerPath);
        var evidenceFactory = new EvidenceInferenceSessionFactory(
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance));
        var registry = new OnnxSessionRegistry(
            new CpuOnlyDiscovery(),
            new WindowsExecutionProviderPolicy(),
            evidenceFactory,
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        string cacheRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderOcrMarkerCSharpPublicV4",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(cacheRoot);
        try
        {
            await using var runtime = new InferenceRuntime(
                registry,
                new BoundedInferenceScheduler(capacity: 2, workerCount: 1),
                new ContentAddressedStageCache(cacheRoot));
            OcrV8ProductionCompositionPipeline ocr = OcrV8ProductionCompositionFactory.Create(
                runtime,
                ocrPayloads,
                [InferenceProvider.Cpu],
                bypassCache: true);
            var markers = new NormalizedMarkerProposalDetector(runtime);
            GateSeal gateSeal = OpenGateSeal(
                repoRoot,
                sourceCommit,
                seal,
                ocrPayloads,
                markerModel);
            DirectGateReport report;
            try
            {
                report = await EvaluateAsync(
                    archivePath,
                    seal,
                    ocr,
                    markers,
                    evidenceFactory,
                    ocrPayloads,
                    markerModel,
                    gateSeal,
                    sourceCommit,
                    CancellationToken.None);
                byte[] reportBytes = JsonSerializer.SerializeToUtf8Bytes(report, JsonOptions);
                Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
                await File.WriteAllBytesAsync(reportPath, reportBytes);
                CompleteGateSeal(
                    gateSeal,
                    report.GatesPassed ? "pass" : "fail",
                    Sha256(reportBytes),
                    null);
            }
            catch (Exception exception) when (exception is not OutOfMemoryException)
            {
                CompleteGateSeal(gateSeal, "failed_runner", null, exception);
                throw;
            }

            Assert.IsTrue(report.GatesPassed, JsonSerializer.Serialize(report.Metrics));
            Assert.IsTrue(report.MarkerCreationEvaluated);
            Assert.AreEqual(0, report.Metrics.TextMarkerCreationCount);
            Assert.IsFalse(report.ArtifactMaskProductionApproved);
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

    private static async Task<DirectGateReport> EvaluateAsync(
        string archivePath,
        SplitSeal seal,
        OcrV8ProductionCompositionPipeline ocr,
        NormalizedMarkerProposalDetector markerDetector,
        EvidenceInferenceSessionFactory evidenceFactory,
        OcrV8ProductionPayloadSet ocrPayloads,
        ModelIdentity markerModel,
        GateSeal gateSeal,
        string sourceCommit,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(archivePath);
        byte[] manifestBytes = await ReadEntryAsync(
            archive.GetEntry("manifest.json") ??
                throw new InvalidDataException("Sealed public archive has no manifest.json."),
            cancellationToken);
        Assert.AreEqual(seal.FixtureManifestSha256, Sha256(manifestBytes));
        FixtureManifest manifest = JsonSerializer.Deserialize<FixtureManifest>(manifestBytes, JsonOptions) ??
            throw new InvalidDataException("Sealed public manifest is invalid.");
        Assert.AreEqual("graphreader.ocr-marker-production-composition-fixtures.v1", manifest.Schema);
        Assert.AreEqual("graphreader-csharp-ocr-marker-composition-v4", manifest.Revision);
        Assert.AreEqual("sealed_public", manifest.Split);
        Assert.AreEqual(ExpectedSceneCount, manifest.SceneCount);
        Assert.AreEqual(seal.SceneCount, manifest.SceneCount);
        Assert.IsTrue(manifest.SyntheticOnly);
        Assert.IsFalse(manifest.PrivateOrArticleImages);
        Assert.IsFalse(manifest.ChandlerIncluded);
        Assert.IsFalse(manifest.GeneralizationLabelIncluded);
        Assert.IsFalse(manifest.SecretSeedSerialized);
        string[] observedRoles = manifest.Cases
            .SelectMany(static item => item.TextTruths)
            .Select(static item => item.Role)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.IsTrue(observedRoles.SequenceEqual(RequiredRoles, StringComparer.Ordinal));

        int ocrCorrect = 0;
        int roleCorrect = 0;
        int characterCount = 0;
        int characterErrors = 0;
        int ocrTruePositives = 0;
        int ocrFalsePositives = 0;
        int ocrFalseNegatives = 0;
        int ocrDuplicates = 0;
        int ocrExactScenes = 0;
        int markerTruePositives = 0;
        int markerFalsePositives = 0;
        int markerFalseNegatives = 0;
        int markerDuplicates = 0;
        int markerExactScenes = 0;
        int textMarkerCreationCount = 0;
        var prohibitedHits = MandatoryProhibitedKinds.ToDictionary(
            static kind => kind,
            static _ => 0,
            StringComparer.Ordinal);
        var familyCounts = new Dictionary<string, (int Correct, int Total)>(StringComparer.Ordinal);
        var scenes = new List<SceneEvidence>(manifest.Cases.Count);
        string projectId = "11111111-1111-1111-1111-111111111111";
        string panelId = "22222222-2222-2222-2222-222222222222";

        foreach (FixtureCase fixture in manifest.Cases)
        {
            cancellationToken.ThrowIfCancellationRequested();
            byte[] sourceBytes = await ReadVerifiedEntryAsync(
                archive,
                fixture.ImagePath,
                fixture.ImageSha256,
                cancellationToken);
            (byte[] pixels, int width, int height) = DecodeGray8(sourceBytes);
            Assert.AreEqual(640, width);
            Assert.AreEqual(320, height);
            Assert.AreEqual(fixture.RasterSha256, Sha256(pixels));
            byte[] artifactPng = await ReadVerifiedEntryAsync(
                archive,
                fixture.ArtifactMaskPath,
                fixture.ArtifactMaskPngSha256,
                cancellationToken);
            (byte[] artifactPixels, int artifactWidth, int artifactHeight) = DecodeGray8(artifactPng);
            Assert.AreEqual(width, artifactWidth);
            Assert.AreEqual(height, artifactHeight);
            Assert.AreEqual(fixture.ArtifactMaskRasterSha256, Sha256(artifactPixels));

            var image = new OcrImage(
                width,
                height,
                width,
                pixels,
                OcrSourceImage.Original,
                OcrFrameTransform.Identity,
                CanonicalOriginalWidth: width,
                CanonicalOriginalHeight: height);
            OcrResult ocrResult = await ocr.RecognizeAsync(
                new OcrRequest(
                    projectId,
                    panelId,
                    fixture.ImageSha256,
                    image,
                    PlotBounds),
                cancellationToken);
            Assert.IsTrue(ocrResult.Succeeded, ocrResult.Failure?.TechnicalMessage);
            OcrSceneMetrics ocrScene = ScoreOcr(fixture, ocrResult);
            ocrCorrect += ocrScene.Correct;
            roleCorrect += ocrScene.RoleCorrect;
            characterCount += ocrScene.CharacterCount;
            characterErrors += ocrScene.CharacterErrors;
            ocrTruePositives += ocrScene.TruePositives;
            ocrFalsePositives += ocrScene.FalsePositives;
            ocrFalseNegatives += ocrScene.FalseNegatives;
            ocrDuplicates += ocrScene.Duplicates;
            ocrExactScenes += ocrScene.Exact ? 1 : 0;
            foreach (FamilyEvidence family in ocrScene.Families)
            {
                (int correct, int total) = familyCounts.GetValueOrDefault(family.Family);
                familyCounts[family.Family] = (correct + family.Correct, total + family.Total);
            }

            float[] textMask = BuildTextMask(width, height, ocrResult);
            float[] artifactMask = artifactPixels.Select(static value => value / 255f).ToArray();
            float[] luminance = pixels.Select(static value => value / 255f).ToArray();
            var markerFrame = new MarkerImageFrame(
                width,
                height,
                1,
                luminance,
                MarkerSourceImage.Original,
                MarkerAffineTransform.Identity,
                new MarkerMask(width, height, textMask),
                new MarkerMask(width, height, artifactMask));
            MarkerDetectionResult markerResult = await markerDetector.DetectAsync(
                new NormalizedMarkerProposalDetectionRequest(
                    projectId,
                    panelId,
                    fixture.ImageSha256,
                    markerModel,
                    markerFrame,
                    new NormalizedMarkerProposalDetectionOptions()),
                cancellationToken);
            Assert.IsTrue(markerResult.Succeeded, markerResult.Failure?.TechnicalMessage);
            Assert.AreEqual(InferenceProvider.Cpu, markerResult.Model.Provider);
            MarkerSceneMetrics markerScene = ScoreMarkers(fixture, markerResult);
            markerTruePositives += markerScene.TruePositives;
            markerFalsePositives += markerScene.FalsePositives;
            markerFalseNegatives += markerScene.FalseNegatives;
            markerDuplicates += markerScene.Duplicates;
            markerExactScenes += markerScene.Exact ? 1 : 0;
            textMarkerCreationCount += markerScene.TextMarkerCreationCount;
            foreach ((string kind, int count) in markerScene.ProhibitedHits)
            {
                prohibitedHits[kind] = prohibitedHits.GetValueOrDefault(kind) + count;
            }

            scenes.Add(new SceneEvidence(
                fixture.SceneId,
                fixture.ImageSha256,
                fixture.RasterSha256,
                fixture.ArtifactMaskRasterSha256,
                ocrScene,
                markerScene,
                Sha256(Float32Bytes(textMask))));
        }

        int truthTextCount = manifest.Cases.Sum(static item => item.TextTruths.Count);
        int truthMarkerCount = manifest.Cases.Sum(static item => item.MarkerCenters.Count);
        var metrics = new DirectMetrics(
            manifest.SceneCount,
            truthTextCount,
            ocrExactScenes,
            ocrTruePositives,
            ocrFalsePositives,
            ocrFalseNegatives,
            ocrDuplicates,
            ocrCorrect / (double)truthTextCount,
            characterErrors / (double)characterCount,
            roleCorrect / (double)truthTextCount,
            Ratio(familyCounts, "numeric"),
            Ratio(familyCounts, "word"),
            Ratio(familyCounts, "ambiguity"),
            truthMarkerCount,
            markerExactScenes,
            markerTruePositives,
            markerFalsePositives,
            markerFalseNegatives,
            markerDuplicates,
            textMarkerCreationCount,
            prohibitedHits.OrderBy(static item => item.Key, StringComparer.Ordinal)
                .ToDictionary(static item => item.Key, static item => item.Value, StringComparer.Ordinal));
        Dictionary<string, ModelExecutionEvidence> executions = evidenceFactory.Snapshot();
        string[] expectedHashes =
        [
            OcrV8ProductionCompositionFactory.DetectorSha256,
            OcrV8ProductionCompositionFactory.OfficialRecognizerSha256,
            OcrV8ProductionCompositionFactory.NumericRecognizerSha256,
            OcrV8ProductionCompositionFactory.AmbiguityRecognizerSha256,
            MarkerSha256,
        ];
        bool directRuntimeEvidencePassed = executions.Count == expectedHashes.Length &&
            executions.Keys.Order(StringComparer.Ordinal).SequenceEqual(
                expectedHashes.Order(StringComparer.Ordinal),
                StringComparer.Ordinal) &&
            executions.Values.All(static item =>
                item.CallCount > 0 &&
                item.InputTensorSha256.Count == item.CallCount &&
                item.OutputTensorSha256.Count == item.CallCount &&
                item.Providers.SequenceEqual(["Cpu"], StringComparer.Ordinal));
        bool allProhibitedZero = metrics.ProhibitedMarkerHits.Values.All(static value => value == 0);
        bool gatesPassed = directRuntimeEvidencePassed &&
            ocrExactScenes == manifest.SceneCount &&
            ocrTruePositives == truthTextCount &&
            ocrFalsePositives == 0 && ocrFalseNegatives == 0 && ocrDuplicates == 0 &&
            metrics.RecognitionExactMatch >= 0.90 &&
            metrics.CharacterErrorRate <= 0.05 &&
            metrics.RoleAccuracy >= 0.90 &&
            metrics.NumericExactMatch >= 0.90 &&
            metrics.WordExactMatch >= 0.90 &&
            metrics.AmbiguityExactMatch >= 0.90 &&
            markerExactScenes == manifest.SceneCount &&
            markerTruePositives == truthMarkerCount &&
            markerFalsePositives == 0 && markerFalseNegatives == 0 && markerDuplicates == 0 &&
            textMarkerCreationCount == 0 && allProhibitedZero;

        return new DirectGateReport(
            "graphreader.ocr-marker-production-composition-csharp-public-gate.v4",
            CompositionId,
            sourceCommit,
            seal.FixtureArchiveSha256,
            seal.FixtureManifestSha256,
            "CPUExecutionProvider",
            PayloadEvidence.From(ocrPayloads, markerModel),
            executions,
            gateSeal.Key,
            gateSeal.OpenedSha256,
            metrics,
            scenes,
            gatesPassed,
            FullEightRoleCoverageProven: false,
            MarkerCreationEvaluated: true,
            ArtifactMaskProductionApproved: false,
            ProductionApproval: false,
            ReleaseEligible: false,
            BlockingGates:
            [
                "full_eight_role_coverage",
                "approved_artifact_mask_provider",
                "approved_manifests_and_production_model_store",
                "packaging_discovery_and_clean_machine_execution",
                "private_chandler_automatic_validation",
            ]);
    }

    private static OcrSceneMetrics ScoreOcr(FixtureCase fixture, OcrResult result)
    {
        var matched = new HashSet<int>();
        int falsePositives = 0;
        int duplicates = 0;
        int correct = 0;
        int roleCorrect = 0;
        int characterCount = 0;
        int characterErrors = 0;
        var familyCounts = new Dictionary<string, (int Correct, int Total)>(StringComparer.Ordinal);
        foreach (OcrRegion region in result.Regions)
        {
            (int Index, double Iou)[] matches = fixture.TextTruths
                .Select((truth, index) =>
                    (Index: index, Iou: IntersectionOverUnion(region.Polygon.Bounds, truth.Bbox)))
                .Where(static item => item.Iou >= TruthMatchIouMinimum)
                .OrderByDescending(static item => item.Iou)
                .ThenBy(static item => item.Index)
                .ToArray();
            if (matches.Length == 0)
            {
                falsePositives++;
                continue;
            }

            int truthIndex = matches[0].Index;
            if (!matched.Add(truthIndex))
            {
                duplicates++;
                continue;
            }

            FixtureTextTruth truth = fixture.TextTruths[truthIndex];
            bool exact = string.Equals(region.Text, truth.TruthText, StringComparison.Ordinal);
            bool role = string.Equals(RoleName(region.Role), truth.Role, StringComparison.Ordinal);
            correct += exact ? 1 : 0;
            roleCorrect += role ? 1 : 0;
            characterCount += truth.TruthText.EnumerateRunes().Count();
            characterErrors += LevenshteinDistance(truth.TruthText, region.Text);
            (int familyCorrect, int familyTotal) = familyCounts.GetValueOrDefault(truth.Family);
            familyCounts[truth.Family] = (familyCorrect + (exact ? 1 : 0), familyTotal + 1);
        }

        int falseNegatives = fixture.TextTruths.Count - matched.Count;
        return new OcrSceneMetrics(
            fixture.TextTruths.Count,
            result.Regions.Count,
            matched.Count,
            falsePositives,
            falseNegatives,
            duplicates,
            falsePositives == 0 && falseNegatives == 0 && duplicates == 0 &&
                result.Regions.Count == fixture.TextTruths.Count,
            correct,
            roleCorrect,
            characterCount,
            characterErrors,
            familyCounts.Select(static item =>
                new FamilyEvidence(item.Key, item.Value.Correct, item.Value.Total)).ToArray());
    }

    private static MarkerSceneMetrics ScoreMarkers(
        FixtureCase fixture,
        MarkerDetectionResult result)
    {
        var matched = new HashSet<int>();
        int falsePositives = 0;
        int duplicates = 0;
        int textHits = 0;
        var prohibited = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (MarkerCenter marker in result.Markers)
        {
            (int Index, double Distance)[] matches = fixture.MarkerCenters
                .Select((truth, index) =>
                    (Index: index, Distance: Distance(marker.Center.X, marker.Center.Y, truth[0], truth[1])))
                .Where(static item => item.Distance <= MarkerMatchTolerance)
                .OrderBy(static item => item.Distance)
                .ThenBy(static item => item.Index)
                .ToArray();
            if (matches.Length == 0)
            {
                falsePositives++;
            }
            else if (!matched.Add(matches[0].Index))
            {
                duplicates++;
            }

            textHits += fixture.TextTruths.Count(truth => Contains(truth.Bbox, marker.Center.X, marker.Center.Y));
            foreach (FixtureProhibited item in fixture.Prohibited)
            {
                if (Distance(marker.Center.X, marker.Center.Y, item.X, item.Y) <= ProhibitedHitTolerance)
                {
                    prohibited[item.Kind] = prohibited.GetValueOrDefault(item.Kind) + 1;
                }
            }
        }

        int falseNegatives = fixture.MarkerCenters.Count - matched.Count;
        return new MarkerSceneMetrics(
            fixture.MarkerCenters.Count,
            result.Markers.Count,
            matched.Count,
            falsePositives,
            falseNegatives,
            duplicates,
            falsePositives == 0 && falseNegatives == 0 && duplicates == 0 &&
                result.Markers.Count == fixture.MarkerCenters.Count,
            textHits,
            prohibited.OrderBy(static item => item.Key, StringComparer.Ordinal)
                .ToDictionary(static item => item.Key, static item => item.Value, StringComparer.Ordinal),
            result.Frames.Select(static frame => frame.CacheKey).ToArray());
    }

    private static float[] BuildTextMask(int width, int height, OcrResult result)
    {
        var mask = new float[checked(width * height)];
        IEnumerable<OcrRectangle> bounds = result.Regions.Select(static item => item.Polygon.Bounds)
            .Concat(result.Masks.Select(static item => item.Polygon.Bounds));
        foreach (OcrRectangle value in bounds)
        {
            int left = Math.Clamp((int)Math.Floor(value.Left - 1), 0, width - 1);
            int top = Math.Clamp((int)Math.Floor(value.Top - 1), 0, height - 1);
            int right = Math.Clamp((int)Math.Ceiling(value.Right + 1), 0, width - 1);
            int bottom = Math.Clamp((int)Math.Ceiling(value.Bottom + 1), 0, height - 1);
            for (int y = top; y <= bottom; y++)
            {
                mask.AsSpan((y * width) + left, right - left + 1).Fill(1f);
            }
        }

        return mask;
    }

    private static SplitSeal ReadSplitSeal(string path, string repoRoot)
    {
        SplitSeal seal = JsonSerializer.Deserialize<SplitSeal>(File.ReadAllBytes(path), JsonOptions) ??
            throw new InvalidDataException("The tracked split seal is invalid.");
        Assert.AreEqual(
            "graphreader.ocr-marker-production-composition-split-seal.v1",
            seal.Schema);
        Assert.AreEqual(ExpectedSceneCount, seal.SceneCount);
        Assert.IsTrue(seal.SyntheticOnly);
        Assert.IsFalse(seal.PrivateOrArticleImages);
        Assert.IsFalse(seal.ChandlerIncluded);
        Assert.IsFalse(seal.GeneralizationLabelIncluded);
        Assert.IsFalse(seal.SecretSeedSerialized);
        Assert.AreEqual(0, seal.ModelExecutionCountAtFreeze);
        Assert.IsFalse(seal.ProductionApproval);
        Assert.IsFalse(seal.ReleaseEligible);
        foreach ((string relativePath, string expectedHash) in seal.SourceSha256)
        {
            string sourcePath = Path.GetFullPath(Path.Combine(repoRoot, relativePath.Replace('/', Path.DirectorySeparatorChar)));
            Assert.IsTrue(sourcePath.StartsWith(repoRoot, StringComparison.OrdinalIgnoreCase));
            AssertHash(sourcePath, expectedHash, $"sealed source {relativePath}");
        }

        return seal;
    }

    private static GateAuthorization ReadAuthorization(
        string path,
        string sealPath,
        SplitSeal seal,
        string repoRoot,
        string reportPath)
    {
        Assert.IsTrue(File.Exists(path), "The separate one-time public gate authorization is absent.");
        GateAuthorization authorization = JsonSerializer.Deserialize<GateAuthorization>(
            File.ReadAllBytes(path),
            JsonOptions) ?? throw new InvalidDataException("The public gate authorization is invalid.");
        Assert.AreEqual("graphreader.ocr-marker-production-composition-authorization.v1", authorization.Schema);
        Assert.AreEqual("ocr-marker-production-composition", authorization.Task);
        Assert.AreEqual("graphreader-csharp-ocr-marker-composition-v4", authorization.Revision);
        Assert.AreEqual(CompositionId, authorization.CompositionId);
        Assert.IsTrue(authorization.ExecutionAuthorized);
        Assert.AreEqual(1, authorization.ExecutionCountAuthorized);
        Assert.AreEqual("CPUExecutionProvider", authorization.Provider);
        Assert.IsTrue(
            authorization.SealedIdentityCommit.Length == 40 &&
            authorization.SealedIdentityCommit.All(Uri.IsHexDigit));
        Assert.AreEqual(seal.FixtureArchiveSha256, authorization.FixtureArchiveSha256);
        Assert.AreEqual(seal.FixtureManifestSha256, authorization.FixtureManifestSha256);
        Assert.AreEqual(Sha256(File.ReadAllBytes(sealPath)), authorization.SplitSealSha256);
        Assert.AreEqual(
            "OcrMarkerDirectPublicGateV4Tests.NewSealedBytesExecuteOnceThroughCSharpOcrAndMarkerCpuComposition",
            authorization.ExactTest);
        Assert.AreEqual(
            authorization.ResultPath,
            Path.GetRelativePath(repoRoot, reportPath).Replace('\\', '/'));
        Assert.IsFalse(authorization.RerunOrRepairAuthorized);
        Assert.IsFalse(authorization.ArtifactMaskProductionApproval);
        Assert.IsFalse(authorization.ManifestCreationAuthorized);
        Assert.IsFalse(authorization.ModelStorePromotionAuthorized);
        Assert.IsFalse(authorization.PrivateValidationAuthorized);
        Assert.IsFalse(authorization.ProductionApproval);
        Assert.IsFalse(authorization.ReleaseEligible);
        return authorization;
    }

    private static GateSeal OpenGateSeal(
        string repoRoot,
        string sourceCommit,
        SplitSeal split,
        OcrV8ProductionPayloadSet ocr,
        ModelIdentity marker)
    {
        string[] hashes =
        [
            ocr.Detector.Sha256,
            ocr.OfficialRecognizer.Sha256,
            ocr.NumericRecognizer.Sha256,
            ocr.AmbiguityRecognizer.Sha256,
            marker.Sha256,
        ];
        string material = string.Join(
            '\n',
            new[] { CompositionId, sourceCommit, split.FixtureArchiveSha256 }
                .Concat(hashes.Order(StringComparer.Ordinal)));
        string key = Sha256(Encoding.UTF8.GetBytes(material));
        string root = Path.Combine(
            repoRoot,
            "ml",
            "markers",
            "gate-seals",
            "ocr-marker-composition",
            key);
        Directory.CreateDirectory(root);
        string openedPath = Path.Combine(root, "opened.json");
        string resultPath = Path.Combine(root, "result.json");
        Assert.IsFalse(File.Exists(openedPath), $"The one-time composed gate was already opened: {openedPath}");
        Assert.IsFalse(File.Exists(resultPath), $"The one-time composed gate already has a result: {resultPath}");
        var opened = new
        {
            schema = "graphreader.ocr-marker-production-composition-opened-seal.v4",
            composition_id = CompositionId,
            canonical_seal_key = key,
            source_commit = sourceCommit,
            fixture_archive_sha256 = split.FixtureArchiveSha256,
            candidate_sha256 = hashes.Order(StringComparer.Ordinal).ToArray(),
            opened_utc = DateTimeOffset.UtcNow,
            production_approval = false,
            release_eligible = false,
        };
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(opened, JsonOptions);
        using (var stream = new FileStream(openedPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
        {
            stream.Write(bytes);
            stream.Flush(flushToDisk: true);
        }

        return new GateSeal(key, openedPath, resultPath, Sha256(bytes));
    }

    private static void CompleteGateSeal(
        GateSeal gate,
        string status,
        string? reportSha256,
        Exception? exception)
    {
        Assert.IsTrue(File.Exists(gate.OpenedPath), "The opened gate seal disappeared during execution.");
        Assert.AreEqual(gate.OpenedSha256, Sha256(File.ReadAllBytes(gate.OpenedPath)));
        var result = new
        {
            schema = "graphreader.ocr-marker-production-composition-result-seal.v4",
            canonical_seal_key = gate.Key,
            opened_seal_sha256 = gate.OpenedSha256,
            status,
            report_sha256 = reportSha256,
            exception_type = exception?.GetType().Name,
            exception_message = exception?.Message,
            completed_utc = DateTimeOffset.UtcNow,
            production_approval = false,
            release_eligible = false,
        };
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(result, JsonOptions);
        using var stream = new FileStream(
            gate.ResultPath,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.Read);
        stream.Write(bytes);
        stream.Flush(flushToDisk: true);
    }

    private static string ReadGitHead(string repoRoot)
    {
        var start = new ProcessStartInfo("git")
        {
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("rev-parse");
        start.ArgumentList.Add("HEAD");
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Git could not be started for source verification.");
        string output = process.StandardOutput.ReadToEnd().Trim();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.AreEqual(0, process.ExitCode, error);
        Assert.IsTrue(output.Length == 40 && output.All(Uri.IsHexDigit));
        return output.ToLowerInvariant();
    }

    private static string ReadGitParent(string repoRoot)
    {
        var start = new ProcessStartInfo("git")
        {
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("rev-parse");
        start.ArgumentList.Add("HEAD^");
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Git could not be started for authorization parent verification.");
        string output = process.StandardOutput.ReadToEnd().Trim();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.AreEqual(0, process.ExitCode, error);
        Assert.IsTrue(output.Length == 40 && output.All(Uri.IsHexDigit));
        return output.ToLowerInvariant();
    }

    private static void RequireGateWorkingTreeScope(string repoRoot)
    {
        var start = new ProcessStartInfo("git")
        {
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("status");
        start.ArgumentList.Add("--porcelain=v1");
        start.ArgumentList.Add("-uall");
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Git could not be started for working-tree verification.");
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.AreEqual(0, process.ExitCode, error);
        string[] dirtyPaths = output
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .Select(static line => line.Length > 3 ? line[3..].Replace('\\', '/') : string.Empty)
            .Select(static path => path.Contains(" -> ", StringComparison.Ordinal)
                ? path[(path.LastIndexOf(" -> ", StringComparison.Ordinal) + 4)..]
                : path)
            .ToArray();
        string[] unexpected = dirtyPaths
            .Where(path => !AllowedDirtyUserPaths.Contains(path, StringComparer.Ordinal))
            .ToArray();
        Assert.IsEmpty(
            unexpected,
            $"The one-time gate requires committed executable sources. Unexpected dirty paths: {string.Join(", ", unexpected)}");
    }

    private static OcrV8ProductionPayloadSet OcrPayloads(
        string detector,
        string official,
        string numeric,
        string ambiguity,
        string alphabet) =>
        new(
            new ModelIdentity(
                "graph-text-spaced-component-recall-v10-p2",
                "0.0.21-v10-p2",
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

    private static string ReadOfficialAlphabet(string path)
    {
        string[] lines = File.ReadAllLines(path, Encoding.UTF8);
        int start = Array.FindIndex(lines, static line =>
            string.Equals(line.Trim(), "character_dict:", StringComparison.Ordinal));
        if (start < 0)
        {
            throw new InvalidDataException("Recognition inference.yml has no character_dict.");
        }

        var values = new List<string>();
        for (int index = start + 1; index < lines.Length; index++)
        {
            string line = lines[index];
            if (!line.StartsWith("  - ", StringComparison.Ordinal))
            {
                if (line.StartsWith("  ", StringComparison.Ordinal) && string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                break;
            }

            string scalar = line[4..].Trim();
            string value = scalar.Length >= 2 && scalar[0] == '\'' && scalar[^1] == '\''
                ? scalar[1..^1].Replace("''", "'", StringComparison.Ordinal)
                : scalar.Length >= 2 && scalar[0] == '"' && scalar[^1] == '"'
                    ? JsonSerializer.Deserialize<string>(scalar) ?? string.Empty
                    : scalar;
            if (value.EnumerateRunes().Count() != 1)
            {
                throw new InvalidDataException($"Recognition alphabet item is not one scalar: {line}");
            }

            values.Add(value);
        }

        if (!values.Contains(" ", StringComparer.Ordinal))
        {
            values.Add(" ");
        }

        string alphabet = string.Concat(values);
        Assert.AreEqual(437, alphabet.EnumerateRunes().Count());
        Assert.AreEqual(
            OcrV8ProductionCompositionFactory.OfficialAlphabetSha256,
            Sha256(Encoding.UTF8.GetBytes(alphabet)));
        return alphabet;
    }

    private static (byte[] Pixels, int Width, int Height) DecodeGray8(byte[] source)
    {
        ReadOnlySpan<byte> png = source;
        ReadOnlySpan<byte> signature = [137, 80, 78, 71, 13, 10, 26, 10];
        if (png.Length < signature.Length || !png[..signature.Length].SequenceEqual(signature))
        {
            throw new InvalidDataException("Public fixture has an invalid PNG signature.");
        }

        var offset = signature.Length;
        var width = 0;
        var height = 0;
        using var compressed = new MemoryStream();
        while (offset < png.Length)
        {
            if (png.Length - offset < 12)
            {
                throw new InvalidDataException("Public fixture PNG chunk is truncated.");
            }

            int length = checked((int)BinaryPrimitives.ReadUInt32BigEndian(png.Slice(offset, 4)));
            ReadOnlySpan<byte> type = png.Slice(offset + 4, 4);
            int dataOffset = offset + 8;
            int next = checked(dataOffset + length + 4);
            if (length < 0 || next > png.Length)
            {
                throw new InvalidDataException("Public fixture PNG chunk length is invalid.");
            }

            ReadOnlySpan<byte> data = png.Slice(dataOffset, length);
            if (type.SequenceEqual("IHDR"u8))
            {
                if (length != 13 || data[8] != 8 || data[9] != 0 ||
                    data[10] != 0 || data[11] != 0 || data[12] != 0)
                {
                    throw new InvalidDataException("Public fixture PNG must be non-interlaced Gray8.");
                }

                width = checked((int)BinaryPrimitives.ReadUInt32BigEndian(data[..4]));
                height = checked((int)BinaryPrimitives.ReadUInt32BigEndian(data.Slice(4, 4)));
            }
            else if (type.SequenceEqual("IDAT"u8))
            {
                compressed.Write(data);
            }
            else if (type.SequenceEqual("IEND"u8))
            {
                break;
            }

            offset = next;
        }

        compressed.Position = 0;
        using var inflated = new MemoryStream(checked(height * (width + 1)));
        using (var zlib = new ZLibStream(compressed, CompressionMode.Decompress, leaveOpen: true))
        {
            zlib.CopyTo(inflated);
        }

        byte[] filtered = inflated.ToArray();
        if (width <= 0 || height <= 0 || filtered.Length != checked(height * (width + 1)))
        {
            throw new InvalidDataException("Public fixture PNG dimensions or raster length are invalid.");
        }

        var pixels = new byte[checked(width * height)];
        for (var y = 0; y < height; y++)
        {
            int sourceRow = y * (width + 1);
            int targetRow = y * width;
            byte filter = filtered[sourceRow];
            for (var x = 0; x < width; x++)
            {
                byte raw = filtered[sourceRow + 1 + x];
                byte left = x == 0 ? (byte)0 : pixels[targetRow + x - 1];
                byte up = y == 0 ? (byte)0 : pixels[targetRow - width + x];
                byte upLeft = x == 0 || y == 0 ? (byte)0 : pixels[targetRow - width + x - 1];
                pixels[targetRow + x] = filter switch
                {
                    0 => raw,
                    1 => unchecked((byte)(raw + left)),
                    2 => unchecked((byte)(raw + up)),
                    3 => unchecked((byte)(raw + ((left + up) / 2))),
                    4 => unchecked((byte)(raw + Paeth(left, up, upLeft))),
                    _ => throw new InvalidDataException($"Public fixture PNG filter {filter} is unsupported."),
                };
            }
        }

        return (pixels, width, height);
    }

    private static byte Paeth(byte left, byte up, byte upLeft)
    {
        int prediction = left + up - upLeft;
        int leftDistance = Math.Abs(prediction - left);
        int upDistance = Math.Abs(prediction - up);
        int diagonalDistance = Math.Abs(prediction - upLeft);
        return leftDistance <= upDistance && leftDistance <= diagonalDistance
            ? left
            : upDistance <= diagonalDistance ? up : upLeft;
    }

    private static async Task<byte[]> ReadVerifiedEntryAsync(
        ZipArchive archive,
        string path,
        string expectedSha256,
        CancellationToken cancellationToken)
    {
        byte[] bytes = await ReadEntryAsync(
            archive.GetEntry(path) ?? throw new InvalidDataException($"Fixture resource is absent: {path}"),
            cancellationToken);
        Assert.AreEqual(expectedSha256, Sha256(bytes), $"Fixture resource changed: {path}");
        return bytes;
    }

    private static async Task<byte[]> ReadEntryAsync(
        ZipArchiveEntry entry,
        CancellationToken cancellationToken)
    {
        await using Stream source = entry.Open();
        using var destination = new MemoryStream(checked((int)entry.Length));
        await source.CopyToAsync(destination, cancellationToken);
        return destination.ToArray();
    }

    private static string FindRepositoryRoot()
    {
        string path = AppContext.BaseDirectory;
        while (!File.Exists(Path.Combine(path, "GraphAutoReader.slnx")))
        {
            DirectoryInfo? parent = Directory.GetParent(path);
            Assert.IsNotNull(parent, "The repository root could not be located.");
            path = parent.FullName;
        }

        return Path.GetFullPath(path);
    }

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
        Assert.IsFalse(File.Exists(path), $"One-time composed C# report already exists: {path}");
        return path;
    }

    private static string RequiredSourceCommit()
    {
        string value = Environment.GetEnvironmentVariable(SourceCommitVariable) ?? string.Empty;
        Assert.IsTrue(
            value.Length == 40 && value.All(Uri.IsHexDigit),
            $"{SourceCommitVariable} must be the exact 40-character committed source identity.");
        return value.ToLowerInvariant();
    }

    private static void AssertHash(string path, string expected, string label) =>
        Assert.AreEqual(expected, Sha256(File.ReadAllBytes(path)), $"{label} checksum changed.");

    private static string Sha256(ReadOnlySpan<byte> bytes) =>
        Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static byte[] Float32Bytes(ReadOnlySpan<float> values)
    {
        var bytes = new byte[checked(values.Length * sizeof(float))];
        for (int index = 0; index < values.Length; index++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(index * sizeof(float), sizeof(float)),
                BitConverter.SingleToInt32Bits(values[index]));
        }

        return bytes;
    }

    private static double IntersectionOverUnion(OcrRectangle left, int[] right)
    {
        double intersectionWidth = Math.Max(0, Math.Min(left.Right, right[2]) - Math.Max(left.Left, right[0]));
        double intersectionHeight = Math.Max(0, Math.Min(left.Bottom, right[3]) - Math.Max(left.Top, right[1]));
        double intersection = intersectionWidth * intersectionHeight;
        double union = (left.Width * left.Height) +
            ((right[2] - right[0]) * (right[3] - right[1])) - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static bool Contains(int[] bounds, double x, double y) =>
        x >= bounds[0] - 1 && x <= bounds[2] + 1 && y >= bounds[1] - 1 && y <= bounds[3] + 1;

    private static double Distance(double leftX, double leftY, double rightX, double rightY)
    {
        double x = leftX - rightX;
        double y = leftY - rightY;
        return Math.Sqrt((x * x) + (y * y));
    }

    private static string RoleName(OcrTextRole role) => role switch
    {
        OcrTextRole.XTick => "x_tick",
        OcrTextRole.YTick => "y_tick",
        OcrTextRole.AxisTitle => "axis_title",
        OcrTextRole.PhaseHeading => "phase_heading",
        OcrTextRole.LegendText => "legend_text",
        OcrTextRole.Participant => "participant",
        OcrTextRole.Annotation => "annotation",
        _ => "other",
    };

    private static int LevenshteinDistance(string expected, string actual)
    {
        string[] left = expected.EnumerateRunes().Select(static rune => rune.ToString()).ToArray();
        string[] right = actual.EnumerateRunes().Select(static rune => rune.ToString()).ToArray();
        var prior = Enumerable.Range(0, right.Length + 1).ToArray();
        var current = new int[right.Length + 1];
        for (int leftIndex = 1; leftIndex <= left.Length; leftIndex++)
        {
            current[0] = leftIndex;
            for (int rightIndex = 1; rightIndex <= right.Length; rightIndex++)
            {
                int substitution = string.Equals(
                    left[leftIndex - 1],
                    right[rightIndex - 1],
                    StringComparison.Ordinal) ? 0 : 1;
                current[rightIndex] = Math.Min(
                    Math.Min(current[rightIndex - 1] + 1, prior[rightIndex] + 1),
                    prior[rightIndex - 1] + substitution);
            }

            (prior, current) = (current, prior);
        }

        return prior[right.Length];
    }

    private static double Ratio(
        IReadOnlyDictionary<string, (int Correct, int Total)> values,
        string family)
    {
        (int correct, int total) = values.GetValueOrDefault(family);
        return total == 0 ? 0 : correct / (double)total;
    }

    private sealed record SplitSeal(
        string Schema,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        int SceneCount,
        bool SyntheticOnly,
        bool PrivateOrArticleImages,
        bool ChandlerIncluded,
        bool GeneralizationLabelIncluded,
        bool SecretSeedSerialized,
        int ModelExecutionCountAtFreeze,
        IReadOnlyDictionary<string, string> SourceSha256,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record GateAuthorization(
        string Schema,
        string Task,
        string Revision,
        string CompositionId,
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
        bool ArtifactMaskProductionApproval,
        bool ManifestCreationAuthorized,
        bool ModelStorePromotionAuthorized,
        bool PrivateValidationAuthorized,
        bool ProductionApproval,
        bool ReleaseEligible);

    private sealed record GateSeal(
        string Key,
        string OpenedPath,
        string ResultPath,
        string OpenedSha256);

    private sealed record FixtureManifest(
        string Schema,
        string Revision,
        string Split,
        int SceneCount,
        bool SyntheticOnly,
        bool PrivateOrArticleImages,
        bool ChandlerIncluded,
        bool GeneralizationLabelIncluded,
        bool SecretSeedSerialized,
        IReadOnlyList<FixtureCase> Cases);

    private sealed record FixtureCase(
        string SceneId,
        string ImagePath,
        string ImageSha256,
        string RasterSha256,
        string ArtifactMaskPath,
        string ArtifactMaskPngSha256,
        string ArtifactMaskRasterSha256,
        IReadOnlyList<FixtureTextTruth> TextTruths,
        IReadOnlyList<double[]> MarkerCenters,
        IReadOnlyList<FixtureProhibited> Prohibited);

    private sealed record FixtureTextTruth(
        int[] Bbox,
        string TruthText,
        string DisplayText,
        string Role,
        string Family);

    private sealed record FixtureProhibited(string Kind, double X, double Y);

    private sealed record DirectGateReport(
        string Schema,
        string CompositionId,
        string SourceCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string Provider,
        IReadOnlyList<PayloadEvidence> Payloads,
        IReadOnlyDictionary<string, ModelExecutionEvidence> ModelExecutions,
        string CanonicalSealKey,
        string OpenedSealSha256,
        DirectMetrics Metrics,
        IReadOnlyList<SceneEvidence> Scenes,
        bool GatesPassed,
        bool FullEightRoleCoverageProven,
        bool MarkerCreationEvaluated,
        bool ArtifactMaskProductionApproved,
        bool ProductionApproval,
        bool ReleaseEligible,
        IReadOnlyList<string> BlockingGates);

    private sealed record DirectMetrics(
        int SceneCount,
        int TextTruthCount,
        int ExactOcrSceneCount,
        int OcrTruePositives,
        int OcrFalsePositives,
        int OcrFalseNegatives,
        int OcrDuplicateRegions,
        double RecognitionExactMatch,
        double CharacterErrorRate,
        double RoleAccuracy,
        double NumericExactMatch,
        double WordExactMatch,
        double AmbiguityExactMatch,
        int MarkerTruthCount,
        int ExactMarkerSceneCount,
        int MarkerTruePositives,
        int MarkerFalsePositives,
        int MarkerFalseNegatives,
        int MarkerDuplicates,
        int TextMarkerCreationCount,
        IReadOnlyDictionary<string, int> ProhibitedMarkerHits);

    private sealed record SceneEvidence(
        string SceneId,
        string SourcePngSha256,
        string SourceRasterSha256,
        string ArtifactMaskRasterSha256,
        OcrSceneMetrics Ocr,
        MarkerSceneMetrics Markers,
        string DerivedOcrTextMaskSha256);

    private sealed record OcrSceneMetrics(
        int TruthCount,
        int PredictionCount,
        int TruePositives,
        int FalsePositives,
        int FalseNegatives,
        int Duplicates,
        bool Exact,
        int Correct,
        int RoleCorrect,
        int CharacterCount,
        int CharacterErrors,
        IReadOnlyList<FamilyEvidence> Families);

    private sealed record FamilyEvidence(string Family, int Correct, int Total);

    private sealed record MarkerSceneMetrics(
        int TruthCount,
        int PredictionCount,
        int TruePositives,
        int FalsePositives,
        int FalseNegatives,
        int Duplicates,
        bool Exact,
        int TextMarkerCreationCount,
        IReadOnlyDictionary<string, int> ProhibitedHits,
        IReadOnlyList<string> CacheKeys);

    private sealed record PayloadEvidence(string ModelId, string Version, string Sha256)
    {
        public static System.Collections.ObjectModel.ReadOnlyCollection<PayloadEvidence> From(
            OcrV8ProductionPayloadSet payloads,
            ModelIdentity marker) =>
            Array.AsReadOnly(new[]
            {
                Create(payloads.Detector),
                Create(payloads.OfficialRecognizer),
                Create(payloads.NumericRecognizer),
                Create(payloads.AmbiguityRecognizer),
                Create(marker),
            });

        private static PayloadEvidence Create(ModelIdentity model) =>
            new(model.ModelId, model.Version, model.Sha256.ToLowerInvariant());
    }

    private sealed record ModelExecutionEvidence(
        string ModelId,
        string ModelSha256,
        int CallCount,
        IReadOnlyList<string> InputTensorSha256,
        IReadOnlyList<string> OutputTensorSha256,
        IReadOnlyList<string> Providers);

    private sealed class EvidenceInferenceSessionFactory : IInferenceSessionFactory
    {
        private readonly IInferenceSessionFactory inner;
        private readonly ConcurrentDictionary<string, ExecutionCollector> executions =
            new(StringComparer.Ordinal);

        public EvidenceInferenceSessionFactory(IInferenceSessionFactory inner) =>
            this.inner = inner ?? throw new ArgumentNullException(nameof(inner));

        public async ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken)
        {
            IInferenceSession session = await inner.CreateAsync(
                model,
                provider,
                cpuConfiguration,
                cancellationToken);
            ExecutionCollector collector = executions.GetOrAdd(
                model.Sha256.ToLowerInvariant(),
                _ => new ExecutionCollector(model));
            return new EvidenceInferenceSession(session, collector);
        }

        public Dictionary<string, ModelExecutionEvidence> Snapshot() =>
            executions.ToDictionary(
                static item => item.Key,
                static item => item.Value.Snapshot(),
                StringComparer.Ordinal);
    }

    private sealed class EvidenceInferenceSession : IInferenceSession
    {
        private readonly IInferenceSession inner;
        private readonly ExecutionCollector collector;

        public EvidenceInferenceSession(IInferenceSession inner, ExecutionCollector collector)
        {
            this.inner = inner;
            this.collector = collector;
        }

        public InferenceProvider Provider => inner.Provider;

        public async ValueTask<InferenceExecution> RunAsync(
            InferenceInput input,
            CancellationToken cancellationToken)
        {
            string inputHash = Sha256(Float32Bytes(input.Values.Span));
            InferenceExecution execution = await inner.RunAsync(input, cancellationToken);
            collector.Add(
                inputHash,
                Sha256(Float32Bytes(execution.Output.ToArray())),
                execution.Provider);
            return execution;
        }

        public ValueTask DisposeAsync() => inner.DisposeAsync();
    }

    private sealed class ExecutionCollector
    {
        private readonly object sync = new();
        private readonly ModelIdentity model;
        private readonly List<string> inputs = [];
        private readonly List<string> outputs = [];
        private readonly List<string> providers = [];

        public ExecutionCollector(ModelIdentity model) => this.model = model;

        public void Add(string input, string output, InferenceProvider provider)
        {
            lock (sync)
            {
                inputs.Add(input);
                outputs.Add(output);
                providers.Add(provider.ToString());
            }
        }

        public ModelExecutionEvidence Snapshot()
        {
            lock (sync)
            {
                return new ModelExecutionEvidence(
                    model.ModelId,
                    model.Sha256.ToLowerInvariant(),
                    inputs.Count,
                    Array.AsReadOnly(inputs.ToArray()),
                    Array.AsReadOnly(outputs.ToArray()),
                    Array.AsReadOnly(providers.Distinct(StringComparer.Ordinal).Order().ToArray()));
            }
        }
    }

    private sealed class CpuOnlyDiscovery : IExecutionProviderDiscovery
    {
        public IReadOnlyList<string> GetAvailableProviders() => ["CPUExecutionProvider"];
    }

    private sealed class SingleCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }
}
