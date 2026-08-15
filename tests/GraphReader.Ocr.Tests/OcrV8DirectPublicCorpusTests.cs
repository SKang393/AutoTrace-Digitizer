// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;
using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV8DirectPublicCorpusTests
{
    private const string RunVariable = "GRAPHREADER_RUN_OCR_V8_CSHARP_PUBLIC";
    private const string ArchiveVariable = "GRAPHREADER_OCR_V8_PUBLIC_FIXTURES";
    private const string DetectorVariable = "GRAPHREADER_OCR_V8_DETECTOR";
    private const string OfficialVariable = "GRAPHREADER_OCR_V8_OFFICIAL";
    private const string NumericVariable = "GRAPHREADER_OCR_V8_NUMERIC";
    private const string AmbiguityVariable = "GRAPHREADER_OCR_V8_AMBIGUITY";
    private const string YamlVariable = "GRAPHREADER_OCR_V8_OFFICIAL_YAML";
    private const string ReportVariable = "GRAPHREADER_OCR_V8_CSHARP_REPORT";
    private const string SourceCommitVariable = "GRAPHREADER_OCR_V8_CSHARP_SOURCE_COMMIT";
    private const string ArchiveSha256 =
        "f138e6c8524557e60d3d78bef80b06be39c457bd749fce15b063b41d483b9327";
    private const string ManifestSha256 =
        "b40fb6c0d8c366e650980a3d9278a7b8c3df13258bf58c3dce21be431fe22d55";
    private const string InferenceYamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";
    private const double TruthMatchIouMinimum = 0.5;
    private static readonly OcrRectangle PlotBounds = new(104, 48, 406, 208);
    private static readonly JsonSerializerOptions ReportJsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    [TestMethod]
    public async Task ExactSealedPublicBytesExecuteOnceThroughCSharpCpuComposition()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable(RunVariable), "1", StringComparison.Ordinal))
        {
            Assert.Inconclusive(
                $"Set {RunVariable}=1 plus the seven exact paths and source commit to run the direct public gate.");
        }

        string archivePath = RequiredPath(ArchiveVariable);
        string detectorPath = RequiredPath(DetectorVariable);
        string officialPath = RequiredPath(OfficialVariable);
        string numericPath = RequiredPath(NumericVariable);
        string ambiguityPath = RequiredPath(AmbiguityVariable);
        string yamlPath = RequiredPath(YamlVariable);
        string reportPath = RequiredOutputPath(ReportVariable);
        string sourceCommit = RequiredSourceCommit();
        AssertHash(archivePath, ArchiveSha256, "sealed public archive");
        AssertHash(yamlPath, InferenceYamlSha256, "official inference YAML");
        string alphabet = ReadOfficialAlphabet(yamlPath);

        OcrV8ProductionPayloadSet payloads = Payloads(
            detectorPath,
            officialPath,
            numericPath,
            ambiguityPath,
            alphabet);
        var evidenceFactory = new EvidenceInferenceSessionFactory(
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance));
        var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            evidenceFactory,
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        string cacheRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderOcrV8CSharpPublic",
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

            DirectGateReport report = await EvaluateAsync(
                archivePath,
                pipeline,
                evidenceFactory,
                payloads,
                sourceCommit,
                cancellationToken: CancellationToken.None);
            byte[] reportBytes = JsonSerializer.SerializeToUtf8Bytes(
                report,
                ReportJsonOptions);
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            await File.WriteAllBytesAsync(reportPath, reportBytes);

            Assert.IsTrue(report.GatesPassed, JsonSerializer.Serialize(report.Metrics));
            Assert.IsFalse(report.MarkerCreationEvaluated);
            Assert.IsFalse(report.ProductionApproval);
            Assert.IsFalse(report.ReleaseEligible);
            Assert.AreEqual(4, evidenceFactory.ModelExecutions.Count);
            Assert.IsTrue(evidenceFactory.ModelExecutions.All(static item =>
                item.Value.CallCount > 0 &&
                item.Value.InputTensorSha256.Count == item.Value.CallCount &&
                item.Value.OutputTensorSha256.Count == item.Value.CallCount));
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
        OcrV8ProductionCompositionPipeline pipeline,
        EvidenceInferenceSessionFactory evidenceFactory,
        OcrV8ProductionPayloadSet payloads,
        string sourceCommit,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(archivePath);
        ZipArchiveEntry manifestEntry = archive.GetEntry("manifest.json") ??
            throw new InvalidDataException("Sealed public archive has no manifest.json.");
        byte[] manifestBytes = await ReadEntryAsync(manifestEntry, cancellationToken);
        Assert.AreEqual(ManifestSha256, Sha256(manifestBytes), "Manifest bytes changed.");
        FixtureManifest manifest = JsonSerializer.Deserialize<FixtureManifest>(
            manifestBytes,
            ReportJsonOptions) ??
            throw new InvalidDataException("Fixture manifest is invalid.");
        Assert.AreEqual("graphreader.ocr-production-composition-fixtures.v8", manifest.Schema);
        Assert.AreEqual("sealed_public", manifest.Split);
        Assert.AreEqual(160, manifest.SceneCount);
        Assert.AreEqual(800, manifest.TruthRegionCount);
        Assert.IsTrue(manifest.SyntheticOnly);
        Assert.IsFalse(manifest.PrivateOrArticleImages);
        Assert.IsFalse(manifest.ChandlerIncluded);
        Assert.IsFalse(manifest.GeneralizationLabelIncluded);

        var scenes = new List<SceneEvidence>(manifest.Cases.Count);
        var familyCounts = new Dictionary<string, (int Correct, int Total)>(StringComparer.Ordinal);
        var totalCorrect = 0;
        var roleCorrect = 0;
        var totalCharacters = 0;
        var editErrors = 0;
        var truePositives = 0;
        var falsePositives = 0;
        var falseNegatives = 0;
        var duplicates = 0;
        var exactScenes = 0;

        foreach (FixtureCase fixture in manifest.Cases)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ZipArchiveEntry imageEntry = archive.GetEntry(fixture.ImagePath) ??
                throw new InvalidDataException($"Fixture image is missing: {fixture.ImagePath}");
            byte[] sourceBytes = await ReadEntryAsync(imageEntry, cancellationToken);
            Assert.AreEqual(fixture.ImageSha256, Sha256(sourceBytes), $"PNG changed: {fixture.SceneId}");
            (byte[] pixels, int width, int height) = DecodeGray8(sourceBytes);
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
                    "ocr-v8-direct-public",
                    fixture.SceneId,
                    fixture.ImageSha256,
                    image,
                    PlotBounds),
                cancellationToken);
            Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);

            HashSet<int> matchedTruths = [];
            var predictions = new List<PredictionEvidence>(result.Regions.Count);
            var sceneFalsePositives = 0;
            var sceneDuplicates = 0;
            foreach (OcrRegion region in result.Regions)
            {
                (int Index, double Iou)[] matches = fixture.Truths
                    .Select((truth, index) => (Index: index, Iou: IntersectionOverUnion(region.Polygon.Bounds, truth.Box)))
                    .Where(static item => item.Iou >= TruthMatchIouMinimum)
                    .OrderByDescending(static item => item.Iou)
                    .ThenBy(static item => item.Index)
                    .ToArray();
                if (matches.Length == 0)
                {
                    sceneFalsePositives++;
                    predictions.Add(new PredictionEvidence(
                        region.Text,
                        region.Role.ToString(),
                        Rectangle(region.Polygon.Bounds),
                        null,
                        null,
                        false,
                        false,
                        0));
                    continue;
                }

                int truthIndex = matches[0].Index;
                if (!matchedTruths.Add(truthIndex))
                {
                    sceneDuplicates++;
                    continue;
                }

                FixtureTruth truth = fixture.Truths[truthIndex];
                bool exact = string.Equals(region.Text, truth.TruthText, StringComparison.Ordinal);
                bool roleMatch = string.Equals(
                    RoleName(region.Role),
                    truth.Role,
                    StringComparison.Ordinal);
                totalCorrect += exact ? 1 : 0;
                roleCorrect += roleMatch ? 1 : 0;
                int distance = LevenshteinDistance(truth.TruthText, region.Text);
                totalCharacters += truth.TruthText.EnumerateRunes().Count();
                editErrors += distance;
                (int correct, int total) = familyCounts.GetValueOrDefault(truth.Family);
                familyCounts[truth.Family] = (correct + (exact ? 1 : 0), total + 1);
                predictions.Add(new PredictionEvidence(
                    region.Text,
                    RoleName(region.Role),
                    Rectangle(region.Polygon.Bounds),
                    truth.TruthText,
                    truth.Role,
                    exact,
                    roleMatch,
                    matches[0].Iou));
            }

            int sceneFalseNegatives = fixture.Truths.Count - matchedTruths.Count;
            bool sceneExact = sceneFalsePositives == 0 &&
                sceneFalseNegatives == 0 &&
                sceneDuplicates == 0 &&
                result.Regions.Count == fixture.Truths.Count;
            exactScenes += sceneExact ? 1 : 0;
            truePositives += matchedTruths.Count;
            falsePositives += sceneFalsePositives;
            falseNegatives += sceneFalseNegatives;
            duplicates += sceneDuplicates;
            scenes.Add(new SceneEvidence(
                fixture.SceneId,
                fixture.ImageSha256,
                fixture.RasterSha256,
                fixture.Truths.Count,
                result.Regions.Count,
                matchedTruths.Count,
                sceneFalsePositives,
                sceneFalseNegatives,
                sceneDuplicates,
                sceneExact,
                predictions));
        }

        double recognitionExact = totalCorrect / (double)manifest.TruthRegionCount;
        double cer = editErrors / (double)totalCharacters;
        double roleAccuracy = roleCorrect / (double)manifest.TruthRegionCount;
        double numericExact = Ratio(familyCounts, "numeric");
        double wordExact = Ratio(familyCounts, "word");
        double ambiguityExact = Ratio(familyCounts, "ambiguity");
        var metrics = new DirectMetrics(
            manifest.SceneCount,
            manifest.TruthRegionCount,
            exactScenes,
            truePositives,
            falsePositives,
            falseNegatives,
            duplicates,
            recognitionExact,
            cer,
            roleAccuracy,
            numericExact,
            wordExact,
            ambiguityExact,
            falsePositives);
        Dictionary<string, ModelExecutionEvidence> modelExecutions = evidenceFactory.Snapshot();
        bool directRuntimeEvidencePassed =
            modelExecutions.Count == 4 &&
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
        bool gatesPassed = directRuntimeEvidencePassed &&
            exactScenes == manifest.SceneCount &&
            truePositives == manifest.TruthRegionCount &&
            falsePositives == 0 && falseNegatives == 0 && duplicates == 0 &&
            recognitionExact >= 0.90 && cer <= 0.05 && roleAccuracy >= 0.90 &&
            numericExact >= 0.90 && wordExact >= 0.90 && ambiguityExact >= 0.90;

        return new DirectGateReport(
            "graphreader.ocr-production-composition-csharp-public-gate.v1",
            OcrV8ProductionCompositionOptions.ReviewedCompositionId,
            sourceCommit,
            ArchiveSha256,
            ManifestSha256,
            pipeline.ConfigurationFingerprint,
            "CPUExecutionProvider",
            PayloadEvidence.From(payloads),
            modelExecutions,
            metrics,
            scenes,
            gatesPassed,
            MarkerCreationEvaluated: false,
            ProductionApproval: false,
            ReleaseEligible: false,
            BlockingGates:
            [
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

    internal static string ReadOfficialAlphabet(string path)
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

    internal static (byte[] Pixels, int Width, int Height) DecodeGray8(byte[] source)
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
                    throw new InvalidDataException(
                        "Public fixture PNG must be non-interlaced Gray8 with standard compression and filtering.");
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

        if (width <= 0 || height <= 0)
        {
            throw new InvalidDataException("Public fixture PNG has no valid IHDR.");
        }

        compressed.Position = 0;
        using var inflated = new MemoryStream(checked(height * (width + 1)));
        using (var zlib = new ZLibStream(compressed, CompressionMode.Decompress, leaveOpen: true))
        {
            zlib.CopyTo(inflated);
        }

        byte[] filtered = inflated.ToArray();
        if (filtered.Length != checked(height * (width + 1)))
        {
            throw new InvalidDataException("Public fixture PNG inflated raster length is invalid.");
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

    internal static async Task<byte[]> ReadEntryAsync(
        ZipArchiveEntry entry,
        CancellationToken cancellationToken)
    {
        await using Stream source = entry.Open();
        using var destination = new MemoryStream(checked((int)entry.Length));
        await source.CopyToAsync(destination, cancellationToken);
        return destination.ToArray();
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
        Assert.IsFalse(File.Exists(path), $"One-time C# public report already exists: {path}");
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

    internal static string Sha256(ReadOnlySpan<byte> bytes) =>
        Convert.ToHexStringLower(SHA256.HashData(bytes));

    internal static double IntersectionOverUnion(OcrRectangle left, int[] right)
    {
        double intersectionWidth = Math.Max(0, Math.Min(left.Right, right[2]) - Math.Max(left.Left, right[0]));
        double intersectionHeight = Math.Max(0, Math.Min(left.Bottom, right[3]) - Math.Max(left.Top, right[1]));
        double intersection = intersectionWidth * intersectionHeight;
        double union = (left.Width * left.Height) +
            ((right[2] - right[0]) * (right[3] - right[1])) - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static double[] Rectangle(OcrRectangle value) =>
        [value.Left, value.Top, value.Right, value.Bottom];

    internal static string RoleName(OcrTextRole role) => role switch
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

    internal static int LevenshteinDistance(string expected, string actual)
    {
        string[] left = expected.EnumerateRunes().Select(static rune => rune.ToString()).ToArray();
        string[] right = actual.EnumerateRunes().Select(static rune => rune.ToString()).ToArray();
        var prior = Enumerable.Range(0, right.Length + 1).ToArray();
        var current = new int[right.Length + 1];
        for (var leftIndex = 1; leftIndex <= left.Length; leftIndex++)
        {
            current[0] = leftIndex;
            for (var rightIndex = 1; rightIndex <= right.Length; rightIndex++)
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
        string DisplayText,
        string Role,
        string Family)
    {
        [JsonIgnore]
        public int[] Box => Bbox;
    }

    private sealed record DirectGateReport(
        string Schema,
        string CompositionId,
        string SourceCommit,
        string FixtureArchiveSha256,
        string FixtureManifestSha256,
        string ConfigurationFingerprint,
        string Provider,
        IReadOnlyList<PayloadEvidence> Payloads,
        IReadOnlyDictionary<string, ModelExecutionEvidence> ModelExecutions,
        DirectMetrics Metrics,
        IReadOnlyList<SceneEvidence> Scenes,
        bool GatesPassed,
        bool MarkerCreationEvaluated,
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
        double NumericExactMatch,
        double WordExactMatch,
        double AmbiguityExactMatch,
        int ProhibitedStructureHits);

    private sealed record SceneEvidence(
        string SceneId,
        string SourcePngSha256,
        string SourceRasterSha256,
        int TruthRegionCount,
        int AcceptedRegionCount,
        int TruePositives,
        int FalsePositives,
        int FalseNegatives,
        int DuplicateRegionCount,
        bool ExactDetection,
        IReadOnlyList<PredictionEvidence> Predictions);

    private sealed record PredictionEvidence(
        string Prediction,
        string PredictedRole,
        double[] PredictionBbox,
        string? TruthText,
        string? TruthRole,
        bool Exact,
        bool RoleCorrect,
        double TruthIou);

    private sealed record PayloadEvidence(string ModelId, string Version, string Sha256)
    {
        public static System.Collections.ObjectModel.ReadOnlyCollection<PayloadEvidence> From(
            OcrV8ProductionPayloadSet payloads) =>
            Array.AsReadOnly(new[]
            {
                Create(payloads.Detector),
                Create(payloads.OfficialRecognizer),
                Create(payloads.NumericRecognizer),
                Create(payloads.AmbiguityRecognizer),
            });

        private static PayloadEvidence Create(ModelIdentity model) =>
            new(model.ModelId, model.Version, model.Sha256.ToLowerInvariant());
    }

    internal sealed record ModelExecutionEvidence(
        string ModelId,
        string ModelSha256,
        int CallCount,
        IReadOnlyList<string> InputTensorSha256,
        IReadOnlyList<string> OutputTensorSha256,
        IReadOnlyList<string> Providers);

    internal sealed class EvidenceInferenceSessionFactory : IInferenceSessionFactory
    {
        private readonly IInferenceSessionFactory inner;
        private readonly ConcurrentDictionary<string, ExecutionCollector> executions =
            new(StringComparer.Ordinal);

        public EvidenceInferenceSessionFactory(IInferenceSessionFactory inner) =>
            this.inner = inner ?? throw new ArgumentNullException(nameof(inner));

        public Dictionary<string, ModelExecutionEvidence> ModelExecutions => Snapshot();

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
            string inputHash = HashFloatTensor(input.Values.Span);
            InferenceExecution execution = await inner.RunAsync(input, cancellationToken);
            collector.Add(inputHash, HashFloatTensor(execution.Output.ToArray()), execution.Provider);
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

    private sealed class SingleCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }

    private static string HashFloatTensor(ReadOnlySpan<float> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        foreach (float value in values)
        {
            BitConverter.TryWriteBytes(bytes, value);
            hash.AppendData(bytes);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
}
