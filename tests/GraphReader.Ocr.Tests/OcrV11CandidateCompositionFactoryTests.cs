// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV11CandidateCompositionFactoryTests
{
    [TestMethod]
    public void ExactCandidateAndReviewedRecognitionPayloadsAreAcceptedWithoutApproval()
    {
        string repoRoot = FindRepositoryRoot();
        OcrV8ProductionPayloadSet payloads = Payloads(repoRoot);

        OcrV11CandidateCompositionFactory.ValidatePayloads(payloads);

        Assert.AreEqual(
            OcrV11CandidateCompositionFactory.DetectorSha256,
            payloads.Detector.Sha256);
        Assert.AreNotEqual(
            OcrV8ProductionCompositionFactory.DetectorSha256,
            payloads.Detector.Sha256);
    }

    [TestMethod]
    public void CandidateFactoryRejectsLegacyDetectorAndUnreviewedProviderPolicy()
    {
        string repoRoot = FindRepositoryRoot();
        OcrV8ProductionPayloadSet payloads = Payloads(repoRoot);
        OcrV8ProductionPayloadSet legacyDetector = payloads with
        {
            Detector = payloads.Detector with
            {
                ModelId = "graph-text-spaced-component-recall-v10-p2",
                Sha256 = OcrV8ProductionCompositionFactory.DetectorSha256,
            },
        };

        InvalidDataException identityError = Assert.ThrowsExactly<InvalidDataException>(() =>
            OcrV11CandidateCompositionFactory.ValidatePayloads(legacyDetector));
        StringAssert.Contains(identityError.Message, "unreviewed identity");

    }

    [TestMethod]
    public async Task ExactCandidateFactoryBuildsOnlyTheLabeledCandidateComposition()
    {
        string repoRoot = FindRepositoryRoot();
        string cacheRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderV11CandidateFactoryTests",
            Guid.NewGuid().ToString("N"));
        try
        {
            var registry = new OnnxSessionRegistry(
                new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
                new WindowsExecutionProviderPolicy(),
                new FakeInferenceSessionFactory(),
                CpuThreadConfiguration.Create(1));
            await using var runtime = new InferenceRuntime(
                registry,
                new BoundedInferenceScheduler(2, 1),
                new ContentAddressedStageCache(cacheRoot));

            OcrV8ProductionCompositionPipeline pipeline =
                OcrV11CandidateCompositionFactory.Create(
                    runtime,
                    Payloads(repoRoot),
                    [InferenceProvider.Cpu],
                    bypassCache: true);

            Assert.AreEqual(OcrV11CandidateCompositionFactory.CandidateCompositionId, pipeline.CompositionId);
            Assert.AreEqual(64, pipeline.ConfigurationFingerprint.Length);
        }
        finally
        {
            if (Directory.Exists(cacheRoot))
            {
                Directory.Delete(cacheRoot, recursive: true);
            }
        }
    }

    private static OcrV8ProductionPayloadSet Payloads(string repoRoot)
    {
        ModelIdentity Model(string id, string version, string path)
        {
            string absolute = Path.Combine(repoRoot, path.Replace('/', Path.DirectorySeparatorChar));
            Assert.IsTrue(File.Exists(absolute), $"Candidate payload is missing: {absolute}");
            string sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(absolute)));
            return new ModelIdentity(id, version, sha256, absolute);
        }

        string yamlPath = Path.Combine(
            repoRoot,
            "ml",
            "ocr",
            "official_bakeoff",
            "runs",
            "extracted",
            "en_PP-OCRv5_mobile_rec_infer",
            "inference.yml");
        Assert.IsTrue(File.Exists(yamlPath), $"Official inference YAML is missing: {yamlPath}");
        string alphabet = ReadOfficialAlphabet(yamlPath);
        return new OcrV8ProductionPayloadSet(
            Model(
                OcrV11CandidateCompositionFactory.DetectorModelId,
                "0.0.21-p2",
                "ml/ocr/composite_proposal_role_v11/artifacts/P2-run/graph-text-composite-proposal-role-v11-p2.onnx"),
            Model(
                "en_PP-OCRv5_mobile_rec",
                "0.0.21-converted",
                "ml/ocr/official_bakeoff/runs/conversion/en_PP-OCRv5_mobile_rec.onnx"),
            Model(
                "graph-numeric-component-ensemble-v5",
                "0.0.21-p1",
                "ml/ocr/component_ensemble_v5/artifacts/P1-run/graph-numeric-component-ensemble-v5-p1.onnx"),
            Model(
                "graph-ambiguity-source-group-v3-p2",
                "0.0.21-p2",
                "ml/ocr/ambiguity_source_group_classifier_v3/artifacts/P2-run/graph-ambiguity-source-group-v3-p2.onnx"),
            alphabet);
    }

    private static string ReadOfficialAlphabet(string path)
    {
        string[] lines = File.ReadAllLines(path, Encoding.UTF8);
        int start = Array.FindIndex(lines, static line =>
            string.Equals(line.Trim(), "character_dict:", StringComparison.Ordinal));
        Assert.IsGreaterThanOrEqualTo(0, start, "Official character_dict is absent.");
        var values = new List<string>();
        for (int index = start + 1; index < lines.Length; index++)
        {
            string line = lines[index];
            if (!line.StartsWith("  - ", StringComparison.Ordinal))
            {
                break;
            }

            string scalar = line[4..].Trim();
            string value = scalar.Length >= 2 && scalar[0] == '\'' && scalar[^1] == '\''
                ? scalar[1..^1].Replace("''", "'", StringComparison.Ordinal)
                : scalar.Length >= 2 && scalar[0] == '"' && scalar[^1] == '"'
                    ? JsonSerializer.Deserialize<string>(scalar) ?? string.Empty
                    : scalar;
            Assert.AreEqual(1, value.EnumerateRunes().Count());
            values.Add(value);
        }

        if (!values.Contains(" ", StringComparer.Ordinal))
        {
            values.Add(" ");
        }

        return string.Concat(values);
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
}
