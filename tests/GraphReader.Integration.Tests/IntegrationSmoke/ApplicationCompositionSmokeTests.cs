// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
using GraphReader.Inference;
using GraphReader.Pdf;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

[TestClass]
public sealed class ApplicationCompositionSmokeTests
{
    private const string ProductionModelStoreEnvironmentVariable =
        "GRAPHREADER_PRODUCTION_MODEL_STORE";

    [TestMethod]
    public void OrdinaryRuntimeSelectionDefaultsToRealEmptyManualPreview()
    {
        Assert.AreEqual(WorkflowRuntimeEnvironment.ManualPreview, RuntimeModeSelector.Select(string.Empty));

        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.ManualPreview);

        Assert.AreEqual(WorkflowRuntimeEnvironment.ManualPreview, composition.Environment);
        var workspace = Assert.IsInstanceOfType<ManualPreviewWorkspaceService>(composition.WorkspaceService);
        Assert.IsNull(composition.StartupError);
        Assert.IsFalse(workspace.UsesFakeGraphData);
        Assert.HasCount(0, workspace.CreateWorkspace());
        Assert.IsTrue(workspace.AutomaticStages.All(status => status.State == AutomaticStageState.Unavailable));
        Assert.IsTrue(workspace.AutomaticStages.All(status => !string.IsNullOrWhiteSpace(status.Explanation)));
    }

    [TestMethod]
    public void RecordedFakeCannotBeSelectedByOrdinaryRuntimeConfiguration()
    {
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.ManualPreview,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.RecordedFake)));
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.Production,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.Production)));
    }

    [TestMethod]
    public void ProductionCompositionKeepsManualWorkflowAvailableAndAutomaticStagesFailClosed()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);

        Assert.AreEqual(WorkflowRuntimeEnvironment.Production, composition.Environment);
        var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
        Assert.IsNull(composition.StartupError);
        Assert.AreEqual(WorkflowRuntimeEnvironment.Production, workspace.RuntimeEnvironment);
        Assert.IsFalse(workspace.UsesFakeGraphData);
        Assert.HasCount(6, workspace.AutomaticStages);
        Assert.IsTrue(workspace.AutomaticStages.All(static stage => stage.State == AutomaticStageState.Unavailable));
        Assert.IsTrue(workspace.AutomaticStages.All(static stage => !string.IsNullOrWhiteSpace(stage.Explanation)));
        Assert.AreEqual(
            "enhancement,axis,ocr,markers,legends,phases",
            string.Join(',', workspace.AutomaticStages.Select(static stage => stage.Stage)));
        Assert.IsInstanceOfType<IAutomaticWorkspaceService>(workspace);
        Assert.IsNull(workspace.LastAutomaticRun);
    }

    [TestMethod]
    public async Task ProductionAutomaticWorkflowRemainsFailClosedWithoutApprovedStages()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);
        var workspace = Assert.IsInstanceOfType<IAutomaticWorkspaceService>(composition.WorkspaceService);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => workspace.RunAutomaticDetectionAsync(CancellationToken.None));

        StringAssert.Contains(exception.Message, "clean-machine runtime approval");
        Assert.IsNull(workspace.LastAutomaticRun);
    }

    [TestMethod]
    public void StageAvailabilityRequiresChecksumResolvedModelsRuntimeAndComposedAdapters()
    {
        var classifierOnly = new ProductionModelAvailabilitySnapshot(
            new HashSet<string>(["marker_classifier"], StringComparer.Ordinal),
            "Checksum-resolved CPU production models: classifier.");

        IReadOnlyList<AutomaticStageStatus> partial = ProductionStageAvailabilityRegistry.Create(
            localEnhancementConfigured: false,
            classifierOnly,
            reviewedPdfiumConfigured: false);

        AutomaticStageStatus markerPartial = partial.Single(stage => stage.Stage == "markers");
        Assert.AreEqual(AutomaticStageState.Unavailable, markerPartial.State);
        StringAssert.Contains(markerPartial.Explanation, "marker_center");
        Assert.IsFalse(
            markerPartial.Explanation.Contains(
                "marker_classifier, marker_center",
                StringComparison.Ordinal));

        var complete = new ProductionModelAvailabilitySnapshot(
            new HashSet<string>(
                ["ocr_detection", "ocr_recognition", "marker_center", "marker_classifier"],
                StringComparer.Ordinal),
            "Checksum-resolved CPU production models: complete fixture.");
        IReadOnlyList<AutomaticStageStatus> modelsWithoutAdapters = ProductionStageAvailabilityRegistry.Create(
            localEnhancementConfigured: false,
            complete,
            reviewedPdfiumConfigured: true,
            new ProductionRuntimeAvailabilitySnapshot(
                true,
                "Checksum-resolved release-approved OpenCV fixture."));

        Assert.AreEqual(
            AutomaticStageState.Unavailable,
            modelsWithoutAdapters.Single(stage => stage.Stage == "axis").State);
        Assert.AreEqual(
            AutomaticStageState.Unavailable,
            modelsWithoutAdapters.Single(stage => stage.Stage == "ocr").State);
        Assert.AreEqual(
            AutomaticStageState.Unavailable,
            modelsWithoutAdapters.Single(stage => stage.Stage == "markers").State);
        StringAssert.Contains(
            modelsWithoutAdapters.Single(stage => stage.Stage == "markers").Explanation,
            "no approved production marker adapter");

        var mutableAdapterStages = new HashSet<string>(
            ["axis", "ocr", "markers", "legends", "phases"],
            StringComparer.Ordinal);
        var completeAdapter = new ProductionDetectionAdapterAvailabilitySnapshot(
            mutableAdapterStages,
            "Checksum-resolved approved production adapter fixture.");
        mutableAdapterStages.Clear();
        Assert.Throws<ArgumentException>(() => new ProductionDetectionAdapterAvailabilitySnapshot(
            ["unregistered"],
            "Invalid fixture."));
        Assert.Throws<ArgumentException>(() => new ProductionDetectionAdapterAvailabilitySnapshot(
            [],
            " "));
        IReadOnlyList<AutomaticStageStatus> resolved = ProductionStageAvailabilityRegistry.Create(
            localEnhancementConfigured: false,
            complete,
            reviewedPdfiumConfigured: true,
            new ProductionRuntimeAvailabilitySnapshot(
                true,
                "Checksum-resolved release-approved OpenCV fixture."),
            inferenceRuntimeConfigured: true,
            completeAdapter);

        Assert.AreEqual(
            AutomaticStageState.Approved,
            resolved.Single(stage => stage.Stage == "axis").State);
        Assert.AreEqual(
            AutomaticStageState.Approved,
            resolved.Single(stage => stage.Stage == "ocr").State);
        Assert.AreEqual(
            AutomaticStageState.Approved,
            resolved.Single(stage => stage.Stage == "markers").State);
        Assert.AreEqual(
            AutomaticStageState.Approved,
            resolved.Single(stage => stage.Stage == "legends").State);
        StringAssert.Contains(
            resolved.Single(stage => stage.Stage == "legends").Explanation,
            "approved");
        Assert.AreEqual(
            AutomaticStageState.Approved,
            resolved.Single(stage => stage.Stage == "phases").State);

        IReadOnlyList<AutomaticStageStatus> missingRuntime = ProductionStageAvailabilityRegistry.Create(
            localEnhancementConfigured: false,
            complete,
            reviewedPdfiumConfigured: true,
            new ProductionRuntimeAvailabilitySnapshot(
                true,
                "Checksum-resolved release-approved OpenCV fixture."),
            inferenceRuntimeConfigured: false);
        Assert.AreEqual(
            AutomaticStageState.Unavailable,
            missingRuntime.Single(stage => stage.Stage == "ocr").State);
        Assert.AreEqual(
            AutomaticStageState.Unavailable,
            missingRuntime.Single(stage => stage.Stage == "markers").State);
        StringAssert.Contains(
            missingRuntime.Single(stage => stage.Stage == "ocr").Explanation,
            "mandatory CPU fallback");
    }

    [TestMethod]
    public async Task ProductionComposesLazyBoundedOnnxRuntimeWithCpuFallback()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.ApplicationComposition.Inference",
            Guid.NewGuid().ToString("N"));
        string modelRoot = Path.Combine(root, "models");
        Directory.CreateDirectory(modelRoot);
        ProductionInferenceRuntimeHost? host = null;
        try
        {
            var paths = new ModelRootApplicationPaths(modelRoot);
            ApplicationCompositionResult composition = await ApplicationComposition.CreateAsync(
                WorkflowRuntimeEnvironment.Production,
                paths,
                applicationRoot: root,
                cancellationToken: CancellationToken.None);

            host = composition.InferenceRuntimeHost;
            Assert.IsNotNull(host);
            Assert.IsFalse(host.IsInitialized);
            Assert.AreEqual(ProductionInferenceRuntimeHost.DefaultQueueCapacity, host.QueueCapacity);
            Assert.AreEqual(ProductionInferenceRuntimeHost.DefaultWorkerCount, host.WorkerCount);
            Assert.AreEqual(InferenceProvider.Cpu, host.ProviderOrder[^1]);
            Assert.IsTrue(host.CpuThreadConfiguration.IntraOperationThreads >= 1);
            Assert.IsTrue(
                host.CacheRoot.StartsWith(Path.GetFullPath(paths.CacheRoot), StringComparison.OrdinalIgnoreCase));

            _ = host.Runtime;
            Assert.IsTrue(host.IsInitialized);
            Assert.IsTrue(Directory.Exists(host.CacheRoot));
        }
        finally
        {
            if (host is not null)
            {
                await host.DisposeAsync();
            }
            Directory.Delete(root, recursive: true);
        }
    }

    [TestMethod]
    public async Task RuntimeAvailabilityRequiresExactBytesAndReleaseApproval()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.ApplicationComposition.Runtime",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string runtimePath = Path.Combine(root, "OpenCvSharpExtern.dll");
            File.WriteAllBytes(runtimePath, [1, 3, 5, 7, 9]);
            WriteRuntimeMetadata(root, RuntimeSha256(runtimePath), cleanMachineEvidence: true, releaseApproved: true);

            ProductionRuntimeAvailabilitySnapshot approved =
                await ProductionRuntimeAvailabilityProbe.InspectAsync(root, CancellationToken.None);
            Assert.IsTrue(approved.AxisApproved);
            Assert.AreEqual(RuntimeSha256(runtimePath), approved.RuntimeSha256);
            StringAssert.Contains(approved.Evidence, RuntimeSha256(runtimePath));

            File.AppendAllText(runtimePath, "tampered", Encoding.UTF8);
            ProductionRuntimeAvailabilitySnapshot tampered =
                await ProductionRuntimeAvailabilityProbe.InspectAsync(root, CancellationToken.None);
            Assert.IsFalse(tampered.AxisApproved);
            StringAssert.Contains(tampered.Evidence, "checksum");

            string currentHash = RuntimeSha256(runtimePath);
            WriteRuntimeMetadata(root, currentHash, cleanMachineEvidence: false, releaseApproved: false);
            ProductionRuntimeAvailabilitySnapshot notApproved =
                await ProductionRuntimeAvailabilityProbe.InspectAsync(root, CancellationToken.None);
            Assert.IsFalse(notApproved.AxisApproved);
            StringAssert.Contains(notApproved.Evidence, "lacks mandatory");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [TestMethod]
    public async Task AsyncProductionCompositionCreatesAxisAdapterFromApprovedExactRuntime()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.ApplicationComposition.Runtime",
            Guid.NewGuid().ToString("N"));
        string modelRoot = Path.Combine(root, "models");
        Directory.CreateDirectory(modelRoot);
        try
        {
            string runtimePath = Path.Combine(root, "OpenCvSharpExtern.dll");
            File.WriteAllBytes(runtimePath, [2, 4, 6, 8]);
            WriteRuntimeMetadata(root, RuntimeSha256(runtimePath), cleanMachineEvidence: true, releaseApproved: true);

            ApplicationCompositionResult composition = await ApplicationComposition.CreateAsync(
                WorkflowRuntimeEnvironment.Production,
                new ModelRootApplicationPaths(modelRoot),
                applicationRoot: root,
                cancellationToken: CancellationToken.None);

            var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
            Assert.AreEqual(
                AutomaticStageState.Approved,
                workspace.AutomaticStages.Single(stage => stage.Stage == "axis").State);
            Assert.IsNotNull(composition.AxisGeometryAdapter);
            Assert.IsTrue(composition.AxisGeometryAdapter.IsApproved);
            StringAssert.Contains(
                workspace.AutomaticStages.Single(stage => stage.Stage == "axis").Explanation,
                RuntimeSha256(runtimePath));
            Assert.AreEqual(
                AutomaticStageState.Unavailable,
                workspace.AutomaticStages.Single(stage => stage.Stage == "ocr").State);
            Assert.IsFalse(workspace.UsesFakeGraphData);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [TestMethod]
    public async Task AsyncProductionCompositionRejectsInvalidInstalledModelEvidence()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.ApplicationComposition",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllText(Path.Combine(root, "production-model-index.json"), "{not-json");
            ApplicationCompositionResult composition = await ApplicationComposition.CreateAsync(
                WorkflowRuntimeEnvironment.Production,
                new ModelRootApplicationPaths(root),
                cancellationToken: CancellationToken.None);

            var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
            AutomaticStageStatus ocr = workspace.AutomaticStages.Single(stage => stage.Stage == "ocr");
            Assert.AreEqual(AutomaticStageState.Unavailable, ocr.State);
            StringAssert.Contains(ocr.Explanation, "MODEL_PACKAGE_INDEX_INVALID");
            Assert.IsFalse(workspace.UsesFakeGraphData);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [TestMethod]
    public async Task AsyncProductionCompositionUsesOnlyChecksumResolvedCpuTasks()
    {
        using var package = ApprovedModelPackageFixture.Create("marker_classifier");

        ApplicationCompositionResult composition = await ApplicationComposition.CreateAsync(
            WorkflowRuntimeEnvironment.Production,
            new ModelRootApplicationPaths(package.Root),
            cancellationToken: CancellationToken.None);

        var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
        AutomaticStageStatus markers = workspace.AutomaticStages.Single(stage => stage.Stage == "markers");
        Assert.AreEqual(AutomaticStageState.Unavailable, markers.State);
        StringAssert.Contains(markers.Explanation, "marker_center");
        StringAssert.Contains(markers.Explanation, "fixture-marker-classifier@1.0.0");
        Assert.IsFalse(
            markers.Explanation.Contains("Missing: marker_classifier", StringComparison.Ordinal));
        Assert.IsNotNull(
            composition.MarkerClassificationAdapter,
            composition.StartupError?.TechnicalMessage);
        Assert.IsTrue(composition.MarkerClassificationAdapter.IsApproved);
        Assert.AreEqual(
            "fixture-marker-classifier",
            composition.MarkerClassificationAdapter.Model.ModelId);
        Assert.IsNotNull(composition.InferenceRuntimeHost);
        Assert.IsFalse(composition.InferenceRuntimeHost.IsInitialized);
        Assert.IsFalse(workspace.UsesFakeGraphData);
    }

    [TestMethod]
    public async Task ExactLocalClassifierManifestComposesLazyProductionAdapter()
    {
        string? modelRoot = Environment.GetEnvironmentVariable(
            ProductionModelStoreEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(modelRoot))
        {
            Assert.Inconclusive(
                $"Set {ProductionModelStoreEnvironmentVariable} to the ignored production model package to run this probe.");
        }

        ApplicationCompositionResult composition = await ApplicationComposition.CreateAsync(
            WorkflowRuntimeEnvironment.Production,
            new ModelRootApplicationPaths(modelRoot),
            cancellationToken: CancellationToken.None);
        try
        {
            Assert.IsNull(composition.StartupError);
            Assert.IsNotNull(composition.MarkerClassificationAdapter);
            Assert.IsTrue(composition.MarkerClassificationAdapter.IsApproved);
            Assert.AreEqual(
                "26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a",
                composition.MarkerClassificationAdapter.Model.Sha256.ToLowerInvariant());
            Assert.IsNotNull(composition.InferenceRuntimeHost);
            Assert.IsFalse(composition.InferenceRuntimeHost.IsInitialized);
        }
        finally
        {
            if (composition.InferenceRuntimeHost is not null)
            {
                await composition.InferenceRuntimeHost.DisposeAsync();
            }
        }
    }

    [TestMethod]
    public void ProductionRejectsMissingPdfiumApprovalWithoutEnablingAutomation()
    {
        string? previous = Environment.GetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(
                ApplicationComposition.PdfiumApprovalEnvironmentVariable,
                Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "reviewed-approval.json"));

            ApplicationCompositionResult composition =
                ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);

            Assert.IsNotNull(composition.StartupError);
            Assert.AreEqual("PDFIUM_APPROVAL_REJECTED", composition.StartupError.Code);
            var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
            Assert.IsTrue(workspace.AutomaticStages.All(static stage =>
                stage.State == AutomaticStageState.Unavailable));
        }
        finally
        {
            Environment.SetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable, previous);
        }
    }

    [TestMethod]
    public void ProductionDiscoversPackagedPdfiumApprovalAndRejectsInvalidBytes()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.ApplicationComposition.PackagedPdfium",
            Guid.NewGuid().ToString("N"));
        string pdfiumRoot = Path.Combine(root, "pdfium");
        string? previous = Environment.GetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable);
        Directory.CreateDirectory(pdfiumRoot);
        try
        {
            Environment.SetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable, null);
            File.WriteAllText(
                Path.Combine(pdfiumRoot, "reviewed-approval.json"),
                "{not-json",
                Encoding.UTF8);

            ApplicationCompositionResult composition = ApplicationComposition.Create(
                WorkflowRuntimeEnvironment.Production,
                applicationRoot: root);

            Assert.IsNotNull(composition.StartupError);
            Assert.AreEqual("PDFIUM_APPROVAL_REJECTED", composition.StartupError.Code);
            StringAssert.Contains(composition.StartupError.TechnicalMessage, "invalid start");
            var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
            Assert.IsTrue(workspace.AutomaticStages.All(static stage =>
                stage.State == AutomaticStageState.Unavailable));
        }
        finally
        {
            Environment.SetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable, previous);
            Directory.Delete(root, recursive: true);
        }
    }

    [TestMethod]
    public void ProductionRejectsMalformedNestedAndAmbiguousPdfiumEvidenceWithoutCrashing()
    {
        string? previous = Environment.GetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable);
        try
        {
            foreach (MalformedEvidenceKind kind in Enum.GetValues<MalformedEvidenceKind>())
            {
                using var fixture = new MalformedPdfiumApprovalFixture(kind);
                Environment.SetEnvironmentVariable(
                    ApplicationComposition.PdfiumApprovalEnvironmentVariable,
                    fixture.ApprovalPath);

                ApplicationCompositionResult composition =
                    ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);

                Assert.IsNotNull(composition.StartupError, kind.ToString());
                Assert.AreEqual("PDFIUM_APPROVAL_REJECTED", composition.StartupError.Code, kind.ToString());
                StringAssert.Contains(composition.StartupError.TechnicalMessage, fixture.ExpectedDiagnostic);
                var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
                Assert.IsTrue(
                    workspace.AutomaticStages.All(static stage => stage.State == AutomaticStageState.Unavailable),
                    kind.ToString());
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(ApplicationComposition.PdfiumApprovalEnvironmentVariable, previous);
        }
    }

    [TestMethod]
    public void RecordedFakeRequiresExplicitEnvironmentSelection()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.RecordedFake);

        Assert.AreEqual(WorkflowRuntimeEnvironment.RecordedFake, composition.Environment);
        Assert.IsInstanceOfType<FakeWorkspaceService>(composition.WorkspaceService);
        Assert.IsNull(composition.StartupError);
        Assert.HasCount(1, composition.WorkspaceService.CreateWorkspace());
    }

    private enum MalformedEvidenceKind
    {
        MissingManifestFeatures,
        WrongShapedSourceLockSources,
        DuplicateApprovalField,
        UnexpectedApprovalField,
    }

    private static void WriteRuntimeMetadata(
        string root,
        string binarySha256,
        bool cleanMachineEvidence,
        bool releaseApproved)
    {
        var metadata = new Dictionary<string, object?>
        {
            ["schema"] = "graphreader.reviewed-opencv-runtime.v1",
            ["runtimeId"] = "opencvsharpextern-source-audited",
            ["profileId"] = "graphreader-axis-minimal-win-x64",
            ["evidenceRootName"] = "integration-fixture",
            ["binarySha256"] = binarySha256.ToLowerInvariant(),
            ["replacedBinarySha256"] = new string('a', 64),
            ["sourceRevisions"] = new Dictionary<string, object?>
            {
                ["openCvSharp"] = "fixture-opencvsharp",
                ["openCv"] = "fixture-opencv",
                ["vcpkg"] = "fixture-vcpkg",
            },
            ["provenanceValidated"] = true,
            ["noticeReviewStatus"] = "complete",
            ["maintainerAttestationStatus"] = "recorded-private",
            ["cleanMachineEvidence"] = cleanMachineEvidence,
            ["releaseApproved"] = releaseApproved,
        };
        File.WriteAllText(
            Path.Combine(root, "reviewed-opencv-runtime.json"),
            JsonSerializer.Serialize(metadata),
            Encoding.UTF8);
    }

    private static string RuntimeSha256(string path) =>
        Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));

    private sealed class ApprovedModelPackageFixture : IDisposable
    {
        private ApprovedModelPackageFixture(string root)
        {
            Root = root;
        }

        public string Root { get; }

        public static ApprovedModelPackageFixture Create(string task)
        {
            string root = Path.Combine(
                Path.GetTempPath(),
                "GraphReader.ApplicationComposition.Models",
                Guid.NewGuid().ToString("N"));
            string modelId = $"fixture-{task.Replace('_', '-')}";
            const string version = "1.0.0";
            string runtimeDirectory = Path.Combine(root, "runtime", modelId, version);
            string manifestDirectory = Path.Combine(root, "manifest", modelId, version);
            string noticeDirectory = Path.Combine(root, "notices", modelId, version);
            string evidenceDirectory = Path.Combine(root, "evidence", modelId, version);
            Directory.CreateDirectory(runtimeDirectory);
            Directory.CreateDirectory(manifestDirectory);
            Directory.CreateDirectory(noticeDirectory);
            Directory.CreateDirectory(evidenceDirectory);

            string payloadPath = Path.Combine(runtimeDirectory, "model.onnx");
            File.WriteAllBytes(payloadPath, [1, 2, 3, 4]);
            string payloadSha256 = Sha256(payloadPath);
            string noticePath = Path.Combine(noticeDirectory, "fixture-model.txt");
            File.WriteAllText(noticePath, "Apache-2.0 synthetic packaging fixture.", Encoding.UTF8);
            string noticeSha256 = Sha256(noticePath);
            string evidencePath = Path.Combine(evidenceDirectory, "fixture-benchmark.json");
            File.WriteAllText(evidencePath, "{\"status\":\"pass\"}", Encoding.UTF8);
            string evidenceSha256 = Sha256(evidencePath);

            var manifest = new Dictionary<string, object?>
            {
                ["manifest_version"] = 1,
                ["model_id"] = modelId,
                ["model_version"] = version,
                ["task"] = task,
                ["source"] = new Dictionary<string, object?>
                {
                    ["name"] = "Application composition synthetic fixture",
                    ["url"] = "local://application-composition-fixture",
                    ["revision"] = "1",
                },
                ["license"] = new Dictionary<string, object?>
                {
                    ["spdx"] = "Apache-2.0",
                    ["notice_path"] = "LICENSES/fixture-model.txt",
                    ["reviewed"] = true,
                },
                ["sha256"] = payloadSha256,
                ["files"] = new[] { "model.onnx" },
                ["inputs"] = task == "marker_classifier"
                    ? new[]
                    {
                        new Dictionary<string, object?>
                        {
                            ["name"] = "marker_patch",
                            ["shape"] = new object[] { "N", 1, 32, 32 },
                        },
                    }
                    : new[] { new Dictionary<string, object?> { ["name"] = "input" } },
                ["outputs"] = task == "marker_classifier"
                    ? new[]
                    {
                        new Dictionary<string, object?>
                        {
                            ["name"] = "classification_probabilities",
                            ["shape"] = new object[] { "N", 25 },
                            ["order"] = new[]
                            {
                                "shape_probabilities[9]",
                                "fill_probabilities[3]",
                                "artifact_probability[1]",
                                "l2_normalized_embedding[12]",
                            },
                        },
                    }
                    : new[] { new Dictionary<string, object?> { ["name"] = "output" } },
                ["commercial_use"] = true,
                ["redistribution"] = true,
                ["providers"] = new[] { "cpu" },
                ["benchmarks"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["profile"] = "application-composition-v1",
                        ["status"] = "pass",
                        ["release_eligible"] = true,
                        ["production_approval"] = true,
                        ["evidence_path"] = "artifacts/evidence/fixture-benchmark.json",
                        ["evidence_sha256"] = evidenceSha256,
                    },
                },
            };
            if (task == "marker_classifier")
            {
                manifest["preprocessing"] = new Dictionary<string, object?>
                {
                    ["normalization_mean"] = 0f,
                    ["normalization_scale"] = 1f,
                };
                manifest["postprocessing"] = new Dictionary<string, object?>
                {
                    ["shape_order"] = new[]
                    {
                        "circle", "square", "triangle_up", "triangle_down", "diamond",
                        "star", "asterisk", "cross", "other",
                    },
                    ["fill_order"] = new[] { "filled", "open", "unknown" },
                    ["shape_and_fill_separate"] = true,
                };
            }

            string manifestPath = Path.Combine(manifestDirectory, "manifest.json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest), Encoding.UTF8);
            string manifestSha256 = Sha256(manifestPath);

            var index = new Dictionary<string, object?>
            {
                ["schema_version"] = 1,
                ["models"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["model_id"] = modelId,
                        ["model_version"] = version,
                        ["manifest"] = new Dictionary<string, object?>
                        {
                            ["path"] = $"manifest/{modelId}/{version}/manifest.json",
                            ["sha256"] = manifestSha256,
                        },
                        ["payloads"] = new[]
                        {
                            new Dictionary<string, object?>
                            {
                                ["declared_path"] = "model.onnx",
                                ["path"] = $"runtime/{modelId}/{version}/model.onnx",
                                ["sha256"] = payloadSha256,
                            },
                        },
                        ["notice"] = new Dictionary<string, object?>
                        {
                            ["declared_path"] = "LICENSES/fixture-model.txt",
                            ["path"] = $"notices/{modelId}/{version}/fixture-model.txt",
                            ["sha256"] = noticeSha256,
                        },
                        ["benchmark_evidence"] = new Dictionary<string, object?>
                        {
                            ["declared_path"] = "artifacts/evidence/fixture-benchmark.json",
                            ["path"] = $"evidence/{modelId}/{version}/fixture-benchmark.json",
                            ["sha256"] = evidenceSha256,
                        },
                    },
                },
            };
            File.WriteAllText(
                Path.Combine(root, "production-model-index.json"),
                JsonSerializer.Serialize(index),
                Encoding.UTF8);
            return new ApprovedModelPackageFixture(root);
        }

        public void Dispose()
        {
            if (Directory.Exists(Root))
            {
                Directory.Delete(Root, recursive: true);
            }
        }

        private static string Sha256(string path) =>
            Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
    }

    private sealed class ModelRootApplicationPaths(string modelRoot) : GraphReader.Domain.IApplicationPaths
    {
        private readonly string _mutableRoot = Path.Combine(modelRoot, "mutable");

        public GraphReader.Domain.DistributionMode Mode => GraphReader.Domain.DistributionMode.Portable;

        public string SettingsRoot => Path.Combine(_mutableRoot, "Settings");

        public string CacheRoot => Path.Combine(_mutableRoot, "Cache");

        public string LogsRoot => Path.Combine(_mutableRoot, "Logs");

        public string AutosaveRoot => Path.Combine(_mutableRoot, "Autosave");

        public string RecoveryRoot => Path.Combine(_mutableRoot, "Recovery");

        public string ModelRoot { get; } = Path.GetFullPath(modelRoot);
    }

    private sealed class MalformedPdfiumApprovalFixture : IDisposable
    {
        private readonly string _root;

        public MalformedPdfiumApprovalFixture(MalformedEvidenceKind kind)
        {
            _root = Path.Combine(
                Path.GetTempPath(),
                "GraphReader.Integration.Tests.Pdfium",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_root);
            ApprovalPath = Path.Combine(_root, "reviewed-approval.json");
            string binaryPath = Path.Combine(_root, "graphreader_pdfium_renderer.exe");
            string sourceLockPath = Path.Combine(_root, "source-lock.json");
            string manifestPath = Path.Combine(_root, "build-manifest.json");
            string noticePath = Path.Combine(_root, "third-party-notices.reviewed.txt");

            File.WriteAllBytes(binaryPath, "controlled-runner"u8.ToArray());
            File.WriteAllText(
                noticePath,
                "REVIEW STATUS: COMPLETE\nControlled integration notice.\n",
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            JsonNode sources = kind == MalformedEvidenceKind.WrongShapedSourceLockSources
                ? JsonValue.Create("not-an-object")!
                : new JsonObject
                {
                    ["pdfium"] = new JsonObject
                    {
                        ["repository"] = ReviewedPdfiumPageRendererBackend.PinnedSource,
                        ["revision"] = ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                        ["rootBuildGnBlob"] = new string('1', 40),
                        ["renderDeviceHeaderBlob"] = new string('2', 40),
                        ["license"] = "BSD-3-Clause",
                    },
                    ["depotTools"] = new JsonObject(),
                };
            var sourceLock = new JsonObject
            {
                ["schemaVersion"] = 1,
                ["profileId"] = ReviewedPdfiumPageRendererBackend.RequiredProfileId,
                ["compatibilityPatchSha256"] = new string('3', 64),
                ["sources"] = sources,
                ["target"] = new JsonObject
                {
                    ["os"] = "win",
                    ["cpu"] = "x64",
                    ["configuration"] = "Release",
                    ["binaryName"] = "graphreader_pdfium_renderer.exe",
                    ["maxParallelCompileJobs"] = 4,
                    ["v8"] = false,
                    ["xfa"] = false,
                    ["skia"] = false,
                    ["fontations"] = false,
                    ["partitionAlloc"] = false,
                    ["icuDataFile"] = false,
                },
                ["toolchain"] = new JsonObject(),
            };
            File.WriteAllText(sourceLockPath, sourceLock.ToJsonString(), Encoding.UTF8);

            string binarySha256 = Hash(binaryPath);
            string sourceLockSha256 = Hash(sourceLockPath);
            var manifest = new JsonObject
            {
                ["schemaVersion"] = 1,
                ["profileId"] = ReviewedPdfiumPageRendererBackend.RequiredProfileId,
                ["generatedUtc"] = "2026-08-04T00:00:00Z",
                ["reviewStatus"] = "requires-review",
                ["source"] = ReviewedPdfiumPageRendererBackend.PinnedSource,
                ["sourceRevision"] = ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                ["sourceLockSha256"] = sourceLockSha256,
                ["argsGnSha256"] = new string('4', 64),
                ["overlayBuildSha256"] = new string('5', 64),
                ["overlayRootTargetSha256"] = new string('6', 64),
                ["overlaySourceSha256"] = new string('7', 64),
                ["compatibilityPatchSha256"] = new string('8', 64),
                ["targetDependenciesSha256"] = new string('9', 64),
                ["peImportsSha256"] = new string('a', 64),
                ["binarySha256"] = binarySha256,
                ["warning"] = "Controlled integration manifest.",
            };
            if (kind != MalformedEvidenceKind.MissingManifestFeatures)
            {
                manifest["features"] = new JsonObject
                {
                    ["v8"] = false,
                    ["xfa"] = false,
                    ["skia"] = false,
                    ["icuDataFile"] = false,
                };
            }
            File.WriteAllText(manifestPath, manifest.ToJsonString(), Encoding.UTF8);

            var approval = new JsonObject
            {
                ["schemaVersion"] = 1,
                ["rendererId"] = "graphreader-pdfium-renderer",
                ["rendererVersion"] = ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                ["binaryPath"] = Path.GetFileName(binaryPath),
                ["binarySha256"] = binarySha256,
                ["source"] = ReviewedPdfiumPageRendererBackend.PinnedSource,
                ["sourceRevision"] = ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                ["sourceLockPath"] = Path.GetFileName(sourceLockPath),
                ["sourceLockSha256"] = sourceLockSha256,
                ["buildManifestPath"] = Path.GetFileName(manifestPath),
                ["buildManifestSha256"] = Hash(manifestPath),
                ["licenseSpdx"] = "BSD-3-Clause",
                ["noticePath"] = Path.GetFileName(noticePath),
                ["noticeSha256"] = Hash(noticePath),
                ["reviewApproved"] = true,
                ["redistributionApproved"] = true,
                ["bundlingApproved"] = true,
            };
            if (kind == MalformedEvidenceKind.UnexpectedApprovalField)
            {
                approval["unexpected"] = "not-reviewed";
            }
            File.WriteAllText(ApprovalPath, approval.ToJsonString(), Encoding.UTF8);
            if (kind == MalformedEvidenceKind.DuplicateApprovalField)
            {
                string text = File.ReadAllText(ApprovalPath, Encoding.UTF8);
                const string marker = "\"reviewApproved\":true";
                File.WriteAllText(
                    ApprovalPath,
                    text.Replace(marker, $"{marker},{marker}", StringComparison.Ordinal),
                    Encoding.UTF8);
            }

            ExpectedDiagnostic = kind switch
            {
                MalformedEvidenceKind.MissingManifestFeatures => "missing field 'features'",
                MalformedEvidenceKind.WrongShapedSourceLockSources => "field 'sources' must be a JSON object",
                MalformedEvidenceKind.DuplicateApprovalField => "duplicate field 'reviewApproved'",
                MalformedEvidenceKind.UnexpectedApprovalField => "unexpected field 'unexpected'",
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }

        public string ApprovalPath { get; }

        public string ExpectedDiagnostic { get; }

        public void Dispose()
        {
            try
            {
                Directory.Delete(_root, recursive: true);
            }
            catch (IOException)
            {
            }
        }

        private static string Hash(string path) =>
            Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    }
}
