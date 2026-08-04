// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
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

        StringAssert.Contains(exception.Message, "native runtime");
        Assert.IsNull(workspace.LastAutomaticRun);
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
