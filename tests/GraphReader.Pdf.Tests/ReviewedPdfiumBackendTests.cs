// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using GraphReader.Pdf;

namespace GraphReader.Pdf.Tests;

[TestClass]
public sealed class ReviewedPdfiumBackendTests
{
    [TestMethod]
    public void ApprovalFailsClosedWhenBinaryIsAbsent()
    {
        using var fixture = new ApprovalFixture();
        File.Delete(fixture.BinaryPath);

        _ = Assert.ThrowsExactly<FileNotFoundException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
    }

    [TestMethod]
    public void ApprovalRejectsCandidateWhoseApprovalFlagsRemainFalse()
    {
        using var fixture = new ApprovalFixture();
        fixture.SetApprovalBoolean("reviewApproved", false);

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "reviewApproved");
    }

    [TestMethod]
    public void ApprovalFailsClosedWhenNoticeChecksumDoesNotMatch()
    {
        using var fixture = new ApprovalFixture();
        File.AppendAllText(fixture.NoticePath, "changed", Encoding.UTF8);

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "notice SHA-256");
    }

    [TestMethod]
    public void ApprovalRejectsAnySourceRevisionOtherThanPinnedOfficialRevision()
    {
        using var fixture = new ApprovalFixture(sourceRevision: new string('b', 40));

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, ReviewedPdfiumPageRendererBackend.PinnedSourceRevision);
    }

    [TestMethod]
    public void ApprovalRejectsWrongShapedBuildManifestFeaturesAsInvalidData()
    {
        using var fixture = new ApprovalFixture();
        fixture.SetBuildManifestNode("features", "not-an-object");

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "features");
        StringAssert.Contains(exception.Message, "JSON object");
    }

    [TestMethod]
    public void ApprovalRejectsMissingNestedSourceLockObjectAsInvalidData()
    {
        using var fixture = new ApprovalFixture();
        fixture.RemoveSourceLockProperty("sources");

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "source lock");
        StringAssert.Contains(exception.Message, "missing field 'sources'");
    }

    [TestMethod]
    public void ApprovalRejectsDuplicateJsonProperty()
    {
        using var fixture = new ApprovalFixture();
        fixture.DuplicateApprovalProperty("reviewApproved");

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "duplicate field 'reviewApproved'");
    }

    [TestMethod]
    public void ApprovalRejectsUnexpectedJsonProperty()
    {
        using var fixture = new ApprovalFixture();
        fixture.SetApprovalPath("unexpected", "not-reviewed");

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "unexpected field 'unexpected'");
    }

    [TestMethod]
    public void ApprovalRejectsDuplicateNestedFeatureProperty()
    {
        using var fixture = new ApprovalFixture();
        fixture.DuplicateBuildManifestProperty("v8", "false");

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "duplicate field 'v8'");
    }

    [TestMethod]
    public void ApprovalRejectsUnexpectedNestedTargetProperty()
    {
        using var fixture = new ApprovalFixture();
        fixture.SetSourceLockTargetNode("unexpected", true);

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "unexpected field 'unexpected'");
    }

    [TestMethod]
    public void ApprovalRejectsAbsoluteResourcePath()
    {
        using var fixture = new ApprovalFixture();
        fixture.SetApprovalPath("noticePath", fixture.NoticePath);

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "must be relative");
    }

    [TestMethod]
    public void ApprovalRejectsResourcePathTraversal()
    {
        using var fixture = new ApprovalFixture();
        fixture.SetApprovalPath("binaryPath", Path.Combine("..", Path.GetFileName(fixture.BinaryPath)));

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
            () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
        StringAssert.Contains(exception.Message, "must remain under");
    }

    [TestMethod]
    public void ApprovalRejectsResourceBehindDirectoryJunction()
    {
        if (!OperatingSystem.IsWindows())
        {
            Assert.Inconclusive("Directory junction validation is Windows-specific.");
        }

        using var fixture = new ApprovalFixture();
        string externalRoot = Path.Combine(Path.GetTempPath(), "GraphReader.Pdf.Tests.External", Guid.NewGuid().ToString("N"));
        string junctionPath = Path.Combine(fixture.Root, "linked");
        Directory.CreateDirectory(externalRoot);
        File.Copy(fixture.BinaryPath, Path.Combine(externalRoot, Path.GetFileName(fixture.BinaryPath)));

        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "cmd.exe",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                ArgumentList =
                {
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    junctionPath,
                    externalRoot,
                },
            })!;
            process.WaitForExit();
            Assert.AreEqual(0, process.ExitCode, $"mklink failed: {process.StandardError.ReadToEnd()}");

            fixture.SetApprovalPath(
                "binaryPath",
                Path.Combine("linked", Path.GetFileName(fixture.BinaryPath)));
            InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(
                () => ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, new FakeRunner()));
            StringAssert.Contains(exception.Message, "reparse point");
        }
        finally
        {
            if (Directory.Exists(junctionPath))
            {
                Directory.Delete(junctionPath);
            }

            Directory.Delete(externalRoot, recursive: true);
        }
    }

    [TestMethod]
    public async Task BackendRechecksApprovedBinaryBeforeEveryRender()
    {
        using var fixture = new ApprovalFixture();
        var runner = new FakeRunner();
        ReviewedPdfiumPageRendererBackend backend =
            ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, runner);
        File.AppendAllBytes(fixture.BinaryPath, [0xff]);

        PdfiumBackendRenderResult result = await backend.RenderPageAsync(
            new PdfiumBackendRenderRequest(
                new ImmutableByteBuffer("%PDF-1.7"u8),
                new string('a', 64),
                pageNumber: 1,
                dpi: 144),
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("PDFIUM_APPROVAL_INPUT_CHANGED", result.Failure!.Code);
        Assert.AreEqual(0, runner.CallCount);
    }

    [TestMethod]
    public async Task ReviewedBackendRendersControlledBgraArtifactThroughExactApproval()
    {
        using var fixture = new ApprovalFixture();
        var runner = new FakeRunner();
        ReviewedPdfiumPageRendererBackend backend =
            ReviewedPdfiumPageRendererBackend.Load(fixture.ApprovalPath, runner);
        IPdfPageRenderingService service = backend.CreateRenderingService();

        PdfPageRenderResult result = await service.RenderAsync(
            new PdfPageRenderRequest("%PDF-1.7\ncontrolled"u8.ToArray(), 1, 144),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded);
        Assert.AreEqual(PdfPageRenderStatus.Succeeded, result.Status);
        Assert.AreEqual(2, result.Page!.Width);
        Assert.AreEqual(1, result.Page.Height);
        Assert.AreEqual("image/png", result.Page.MediaType);
        Assert.AreEqual(ReviewedPdfiumPageRendererBackend.PinnedSourceRevision, result.Metadata!.RendererVersion);
        Assert.AreEqual(1, runner.CallCount);

        string? evidenceDirectory = Environment.GetEnvironmentVariable("GRAPHREADER_PDFIUM_TEST_EVIDENCE_DIR");
        if (!string.IsNullOrWhiteSpace(evidenceDirectory))
        {
            Directory.CreateDirectory(evidenceDirectory);
            File.WriteAllBytes(Path.Combine(evidenceDirectory, "reviewed-render.png"), result.Page.PngBytes.ToArray());
            File.Copy(fixture.BinaryPath, Path.Combine(evidenceDirectory, Path.GetFileName(fixture.BinaryPath)), overwrite: true);
            File.Copy(fixture.SourceLockPath, Path.Combine(evidenceDirectory, Path.GetFileName(fixture.SourceLockPath)), overwrite: true);
            File.Copy(fixture.BuildManifestPath, Path.Combine(evidenceDirectory, Path.GetFileName(fixture.BuildManifestPath)), overwrite: true);
            File.Copy(fixture.NoticePath, Path.Combine(evidenceDirectory, Path.GetFileName(fixture.NoticePath)), overwrite: true);
            File.Copy(fixture.ApprovalPath, Path.Combine(evidenceDirectory, "reviewed-approval.controlled-test.json"), overwrite: true);
        }
    }

    [TestMethod]
    public async Task ExactIgnoredReviewedRunnerRendersControlledPdf()
    {
        string? approvalPath = Environment.GetEnvironmentVariable(
            "GRAPHREADER_PDFIUM_APPROVAL_PATH");
        string? inputPath = Environment.GetEnvironmentVariable(
            "GRAPHREADER_PDFIUM_CONTROLLED_INPUT_PATH");
        if (string.IsNullOrWhiteSpace(approvalPath) ||
            string.IsNullOrWhiteSpace(inputPath))
        {
            Assert.Inconclusive(
                "Set GRAPHREADER_PDFIUM_APPROVAL_PATH and GRAPHREADER_PDFIUM_CONTROLLED_INPUT_PATH to run the exact ignored local PDFium evidence.");
        }

        byte[] pdfBytes = await File.ReadAllBytesAsync(inputPath);
        string pdfSha256 = Convert.ToHexStringLower(SHA256.HashData(pdfBytes));
        ReviewedPdfiumPageRendererBackend backend =
            ReviewedPdfiumPageRendererBackend.Load(approvalPath);
        IPdfPageRenderingService service = backend.CreateRenderingService();

        PdfPageRenderResult result = await service.RenderAsync(
            new PdfPageRenderRequest(pdfBytes, pageNumber: 1, dpi: 144),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(PdfPageRenderStatus.Succeeded, result.Status);
        Assert.IsGreaterThan(0, result.Page!.Width);
        Assert.IsGreaterThan(0, result.Page.Height);
        CollectionAssert.AreEqual(
            new byte[] { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a },
            result.Page.PngBytes.ToArray()[..8]);
        Assert.AreEqual(pdfSha256, result.Metadata!.PdfSha256);
        Assert.AreEqual(
            ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
            result.Metadata.RendererVersion);
        Assert.AreEqual(
            "efd13a38cf3cd8e04d8284a42fff42923267293170424153b1a2a96dbf6fe8ea",
            result.Metadata.RendererSha256);
    }

    [TestMethod]
    public async Task TempCleanupRetriesUntilRunnerFileHandleIsReleased()
    {
        if (!OperatingSystem.IsWindows())
        {
            Assert.Inconclusive("The locked-file cleanup behavior is Windows-specific.");
        }

        string tempRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.Pdf.Tests.Cleanup",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string rawPath = Path.Combine(tempRoot, "page.raw");
        await File.WriteAllBytesAsync(rawPath, [1, 2, 3]);

        await using FileStream lockStream = new(
            rawPath,
            FileMode.Open,
            FileAccess.ReadWrite,
            FileShare.None);
        Task cleanup = LocalPdfiumRunnerProcess.DeleteDirectoryWithRetriesAsync(tempRoot);
        await Task.Delay(120);
        await lockStream.DisposeAsync();
        await cleanup;

        Assert.IsFalse(Directory.Exists(tempRoot));
    }

    [TestMethod]
    public void PngEncoderCancelsDuringScanlineStreaming()
    {
        const int width = 64;
        const int height = 64;
        byte[] bgra = new byte[width * height * 4];
        using var cancellation = new CancellationTokenSource();
        var page = new PdfiumRawPage(new ImmutableByteBuffer(bgra), width, height, width * 4);

        _ = Assert.ThrowsExactly<OperationCanceledException>(() =>
            PdfiumPngEncoder.Encode(
                page,
                scanlineCompleted: row =>
                {
                    if (row == 0)
                    {
                        cancellation.Cancel();
                    }
                },
                cancellation.Token));
    }

    [TestMethod]
    public void PngEncoderRejectsOverLimitLayoutWithoutAllocatingPixels()
    {
        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(() =>
            PdfiumPngEncoder.ValidateLayout(
                pixelBufferLength: 0,
                width: 10_000,
                height: 5_000,
                stride: 40_000));

        StringAssert.Contains(exception.Message, "160 MiB managed pixel limit");
    }

    [TestMethod]
    public void PngEncoderEmitsBoundedMultipleIdatChunks()
    {
        const int width = 512;
        const int height = 512;
        byte[] bgra = new byte[width * height * 4];
        new Random(1904).NextBytes(bgra);
        var page = new PdfiumRawPage(new ImmutableByteBuffer(bgra), width, height, width * 4);

        byte[] png = PdfiumPngEncoder.Encode(page, CancellationToken.None);
        int idatChunks = CountPngChunks(png, "IDAT"u8);

        Assert.IsGreaterThan(1, idatChunks);
    }

    private static int CountPngChunks(ReadOnlySpan<byte> png, ReadOnlySpan<byte> requestedType)
    {
        int offset = 8;
        int count = 0;
        while (offset < png.Length)
        {
            int length = checked((int)BinaryPrimitives.ReadUInt32BigEndian(png.Slice(offset, 4)));
            ReadOnlySpan<byte> type = png.Slice(offset + 4, 4);
            if (type.SequenceEqual(requestedType))
            {
                count++;
            }

            offset = checked(offset + 12 + length);
        }

        Assert.AreEqual(png.Length, offset);
        return count;
    }

    private sealed class FakeRunner : IPdfiumRunnerProcess
    {
        public int CallCount { get; private set; }

        public Task<PdfiumRawPage> RenderAsync(
            string binaryPath,
            ImmutableByteBuffer pdfBytes,
            int pageNumber,
            int dpi,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CallCount++;
            byte[] bgra =
            [
                0x00, 0x00, 0xff, 0xff,
                0x00, 0xff, 0x00, 0xff,
            ];
            return Task.FromResult(new PdfiumRawPage(new ImmutableByteBuffer(bgra), 2, 1, 8));
        }
    }

    private sealed class ApprovalFixture : IDisposable
    {
        private readonly string _root;

        public ApprovalFixture(string? sourceRevision = null)
        {
            _root = Path.Combine(Path.GetTempPath(), "GraphReader.Pdf.Tests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_root);
            BinaryPath = Path.Combine(_root, "graphreader_pdfium_renderer.exe");
            SourceLockPath = Path.Combine(_root, "source-lock.json");
            BuildManifestPath = Path.Combine(_root, "build-manifest.json");
            NoticePath = Path.Combine(_root, "third-party-notices.reviewed.txt");
            ApprovalPath = Path.Combine(_root, "reviewed-approval.json");

            File.WriteAllBytes(BinaryPath, "controlled-pdfium-runner"u8.ToArray());
            File.WriteAllText(
                SourceLockPath,
                JsonSerializer.Serialize(new
                {
                    schemaVersion = 1,
                    profileId = ReviewedPdfiumPageRendererBackend.RequiredProfileId,
                    compatibilityPatchSha256 = new string('1', 64),
                    sources = new
                    {
                        pdfium = new
                        {
                            repository = ReviewedPdfiumPageRendererBackend.PinnedSource,
                            revision = ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                            rootBuildGnBlob = new string('2', 40),
                            renderDeviceHeaderBlob = new string('3', 40),
                            license = "BSD-3-Clause",
                        },
                        depotTools = new { },
                    },
                    target = new
                    {
                        os = "win",
                        cpu = "x64",
                        configuration = "Release",
                        binaryName = "graphreader_pdfium_renderer.exe",
                        maxParallelCompileJobs = 4,
                        v8 = false,
                        xfa = false,
                        skia = false,
                        fontations = false,
                        partitionAlloc = false,
                        icuDataFile = false,
                    },
                    toolchain = new { },
                }),
                Encoding.UTF8);
            File.WriteAllText(
                NoticePath,
                "REVIEW STATUS: COMPLETE\nControlled test notice.\n",
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            string binarySha256 = Hash(BinaryPath);
            string sourceLockSha256 = Hash(SourceLockPath);
            File.WriteAllText(
                BuildManifestPath,
                JsonSerializer.Serialize(new
                {
                    schemaVersion = 1,
                    profileId = ReviewedPdfiumPageRendererBackend.RequiredProfileId,
                    generatedUtc = "2026-08-04T00:00:00Z",
                    reviewStatus = "requires-review",
                    source = ReviewedPdfiumPageRendererBackend.PinnedSource,
                    sourceRevision = ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                    sourceLockSha256,
                    argsGnSha256 = new string('4', 64),
                    overlayBuildSha256 = new string('5', 64),
                    overlayRootTargetSha256 = new string('6', 64),
                    overlaySourceSha256 = new string('7', 64),
                    compatibilityPatchSha256 = new string('8', 64),
                    targetDependenciesSha256 = new string('9', 64),
                    peImportsSha256 = new string('a', 64),
                    binarySha256,
                    features = new { v8 = false, xfa = false, skia = false, icuDataFile = false },
                    warning = "Controlled test manifest.",
                }),
                Encoding.UTF8);

            File.WriteAllText(
                ApprovalPath,
                JsonSerializer.Serialize(new
                {
                    schemaVersion = 1,
                    rendererId = "graphreader-pdfium-renderer",
                    rendererVersion = sourceRevision ?? ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                    binaryPath = Path.GetFileName(BinaryPath),
                    binarySha256,
                    source = ReviewedPdfiumPageRendererBackend.PinnedSource,
                    sourceRevision = sourceRevision ?? ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                    sourceLockPath = Path.GetFileName(SourceLockPath),
                    sourceLockSha256,
                    buildManifestPath = Path.GetFileName(BuildManifestPath),
                    buildManifestSha256 = Hash(BuildManifestPath),
                    licenseSpdx = "BSD-3-Clause",
                    noticePath = Path.GetFileName(NoticePath),
                    noticeSha256 = Hash(NoticePath),
                    reviewApproved = true,
                    redistributionApproved = true,
                    bundlingApproved = true,
                }),
                Encoding.UTF8);
        }

        public string BinaryPath { get; }

        public string SourceLockPath { get; }

        public string BuildManifestPath { get; }

        public string NoticePath { get; }

        public string ApprovalPath { get; }

        public string Root => _root;

        public void SetApprovalPath(string propertyName, string value)
        {
            JsonObject approval = JsonNode.Parse(File.ReadAllText(ApprovalPath, Encoding.UTF8))!.AsObject();
            approval[propertyName] = value;
            File.WriteAllText(ApprovalPath, approval.ToJsonString(), Encoding.UTF8);
        }

        public void SetApprovalBoolean(string propertyName, bool value)
        {
            JsonObject approval = JsonNode.Parse(File.ReadAllText(ApprovalPath, Encoding.UTF8))!.AsObject();
            approval[propertyName] = value;
            File.WriteAllText(ApprovalPath, approval.ToJsonString(), Encoding.UTF8);
        }

        public void SetBuildManifestNode(string propertyName, JsonNode value)
        {
            JsonObject manifest = JsonNode.Parse(File.ReadAllText(BuildManifestPath, Encoding.UTF8))!.AsObject();
            manifest[propertyName] = value;
            File.WriteAllText(BuildManifestPath, manifest.ToJsonString(), Encoding.UTF8);
            RefreshApprovalHashes();
        }

        public void RemoveSourceLockProperty(string propertyName)
        {
            JsonObject sourceLock = JsonNode.Parse(File.ReadAllText(SourceLockPath, Encoding.UTF8))!.AsObject();
            sourceLock.Remove(propertyName);
            File.WriteAllText(SourceLockPath, sourceLock.ToJsonString(), Encoding.UTF8);

            JsonObject manifest = JsonNode.Parse(File.ReadAllText(BuildManifestPath, Encoding.UTF8))!.AsObject();
            manifest["sourceLockSha256"] = Hash(SourceLockPath);
            File.WriteAllText(BuildManifestPath, manifest.ToJsonString(), Encoding.UTF8);
            RefreshApprovalHashes();
        }

        public void DuplicateApprovalProperty(string propertyName)
        {
            string approval = File.ReadAllText(ApprovalPath, Encoding.UTF8);
            string marker = $"\"{propertyName}\":true";
            string duplicated = approval.Replace(marker, $"{marker},{marker}", StringComparison.Ordinal);
            Assert.AreNotEqual(approval, duplicated);
            File.WriteAllText(ApprovalPath, duplicated, Encoding.UTF8);
        }

        public void DuplicateBuildManifestProperty(string propertyName, string serializedValue)
        {
            string manifest = File.ReadAllText(BuildManifestPath, Encoding.UTF8);
            string marker = $"\"{propertyName}\":{serializedValue}";
            string duplicated = manifest.Replace(marker, $"{marker},{marker}", StringComparison.Ordinal);
            Assert.AreNotEqual(manifest, duplicated);
            File.WriteAllText(BuildManifestPath, duplicated, Encoding.UTF8);
            RefreshApprovalHashes();
        }

        public void SetSourceLockTargetNode(string propertyName, JsonNode value)
        {
            JsonObject sourceLock = JsonNode.Parse(File.ReadAllText(SourceLockPath, Encoding.UTF8))!.AsObject();
            sourceLock["target"]!.AsObject()[propertyName] = value;
            File.WriteAllText(SourceLockPath, sourceLock.ToJsonString(), Encoding.UTF8);

            JsonObject manifest = JsonNode.Parse(File.ReadAllText(BuildManifestPath, Encoding.UTF8))!.AsObject();
            manifest["sourceLockSha256"] = Hash(SourceLockPath);
            File.WriteAllText(BuildManifestPath, manifest.ToJsonString(), Encoding.UTF8);
            RefreshApprovalHashes();
        }

        private void RefreshApprovalHashes()
        {
            JsonObject approval = JsonNode.Parse(File.ReadAllText(ApprovalPath, Encoding.UTF8))!.AsObject();
            approval["sourceLockSha256"] = Hash(SourceLockPath);
            approval["buildManifestSha256"] = Hash(BuildManifestPath);
            File.WriteAllText(ApprovalPath, approval.ToJsonString(), Encoding.UTF8);
        }

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
