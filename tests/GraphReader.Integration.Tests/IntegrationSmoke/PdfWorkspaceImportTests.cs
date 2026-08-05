// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using GraphReader.App.Integration;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
using GraphReader.App.ViewModels;
using GraphReader.Domain;
using GraphReader.Imaging;
using GraphReader.Pdf;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

[TestClass]
public sealed class PdfWorkspaceImportTests
{
    [TestMethod]
    public async Task ManualCompositionImportsBornDigitalPdfThroughRealPdfPipeline()
    {
        using var directory = new TemporaryDirectory();
        string pdfPath = Path.Combine(directory.Path, "born digital article.pdf");
        byte[] pdfBytes = CreateBornDigitalPdf();
        await File.WriteAllBytesAsync(pdfPath, pdfBytes);
        ApplicationCompositionResult composition = ApplicationComposition.Create(
            WorkflowRuntimeEnvironment.ManualPreview,
            applicationRoot: directory.Path);
        var workspace = Assert.IsInstanceOfType<ManualPreviewWorkspaceService>(
            composition.WorkspaceService);

        IReadOnlyList<WorkspaceTabViewModel> tabs = await workspace.ImportImagesAsync(
            [pdfPath],
            CancellationToken.None);
        Assert.IsTrue(
            tabs.Count > 0,
            string.Join(
                " | ",
                workspace.LastImportErrors.Select(static error =>
                    $"{error.Code}:{error.TechnicalMessage}")));
        WorkspaceTabViewModel tab = tabs.Single();

        Assert.IsFalse(workspace.UsesFakeGraphData);
        Assert.IsNotNull(tab.ImageSource);
        Assert.AreEqual(640, tab.PixelWidth);
        Assert.AreEqual(480, tab.PixelHeight);
        Assert.AreEqual(1, tab.PageNumber);
        Assert.AreEqual(pdfPath, tab.SourcePath);
        Assert.IsEmpty(workspace.LastImportErrors);
        SourceReference source = workspace.CurrentProject.Sources.Single();
        Assert.AreEqual(SourceKind.Pdf, source.Kind);
        Assert.AreEqual(
            Convert.ToHexStringLower(SHA256.HashData(pdfBytes)),
            source.Sha256);
        Assert.AreEqual(
            source.Sha256,
            Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(pdfPath))));
    }

    [TestMethod]
    public async Task PdfPanelsOpenAsRealTabsAndReopenWithStableIdentity()
    {
        using var directory = new TemporaryDirectory();
        string pdfPath = Path.Combine(directory.Path, "article graph.pdf");
        byte[] pdfBytes = "%PDF-1.7\n% procedural workspace test\n"u8.ToArray();
        await File.WriteAllBytesAsync(pdfPath, pdfBytes);
        byte[] panelPng = CreatePanelPng();
        string unchangedPdfSha256 = Convert.ToHexStringLower(SHA256.HashData(pdfBytes));
        var pdfImporter = new DeterministicPdfImportService(panelPng);
        var workspace = new ManualPreviewWorkspaceService(pdfImportService: pdfImporter);

        WorkspaceTabViewModel tab = (await workspace.ImportImagesAsync(
            [pdfPath],
            CancellationToken.None)).Single();

        Assert.IsNotNull(tab.ImageSource);
        Assert.AreEqual(32, tab.PixelWidth);
        Assert.AreEqual(24, tab.PixelHeight);
        Assert.AreEqual(2, tab.PageNumber);
        Assert.AreEqual(pdfPath, tab.SourcePath);
        Assert.AreEqual(Convert.ToHexStringLower(SHA256.HashData(panelPng)), tab.SourceSha256);
        SourceReference source = workspace.CurrentProject.Sources.Single();
        Assert.AreEqual(SourceKind.Pdf, source.Kind);
        Assert.AreEqual(unchangedPdfSha256, source.Sha256);
        PanelRecord panel = workspace.CurrentProject.Panels.Single();
        Assert.AreEqual(tab.PanelId, panel.PanelId.Value.ToString("D"));
        Assert.AreEqual(2, panel.PageNumber);
        Assert.AreEqual(unchangedPdfSha256, Convert.ToHexStringLower(SHA256.HashData(
            await File.ReadAllBytesAsync(pdfPath))));

        string projectPath = Path.Combine(directory.Path, "pdf-workspace.garproj");
        DomainResult<ProjectSaveReceipt> saved = await workspace.SaveProjectAsync(
            projectPath,
            CancellationToken.None);
        Assert.IsTrue(saved.IsSuccess);

        var reopenedWorkspace = new ManualPreviewWorkspaceService(pdfImportService: pdfImporter);
        WorkspaceTabViewModel reopened = (await reopenedWorkspace.OpenProjectAsync(
            projectPath,
            CancellationToken.None)).Single();

        Assert.AreEqual(tab.PanelId, reopened.PanelId);
        Assert.AreEqual(tab.SourceSha256, reopened.SourceSha256);
        Assert.AreEqual(2, reopened.PageNumber);
        Assert.IsNotNull(reopened.ImageSource);
        Assert.IsEmpty(reopenedWorkspace.LastImportErrors);
        Assert.AreEqual(2, pdfImporter.CallCount);
    }

    [TestMethod]
    public async Task ExactReviewedPdfiumRendersScannedPageIntoRealWorkspaceTab()
    {
        string? approvalPath = Environment.GetEnvironmentVariable(
            ApplicationComposition.PdfiumApprovalEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(approvalPath))
        {
            Assert.Inconclusive(
                $"Set {ApplicationComposition.PdfiumApprovalEnvironmentVariable} to run the exact ignored local scanned-PDF workflow.");
        }

        using var directory = new TemporaryDirectory();
        string pdfPath = Path.Combine(directory.Path, "procedural scanned graph.pdf");
        byte[] pdfBytes = CreateAnnotationScannedPdf();
        await File.WriteAllBytesAsync(pdfPath, pdfBytes);
        string sourceSha256 = Convert.ToHexStringLower(SHA256.HashData(pdfBytes));
        string? previous = Environment.GetEnvironmentVariable(
            ApplicationComposition.PdfiumApprovalEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(
                ApplicationComposition.PdfiumApprovalEnvironmentVariable,
                approvalPath);
            ReviewedPdfiumPageRendererBackend backend =
                ReviewedPdfiumPageRendererBackend.Load(approvalPath);
            var importer = new PdfImportService(
                new PdfPigDocumentInspector(),
                new PanelizationEngine(),
                backend.CreateRenderingService());
            PdfImportResult imported = await importer.ImportAsync(
                new PdfImportRequest(
                    Guid.Parse("3a08308e-c867-4df7-9184-a6f70a6e7efd"),
                    Guid.Parse("386d0644-c94a-43ec-b62c-91ebde44014f"),
                    new ImmutableByteBuffer(pdfBytes),
                    Path.GetFileName(pdfPath),
                    Password: null,
                    new PdfPanelizationOptions()),
                CancellationToken.None);

            Assert.IsTrue(imported.Succeeded, string.Join(" | ", imported.Failures.Select(static failure => failure.TechnicalMessage)));
            Assert.IsGreaterThan(0d, imported.Timing.RenderingMilliseconds);
            Assert.IsNotEmpty(imported.Panels);
            Assert.IsTrue(imported.Figures.Any(static figure =>
                figure.SourceKind == PdfFigureSourceKind.RenderedPage));

            ApplicationCompositionResult composition = ApplicationComposition.Create(
                WorkflowRuntimeEnvironment.ManualPreview,
                applicationRoot: directory.Path);
            Assert.IsNull(composition.StartupError);
            var workspace = Assert.IsInstanceOfType<ManualPreviewWorkspaceService>(
                composition.WorkspaceService);
            IReadOnlyList<WorkspaceTabViewModel> tabs = await workspace.ImportImagesAsync(
                [pdfPath],
                CancellationToken.None);

            Assert.IsTrue(
                tabs.Count > 0,
                string.Join(
                    " | ",
                    workspace.LastImportErrors.Select(static error =>
                        $"{error.Code}:{error.TechnicalMessage}")));
            Assert.IsTrue(tabs.All(static tab => tab.ImageSource is not null));
            Assert.IsTrue(tabs.All(static tab => tab.PageNumber == 1));
            Assert.IsFalse(workspace.UsesFakeGraphData);
            Assert.IsEmpty(workspace.LastImportErrors);
            Assert.AreEqual(sourceSha256, workspace.CurrentProject.Sources.Single().Sha256);
            Assert.AreEqual(
                sourceSha256,
                Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(pdfPath))));
        }
        finally
        {
            Environment.SetEnvironmentVariable(
                ApplicationComposition.PdfiumApprovalEnvironmentVariable,
                previous);
        }
    }

    [TestMethod]
    public async Task PdfWithoutConfiguredImporterFailsWithoutAddingTabsOrSources()
    {
        using var directory = new TemporaryDirectory();
        string pdfPath = Path.Combine(directory.Path, "unavailable.pdf");
        await File.WriteAllBytesAsync(pdfPath, "%PDF-1.7\n"u8.ToArray());
        var workspace = new ManualPreviewWorkspaceService();

        IReadOnlyList<WorkspaceTabViewModel> tabs = await workspace.ImportImagesAsync(
            [pdfPath],
            CancellationToken.None);

        Assert.IsEmpty(tabs);
        Assert.IsEmpty(workspace.CurrentProject.Sources);
        Assert.IsEmpty(workspace.CurrentProject.Panels);
        Assert.HasCount(1, workspace.LastImportErrors);
        Assert.AreEqual(ImageImportErrorCode.UnsupportedFormat, workspace.LastImportErrors[0].Code);
        Assert.AreEqual("Errors.PdfRendererUnavailable", workspace.LastImportErrors[0].UserMessageKey);
    }

    private static byte[] CreatePanelPng()
    {
        const int width = 32;
        const int height = 24;
        byte[] pixels = Enumerable.Repeat((byte)0xff, width * height).ToArray();
        for (int x = 4; x < 29; x++)
        {
            pixels[(20 * width) + x] = 0x20;
        }

        for (int y = 3; y < 21; y++)
        {
            pixels[(y * width) + 4] = 0x20;
        }

        BitmapSource bitmap = BitmapSource.Create(
            width,
            height,
            96,
            96,
            PixelFormats.Gray8,
            palette: null,
            pixels,
            stride: width);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = new MemoryStream();
        encoder.Save(stream);
        return stream.ToArray();
    }

    private static byte[] CreateAnnotationScannedPdf()
    {
        var appearance = new StringBuilder();
        appearance.Append("q\n1 1 1 rg\n0 0 540 648 re f\n");
        appearance.Append("0 0 0 RG\n2 w\n60 60 m 500 60 l S\n60 60 m 60 590 l S\n");
        appearance.Append("1 w\n250 60 m 250 590 l S\n400 60 m 400 590 l S\n");
        int[] markerX = [90, 125, 160, 195, 230, 275, 310, 345, 380, 425, 460, 490];
        int[] markerY = [120, 160, 145, 220, 205, 300, 270, 360, 340, 455, 430, 510];
        appearance.Append("2 w\n90 120 m\n");
        for (int index = 1; index < markerX.Length; index++)
        {
            appearance.Append(markerX[index]).Append(' ').Append(markerY[index]).Append(" l\n");
        }

        appearance.Append("S\n0 0 0 rg\n");
        for (int index = 0; index < markerX.Length; index++)
        {
            appearance.Append(markerX[index] - 5).Append(' ')
                .Append(markerY[index] - 5).Append(" 10 10 re f\n");
        }

        appearance.Append("Q\n");
        string appearanceStream = appearance.ToString();
        int appearanceLength = Encoding.ASCII.GetByteCount(appearanceStream);
        string[] objects =
        [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Annots [4 0 R] >>",
            "<< /Type /Annot /Subtype /Stamp /Rect [36 72 576 720] /F 4 /AP << /N 5 0 R >> >>",
            $"<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 540 648] /Resources << >> /Length {appearanceLength} >>\nstream\n{appearanceStream}endstream",
        ];

        using var output = new MemoryStream();
        WriteAscii(output, "%PDF-1.7\n% procedural scanned-page workflow fixture\n");
        long[] offsets = new long[objects.Length + 1];
        for (int index = 0; index < objects.Length; index++)
        {
            offsets[index + 1] = output.Position;
            WriteAscii(output, $"{index + 1} 0 obj\n{objects[index]}\nendobj\n");
        }

        long crossReferenceOffset = output.Position;
        WriteAscii(output, $"xref\n0 {objects.Length + 1}\n0000000000 65535 f \n");
        for (int index = 1; index < offsets.Length; index++)
        {
            WriteAscii(output, $"{offsets[index]:D10} 00000 n \n");
        }

        WriteAscii(
            output,
            $"trailer\n<< /Size {objects.Length + 1} /Root 1 0 R >>\nstartxref\n{crossReferenceOffset}\n%%EOF\n");
        return output.ToArray();
    }

    private static void WriteAscii(Stream stream, string value)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(value);
        stream.Write(bytes, 0, bytes.Length);
    }

    private static byte[] CreateBornDigitalPdf()
    {
        using var builder = new PdfDocumentBuilder
        {
            DocumentInformation = new PdfDocumentBuilder.DocumentInformationBuilder
            {
                Title = "Procedural graph import fixture",
                Creator = "GraphReader.Integration.Tests",
                Producer = "PdfPig",
            },
        };
        PdfDocumentBuilder.AddedFont font = builder.AddStandard14Font(Standard14Font.Helvetica);
        PdfPageBuilder page = builder.AddPage(width: 612, height: 792);
        page.AddText(
            "Figure 1. Procedural graph import fixture.",
            fontSize: 11,
            new PdfPoint(72, 264),
            font);
        page.AddPng(CreateLargeGraphPng(), new PdfRectangle(72, 288, 540, 648));
        page.DrawLine(new PdfPoint(96, 312), new PdfPoint(96, 624), lineWidth: 1.5);
        page.DrawLine(new PdfPoint(96, 312), new PdfPoint(516, 312), lineWidth: 1.5);
        return builder.Build();
    }

    private static byte[] CreateLargeGraphPng()
    {
        const int width = 640;
        const int height = 480;
        byte[] pixels = Enumerable.Repeat((byte)0xff, width * height).ToArray();
        for (int y = 48; y < height - 48; y++)
        {
            pixels[(y * width) + 64] = 0x20;
        }

        for (int x = 64; x < width - 32; x++)
        {
            pixels[((height - 48) * width) + x] = 0x20;
        }

        for (int x = 96; x < width - 64; x += 48)
        {
            int y = height - 96 - (((x / 48) % 5) * 40);
            for (int offsetY = -3; offsetY <= 3; offsetY++)
            {
                for (int offsetX = -3; offsetX <= 3; offsetX++)
                {
                    pixels[((y + offsetY) * width) + x + offsetX] = 0x20;
                }
            }
        }

        BitmapSource bitmap = BitmapSource.Create(
            width,
            height,
            96,
            96,
            PixelFormats.Gray8,
            palette: null,
            pixels,
            stride: width);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = new MemoryStream();
        encoder.Save(stream);
        return stream.ToArray();
    }

    private sealed class DeterministicPdfImportService(byte[] encodedPanel) : IPdfImportService
    {
        private readonly byte[] panelBytes = (byte[])encodedPanel.Clone();

        public int CallCount { get; private set; }

        public Task<PdfImportResult> ImportAsync(
            PdfImportRequest request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CallCount++;
            string documentSha256 = Convert.ToHexStringLower(SHA256.HashData(request.PdfBytes.ToArray()));
            Guid figureId = Guid.Parse("50000000-0000-0000-0000-000000000031");
            Guid panelId = Guid.Parse("60000000-0000-0000-0000-000000000031");
            var bounds = new PdfRectD(0, 0, 32, 24);
            var figure = new PdfFigureCandidate(
                figureId,
                pageNumber: 2,
                PdfFigureSourceKind.EmbeddedImage,
                embeddedImageId: null,
                bounds,
                bounds,
                sourcePixelWidth: 32,
                sourcePixelHeight: 24,
                new ImmutableByteBuffer(panelBytes),
                mediaType: "image/png",
                caption: null,
                evidence: [],
                confidence: 1);
            var panel = new PdfPanelRecord(
                panelId,
                figureId,
                pageNumber: 2,
                order: 0,
                bounds,
                bounds,
                bounds,
                participantLabel: null,
                caption: null,
                semanticSuggestions: [],
                evidence: [],
                confidence: 1);
            var document = new PdfDocumentSnapshot(
                documentSha256,
                new PdfDocumentMetadata(null, null, null, null, null, null),
                []);
            return Task.FromResult(new PdfImportResult(
                request.RunId,
                request.ProjectId,
                document,
                [figure],
                [panel],
                failures: [],
                warnings: [],
                new PdfImportTiming(1, 0, 1, 2)));
        }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "GraphReaderPdfWorkspaceTests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
