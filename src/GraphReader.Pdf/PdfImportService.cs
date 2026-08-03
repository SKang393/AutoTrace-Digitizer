// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;

namespace GraphReader.Pdf;

/// <summary>
/// Coordinates local structure inspection, render fallback, and panel proposals.
/// </summary>
public sealed class PdfImportService : IPdfImportService
{
    private readonly IPdfDocumentInspector _inspector;
    private readonly IPdfPanelizationEngine _panelization;
    private readonly IPdfPageRenderingService? _renderer;

    public PdfImportService(
        IPdfDocumentInspector inspector,
        IPdfPanelizationEngine panelization,
        IPdfPageRenderingService? renderer = null)
    {
        _inspector = inspector ?? throw new ArgumentNullException(nameof(inspector));
        _panelization = panelization ?? throw new ArgumentNullException(nameof(panelization));
        _renderer = renderer;
    }

    public async Task<PdfImportResult> ImportAsync(
        PdfImportRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var total = Stopwatch.StartNew();
        List<PdfFailure> failures = Validate(request);
        if (failures.Any(IsError))
        {
            return Empty(request, failures, total.Elapsed.TotalMilliseconds);
        }

        PdfInspectionResult inspection;
        try
        {
            inspection = await _inspector.InspectAsync(
                    new PdfInspectionRequest(
                        new ImmutableByteBuffer(request.PdfBytes.ToArray()),
                        request.SourceDisplayName,
                        request.Password,
                        request.ContractVersion),
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception exception) when (IsUnexpectedCollaboratorFault(exception))
        {
            failures.Add(Error(
                "PDF_INSPECTION_UNEXPECTED",
                "Errors.PdfInspectionUnexpected",
                "The local PDF inspector failed unexpectedly.",
                recoverable: true,
                "Retry the import or choose another PDF."));
            return Empty(request, failures, total.Elapsed.TotalMilliseconds);
        }
        cancellationToken.ThrowIfCancellationRequested();
        failures.AddRange(inspection.Failures);
        if (!inspection.Succeeded || inspection.Document is null)
        {
            return new PdfImportResult(
                request.RunId,
                request.ProjectId,
                inspection.Document,
                [],
                [],
                failures,
                [],
                new PdfImportTiming(
                    inspection.Timing.TotalMilliseconds,
                    0d,
                    0d,
                    total.Elapsed.TotalMilliseconds));
        }

        var figures = new List<PdfFigureCandidate>();
        var panels = new List<PdfPanelRecord>();
        var warnings = new List<string>();
        double renderingMilliseconds = 0d;
        double panelizationMilliseconds = 0d;

        foreach (PdfPageSnapshot page in inspection.Document.Pages.OrderBy(static page => page.PageNumber))
        {
            cancellationToken.ThrowIfCancellationRequested();
            PdfPanelizationResult pageResult = await ProposeAsync(
                    inspection.Document.DocumentSha256,
                    page,
                    renderedPage: null,
                    request.PanelizationOptions,
                    cancellationToken)
                .ConfigureAwait(false);
            panelizationMilliseconds += pageResult.ElapsedMilliseconds;

            if (pageResult.Panels.Count == 0)
            {
                (PdfRenderedPage? renderedPage, PdfFailure? renderFailure, double renderMs) =
                    await RenderPageAsync(request, page.PageNumber, cancellationToken).ConfigureAwait(false);
                renderingMilliseconds += renderMs;
                cancellationToken.ThrowIfCancellationRequested();

                if (renderedPage is not null)
                {
                    pageResult = await ProposeAsync(
                            inspection.Document.DocumentSha256,
                            page,
                            renderedPage,
                            request.PanelizationOptions,
                            cancellationToken)
                        .ConfigureAwait(false);
                    panelizationMilliseconds += pageResult.ElapsedMilliseconds;
                }
                else if (renderFailure is not null)
                {
                    failures.Add(renderFailure);
                }
            }

            figures.AddRange(pageResult.Figures);
            panels.AddRange(pageResult.Panels);
            failures.AddRange(pageResult.Failures);
            warnings.AddRange(pageResult.Warnings);
        }

        PdfFigureCandidate[] orderedFigures = figures
            .OrderBy(static figure => figure.PageNumber)
            .ThenBy(static figure => figure.BoundsPagePixels.Y)
            .ThenBy(static figure => figure.BoundsPagePixels.X)
            .ThenBy(static figure => figure.FigureId)
            .ToArray();
        PdfPanelRecord[] orderedPanels = panels
            .OrderBy(static panel => panel.PageNumber)
            .ThenBy(static panel => panel.BoundsPagePixels.Y)
            .ThenBy(static panel => panel.BoundsPagePixels.X)
            .ThenBy(static panel => panel.PanelId)
            .ToArray();

        if (orderedPanels.Length == 0 && !failures.Any(IsError))
        {
            warnings.Add("No graph-like figure passed the deterministic panelization threshold.");
        }

        return new PdfImportResult(
            request.RunId,
            request.ProjectId,
            inspection.Document,
            orderedFigures,
            orderedPanels,
            failures,
            warnings.Distinct(StringComparer.Ordinal).ToArray(),
            new PdfImportTiming(
                inspection.Timing.TotalMilliseconds,
                renderingMilliseconds,
                panelizationMilliseconds,
                total.Elapsed.TotalMilliseconds));
    }

    private async Task<PdfPanelizationResult> ProposeAsync(
        string documentSha256,
        PdfPageSnapshot page,
        PdfRenderedPage? renderedPage,
        PdfPanelizationOptions options,
        CancellationToken cancellationToken)
    {
        try
        {
            return await _panelization.ProposeAsync(
                    new PdfPanelizationInput(documentSha256, page, renderedPage, options),
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception exception) when (IsUnexpectedCollaboratorFault(exception))
        {
            return new PdfPanelizationResult(
                [],
                [],
                [Error(
                    "PDF_PANELIZATION_UNEXPECTED",
                    "Errors.PdfPanelizationUnexpected",
                    "The local PDF panelizer failed unexpectedly.",
                    recoverable: true,
                    "Review the page manually or retry the import.",
                    page.PageNumber)],
                [],
                0d);
        }
    }

    private async Task<(PdfRenderedPage? Page, PdfFailure? Failure, double Milliseconds)> RenderPageAsync(
        PdfImportRequest request,
        int pageNumber,
        CancellationToken cancellationToken)
    {
        if (_renderer is null)
        {
            return (
                null,
                Error(
                    "PDF_RENDERER_UNAVAILABLE",
                    "Errors.PdfRendererUnavailable",
                    "No reviewed local PDFium renderer is configured for page fallback.",
                    recoverable: true,
                    "Install or configure a reviewed local PDF renderer.",
                    pageNumber),
                0d);
        }

        var stopwatch = Stopwatch.StartNew();
        PdfPageRenderResult result;
        try
        {
            result = await _renderer.RenderAsync(
                    new PdfPageRenderRequest(
                        new ImmutableByteBuffer(request.PdfBytes.ToArray()),
                        pageNumber,
                        request.PanelizationOptions.RenderDpi,
                        request.ContractVersion),
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception exception) when (IsUnexpectedCollaboratorFault(exception))
        {
            return (
                null,
                Error(
                    "PDF_RENDER_UNEXPECTED",
                    "Errors.PdfRenderUnexpected",
                    "The reviewed local PDF renderer failed unexpectedly.",
                    recoverable: true,
                    "Retry the import or inspect the page manually.",
                    pageNumber),
                stopwatch.Elapsed.TotalMilliseconds);
        }
        cancellationToken.ThrowIfCancellationRequested();

        if (result.Succeeded && result.Page is not null)
        {
            return (result.Page, null, stopwatch.Elapsed.TotalMilliseconds);
        }

        PdfFailure? renderFailure = result.Failure;
        return (
            null,
            Error(
                renderFailure?.Code ?? "PDF_RENDER_FAILED",
                renderFailure?.UserMessageKey ?? "Errors.PdfRenderFailed",
                renderFailure?.TechnicalMessage ?? "The local renderer returned no page.",
                renderFailure?.Recoverable ?? true,
                renderFailure?.SuggestedAction ?? "Retry or inspect another page.",
                pageNumber),
            stopwatch.Elapsed.TotalMilliseconds);
    }

    private static List<PdfFailure> Validate(PdfImportRequest request)
    {
        var failures = new List<PdfFailure>();
        if (request.RunId == Guid.Empty || request.ProjectId == Guid.Empty)
        {
            failures.Add(Error(
                "PDF_INVALID_ID",
                "Errors.PdfInvalidId",
                "Run and project identifiers must be non-empty.",
                recoverable: true,
                "Create stable run and project identifiers."));
        }

        if (request.PdfBytes is null || request.PdfBytes.Length == 0)
        {
            failures.Add(Error(
                "PDF_EMPTY_INPUT",
                "Errors.PdfEmptyInput",
                "PDF content must not be empty.",
                recoverable: true,
                "Choose a readable PDF file."));
        }

        if (string.IsNullOrWhiteSpace(request.SourceDisplayName) ||
            request.ContractVersion != PdfImportContract.Version ||
            request.PanelizationOptions is null ||
            request.PanelizationOptions.RenderDpi is < 72 or > 600 ||
            request.PanelizationOptions.MinimumFigureConfidence is < 0d or > 1d ||
            request.PanelizationOptions.MinimumWhitespaceFraction is < 0d or > 0.5d ||
            request.PanelizationOptions.MaximumPanelsPerFigure is < 1 or > 100)
        {
            failures.Add(Error(
                "PDF_INVALID_REQUEST",
                "Errors.PdfInvalidRequest",
                "The PDF import name, contract version, or fixed panelization options are invalid.",
                recoverable: true,
                "Use the current contract and supported panelization defaults."));
        }

        return failures;
    }

    private static PdfImportResult Empty(
        PdfImportRequest request,
        IEnumerable<PdfFailure> failures,
        double totalMilliseconds) => new(
            request.RunId,
            request.ProjectId,
            null,
            [],
            [],
            failures,
            [],
            new PdfImportTiming(0d, 0d, 0d, totalMilliseconds));

    private static bool IsError(PdfFailure failure) =>
        failure.Severity == PdfFailureSeverity.Error;

    private static bool IsUnexpectedCollaboratorFault(Exception exception) =>
        exception is not OperationCanceledException and
        not OutOfMemoryException and
        not StackOverflowException and
        not AccessViolationException;

    private static PdfFailure Error(
        string code,
        string userMessageKey,
        string technicalMessage,
        bool recoverable,
        string suggestedAction,
        int? pageNumber = null) => new(
            code,
            PdfFailureSeverity.Error,
            userMessageKey,
            technicalMessage,
            recoverable,
            suggestedAction,
            pageNumber);
}
