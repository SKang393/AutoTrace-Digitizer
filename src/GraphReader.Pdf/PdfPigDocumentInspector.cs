// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;
using UglyToad.PdfPig.Exceptions;
using UglyToad.PdfPig.Tokens;

namespace GraphReader.Pdf;

public static class PdfInspectionFailureCodes
{
    public const string InvalidRequest = "PDF_INSPECTION_INVALID_REQUEST";
    public const string UnsupportedContract = "PDF_INSPECTION_UNSUPPORTED_CONTRACT";
    public const string PasswordRequired = "PDF_PASSWORD_REQUIRED";
    public const string PasswordRejected = "PDF_PASSWORD_REJECTED";
    public const string CorruptDocument = "PDF_DOCUMENT_CORRUPT";
    public const string ExtractionFailed = "PDF_INSPECTION_FAILED";
    public const string TextSegmentationFallback = "PDF_TEXT_SEGMENTATION_FALLBACK";
    public const string EmbeddedImageUnsupported = "PDF_EMBEDDED_IMAGE_UNSUPPORTED";
}

/// <summary>
/// Inspects local PDF bytes without rendering pages or interpreting figure semantics.
/// </summary>
public sealed class PdfPigDocumentInspector : IPdfDocumentInspector
{
    private static readonly string[] ParticipantLabelPrefixes =
    [
        "participant",
        "student",
        "subject",
        "client",
        "child",
    ];

    private static readonly DefaultPageSegmenter TextSegmenter = new(
        new DefaultPageSegmenter.DefaultPageSegmenterOptions
        {
            MaxDegreeOfParallelism = 1,
            WordSeparator = " ",
            LineSeparator = "\n",
        });

    public Task<PdfInspectionResult> InspectAsync(
        PdfInspectionRequest request,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        PdfFailure? requestFailure = ValidateRequest(request);
        if (requestFailure is not null)
        {
            return Task.FromResult(Failed(requestFailure));
        }

        return Task.Run(
            () => InspectCore(request, cancellationToken),
            cancellationToken);
    }

    private static PdfInspectionResult InspectCore(
        PdfInspectionRequest request,
        CancellationToken cancellationToken)
    {
        var totalTimer = Stopwatch.StartNew();
        var openTimer = new Stopwatch();
        var extractTimer = new Stopwatch();
        string documentSha256;

        try
        {
            documentSha256 = ComputeSha256(request.PdfBytes.Memory, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }

        PdfDocument? document = null;

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            openTimer.Start();
            document = PdfDocument.Open(request.PdfBytes.Memory, CreateParsingOptions(request.Password));
            openTimer.Stop();
            cancellationToken.ThrowIfCancellationRequested();
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (PdfDocumentEncryptedException)
        {
            openTimer.Stop();
            totalTimer.Stop();
            return Failed(
                EncryptedFailure(request.Password),
                openTimer.Elapsed.TotalMilliseconds,
                0d,
                totalTimer.Elapsed.TotalMilliseconds);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            openTimer.Stop();
            totalTimer.Stop();
            return Failed(
                CorruptFailure(),
                openTimer.Elapsed.TotalMilliseconds,
                0d,
                totalTimer.Elapsed.TotalMilliseconds);
        }

        using (document)
        {
            try
            {
                extractTimer.Start();
                var failures = new List<PdfFailure>();
                PdfDocumentMetadata metadata = ExtractMetadata(document);
                var pages = new List<PdfPageSnapshot>(document.NumberOfPages);

                for (var pageNumber = 1; pageNumber <= document.NumberOfPages; pageNumber++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Page page = document.GetPage(pageNumber);
                    pages.Add(ExtractPage(
                        page,
                        documentSha256,
                        failures,
                        cancellationToken));
                }

                cancellationToken.ThrowIfCancellationRequested();
                extractTimer.Stop();
                totalTimer.Stop();
                var snapshot = new PdfDocumentSnapshot(documentSha256, metadata, pages);
                return new PdfInspectionResult(
                    snapshot,
                    failures,
                    new PdfInspectionTiming(
                        openTimer.Elapsed.TotalMilliseconds,
                        extractTimer.Elapsed.TotalMilliseconds,
                        totalTimer.Elapsed.TotalMilliseconds));
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (PdfDocumentEncryptedException)
            {
                extractTimer.Stop();
                totalTimer.Stop();
                return Failed(
                    EncryptedFailure(request.Password),
                    openTimer.Elapsed.TotalMilliseconds,
                    extractTimer.Elapsed.TotalMilliseconds,
                    totalTimer.Elapsed.TotalMilliseconds);
            }
            catch (PdfDocumentFormatException)
            {
                extractTimer.Stop();
                totalTimer.Stop();
                return Failed(
                    CorruptFailure(),
                    openTimer.Elapsed.TotalMilliseconds,
                    extractTimer.Elapsed.TotalMilliseconds,
                    totalTimer.Elapsed.TotalMilliseconds);
            }
            catch (InvalidDataException)
            {
                extractTimer.Stop();
                totalTimer.Stop();
                return Failed(
                    CorruptFailure(),
                    openTimer.Elapsed.TotalMilliseconds,
                    extractTimer.Elapsed.TotalMilliseconds,
                    totalTimer.Elapsed.TotalMilliseconds);
            }
            catch (Exception exception) when (exception is not OutOfMemoryException)
            {
                extractTimer.Stop();
                totalTimer.Stop();
                return Failed(
                    ExtractionFailure(),
                    openTimer.Elapsed.TotalMilliseconds,
                    extractTimer.Elapsed.TotalMilliseconds,
                    totalTimer.Elapsed.TotalMilliseconds);
            }
        }
    }

    private static PdfPageSnapshot ExtractPage(
        Page page,
        string documentSha256,
        List<PdfFailure> failures,
        CancellationToken cancellationToken)
    {
        if (!double.IsFinite(page.Width) ||
            !double.IsFinite(page.Height) ||
            page.Width <= 0d ||
            page.Height <= 0d)
        {
            throw new InvalidDataException("The PDF contains an invalid page boundary.");
        }

        PageGeometry geometry = CreatePageGeometry(page);

        PdfVectorLine[] vectorLines = ExtractVectorLines(
            page,
            cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();

        PdfTextBlock[] textBlocks = ExtractTextBlocks(
            page,
            documentSha256,
            geometry.WidthPoints,
            geometry.HeightPoints,
            vectorLines,
            failures,
            cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();

        PdfEmbeddedImage[] embeddedImages = ExtractImages(
            page,
            documentSha256,
            failures,
            cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();

        return new PdfPageSnapshot(
            page.Number,
            geometry.WidthPoints,
            geometry.HeightPoints,
            textBlocks,
            embeddedImages,
            vectorLines,
            geometry.OriginalVisibleBoundsPoints,
            geometry.RotationDegrees,
            geometry.NormalizedToOriginalPagePoints);
    }

    private static PdfTextBlock[] ExtractTextBlocks(
        Page page,
        string documentSha256,
        double pageWidth,
        double pageHeight,
        IReadOnlyList<PdfVectorLine> vectorLines,
        List<PdfFailure> failures,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        List<Word> words = page.GetWords(NearestNeighbourWordExtractor.Instance)
            .Where(IsUsableWord)
            .OrderByDescending(static word => MaximumY(word.BoundingBox))
            .ThenBy(static word => MinimumX(word.BoundingBox))
            .ThenBy(static word => word.Text, StringComparer.Ordinal)
            .ToList();
        cancellationToken.ThrowIfCancellationRequested();

        List<TextBlockCandidate> candidates;
        try
        {
            candidates = TextSegmenter.GetBlocks(words)
                .SelectMany(static block => block.TextLines)
                .Select(static line => new TextBlockCandidate(
                    line.Text,
                    ToRect(line.BoundingBox),
                    line.TextOrientation))
                .Where(static block =>
                    !string.IsNullOrWhiteSpace(block.Text) &&
                    block.Bounds.IsValid)
                .ToList();
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            candidates = CreateLineBlocks(words);
            failures.Add(Warning(
                PdfInspectionFailureCodes.TextSegmentationFallback,
                "Warnings.PdfTextSegmentationFallback",
                "PDF text was retained using deterministic line grouping.",
                page.Number));
        }

        if (candidates.Count == 0 && words.Count > 0)
        {
            candidates = CreateLineBlocks(words);
        }

        TextBlockCandidate[] ordered = candidates
            .OrderByDescending(static block => block.Bounds.Bottom)
            .ThenBy(static block => block.Bounds.X)
            .ThenBy(static block => block.Text, StringComparer.Ordinal)
            .ThenBy(static block => block.Orientation)
            .ToArray();
        var result = new PdfTextBlock[ordered.Length];

        for (var index = 0; index < ordered.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TextBlockCandidate candidate = ordered[index];
            TextRoleDecision role = ClassifyTextRole(
                candidate,
                pageWidth,
                pageHeight,
                vectorLines);
            result[index] = new PdfTextBlock(
                CreateStableId(
                    "text-block",
                    documentSha256,
                    page.Number,
                    index,
                    candidate.Bounds,
                    candidate.Text),
                candidate.Text,
                candidate.Bounds,
                role.Role,
                role.Confidence);
        }

        return result;
    }

    private static PdfEmbeddedImage[] ExtractImages(
        Page page,
        string documentSha256,
        List<PdfFailure> failures,
        CancellationToken cancellationToken)
    {
        var candidates = new List<EmbeddedImageCandidate>();

        foreach (IPdfImage image in page.GetImages())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (image.IsImageMask ||
                image.WidthInSamples <= 0 ||
                image.HeightInSamples <= 0)
            {
                continue;
            }

            PdfRectangle boundingBox = image.BoundingBox;
            PdfRectD bounds = ToRect(boundingBox);
            PdfAffineTransform sourcePixelsToPagePoints = CreateImagePlacementTransform(
                boundingBox,
                image.WidthInSamples,
                image.HeightInSamples);
            if (!bounds.IsValid || !sourcePixelsToPagePoints.IsInvertible)
            {
                failures.Add(UnsupportedImageWarning(page.Number));
                continue;
            }

            try
            {
                if (!TryExtractEncodedImage(image, out byte[]? encodedBytes, out string? mediaType) ||
                    encodedBytes is null ||
                    encodedBytes.Length == 0 ||
                    mediaType is null)
                {
                    failures.Add(UnsupportedImageWarning(page.Number));
                    continue;
                }

                string sha256 = ComputeSha256(encodedBytes, cancellationToken);
                candidates.Add(new EmbeddedImageCandidate(
                    bounds,
                    image.WidthInSamples,
                    image.HeightInSamples,
                    mediaType,
                    encodedBytes,
                    sha256,
                    sourcePixelsToPagePoints));
            }
            catch (Exception exception) when (exception is not OutOfMemoryException)
            {
                failures.Add(UnsupportedImageWarning(page.Number));
            }
        }

        EmbeddedImageCandidate[] ordered = candidates
            .OrderByDescending(static image => image.Bounds.Bottom)
            .ThenBy(static image => image.Bounds.X)
            .ThenByDescending(static image => image.Bounds.Area)
            .ThenBy(static image => image.Sha256, StringComparer.Ordinal)
            .ThenBy(static image => ImagePlacementDiscriminator(image), StringComparer.Ordinal)
            .ToArray();
        var result = new PdfEmbeddedImage[ordered.Length];

        for (var index = 0; index < ordered.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EmbeddedImageCandidate candidate = ordered[index];
            result[index] = new PdfEmbeddedImage(
                CreateStableId(
                    "embedded-image",
                    documentSha256,
                    page.Number,
                    index,
                    candidate.Bounds,
                    ImagePlacementDiscriminator(candidate)),
                candidate.Bounds,
                candidate.PixelWidth,
                candidate.PixelHeight,
                candidate.MediaType,
                new ImmutableByteBuffer(candidate.EncodedBytes),
                candidate.Sha256,
                candidate.SourcePixelsToPagePoints);
        }

        return result;
    }

    private static PdfVectorLine[] ExtractVectorLines(
        Page page,
        CancellationToken cancellationToken)
    {
        var lines = new List<PdfVectorLine>();

        foreach (var path in page.Paths)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!path.IsStroked)
            {
                continue;
            }

            double width = double.IsFinite(path.LineWidth) && path.LineWidth >= 0d
                ? path.LineWidth
                : 0d;

            foreach (PdfSubpath subpath in path)
            {
                foreach (PdfSubpath.IPathCommand command in subpath.Commands)
                {
                    if (command is not PdfSubpath.Line line ||
                        !IsFinite(line.From) ||
                        !IsFinite(line.To) ||
                        line.Length <= 0d)
                    {
                        continue;
                    }

                    PdfPointD start = ToPoint(line.From);
                    PdfPointD end = ToPoint(line.To);
                    NormalizeLineDirection(ref start, ref end);
                    lines.Add(new PdfVectorLine(start, end, width));
                }
            }
        }

        cancellationToken.ThrowIfCancellationRequested();
        return lines
            .OrderByDescending(static line => Math.Max(
                line.StartPagePoints.Y,
                line.EndPagePoints.Y))
            .ThenBy(static line => line.StartPagePoints.X)
            .ThenBy(static line => line.StartPagePoints.Y)
            .ThenBy(static line => line.EndPagePoints.X)
            .ThenBy(static line => line.EndPagePoints.Y)
            .ThenBy(static line => line.WidthPoints)
            .ToArray();
    }

    private static List<TextBlockCandidate> CreateLineBlocks(IReadOnlyList<Word> words)
    {
        var lines = new List<LineCandidate>();

        foreach (Word word in words)
        {
            PdfRectD bounds = ToRect(word.BoundingBox);
            double centerY = bounds.Y + (bounds.Height / 2d);
            LineCandidate? line = lines.Count == 0 ? null : lines[^1];
            double tolerance = line is null
                ? 0d
                : Math.Max(line.Bounds.Height, bounds.Height) * 0.6d;

            if (line is null || Math.Abs(line.CenterY - centerY) > tolerance)
            {
                lines.Add(new LineCandidate(word, bounds));
            }
            else
            {
                line.Add(word, bounds);
            }
        }

        return lines
            .Select(static line => new TextBlockCandidate(
                string.Join(
                    " ",
                    line.Words
                        .OrderBy(static item => MinimumX(item.BoundingBox))
                        .ThenBy(static item => item.Text, StringComparer.Ordinal)
                        .Select(static item => item.Text)),
                line.Bounds,
                line.Orientation))
            .Where(static block =>
                !string.IsNullOrWhiteSpace(block.Text) &&
                block.Bounds.IsValid)
            .ToList();
    }

    private static bool TryExtractEncodedImage(
        IPdfImage image,
        out byte[]? encodedBytes,
        out string? mediaType)
    {
        if (image.TryGetPng(out byte[]? pngBytes) &&
            pngBytes is { Length: > 0 } &&
            HasPngSignature(pngBytes))
        {
            encodedBytes = pngBytes;
            mediaType = "image/png";
            return true;
        }

        ReadOnlyMemory<byte> rawMemory = image.RawMemory;
        if (HasSingleImageFilter(image, NameToken.DctDecode, NameToken.DctDecodeAbbreviation) &&
            HasJpegSignature(rawMemory.Span))
        {
            encodedBytes = rawMemory.ToArray();
            mediaType = "image/jpeg";
            return encodedBytes.Length > 0;
        }

        if (HasSingleImageFilter(image, NameToken.JpxDecode) &&
            HasJpeg2000Signature(rawMemory.Span))
        {
            encodedBytes = rawMemory.ToArray();
            mediaType = "image/jp2";
            return encodedBytes.Length > 0;
        }

        encodedBytes = null;
        mediaType = null;
        return false;
    }

    private static bool HasSingleImageFilter(IPdfImage image, params NameToken[] expected)
    {
        if (!image.ImageDictionary.TryGet(NameToken.Filter, out IToken? token) || token is null)
        {
            return false;
        }

        return token switch
        {
            NameToken name => expected.Any(candidate => candidate.Equals(name)),
            ArrayToken { Length: 1 } array when array[0] is NameToken name =>
                expected.Any(candidate => candidate.Equals(name)),
            _ => false,
        };
    }

    private static bool HasPngSignature(ReadOnlySpan<byte> bytes)
    {
        ReadOnlySpan<byte> signature = [137, 80, 78, 71, 13, 10, 26, 10];
        return bytes.StartsWith(signature);
    }

    private static bool HasJpegSignature(ReadOnlySpan<byte> bytes) =>
        bytes.Length >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff;

    private static bool HasJpeg2000Signature(ReadOnlySpan<byte> bytes)
    {
        ReadOnlySpan<byte> fileSignature = [0, 0, 0, 12, 106, 80, 32, 32, 13, 10, 135, 10];
        ReadOnlySpan<byte> codestreamSignature = [0xff, 0x4f, 0xff, 0x51];
        return bytes.StartsWith(fileSignature) || bytes.StartsWith(codestreamSignature);
    }

    private static ParsingOptions CreateParsingOptions(string? password)
    {
        var options = new ParsingOptions
        {
            ClipPaths = false,
            SkipMissingFonts = true,
            UseLenientParsing = true,
        };

        // PdfPig initializes ParsingOptions.Logger to its internal no-op logger.
        // Keeping that default prevents document content from reaching application logs.

        if (password is not null)
        {
            options.Password = password;
        }

        return options;
    }

    private static PdfDocumentMetadata ExtractMetadata(PdfDocument document)
    {
        DocumentInformation information = document.Information;
        return new PdfDocumentMetadata(
            NullIfWhiteSpace(information.Title),
            NullIfWhiteSpace(information.Author),
            NullIfWhiteSpace(information.Subject),
            NullIfWhiteSpace(information.Keywords),
            information.GetCreatedDateTimeOffset(),
            information.GetModifiedDateTimeOffset());
    }

    private static PdfFailure? ValidateRequest(PdfInspectionRequest? request)
    {
        if (request is null ||
            request.PdfBytes is null ||
            request.PdfBytes.Length == 0 ||
            string.IsNullOrWhiteSpace(request.SourceDisplayName))
        {
            return Error(
                PdfInspectionFailureCodes.InvalidRequest,
                "Errors.PdfInspectionInvalidRequest",
                "PDF bytes and a display name are required.",
                recoverable: true,
                "correct_input");
        }

        if (request.ContractVersion != PdfImportContract.Version)
        {
            return Error(
                PdfInspectionFailureCodes.UnsupportedContract,
                "Errors.PdfInspectionUnsupportedContract",
                "The requested PDF contract version is not supported.",
                recoverable: false,
                "upgrade_application");
        }

        return null;
    }

    private static PdfInspectionResult Failed(PdfFailure failure) =>
        Failed(failure, 0d, 0d, 0d);

    private static PdfInspectionResult Failed(
        PdfFailure failure,
        double openMilliseconds,
        double extractMilliseconds,
        double totalMilliseconds) =>
        new(
            document: null,
            [failure],
            new PdfInspectionTiming(
                openMilliseconds,
                extractMilliseconds,
                totalMilliseconds));

    private static PdfFailure EncryptedFailure(string? password) =>
        string.IsNullOrEmpty(password)
            ? Error(
                PdfInspectionFailureCodes.PasswordRequired,
                "Errors.PdfPasswordRequired",
                "The PDF is encrypted and requires a password.",
                recoverable: true,
                "provide_password")
            : Error(
                PdfInspectionFailureCodes.PasswordRejected,
                "Errors.PdfPasswordRejected",
                "The supplied password could not open the encrypted PDF.",
                recoverable: true,
                "provide_password");

    private static PdfFailure CorruptFailure() =>
        Error(
            PdfInspectionFailureCodes.CorruptDocument,
            "Errors.PdfDocumentCorrupt",
            "The PDF structure could not be parsed.",
            recoverable: false,
            "select_another_file");

    private static PdfFailure ExtractionFailure() =>
        Error(
            PdfInspectionFailureCodes.ExtractionFailed,
            "Errors.PdfInspectionFailed",
            "PDF inspection could not complete.",
            recoverable: true,
            "retry");

    private static PdfFailure UnsupportedImageWarning(int pageNumber) =>
        Warning(
            PdfInspectionFailureCodes.EmbeddedImageUnsupported,
            "Warnings.PdfEmbeddedImageUnsupported",
            "An embedded image could not be represented as a supported local image encoding.",
            pageNumber);

    private static PdfFailure Error(
        string code,
        string userMessageKey,
        string technicalMessage,
        bool recoverable,
        string suggestedAction) =>
        new(
            code,
            PdfFailureSeverity.Error,
            userMessageKey,
            technicalMessage,
            recoverable,
            suggestedAction);

    private static PdfFailure Warning(
        string code,
        string userMessageKey,
        string technicalMessage,
        int pageNumber) =>
        new(
            code,
            PdfFailureSeverity.Warning,
            userMessageKey,
            technicalMessage,
            Recoverable: true,
            "continue_with_rendering",
            pageNumber);

    private static string ComputeSha256(
        ReadOnlyMemory<byte> bytes,
        CancellationToken cancellationToken)
    {
        const int chunkSize = 64 * 1024;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        for (var offset = 0; offset < bytes.Length; offset += chunkSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, bytes.Length - offset);
            hash.AppendData(bytes.Span.Slice(offset, count));
        }

        cancellationToken.ThrowIfCancellationRequested();
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private static Guid CreateStableId(
        string kind,
        string documentSha256,
        int pageNumber,
        int index,
        PdfRectD bounds,
        string discriminator)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Append(hash, "graph-reader-pdf-object-v1");
        Append(hash, kind);
        Append(hash, documentSha256);
        Append(hash, pageNumber.ToString(CultureInfo.InvariantCulture));
        Append(hash, index.ToString(CultureInfo.InvariantCulture));
        Append(hash, bounds.X.ToString("R", CultureInfo.InvariantCulture));
        Append(hash, bounds.Y.ToString("R", CultureInfo.InvariantCulture));
        Append(hash, bounds.Width.ToString("R", CultureInfo.InvariantCulture));
        Append(hash, bounds.Height.ToString("R", CultureInfo.InvariantCulture));
        Append(hash, discriminator);
        byte[] digest = hash.GetHashAndReset();
        return new Guid(digest.AsSpan(0, 16));
    }

    private static string ImagePlacementDiscriminator(EmbeddedImageCandidate image)
    {
        PdfAffineTransform transform = image.SourcePixelsToPagePoints;
        return string.Join(
            "|",
            image.Sha256,
            transform.A.ToString("R", CultureInfo.InvariantCulture),
            transform.B.ToString("R", CultureInfo.InvariantCulture),
            transform.C.ToString("R", CultureInfo.InvariantCulture),
            transform.D.ToString("R", CultureInfo.InvariantCulture),
            transform.E.ToString("R", CultureInfo.InvariantCulture),
            transform.F.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void Append(IncrementalHash hash, string value)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(value);
        Span<byte> length = stackalloc byte[sizeof(int)];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(length, encoded.Length);
        hash.AppendData(length);
        hash.AppendData(encoded);
    }

    private static bool IsUsableWord(Word word) =>
        !string.IsNullOrWhiteSpace(word.Text) && ToRect(word.BoundingBox).IsValid;

    private static TextRoleDecision ClassifyTextRole(
        TextBlockCandidate candidate,
        double pageWidth,
        double pageHeight,
        IReadOnlyList<PdfVectorLine> vectorLines)
    {
        if (LooksLikeCaption(candidate.Text))
        {
            return new TextRoleDecision(PdfTextRole.Caption, 0.96d);
        }

        if (LooksLikeExplicitParticipantLabel(candidate.Text))
        {
            return new TextRoleDecision(PdfTextRole.ParticipantLabel, 0.88d);
        }

        if (LooksLikeAxisTitle(candidate, pageWidth, pageHeight, vectorLines, out double confidence))
        {
            return new TextRoleDecision(PdfTextRole.AxisTitle, confidence);
        }

        return new TextRoleDecision(PdfTextRole.Body, 0.95d);
    }

    private static bool LooksLikeExplicitParticipantLabel(string text)
    {
        ReadOnlySpan<char> value = text.AsSpan().Trim();
        if (value.Length == 0 || value.Contains('\n') || value.Contains('\r'))
        {
            return false;
        }

        foreach (string prefix in ParticipantLabelPrefixes)
        {
            if (!value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ||
                value.Length == prefix.Length)
            {
                continue;
            }

            char boundary = value[prefix.Length];
            if (!char.IsWhiteSpace(boundary) && boundary is not ':' and not '#' and not '-')
            {
                continue;
            }

            ReadOnlySpan<char> identifier = value[(prefix.Length + 1)..].Trim();
            if (identifier.Length > 0 && identifier.IndexOfAnyInRange('0', '9') >= 0)
            {
                return true;
            }

            if (identifier.Length > 0 && identifier.ToString().Any(char.IsLetter))
            {
                return true;
            }
        }

        return false;
    }

    private static bool LooksLikeAxisTitle(
        TextBlockCandidate candidate,
        double pageWidth,
        double pageHeight,
        IReadOnlyList<PdfVectorLine> vectorLines,
        out double confidence)
    {
        confidence = 0d;
        if (!IsShortLabelText(candidate.Text))
        {
            return false;
        }

        bool rotated = candidate.Orientation is TextOrientation.Rotate90 or TextOrientation.Rotate270;
        if (rotated && HasAdjacentVerticalAxis(candidate.Bounds, pageWidth, pageHeight, vectorLines))
        {
            confidence = 0.86d;
            return true;
        }

        bool horizontal = candidate.Orientation is TextOrientation.Horizontal or TextOrientation.Other;
        if (horizontal && HasCenteredHorizontalAxisAbove(candidate.Bounds, pageWidth, vectorLines))
        {
            confidence = 0.78d;
            return true;
        }

        return false;
    }

    private static bool IsShortLabelText(string text)
    {
        ReadOnlySpan<char> value = text.AsSpan().Trim();
        if (value.Length is < 2 or > 80 ||
            value.Contains('\n') ||
            value.Contains('\r') ||
            value.Contains(';') ||
            value.Contains('!') ||
            value.Contains('?'))
        {
            return false;
        }

        int wordCount = 0;
        bool insideWord = false;
        int letterCount = 0;
        foreach (char character in value)
        {
            if (char.IsLetter(character))
            {
                letterCount++;
            }

            if (char.IsWhiteSpace(character))
            {
                insideWord = false;
            }
            else if (!insideWord)
            {
                wordCount++;
                insideWord = true;
            }
        }

        return letterCount >= 2 && wordCount is >= 1 and <= 8;
    }

    private static bool HasAdjacentVerticalAxis(
        PdfRectD bounds,
        double pageWidth,
        double pageHeight,
        IReadOnlyList<PdfVectorLine> vectorLines)
    {
        double minimumAxisLength = Math.Max(36d, pageHeight * 0.08d);
        double maximumGap = Math.Max(24d, pageWidth * 0.05d);
        double centerY = bounds.Y + (bounds.Height / 2d);

        return vectorLines.Any(line =>
        {
            if (!line.IsVertical(0.75d))
            {
                return false;
            }

            double lower = Math.Min(line.StartPagePoints.Y, line.EndPagePoints.Y);
            double upper = Math.Max(line.StartPagePoints.Y, line.EndPagePoints.Y);
            double length = upper - lower;
            double x = (line.StartPagePoints.X + line.EndPagePoints.X) / 2d;
            double horizontalGap = x < bounds.X
                ? bounds.X - x
                : x > bounds.Right
                    ? x - bounds.Right
                    : 0d;
            return length >= minimumAxisLength &&
                horizontalGap <= maximumGap &&
                centerY >= lower - 12d &&
                centerY <= upper + 12d;
        });
    }

    private static bool HasCenteredHorizontalAxisAbove(
        PdfRectD bounds,
        double pageWidth,
        IReadOnlyList<PdfVectorLine> vectorLines)
    {
        double minimumAxisLength = Math.Max(72d, pageWidth * 0.15d);
        double blockCenterX = bounds.X + (bounds.Width / 2d);

        return vectorLines.Any(line =>
        {
            if (!line.IsHorizontal(0.75d))
            {
                return false;
            }

            double left = Math.Min(line.StartPagePoints.X, line.EndPagePoints.X);
            double right = Math.Max(line.StartPagePoints.X, line.EndPagePoints.X);
            double length = right - left;
            double y = (line.StartPagePoints.Y + line.EndPagePoints.Y) / 2d;
            double verticalGap = y - bounds.Bottom;
            return length >= minimumAxisLength &&
                bounds.Width <= length * 0.65d &&
                verticalGap >= -2d &&
                verticalGap <= 36d &&
                blockCenterX >= left + (length * 0.20d) &&
                blockCenterX <= right - (length * 0.20d);
        });
    }

    private static bool LooksLikeCaption(string text)
    {
        ReadOnlySpan<char> value = text.AsSpan().TrimStart();
        if (value.StartsWith("Fig.", StringComparison.OrdinalIgnoreCase))
        {
            value = value[4..].TrimStart();
        }
        else if (value.StartsWith("Figure", StringComparison.OrdinalIgnoreCase))
        {
            value = value[6..].TrimStart();
        }
        else
        {
            return false;
        }

        return value.Length > 0 &&
            (char.IsDigit(value[0]) ||
             (value.Length > 1 &&
              (value[0] is 'S' or 's') &&
              char.IsDigit(value[1])));
    }

    private static PdfPointD ToPoint(PdfPoint value) => new(value.X, value.Y);

    private static PageGeometry CreatePageGeometry(Page page)
    {
        PdfRectD normalizedCropBounds = ToRect(page.CropBox.Bounds);
        PdfRectD normalizedMediaBounds = ToRect(page.MediaBox.Bounds);
        PdfRectD normalizedVisibleBounds =
            Intersect(normalizedMediaBounds, normalizedCropBounds) ?? normalizedCropBounds;
        int rotationDegrees = page.Rotation.Value;
        if (!normalizedVisibleBounds.IsValid || rotationDegrees is not (0 or 90 or 180 or 270))
        {
            throw new InvalidDataException("The PDF contains invalid page coordinate geometry.");
        }

        if (!TryGetOriginalVisibleBounds(page, rotationDegrees, out PdfRectD originalVisibleBounds))
        {
            return new PageGeometry(
                page.Width,
                page.Height,
                normalizedVisibleBounds,
                rotationDegrees,
                PdfAffineTransform.Identity);
        }

        PdfAffineTransform normalizedToOriginal = rotationDegrees switch
        {
            0 => new PdfAffineTransform(
                1d, 0d, 0d, 1d,
                originalVisibleBounds.X,
                originalVisibleBounds.Y),
            90 => new PdfAffineTransform(
                0d, 1d, -1d, 0d,
                originalVisibleBounds.Right,
                originalVisibleBounds.Y),
            180 => new PdfAffineTransform(
                -1d, 0d, 0d, -1d,
                originalVisibleBounds.Right,
                originalVisibleBounds.Bottom),
            270 => new PdfAffineTransform(
                0d, -1d, 1d, 0d,
                originalVisibleBounds.X,
                originalVisibleBounds.Bottom),
            _ => throw new InvalidDataException("The PDF contains an unsupported page rotation."),
        };

        return new PageGeometry(
            page.Width,
            page.Height,
            originalVisibleBounds,
            rotationDegrees,
            normalizedToOriginal);
    }

    private static bool TryGetOriginalVisibleBounds(
        Page page,
        int rotationDegrees,
        out PdfRectD visibleBounds)
    {
        bool hasCrop = TryGetDirectPageBounds(page.Dictionary, NameToken.CropBox, out PdfRectD cropBounds);
        bool hasMedia = TryGetDirectPageBounds(page.Dictionary, NameToken.MediaBox, out PdfRectD mediaBounds);
        if (!hasCrop && !hasMedia)
        {
            visibleBounds = default;
            return false;
        }

        visibleBounds = hasCrop && hasMedia
            ? Intersect(mediaBounds, cropBounds) ?? cropBounds
            : hasCrop
                ? cropBounds
                : mediaBounds;
        double expectedWidth = rotationDegrees is 90 or 270
            ? visibleBounds.Height
            : visibleBounds.Width;
        double expectedHeight = rotationDegrees is 90 or 270
            ? visibleBounds.Width
            : visibleBounds.Height;
        return visibleBounds.IsValid &&
            NearlyEqual(expectedWidth, page.Width) &&
            NearlyEqual(expectedHeight, page.Height);
    }

    private static bool TryGetDirectPageBounds(
        DictionaryToken dictionary,
        NameToken name,
        out PdfRectD bounds)
    {
        bounds = default;
        if (!dictionary.TryGet(name, out IToken? token) ||
            token is not ArrayToken { Length: 4 } array ||
            array.Data.Any(static item => item is not NumericToken))
        {
            return false;
        }

        double x1 = ((NumericToken)array[0]).Double;
        double y1 = ((NumericToken)array[1]).Double;
        double x2 = ((NumericToken)array[2]).Double;
        double y2 = ((NumericToken)array[3]).Double;
        bounds = new PdfRectD(
            Math.Min(x1, x2),
            Math.Min(y1, y2),
            Math.Abs(x2 - x1),
            Math.Abs(y2 - y1));
        return bounds.IsValid;
    }

    private static bool NearlyEqual(double first, double second) =>
        Math.Abs(first - second) <= Math.Max(1e-7d, Math.Max(Math.Abs(first), Math.Abs(second)) * 1e-9d);

    private static PdfRectD? Intersect(PdfRectD first, PdfRectD second)
    {
        if (!first.IsValid || !second.IsValid)
        {
            return null;
        }

        double left = Math.Max(first.X, second.X);
        double bottom = Math.Max(first.Y, second.Y);
        double right = Math.Min(first.Right, second.Right);
        double top = Math.Min(first.Bottom, second.Bottom);
        return right > left && top > bottom
            ? new PdfRectD(left, bottom, right - left, top - bottom)
            : null;
    }

    private static PdfAffineTransform CreateImagePlacementTransform(
        PdfRectangle bounds,
        int pixelWidth,
        int pixelHeight)
    {
        if (pixelWidth <= 0 || pixelHeight <= 0)
        {
            return default;
        }

        PdfPoint topLeft = bounds.TopLeft;
        PdfPoint topRight = bounds.TopRight;
        PdfPoint bottomLeft = bounds.BottomLeft;
        return new PdfAffineTransform(
            (topRight.X - topLeft.X) / pixelWidth,
            (topRight.Y - topLeft.Y) / pixelWidth,
            (bottomLeft.X - topLeft.X) / pixelHeight,
            (bottomLeft.Y - topLeft.Y) / pixelHeight,
            topLeft.X,
            topLeft.Y);
    }

    private static PdfRectD ToRect(PdfRectangle value)
    {
        double minX = MinimumX(value);
        double minY = Math.Min(
            Math.Min(value.TopLeft.Y, value.TopRight.Y),
            Math.Min(value.BottomLeft.Y, value.BottomRight.Y));
        double maxX = Math.Max(
            Math.Max(value.TopLeft.X, value.TopRight.X),
            Math.Max(value.BottomLeft.X, value.BottomRight.X));
        double maxY = MaximumY(value);
        return new PdfRectD(minX, minY, maxX - minX, maxY - minY);
    }

    private static double MinimumX(PdfRectangle value) =>
        Math.Min(
            Math.Min(value.TopLeft.X, value.TopRight.X),
            Math.Min(value.BottomLeft.X, value.BottomRight.X));

    private static double MaximumY(PdfRectangle value) =>
        Math.Max(
            Math.Max(value.TopLeft.Y, value.TopRight.Y),
            Math.Max(value.BottomLeft.Y, value.BottomRight.Y));

    private static bool IsFinite(PdfPoint point) =>
        double.IsFinite(point.X) && double.IsFinite(point.Y);

    private static void NormalizeLineDirection(ref PdfPointD start, ref PdfPointD end)
    {
        if (start.X > end.X || (start.X.Equals(end.X) && start.Y > end.Y))
        {
            (start, end) = (end, start);
        }
    }

    private static string? NullIfWhiteSpace(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value;

    private sealed record TextBlockCandidate(
        string Text,
        PdfRectD Bounds,
        TextOrientation Orientation);

    private sealed record TextRoleDecision(PdfTextRole Role, double Confidence);

    private sealed record EmbeddedImageCandidate(
        PdfRectD Bounds,
        int PixelWidth,
        int PixelHeight,
        string MediaType,
        byte[] EncodedBytes,
        string Sha256,
        PdfAffineTransform SourcePixelsToPagePoints);

    private sealed record PageGeometry(
        double WidthPoints,
        double HeightPoints,
        PdfRectD OriginalVisibleBoundsPoints,
        int RotationDegrees,
        PdfAffineTransform NormalizedToOriginalPagePoints);

    private sealed class LineCandidate
    {
        public LineCandidate(Word word, PdfRectD bounds)
        {
            Words = [word];
            Bounds = bounds;
            CenterY = bounds.Y + (bounds.Height / 2d);
            Orientation = word.TextOrientation;
        }

        public List<Word> Words { get; }

        public PdfRectD Bounds { get; private set; }

        public double CenterY { get; private set; }

        public TextOrientation Orientation { get; }

        public void Add(Word word, PdfRectD bounds)
        {
            Words.Add(word);
            double left = Math.Min(Bounds.X, bounds.X);
            double bottom = Math.Min(Bounds.Y, bounds.Y);
            double right = Math.Max(Bounds.Right, bounds.Right);
            double top = Math.Max(Bounds.Bottom, bounds.Bottom);
            Bounds = new PdfRectD(left, bottom, right - left, top - bottom);
            CenterY = Words.Average(static item =>
            {
                PdfRectD itemBounds = ToRect(item.BoundingBox);
                return itemBounds.Y + (itemBounds.Height / 2d);
            });
        }
    }
}
