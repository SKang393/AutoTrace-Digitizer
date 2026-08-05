// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using GraphReader.Axis;

namespace GraphReader.App.Integration.Workflow;

public interface IProductionAxisGeometryAdapter
{
    string AdapterId { get; }

    bool IsApproved { get; }

    Task<ProductionAxisGeometryEvidence> DetectAsync(
        ProductionWorkflowDetectionRequest request,
        CancellationToken cancellationToken);
}

public sealed record ProductionAxisGeometryEvidence(
    WorkflowVisionEnvelope Envelope,
    AxisGeometryResult Geometry);

/// <summary>
/// Decodes immutable original image bytes to a grayscale frame and composes the
/// production OpenCV line provider with the deterministic axis geometry fitter.
/// </summary>
public sealed class ProductionAxisGeometryAdapter : IProductionAxisGeometryAdapter
{
    public const string StageVersion = "axis-opencv-v1";

    private readonly IAxisGeometryDetector detector;
    private readonly ILineCandidateProvider candidateProvider;
    private readonly string runtimeSha256;

    public ProductionAxisGeometryAdapter(
        string runtimeSha256,
        bool isApproved,
        IAxisGeometryDetector? detector = null,
        ILineCandidateProvider? candidateProvider = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(runtimeSha256);
        if (runtimeSha256.Length != 64 || !runtimeSha256.All(Uri.IsHexDigit))
        {
            throw new ArgumentException(
                "The axis runtime SHA-256 must contain exactly 64 hexadecimal characters.",
                nameof(runtimeSha256));
        }

        this.runtimeSha256 = runtimeSha256.ToLowerInvariant();
        IsApproved = isApproved;
        this.detector = detector ?? new AxisGeometryDetector();
        this.candidateProvider = candidateProvider ?? new OpenCvLineCandidateProvider();
    }

    public string AdapterId => $"graphreader-axis-opencv:{runtimeSha256[..12]}";

    public bool IsApproved { get; }

    public async Task<ProductionAxisGeometryEvidence> DetectAsync(
        ProductionWorkflowDetectionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsApproved)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
                "Errors.AxisGeometryNotFound",
                $"Axis adapter '{AdapterId}' does not have release-approved native runtime evidence.",
                "Continue with manual calibration or install the exact approved runtime.");
        }

        if (request.ImageVariant != WorkflowImageVariant.Original ||
            request.Image.Variant != WorkflowImageVariant.Original)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Production axis geometry must run on the immutable original image.",
                "Run axis geometry on the original image before derivative stages.");
        }

        var total = Stopwatch.StartNew();
        var preprocessing = Stopwatch.StartNew();
        byte[] encodedBytes = request.CopyImageBytes();
        if (!string.Equals(
                Convert.ToHexStringLower(SHA256.HashData(encodedBytes)),
                request.Image.Sha256,
                StringComparison.Ordinal))
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.SourceChanged",
                "Axis input bytes do not match the immutable original checksum.",
                "Re-import the source and retry axis detection.");
        }

        GrayscaleLineCandidateFrame frame;
        try
        {
            frame = DecodeGrayscale(encodedBytes, request.Image.Width, request.Image.Height);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidDataException or
            NotSupportedException or FileFormatException)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                $"Axis input decoding failed: {exception.Message}",
                "Re-import a supported image and retry axis detection.");
        }

        preprocessing.Stop();
        AxisGeometryResult geometry;
        try
        {
            geometry = await detector.DetectAsync(
                    frame,
                    candidateProvider,
                    cancellationToken: cancellationToken)
                .ConfigureAwait(false);
        }
        catch (AxisGeometryDetectionException exception)
        {
            throw Failure(
                exception.Code,
                exception.UserMessageKey,
                exception.Message,
                exception.SuggestedAction);
        }
        catch (Exception exception) when (exception is DllNotFoundException or
            EntryPointNotFoundException or BadImageFormatException)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.AxisGeometryNotFound",
                $"The approved OpenCV axis runtime could not be loaded: {exception.Message}",
                "Restore the exact reviewed native runtime or continue with manual calibration.");
        }
        cancellationToken.ThrowIfCancellationRequested();
        total.Stop();

        var envelope = new WorkflowVisionEnvelope(
            contractVersion: 1,
            request.RunId,
            request.ProjectId,
            request.Panel.ImportedPanel.PanelId,
            stage: "axis",
            StageVersion,
            request.Image.Sha256,
            new WorkflowVisionModel(
                "OpenCvSharpExtern",
                StageVersion,
                runtimeSha256,
                "cpu"),
            new WorkflowVisionTiming(
                preprocessing.Elapsed.TotalMilliseconds,
                InferenceMilliseconds: null,
                geometry.Diagnostics.Elapsed.TotalMilliseconds,
                total.Elapsed.TotalMilliseconds),
            geometry.Confidence,
            geometry.Diagnostics.Warnings,
            request.Transforms);
        return new ProductionAxisGeometryEvidence(envelope, geometry);
    }

    private static GrayscaleLineCandidateFrame DecodeGrayscale(
        byte[] encodedBytes,
        int expectedWidth,
        int expectedHeight)
    {
        using var stream = new MemoryStream(encodedBytes, writable: false);
        BitmapDecoder decoder = BitmapDecoder.Create(
            stream,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad);
        if (decoder.Frames.Count == 0)
        {
            throw new InvalidDataException("The image contains no decodable frame.");
        }

        BitmapSource source = decoder.Frames[0];
        if (source.PixelWidth != expectedWidth || source.PixelHeight != expectedHeight)
        {
            throw new InvalidDataException(
                "Decoded image dimensions do not match the retained import evidence.");
        }

        var grayscale = new FormatConvertedBitmap(source, PixelFormats.Gray8, null, 0);
        grayscale.Freeze();
        int stride = checked(grayscale.PixelWidth);
        var pixels = new byte[checked(stride * grayscale.PixelHeight)];
        grayscale.CopyPixels(pixels, stride, 0);
        return new GrayscaleLineCandidateFrame(
            grayscale.PixelWidth,
            grayscale.PixelHeight,
            stride,
            pixels);
    }

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction) =>
        new(new ProductionWorkflowFailure(
            code,
            userMessageKey,
            technicalMessage,
            Recoverable: true,
            suggestedAction));
}
