// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;

namespace GraphReader.Ocr.Tests;

internal static class OcrTestFixtures
{
    public static OcrImage Image(
        OcrSourceImage source = OcrSourceImage.Original,
        int width = 160,
        int height = 100,
        OcrFrameTransform? transform = null)
    {
        byte[] pixels = Enumerable.Range(0, width * height)
            .Select(static index => (byte)((index * 37) % 251))
            .ToArray();
        return new OcrImage(
            width,
            height,
            width,
            pixels,
            source,
            transform ?? OcrFrameTransform.Identity);
    }

    public static OcrDetectedRegion Region(
        string id,
        double x,
        double y,
        double width,
        double height,
        double orientationDegrees = 0d,
        double confidence = 0.9d,
        OcrRegionContext? context = null) =>
        new(
            id,
            OcrPolygon.FromRectangle(new OcrRectangle(x, y, width, height)),
            orientationDegrees,
            confidence,
            context);

    public static OcrRequest Request(
        IReadOnlyList<OcrDetectedRegion>? regions = null,
        OcrImage? enhanced = null,
        string inputHash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") =>
        new(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            inputHash,
            Image(),
            new OcrRectangle(30, 15, 110, 70),
            enhanced,
            regions);
}

internal sealed class StubTextRegionDetector : ITextRegionDetector
{
    private readonly Func<OcrImage, CancellationToken, ValueTask<IReadOnlyList<OcrDetectedRegion>>> _detect;
    private int _callCount;

    public StubTextRegionDetector(IReadOnlyList<OcrDetectedRegion> regions)
        : this((_, _) => ValueTask.FromResult(regions))
    {
    }

    public StubTextRegionDetector(
        Func<OcrImage, CancellationToken, ValueTask<IReadOnlyList<OcrDetectedRegion>>> detect) =>
        _detect = detect;

    public int CallCount => Volatile.Read(ref _callCount);

    public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
        OcrImage image,
        CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _callCount);
        return _detect(image, cancellationToken);
    }
}

internal sealed class StubTextRecognizer : ITextRecognizer
{
    private readonly Func<IReadOnlyList<OcrCrop>, CancellationToken, ValueTask<IReadOnlyList<OcrRecognition>>> _recognize;
    private readonly ConcurrentQueue<int> _batchSizes = new();
    private int _callCount;

    public StubTextRecognizer(
        IReadOnlyDictionary<(string RegionId, OcrSourceImage Source), IReadOnlyList<OcrRecognitionAlternative>>
            alternatives)
        : this((crops, _) => ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
            crops.Select(crop => new OcrRecognition(
                crop.RegionId,
                crop.SourceImage,
                alternatives[(crop.RegionId, crop.SourceImage)],
                0.25d)).ToArray()))
    {
    }

    public StubTextRecognizer(
        Func<IReadOnlyList<OcrCrop>, CancellationToken, ValueTask<IReadOnlyList<OcrRecognition>>> recognize) =>
        _recognize = recognize;

    public string ModelId => "deterministic-test-recognizer";

    public string ModelVersion => "1";

    public string ModelSha256 => "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    public string ConfigurationFingerprint => "fixed-fixture-v1";

    public int CallCount => Volatile.Read(ref _callCount);

    public IReadOnlyList<int> BatchSizes => _batchSizes.ToArray();

    public ValueTask<IReadOnlyList<OcrRecognition>> RecognizeBatchAsync(
        IReadOnlyList<OcrCrop> crops,
        CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _callCount);
        _batchSizes.Enqueue(crops.Count);
        return _recognize(crops, cancellationToken);
    }
}

internal sealed class InMemoryOcrResultCache : IOcrResultCache
{
    private readonly ConcurrentDictionary<string, OcrCachedPayload> _entries = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, OcrRecognitionCachePayload> _recognitionEntries =
        new(StringComparer.Ordinal);
    private int _readCount;
    private int _recognitionReadCount;
    private int _recognitionWriteCount;
    private int _writeCount;

    public int ReadCount => Volatile.Read(ref _readCount);

    public int RecognitionReadCount => Volatile.Read(ref _recognitionReadCount);

    public int RecognitionWriteCount => Volatile.Read(ref _recognitionWriteCount);

    public int WriteCount => Volatile.Read(ref _writeCount);

    public ValueTask<OcrCachedPayload?> TryGetAsync(string key, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _readCount);
        _entries.TryGetValue(key, out OcrCachedPayload? result);
        return ValueTask.FromResult(result);
    }

    public ValueTask PutAsync(string key, OcrCachedPayload payload, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _writeCount);
        _entries[key] = payload;
        return ValueTask.CompletedTask;
    }

    public ValueTask<OcrRecognitionCachePayload?> TryGetRecognitionAsync(
        string key,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _recognitionReadCount);
        _recognitionEntries.TryGetValue(key, out OcrRecognitionCachePayload? result);
        return ValueTask.FromResult(result);
    }

    public ValueTask PutRecognitionAsync(
        string key,
        OcrRecognitionCachePayload payload,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _recognitionWriteCount);
        _recognitionEntries[key] = payload;
        return ValueTask.CompletedTask;
    }
}
