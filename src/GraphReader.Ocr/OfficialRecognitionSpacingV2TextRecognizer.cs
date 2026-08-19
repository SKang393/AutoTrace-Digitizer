// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;

namespace GraphReader.Ocr;

/// <summary>
/// Applies the public-passing image-spacing V2 P2 source-byte rule to exact
/// official recognizer output without changing geometry or confidence.
/// </summary>
public sealed class OfficialRecognitionSpacingV2TextRecognizer : ITextRecognizer
{
    private readonly ITextRecognizer recognizer;

    public OfficialRecognitionSpacingV2TextRecognizer(ITextRecognizer recognizer)
    {
        this.recognizer = recognizer ?? throw new ArgumentNullException(nameof(recognizer));
    }

    public string ModelId => recognizer.ModelId;

    public string ModelVersion => recognizer.ModelVersion;

    public string ModelSha256 => recognizer.ModelSha256;

    public string ConfigurationFingerprint => HashStrings(
    [
        OfficialRecognitionSpacingV2Postprocessor.Revision,
        OfficialRecognitionSpacingV2Postprocessor.ConfigurationFingerprint(),
        recognizer.ModelId,
        recognizer.ModelVersion,
        recognizer.ModelSha256,
        recognizer.ConfigurationFingerprint,
    ]);

    public async ValueTask<IReadOnlyList<OcrRecognition>> RecognizeBatchAsync(
        IReadOnlyList<OcrCrop> crops,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(crops);
        cancellationToken.ThrowIfCancellationRequested();
        if (crops.Any(static crop => crop.SourceCrop is null))
        {
            throw new ArgumentException(
                "Official recognition spacing V2 requires checksum-bound immutable source crops.",
                nameof(crops));
        }

        IReadOnlyList<OcrRecognition> raw = await recognizer
            .RecognizeBatchAsync(crops, cancellationToken)
            .ConfigureAwait(false);
        if (raw.Count != crops.Count)
        {
            throw new InvalidDataException(
                $"Official recognizer returned {raw.Count} results for {crops.Count} source crops.");
        }

        var output = new OcrRecognition[raw.Count];
        for (var index = 0; index < raw.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            OcrRecognition recognition = raw[index];
            if (recognition.Failure is not null || recognition.Alternatives.Count == 0)
            {
                output[index] = recognition;
                continue;
            }

            OcrV8SourceCrop sourceCrop = crops[index].SourceCrop!;
            output[index] = recognition with
            {
                Alternatives = OcrCollections.Freeze(
                    recognition.Alternatives.Select(alternative => alternative with
                    {
                        Text = OfficialRecognitionSpacingV2Postprocessor.Restore(
                            sourceCrop,
                            alternative.Text),
                    })),
            };
        }

        return Array.AsReadOnly(output);
    }

    private static string HashStrings(IEnumerable<string> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string value in values)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            hash.AppendData(BitConverter.GetBytes(bytes.Length));
            hash.AppendData(bytes);
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
}
