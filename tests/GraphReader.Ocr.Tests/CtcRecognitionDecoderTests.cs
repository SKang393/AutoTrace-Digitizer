// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class CtcRecognitionDecoderTests
{
    [TestMethod]
    public void DecoderCollapsesRepeatsAndBlankClassesDeterministically()
    {
        float[] logits = Logits(
            classCount: 3,
            1,
            1,
            0,
            2,
            0);

        IReadOnlyList<CtcDecodedAlternative> alternatives = CtcRecognitionDecoder.Decode(
            logits,
            timeSteps: 5,
            alphabet: "01",
            blankClassIndex: 0);

        Assert.IsNotEmpty(alternatives);
        Assert.AreEqual("01", alternatives[0].Text);
        Assert.IsGreaterThan(0.95d, alternatives[0].Confidence);
    }

    [TestMethod]
    public void DecoderRetainsAUniqueLowerConfidenceAlternative()
    {
        float[] logits =
        [
            0f, 4f, 3.8f,
            4f, 0f, 0f,
        ];

        IReadOnlyList<CtcDecodedAlternative> alternatives = CtcRecognitionDecoder.Decode(
            logits,
            timeSteps: 2,
            alphabet: "01",
            blankClassIndex: 0,
            maximumAlternatives: 2);

        Assert.HasCount(2, alternatives);
        Assert.AreEqual("0", alternatives[0].Text);
        Assert.AreEqual("1", alternatives[1].Text);
        Assert.IsGreaterThan(alternatives[1].Confidence, alternatives[0].Confidence);
    }

    [TestMethod]
    public void DecoderRejectsTensorShapeThatDoesNotMatchAlphabet()
    {
        Assert.Throws<ArgumentException>(() => CtcRecognitionDecoder.Decode(
            [1f, 2f, 3f, 4f],
            timeSteps: 2,
            alphabet: "01",
            blankClassIndex: 0));
    }

    private static float[] Logits(int classCount, params int[] winningClasses)
    {
        var values = Enumerable.Repeat(-6f, classCount * winningClasses.Length).ToArray();
        for (var time = 0; time < winningClasses.Length; time++)
        {
            values[(time * classCount) + winningClasses[time]] = 6f;
        }

        return values;
    }
}
