// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV8ProductionCompositionFactoryTests
{
    [TestMethod]
    public void UnreviewedPayloadMetadataFailsBeforeModelExposure()
    {
        var wrong = new ModelIdentity("wrong", "1", new string('a', 64), "missing.onnx");
        var payloads = new OcrV8ProductionPayloadSet(wrong, wrong, wrong, wrong, "not-reviewed");

        InvalidDataException exception = Assert.ThrowsExactly<InvalidDataException>(() =>
            OcrV8ProductionCompositionFactory.ValidatePayloads(payloads));

        StringAssert.Contains(exception.Message, "unreviewed identity");
    }
}
