// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Axis.Tests;

[TestClass]
public sealed class RobustCalibrationAcceptanceTests
{
    [TestMethod]
    public void OmittedZeroAndOutlierOcrStillRecoverYTransform()
    {
        NumericTickEvidence[] ticks =
        [
            new("y20", 250, 20),
            new("y40", 200, 40),
            new("y60", 150, 60),
            new("y80", 100, 80),
            new("y100", 50, 100),
            new("bad-600", 120, 600, 0.95),
        ];

        LinearTransformFitResult result = RobustCalibration.FitY(ticks);

        Assert.IsTrue(result.IsValid);
        Assert.IsNotNull(result.Transform);
        Assert.AreEqual(300, result.Transform.GraphToPixel(0), 2);
        Assert.AreEqual(50, result.Transform.GraphToPixel(100), 2);
        CollectionAssert.Contains(result.Diagnostics.OutlierIds.ToArray(), "bad-600");
        Assert.IsTrue(result.Uncertainty.RootMeanSquareErrorPixels <= 2);
    }

    [TestMethod]
    public void DuplicateNonmonotonicAndOutlierXTicksDoNotCorruptFit()
    {
        PrintedXTickEvidence[] ticks =
        [
            new("x1", 100, 1),
            new("x6", 225, 6),
            new("x11", 350, 11),
            new("x16", 475, 16),
            new("x21", 600, 21),
            new("duplicate-wrong", 350, 18, 0.5),
            new("reversed", 525, 4, 0.9),
            new("ocr-100", 260, 100, 0.95),
        ];

        LinearTransformFitResult result = RobustCalibration.FitX(ticks);

        Assert.IsTrue(result.IsValid);
        Assert.IsNotNull(result.Transform);
        Assert.AreEqual(100, result.Transform.GraphToPixel(1), 2);
        Assert.AreEqual(600, result.Transform.GraphToPixel(21), 2);
        Assert.IsTrue(result.Diagnostics.OutlierIds.Count >= 3);
    }

    [TestMethod]
    public void DecreasingPrintedSessionsAreNotAcceptedAsAValidXTransform()
    {
        PrintedXTickEvidence[] ticks =
        [
            new("x10", 100, 10),
            new("x5", 200, 5),
            new("x1", 300, 1),
        ];

        LinearTransformFitResult result = RobustCalibration.FitX(ticks);

        Assert.AreNotEqual(CalibrationValidity.Valid, result.Validity);
        Assert.IsFalse(result.IsValid);
        Assert.IsTrue(result.Reasons.Any(reason =>
            reason.Contains("increase", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("monotonic", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("direction", StringComparison.OrdinalIgnoreCase)));
    }

    [TestMethod]
    public void YValuesIncreasingDownwardAreNotAcceptedAsAValidYTransform()
    {
        NumericTickEvidence[] ticks =
        [
            new("y0", 50, 0),
            new("y20", 100, 20),
            new("y40", 150, 40),
        ];

        LinearTransformFitResult result = RobustCalibration.FitY(ticks);

        Assert.AreNotEqual(CalibrationValidity.Valid, result.Validity);
        Assert.IsFalse(result.IsValid);
        Assert.IsTrue(result.Reasons.Any(reason =>
            reason.Contains("downward", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("direction", StringComparison.OrdinalIgnoreCase)));
    }

    [TestMethod]
    public void EqualWeightCompetingYLinesRequireReviewAsAmbiguous()
    {
        NumericTickEvidence[] ticks =
        [
            new("line-a-1", 50, 100),
            new("line-a-2", 100, 80),
            new("line-b-1", 150, 100),
            new("line-b-2", 200, 80),
        ];

        LinearTransformFitResult result = RobustCalibration.FitY(ticks);

        Assert.AreEqual(CalibrationValidity.NeedsReview, result.Validity);
        Assert.IsFalse(result.IsValid);
        Assert.IsTrue(result.Reasons.Any(reason =>
            reason.Contains("ambiguous", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("majority", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("competing", StringComparison.OrdinalIgnoreCase)));
    }

    [TestMethod]
    public void EqualConfidenceNoncollinearYTicksRemainAmbiguousAcrossInputOrder()
    {
        var first = new NumericTickEvidence("first", 50, 100);
        var second = new NumericTickEvidence("second", 100, 80);
        var third = new NumericTickEvidence("third", 150, 20);
        NumericTickEvidence[][] permutations =
        [
            [first, second, third],
            [third, second, first],
            [second, first, third],
            [third, first, second],
        ];

        foreach (NumericTickEvidence[] ticks in permutations)
        {
            LinearTransformFitResult result = RobustCalibration.FitY(ticks);

            Assert.AreEqual(CalibrationValidity.NeedsReview, result.Validity);
            Assert.IsFalse(result.IsValid);
            Assert.IsTrue(result.Reasons.Any(reason =>
                reason.Contains("ambiguous", StringComparison.OrdinalIgnoreCase) ||
                reason.Contains("competing", StringComparison.OrdinalIgnoreCase) ||
                reason.Contains("alternative", StringComparison.OrdinalIgnoreCase)));
        }
    }

    [TestMethod]
    public void NearEqualConfidenceCompetingYModelsRemainAmbiguous()
    {
        double[][] confidenceSets =
        [
            [1.00, 0.99, 0.98],
            [0.91, 0.90, 0.89],
        ];

        foreach (double[] confidence in confidenceSets)
        {
            NumericTickEvidence[] ticks =
            [
                new("first", 50, 100, confidence[0]),
                new("second", 100, 80, confidence[1]),
                new("third", 150, 20, confidence[2]),
            ];

            LinearTransformFitResult result = RobustCalibration.FitY(ticks);

            Assert.AreEqual(CalibrationValidity.NeedsReview, result.Validity);
            Assert.IsTrue(result.Reasons.Any(reason =>
                reason.Contains("competing", StringComparison.OrdinalIgnoreCase) ||
                reason.Contains("ambiguous", StringComparison.OrdinalIgnoreCase)));
        }
    }

    [TestMethod]
    public void SessionFirstAnchorsUseSessionColumnsWithinTwoPixels()
    {
        var request = new SessionFirstCalibrationRequest
        {
            YTicks =
            [
                new("y20", 250, 20),
                new("y40", 200, 40),
                new("y60", 150, 60),
                new("y80", 100, 80),
                new("y100", 50, 100),
            ],
            PrintedXTicks =
            [
                new("x1", 100, 1),
                new("x6", 225, 6),
                new("x11", 350, 11),
                new("x16", 475, 16),
                new("x21", 600, 21),
            ],
            Lattice = new SessionLatticeRequest
            {
                PrintedTicks =
                [
                    new("x1", 100, 1),
                    new("x6", 225, 6),
                    new("x11", 350, 11),
                    new("x16", 475, 16),
                    new("x21", 600, 21),
                ],
            },
            YMaximum = 100,
            XMaximum = 21,
        };

        SessionFirstCalibrationResult result = RobustCalibration.FitSessionFirst(request);

        Assert.AreEqual(CalibrationValidity.Valid, result.Validity);
        CalibrationAnchor session1Y0 = Anchor(result, CalibrationAnchorKind.Session1Y0);
        CalibrationAnchor session1YMax = Anchor(result, CalibrationAnchorKind.Session1YMaximum);
        CalibrationAnchor sessionMaxY0 = Anchor(result, CalibrationAnchorKind.SessionMaximumY0);
        AssertPointWithin(session1Y0.Screen, 100, 300, 2);
        AssertPointWithin(session1YMax.Screen, 100, 50, 2);
        AssertPointWithin(sessionMaxY0.Screen, 600, 300, 2);
        Assert.AreEqual(session1Y0.Screen.X, session1YMax.Screen.X, 0.01);
        Assert.AreEqual(21d, sessionMaxY0.GraphX);
        Assert.IsTrue(result.Anchors.All(anchor => anchor.IsExact));
        Assert.IsTrue(result.Anchors.All(anchor =>
            anchor.CoordinateSpace == AxisGeometryCoordinateSpaces.OriginalPixels));
    }

    [TestMethod]
    public void CalibrationHonorsCancellation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.ThrowsExactly<OperationCanceledException>(() =>
            RobustCalibration.FitY(
                [new NumericTickEvidence("a", 10, 0), new NumericTickEvidence("b", 20, 1)],
                cancellationToken: cancellation.Token));
        Assert.ThrowsExactly<OperationCanceledException>(() =>
            RobustCalibration.FitX(
                [new PrintedXTickEvidence("a", 10, 1), new PrintedXTickEvidence("b", 20, 2)],
                cancellationToken: cancellation.Token));
        Assert.ThrowsExactly<OperationCanceledException>(() =>
            RobustCalibration.FitSessionFirst(new SessionFirstCalibrationRequest(), cancellation.Token));
    }

    [TestMethod]
    public void CalibrationRejectsEvidenceOutsideOriginalPixelCoordinates()
    {
        NumericTickEvidence[] yTicks =
        [
            new("y0", 300, 0, CoordinateSpace: "enhanced_pixels"),
            new("y100", 50, 100),
        ];
        PrintedXTickEvidence[] xTicks =
        [
            new("x1", 100, 1),
            new("x10", 325, 10, CoordinateSpace: "deskewed_pixels"),
        ];

        Assert.Throws<ArgumentException>(() => RobustCalibration.FitY(yTicks));
        Assert.Throws<ArgumentException>(() => RobustCalibration.FitX(xTicks));
    }

    private static CalibrationAnchor Anchor(
        SessionFirstCalibrationResult result,
        CalibrationAnchorKind kind) =>
        result.Anchors.Single(anchor => anchor.Kind == kind);

    private static void AssertPointWithin(
        PixelPoint actual,
        double expectedX,
        double expectedY,
        double tolerance)
    {
        Assert.AreEqual(expectedX, actual.X, tolerance);
        Assert.AreEqual(expectedY, actual.Y, tolerance);
    }
}
