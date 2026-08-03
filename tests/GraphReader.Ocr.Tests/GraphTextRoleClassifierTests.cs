// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class GraphTextRoleClassifierTests
{
    private static readonly OcrRectangle Plot = new(30, 15, 110, 70);

    [TestMethod]
    public void RotatedYLabelIsAxisTitleRatherThanTickOrParticipant()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region(
            "y-title",
            3,
            30,
            12,
            42,
            orientationDegrees: -90,
            context: new OcrRegionContext(AxisTitleExpected: true));

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "Percentage Correct", Plot);

        Assert.AreEqual(OcrTextRole.AxisTitle, result.Role);
        Assert.IsGreaterThan(0.5d, result.Confidence);
    }

    [TestMethod]
    public void ParticipantBandTextIsParticipantMetadataEvidence()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region(
            "participant",
            62,
            2,
            48,
            10,
            context: new OcrRegionContext(InParticipantBand: true));

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "Chandler", Plot);

        Assert.AreEqual(OcrTextRole.Participant, result.Role);
        Assert.AreNotEqual(OcrTextRole.LegendText, result.Role);
    }

    [TestMethod]
    public void LegendProximityClassifiesLegendTextWithoutUsingColor()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region(
            "legend",
            92,
            28,
            35,
            9,
            context: new OcrRegionContext(NearLegendGlyph: true));

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "Treatment", Plot);

        Assert.AreEqual(OcrTextRole.LegendText, result.Role);
    }

    [TestMethod]
    public void GeneralizationNearCalloutIsAnnotationTextNotMarkerSemantics()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region(
            "generalization",
            80,
            42,
            48,
            9,
            context: new OcrRegionContext(NearAnnotationArrow: true));

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "Generalization", Plot);

        Assert.AreEqual(OcrTextRole.Annotation, result.Role);
        Assert.AreNotEqual(OcrTextRole.LegendText, result.Role);
    }

    [TestMethod]
    public void GeneralizationNearPhaseDividerIsPhaseHeadingEvidence()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region(
            "generalization-phase",
            76,
            3,
            52,
            9,
            context: new OcrRegionContext(NearPhaseDivider: true));

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "Generalization", Plot);

        Assert.AreEqual(OcrTextRole.PhaseHeading, result.Role);
    }

    [TestMethod]
    public void PhaseBAbovePlotIsNotMisclassifiedAsNumericTickEight()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region("phase-b", 72, 3, 8, 9);

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "B", Plot);

        Assert.AreEqual(OcrTextRole.PhaseHeading, result.Role);
        Assert.IsFalse(GraphNumericParser.Parse("B").IsSuccess);
    }

    [TestMethod]
    public void PhaseHeadingRequiresDividerContext()
    {
        OcrDetectedRegion withDivider = OcrTestFixtures.Region(
            "phase",
            74,
            3,
            38,
            9,
            context: new OcrRegionContext(NearPhaseDivider: true));
        OcrDetectedRegion withoutDivider = OcrTestFixtures.Region(
            "plain",
            74,
            30,
            38,
            9);

        RoleClassification heading = GraphTextRoleClassifier.Classify(withDivider, "Intervention", Plot);
        RoleClassification plain = GraphTextRoleClassifier.Classify(withoutDivider, "Intervention", Plot);

        Assert.AreEqual(OcrTextRole.PhaseHeading, heading.Role);
        Assert.AreNotEqual(OcrTextRole.PhaseHeading, plain.Role);
    }

    [TestMethod]
    public void NumericLocationSeparatesXAndYTicks()
    {
        OcrDetectedRegion xRegion = OcrTestFixtures.Region(
            "x-tick",
            60,
            89,
            10,
            7,
            context: new OcrRegionContext(NumericExpected: true));
        OcrDetectedRegion yRegion = OcrTestFixtures.Region(
            "y-tick",
            15,
            42,
            11,
            7,
            context: new OcrRegionContext(NumericExpected: true));

        Assert.AreEqual(OcrTextRole.XTick, GraphTextRoleClassifier.Classify(xRegion, "10", Plot).Role);
        Assert.AreEqual(OcrTextRole.YTick, GraphTextRoleClassifier.Classify(yRegion, "50", Plot).Role);
    }

    [TestMethod]
    public void ExplicitRoleHintWinsWhenGeometricSignalsConflict()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region(
            "confirmed-participant",
            95,
            30,
            30,
            8,
            context: new OcrRegionContext(
                NearLegendGlyph: true,
                InParticipantBand: true,
                ExplicitRoleHint: OcrTextRole.Participant));

        RoleClassification result = GraphTextRoleClassifier.Classify(region, "Morgan", Plot);

        Assert.AreEqual(OcrTextRole.Participant, result.Role);
        Assert.IsTrue(result.Reasons.Any(reason =>
            reason.Contains("explicit", StringComparison.OrdinalIgnoreCase)));
    }
}
