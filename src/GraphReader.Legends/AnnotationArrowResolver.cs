// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Ocr;

namespace GraphReader.Legends;

/// <summary>
/// Resolves annotation callouts from pre-detected strokes, triangular arrowheads,
/// OCR regions, and plot-marker centers. All coordinates remain in original pixels.
/// </summary>
public sealed class AnnotationArrowResolver : IAnnotationArrowResolver
{
    private const double GeometryEpsilon = 1e-9;

    public (IReadOnlyList<LegendAnnotationCallout> Callouts, IReadOnlyList<LegendArtifact> Artifacts) Resolve(
        LegendReasoningRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        ValidateOptions(request.Options);

        List<ArrowCandidate> candidates = BuildCandidates(request, cancellationToken);
        var callouts = new List<LegendAnnotationCallout>(candidates.Count);
        var artifacts = new List<LegendArtifact>(candidates.Count * 2);

        foreach (ArrowCandidate candidate in candidates)
        {
            cancellationToken.ThrowIfCancellationRequested();

            LegendTextRegion? text = FindAnnotationText(
                request.TextRegions,
                candidate.Tail,
                request.Options.MaximumAnnotationTextDistance,
                cancellationToken);

            // Annotation-role evidence is required before a triangular glyph is
            // suppressed. This keeps connected triangle data markers from being
            // misclassified as arrowheads based on geometry alone.
            if (text is null)
            {
                continue;
            }

            double joinConfidence = DistanceConfidence(
                candidate.JoinDistance,
                request.Options.MaximumArrowJoinDistance);
            double pairConfidence = Math.Min(
                Math.Min(candidate.Stroke.Confidence, candidate.Triangle.Confidence),
                joinConfidence);

            artifacts.Add(new LegendArtifact(
                $"arrow-shaft:{candidate.Stroke.StrokeId}",
                LegendArtifactKind.ArrowShaft,
                GetStrokeBounds(candidate.Stroke),
                pairConfidence));
            artifacts.Add(new LegendArtifact(
                $"arrowhead:{candidate.Triangle.TriangleId}",
                LegendArtifactKind.Arrowhead,
                candidate.Triangle.Bounds,
                pairConfidence));

            LegendPlotMarker? target = FindTargetMarker(
                request.PlotMarkers,
                request.PlotBounds,
                candidate,
                request.Options,
                cancellationToken);

            if (target is null)
            {
                continue;
            }

            double textDistance = DistanceToRectangle(candidate.Tail, text.Bounds);
            double targetDistance = candidate.Tip.DistanceTo(target.Center);
            double confidence = Math.Min(
                pairConfidence,
                Math.Min(
                    Math.Min(text.Confidence, DistanceConfidence(
                        textDistance,
                        request.Options.MaximumAnnotationTextDistance)),
                    DistanceConfidence(targetDistance, request.Options.MaximumArrowTargetDistance)));

            callouts.Add(new LegendAnnotationCallout(
                $"arrow-callout:{request.PanelId}:{candidate.Stroke.StrokeId}:{candidate.Triangle.TriangleId}",
                text.RegionId,
                text.Text,
                target.MarkerId,
                candidate.Stroke.StrokeId,
                candidate.Triangle.TriangleId,
                confidence));
        }

        callouts.Sort(static (left, right) => StringComparer.Ordinal.Compare(left.CalloutId, right.CalloutId));
        artifacts.Sort(static (left, right) => StringComparer.Ordinal.Compare(left.ArtifactId, right.ArtifactId));
        cancellationToken.ThrowIfCancellationRequested();

        return (LegendCollections.Freeze(callouts), LegendCollections.Freeze(artifacts));
    }

    private static List<ArrowCandidate> BuildCandidates(
        LegendReasoningRequest request,
        CancellationToken cancellationToken)
    {
        var joins = new List<ArrowCandidate>();

        foreach (LegendTriangleCandidate triangle in request.Triangles)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!IsValidTriangle(triangle))
            {
                continue;
            }

            foreach (LegendStrokeCandidate stroke in request.Strokes)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!IsValidStroke(stroke))
                {
                    continue;
                }

                double startDistance = DistanceToTriangle(stroke.Start, triangle.Points);
                double endDistance = DistanceToTriangle(stroke.End, triangle.Points);
                double joinDistance = Math.Min(startDistance, endDistance);
                if (joinDistance > request.Options.MaximumArrowJoinDistance)
                {
                    continue;
                }

                LegendPoint head = startDistance <= endDistance ? stroke.Start : stroke.End;
                LegendPoint tail = startDistance <= endDistance ? stroke.End : stroke.Start;
                if (FindAnnotationText(
                        request.TextRegions,
                        tail,
                        request.Options.MaximumAnnotationTextDistance,
                        cancellationToken) is null)
                {
                    continue;
                }

                double directionX = head.X - tail.X;
                double directionY = head.Y - tail.Y;
                double length = Math.Sqrt((directionX * directionX) + (directionY * directionY));
                if (length <= GeometryEpsilon)
                {
                    continue;
                }

                directionX /= length;
                directionY /= length;
                LegendPoint tip = triangle.Points
                    .OrderByDescending(point => Project(point, tail, directionX, directionY))
                    .ThenBy(static point => point.X)
                    .ThenBy(static point => point.Y)
                    .First();

                joins.Add(new ArrowCandidate(
                    stroke,
                    triangle,
                    tail,
                    tip,
                    directionX,
                    directionY,
                    joinDistance));
            }
        }

        joins.Sort(static (left, right) =>
        {
            int comparison = left.JoinDistance.CompareTo(right.JoinDistance);
            if (comparison != 0)
            {
                return comparison;
            }

            comparison = right.PairConfidence.CompareTo(left.PairConfidence);
            if (comparison != 0)
            {
                return comparison;
            }

            comparison = StringComparer.Ordinal.Compare(left.Triangle.TriangleId, right.Triangle.TriangleId);
            return comparison != 0
                ? comparison
                : StringComparer.Ordinal.Compare(left.Stroke.StrokeId, right.Stroke.StrokeId);
        });

        var selected = new List<ArrowCandidate>(Math.Min(request.Strokes.Count, request.Triangles.Count));
        var usedStrokes = new HashSet<string>(StringComparer.Ordinal);
        var usedTriangles = new HashSet<string>(StringComparer.Ordinal);
        foreach (ArrowCandidate join in joins)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (usedStrokes.Contains(join.Stroke.StrokeId) ||
                usedTriangles.Contains(join.Triangle.TriangleId))
            {
                continue;
            }

            usedStrokes.Add(join.Stroke.StrokeId);
            usedTriangles.Add(join.Triangle.TriangleId);
            selected.Add(join);
        }

        return selected;
    }

    private static LegendTextRegion? FindAnnotationText(
        IReadOnlyList<LegendTextRegion> textRegions,
        LegendPoint tail,
        double maximumDistance,
        CancellationToken cancellationToken)
    {
        LegendTextRegion? best = null;
        double bestDistance = double.PositiveInfinity;

        foreach (LegendTextRegion text in textRegions)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (text.Role != OcrTextRole.Annotation ||
                text.ReviewStatus == OcrReviewStatus.Rejected ||
                string.IsNullOrWhiteSpace(text.RegionId) ||
                string.IsNullOrWhiteSpace(text.Text) ||
                !text.Bounds.IsValid ||
                !IsUnitConfidence(text.Confidence))
            {
                continue;
            }

            double distance = DistanceToRectangle(tail, text.Bounds);
            if (distance > maximumDistance)
            {
                continue;
            }

            if (distance < bestDistance - GeometryEpsilon ||
                (Math.Abs(distance - bestDistance) <= GeometryEpsilon &&
                 (best is null || StringComparer.Ordinal.Compare(text.RegionId, best.RegionId) < 0)))
            {
                best = text;
                bestDistance = distance;
            }
        }

        return best;
    }

    private static LegendPlotMarker? FindTargetMarker(
        IReadOnlyList<LegendPlotMarker> plotMarkers,
        LegendRectangle plotBounds,
        ArrowCandidate candidate,
        LegendReasoningOptions options,
        CancellationToken cancellationToken)
    {
        LegendPlotMarker? best = null;
        double bestDistance = double.PositiveInfinity;
        LegendRectangle arrowheadBounds = candidate.Triangle.Bounds;

        foreach (LegendPlotMarker marker in plotMarkers)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(marker.MarkerId) ||
                string.IsNullOrWhiteSpace(marker.SeriesId) ||
                !marker.Center.IsFinite ||
                !plotBounds.Contains(marker.Center) ||
                arrowheadBounds.Contains(marker.Center))
            {
                continue;
            }

            double targetProjection = Project(
                marker.Center,
                candidate.Tip,
                candidate.DirectionX,
                candidate.DirectionY);
            if (targetProjection < -options.MaximumArrowJoinDistance)
            {
                continue;
            }

            double distance = candidate.Tip.DistanceTo(marker.Center);
            if (distance > options.MaximumArrowTargetDistance)
            {
                continue;
            }

            if (distance < bestDistance - GeometryEpsilon ||
                (Math.Abs(distance - bestDistance) <= GeometryEpsilon &&
                 (best is null || StringComparer.Ordinal.Compare(marker.MarkerId, best.MarkerId) < 0)))
            {
                best = marker;
                bestDistance = distance;
            }
        }

        return best;
    }

    private static bool IsValidTriangle(LegendTriangleCandidate triangle)
    {
        if (string.IsNullOrWhiteSpace(triangle.TriangleId) ||
            triangle.Points.Count != 3 ||
            !IsUnitConfidence(triangle.Confidence) ||
            triangle.Points.Any(static point => !point.IsFinite))
        {
            return false;
        }

        LegendPoint first = triangle.Points[0];
        LegendPoint second = triangle.Points[1];
        LegendPoint third = triangle.Points[2];
        double twiceArea = Math.Abs(
            ((second.X - first.X) * (third.Y - first.Y)) -
            ((second.Y - first.Y) * (third.X - first.X)));
        return twiceArea > GeometryEpsilon;
    }

    private static bool IsValidStroke(LegendStrokeCandidate stroke) =>
        !string.IsNullOrWhiteSpace(stroke.StrokeId) &&
        stroke.Start.IsFinite &&
        stroke.End.IsFinite &&
        double.IsFinite(stroke.Thickness) &&
        stroke.Thickness > 0 &&
        IsUnitConfidence(stroke.Confidence) &&
        stroke.Start.DistanceTo(stroke.End) > GeometryEpsilon;

    private static bool IsUnitConfidence(double confidence) =>
        double.IsFinite(confidence) && confidence >= 0 && confidence <= 1;

    private static double DistanceToTriangle(LegendPoint point, IReadOnlyList<LegendPoint> triangle)
    {
        if (IsInsideTriangle(point, triangle[0], triangle[1], triangle[2]))
        {
            return 0;
        }

        return Math.Min(
            DistanceToSegment(point, triangle[0], triangle[1]),
            Math.Min(
                DistanceToSegment(point, triangle[1], triangle[2]),
                DistanceToSegment(point, triangle[2], triangle[0])));
    }

    private static bool IsInsideTriangle(
        LegendPoint point,
        LegendPoint first,
        LegendPoint second,
        LegendPoint third)
    {
        double firstSign = Cross(point, first, second);
        double secondSign = Cross(point, second, third);
        double thirdSign = Cross(point, third, first);
        bool hasNegative = firstSign < -GeometryEpsilon ||
            secondSign < -GeometryEpsilon ||
            thirdSign < -GeometryEpsilon;
        bool hasPositive = firstSign > GeometryEpsilon ||
            secondSign > GeometryEpsilon ||
            thirdSign > GeometryEpsilon;
        return !(hasNegative && hasPositive);
    }

    private static double Cross(LegendPoint point, LegendPoint first, LegendPoint second) =>
        ((point.X - second.X) * (first.Y - second.Y)) -
        ((first.X - second.X) * (point.Y - second.Y));

    private static double DistanceToSegment(
        LegendPoint point,
        LegendPoint start,
        LegendPoint end)
    {
        double segmentX = end.X - start.X;
        double segmentY = end.Y - start.Y;
        double squaredLength = (segmentX * segmentX) + (segmentY * segmentY);
        if (squaredLength <= GeometryEpsilon)
        {
            return point.DistanceTo(start);
        }

        double projection = (((point.X - start.X) * segmentX) + ((point.Y - start.Y) * segmentY)) /
            squaredLength;
        projection = Math.Clamp(projection, 0, 1);
        return point.DistanceTo(new LegendPoint(
            start.X + (projection * segmentX),
            start.Y + (projection * segmentY)));
    }

    private static double DistanceToRectangle(LegendPoint point, LegendRectangle rectangle)
    {
        double horizontalDistance = Math.Max(rectangle.Left - point.X, Math.Max(0, point.X - rectangle.Right));
        double verticalDistance = Math.Max(rectangle.Top - point.Y, Math.Max(0, point.Y - rectangle.Bottom));
        return Math.Sqrt(
            (horizontalDistance * horizontalDistance) +
            (verticalDistance * verticalDistance));
    }

    private static LegendRectangle GetStrokeBounds(LegendStrokeCandidate stroke)
    {
        double halfThickness = Math.Max(stroke.Thickness / 2, 0.5);
        double left = Math.Min(stroke.Start.X, stroke.End.X) - halfThickness;
        double top = Math.Min(stroke.Start.Y, stroke.End.Y) - halfThickness;
        double right = Math.Max(stroke.Start.X, stroke.End.X) + halfThickness;
        double bottom = Math.Max(stroke.Start.Y, stroke.End.Y) + halfThickness;
        return new LegendRectangle(left, top, right - left, bottom - top);
    }

    private static double Project(
        LegendPoint point,
        LegendPoint origin,
        double directionX,
        double directionY) =>
        ((point.X - origin.X) * directionX) + ((point.Y - origin.Y) * directionY);

    private static double DistanceConfidence(double distance, double maximumDistance) =>
        Math.Clamp(1 - (distance / maximumDistance), 0, 1);

    private static void ValidateOptions(LegendReasoningOptions options)
    {
        if (!double.IsFinite(options.MaximumArrowJoinDistance) || options.MaximumArrowJoinDistance <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                options.MaximumArrowJoinDistance,
                "Maximum arrow join distance must be finite and positive.");
        }

        if (!double.IsFinite(options.MaximumArrowTargetDistance) || options.MaximumArrowTargetDistance <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                options.MaximumArrowTargetDistance,
                "Maximum arrow target distance must be finite and positive.");
        }

        if (!double.IsFinite(options.MaximumAnnotationTextDistance) || options.MaximumAnnotationTextDistance <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                options.MaximumAnnotationTextDistance,
                "Maximum annotation text distance must be finite and positive.");
        }
    }

    private sealed record ArrowCandidate(
        LegendStrokeCandidate Stroke,
        LegendTriangleCandidate Triangle,
        LegendPoint Tail,
        LegendPoint Tip,
        double DirectionX,
        double DirectionY,
        double JoinDistance)
    {
        public double PairConfidence => Math.Min(Stroke.Confidence, Triangle.Confidence);
    }
}
