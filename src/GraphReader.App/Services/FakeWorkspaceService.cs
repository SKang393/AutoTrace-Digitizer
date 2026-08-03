// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Media;
using GraphReader.App.Models;
using GraphReader.App.ViewModels;

namespace GraphReader.App.Services;

public sealed class FakeWorkspaceService : IWorkspaceService
{
    private const string FilledSeriesId = "series-filled-circle";
    private const string OpenSeriesId = "series-open-circle";

    public IReadOnlyList<WorkspaceTabViewModel> CreateWorkspace()
    {
        ObservableCollection<GraphPoint> points =
        [
            new("point-001", FilledSeriesId, 88, 222, 1, 18, "a"),
            new("point-002", FilledSeriesId, 146, 196, 2, 29, "a"),
            new("point-003", FilledSeriesId, 204, 172, 3, 39, "a"),
            new("point-004", FilledSeriesId, 282, 128, 4, 58, "b"),
            new("point-005", FilledSeriesId, 340, 105, 5, 68, "b"),
            new("point-006", OpenSeriesId, 398, 86, 6, 76, "b"),
            new("point-007", OpenSeriesId, 456, 74, 7, 81, "b"),
        ];

        ObservableCollection<SeriesCardViewModel> series = [];
        series.Add(new SeriesCardViewModel(
            FilledSeriesId,
            "●",
            "Filled circle",
            "Primary intervention",
            0.96,
            points));
        series.Add(new SeriesCardViewModel(
            OpenSeriesId,
            "○",
            "Open circle",
            "Generalization probes",
            0.91,
            points));

        WorkspaceTabViewModel tab = new(
            "tab-graph-001",
            "Example graph",
            points,
            series,
            CreateGraphDrawing(false),
            CreateGraphDrawing(true),
            new PhaseDivider(244d / 520d, 28d / 280d, 242d / 280d));

        return [tab];
    }

    public static Task RunStageAsync(
        WorkflowStage stage,
        CancellationToken cancellationToken)
    {
        _ = stage;
        return Task.Delay(TimeSpan.FromMilliseconds(20), cancellationToken);
    }

    Task IWorkspaceService.RunStageAsync(
        WorkflowStage stage,
        CancellationToken cancellationToken) => RunStageAsync(stage, cancellationToken);

    private static DrawingImage CreateGraphDrawing(bool enhanced)
    {
        DrawingGroup group = new();
        using (DrawingContext context = group.Open())
        {
            context.DrawRectangle(Brushes.White, null, new Rect(0, 0, 520, 280));
            Pen axisPen = new(Brushes.Black, enhanced ? 2.2 : 1.6);
            context.DrawLine(axisPen, new Point(54, 24), new Point(54, 242));
            context.DrawLine(axisPen, new Point(54, 242), new Point(492, 242));

            Pen seriesPen = new(Brushes.Black, enhanced ? 1.8 : 1.2);
            Point[] filled =
            [
                new(88, 222), new(146, 196), new(204, 172), new(282, 128), new(340, 105),
            ];
            for (int index = 1; index < filled.Length; index++)
            {
                context.DrawLine(seriesPen, filled[index - 1], filled[index]);
            }

            foreach (Point point in filled)
            {
                context.DrawEllipse(Brushes.Black, null, point, 5, 5);
            }

            foreach (Point point in new[] { new Point(398, 86), new Point(456, 74) })
            {
                context.DrawEllipse(Brushes.White, seriesPen, point, 6, 6);
            }

        }

        group.Freeze();
        DrawingImage image = new(group);
        image.Freeze();
        return image;
    }
}
