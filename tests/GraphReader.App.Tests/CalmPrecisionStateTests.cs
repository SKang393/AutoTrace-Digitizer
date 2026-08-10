// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.IO;
using GraphReader.App.Models;
using GraphReader.App.Services;
using GraphReader.App.ViewModels;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class CalmPrecisionStateTests
{
    [TestMethod]
    public void EmptyReadyAndRecordedReviewStatesAreDerivedWithoutGuessing()
    {
        using var empty = new MainWindowViewModel(new UnavailableWorkspaceService());
        using var ready = new MainWindowViewModel(new ControlledWorkspaceService(
            CreateEmptyTab(),
            static (_, _) => Task.CompletedTask));
        using var reviewing = new MainWindowViewModel(new FakeWorkspaceService());

        Assert.AreEqual(WorkspaceSurfaceState.Empty, empty.SurfaceState);
        Assert.AreEqual(WorkspaceSurfaceState.Ready, ready.SurfaceState);
        Assert.AreEqual(WorkspaceSurfaceState.Reviewing, reviewing.SurfaceState);
    }

    [TestMethod]
    public async Task AnalyzeIsExplicitCancelableAndRestoresPriorSurface()
    {
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var viewModel = new MainWindowViewModel(new ControlledWorkspaceService(
            CreateEmptyTab(),
            async (stage, cancellationToken) =>
            {
                Assert.AreEqual(WorkflowStage.Detect, stage);
                started.SetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            }));

        Assert.AreEqual(WorkspaceSurfaceState.Ready, viewModel.SurfaceState);
        viewModel.AutoDetectCommand.Execute(null);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.AreEqual(WorkspaceSurfaceState.Analyzing, viewModel.SurfaceState);
        Assert.IsTrue(viewModel.IsBusy);

        viewModel.CancelCommand.Execute(null);
        await WaitUntilAsync(() => !viewModel.IsBusy);

        Assert.AreEqual(WorkspaceSurfaceState.Ready, viewModel.SurfaceState);
    }

    [TestMethod]
    public void ReviewProjectionAndTableSelectionStaySynchronized()
    {
        using var viewModel = new MainWindowViewModel(new FakeWorkspaceService());

        Assert.HasCount(1, viewModel.ReviewIssues);
        Assert.AreEqual(ReviewIssueKind.Calibration, viewModel.ReviewIssues[0].Kind);
        Assert.AreEqual(ReviewIssueSeverity.Blocking, viewModel.ReviewIssues[0].Severity);
        Assert.AreEqual(viewModel.SelectedTab!.Points.Count, viewModel.DataPreviewRows.Count);

        DataPreviewRowViewModel target = viewModel.DataPreviewRows[^1];
        viewModel.SelectDataPreviewRowCommand.Execute(target);

        Assert.AreEqual(target.PointId, viewModel.SelectedPointId);
        Assert.AreSame(target, viewModel.SelectedDataPreviewRow);
        Assert.AreEqual(target.PixelX, viewModel.Magnifier.PixelPosition.X);
        Assert.AreEqual(target.PixelY, viewModel.Magnifier.PixelPosition.Y);
    }

    [TestMethod]
    public void EditHistoryIsPerTabAndRestoresManualPointMovement()
    {
        using var viewModel = new MainWindowViewModel(new FakeWorkspaceService());
        WorkspaceTabViewModel firstTab = viewModel.SelectedTab!;
        GraphPoint point = firstTab.Points[0];
        double originalX = point.PixelX;
        double originalY = point.PixelY;

        viewModel.MovePoint(point.PointId, originalX + 11, originalY + 7);

        Assert.IsTrue(viewModel.CanUndo);
        Assert.AreEqual(originalX + 11, viewModel.SelectedTab!.Points[0].PixelX);
        viewModel.UndoCommand.Execute(null);
        Assert.AreEqual(originalX, viewModel.SelectedTab.Points[0].PixelX);
        Assert.AreEqual(originalY, viewModel.SelectedTab.Points[0].PixelY);
        Assert.IsTrue(viewModel.CanRedo);

        viewModel.RedoCommand.Execute(null);
        Assert.AreEqual(originalX + 11, viewModel.SelectedTab.Points[0].PixelX);
        Assert.AreEqual(originalY + 7, viewModel.SelectedTab.Points[0].PixelY);

        var secondTab = new WorkspaceTabViewModel(
            "second-tab",
            "Second graph",
            new ObservableCollection<GraphPoint>(),
            new ObservableCollection<SeriesCardViewModel>(),
            imageSource: null,
            enhancedImageSource: null,
            phaseOverlayContent: null,
            overlayContent: new object());
        viewModel.Tabs.Add(secondTab);
        viewModel.SelectedTab = secondTab;
        Assert.IsFalse(viewModel.CanUndo);
        viewModel.SelectedTab = firstTab;
        Assert.IsTrue(viewModel.CanUndo);
    }

    [TestMethod]
    public void ExportPreviewSummarizesEvidenceAndHonorsBlockingIssues()
    {
        using var viewModel = new MainWindowViewModel(new FakeWorkspaceService());

        viewModel.OpenExportPreviewCommand.Execute(null);

        Assert.AreEqual(WorkspaceSurfaceState.ExportPreview, viewModel.SurfaceState);
        Assert.IsNotNull(viewModel.ExportSummary);
        Assert.AreEqual(viewModel.SelectedTab!.Points.Count, viewModel.ExportSummary.PointCount);
        Assert.AreEqual(viewModel.SeriesCards.Count, viewModel.ExportSummary.SeriesCount);
        Assert.AreEqual(viewModel.BlockingReviewIssueCount, viewModel.ExportSummary.BlockingIssueCount);
        Assert.IsFalse(string.IsNullOrWhiteSpace(viewModel.ExportSummary.ProvenanceSummary));
        Assert.IsFalse(string.IsNullOrWhiteSpace(viewModel.ExportSummary.OutputDirectoryDisplay));
        Assert.IsFalse(viewModel.ExportSummary.CanExport);
        Assert.IsFalse(viewModel.ConfirmExportCommand.CanExecute(null));
    }

    [TestMethod]
    public void ExportWarningsRequireExplicitAcknowledgement()
    {
        var summary = new ExportSummaryViewModel(
            pointCount: 4,
            seriesCount: 1,
            phaseCount: 2,
            blockingIssueCount: 0,
            warningCount: 1,
            outputDirectory: null,
            outputFileNames: Array.Empty<string>());

        Assert.IsTrue(summary.RequiresWarningAcknowledgement);
        Assert.IsFalse(summary.CanExport);

        summary.WarningsAcknowledged = true;

        Assert.IsTrue(summary.CanExport);
    }

    [TestMethod]
    public void RecentProjectsLoadOnlyExistingDistinctPaths()
    {
        string root = Path.Combine(Path.GetTempPath(), $"graph-reader-recent-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        string existingProject = Path.Combine(root, "example.garproj");
        string recentList = Path.Combine(root, "recent-projects.txt");
        File.WriteAllText(existingProject, "test");
        File.WriteAllLines(recentList, [existingProject, existingProject, Path.Combine(root, "missing.garproj")]);

        try
        {
            using var viewModel = new MainWindowViewModel(
                new UnavailableWorkspaceService(),
                localizationService: null,
                recentProjectsPath: recentList);

            Assert.IsTrue(viewModel.HasRecentProjects);
            Assert.HasCount(1, viewModel.RecentProjects);
            Assert.AreEqual(existingProject, viewModel.RecentProjects[0]);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static WorkspaceTabViewModel CreateEmptyTab() => new(
        "ready-tab",
        "Ready graph",
        new ObservableCollection<GraphPoint>(),
        new ObservableCollection<SeriesCardViewModel>(),
        imageSource: null,
        enhancedImageSource: null,
        phaseOverlayContent: null,
        pixelWidth: 640,
        pixelHeight: 480,
        overlayContent: new object());

    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        while (!condition())
        {
            await Task.Delay(10, timeout.Token);
        }
    }

    private sealed class ControlledWorkspaceService(
        WorkspaceTabViewModel initialTab,
        Func<WorkflowStage, CancellationToken, Task> runStage) : IWorkspaceService
    {
        public IReadOnlyList<WorkspaceTabViewModel> CreateWorkspace() => [initialTab];

        public Task RunStageAsync(WorkflowStage stage, CancellationToken cancellationToken) =>
            runStage(stage, cancellationToken);
    }
}
