// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

[TestClass]
public sealed class ApplicationCompositionSmokeTests
{
    [TestMethod]
    public void OrdinaryRuntimeSelectionDefaultsToRealEmptyManualPreview()
    {
        Assert.AreEqual(WorkflowRuntimeEnvironment.ManualPreview, RuntimeModeSelector.Select(string.Empty));

        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.ManualPreview);

        Assert.AreEqual(WorkflowRuntimeEnvironment.ManualPreview, composition.Environment);
        var workspace = Assert.IsInstanceOfType<ManualPreviewWorkspaceService>(composition.WorkspaceService);
        Assert.IsNull(composition.StartupError);
        Assert.IsFalse(workspace.UsesFakeGraphData);
        Assert.HasCount(0, workspace.CreateWorkspace());
        Assert.IsTrue(workspace.AutomaticStages.All(status => status.State == AutomaticStageState.Unavailable));
        Assert.IsTrue(workspace.AutomaticStages.All(status => !string.IsNullOrWhiteSpace(status.Explanation)));
    }

    [TestMethod]
    public void RecordedFakeCannotBeSelectedByOrdinaryRuntimeConfiguration()
    {
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.ManualPreview,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.RecordedFake)));
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.Production,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.Production)));
    }

    [TestMethod]
    public void ProductionCompositionKeepsManualWorkflowAvailableAndAutomaticStagesFailClosed()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);

        Assert.AreEqual(WorkflowRuntimeEnvironment.Production, composition.Environment);
        var workspace = Assert.IsInstanceOfType<ProductionWorkspaceService>(composition.WorkspaceService);
        Assert.IsNull(composition.StartupError);
        Assert.AreEqual(WorkflowRuntimeEnvironment.Production, workspace.RuntimeEnvironment);
        Assert.IsFalse(workspace.UsesFakeGraphData);
        Assert.HasCount(6, workspace.AutomaticStages);
        Assert.IsTrue(workspace.AutomaticStages.All(static stage => stage.State == AutomaticStageState.Unavailable));
        Assert.IsTrue(workspace.AutomaticStages.All(static stage => !string.IsNullOrWhiteSpace(stage.Explanation)));
        Assert.AreEqual(
            "enhancement,axis,ocr,markers,legends,phases",
            string.Join(',', workspace.AutomaticStages.Select(static stage => stage.Stage)));
        Assert.IsInstanceOfType<IAutomaticWorkspaceService>(workspace);
        Assert.IsNull(workspace.LastAutomaticRun);
    }

    [TestMethod]
    public async Task ProductionAutomaticWorkflowRemainsFailClosedWithoutApprovedStages()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);
        var workspace = Assert.IsInstanceOfType<IAutomaticWorkspaceService>(composition.WorkspaceService);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => workspace.RunAutomaticDetectionAsync(CancellationToken.None));

        StringAssert.Contains(exception.Message, "native runtime");
        Assert.IsNull(workspace.LastAutomaticRun);
    }

    [TestMethod]
    public void RecordedFakeRequiresExplicitEnvironmentSelection()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.RecordedFake);

        Assert.AreEqual(WorkflowRuntimeEnvironment.RecordedFake, composition.Environment);
        Assert.IsInstanceOfType<FakeWorkspaceService>(composition.WorkspaceService);
        Assert.IsNull(composition.StartupError);
        Assert.HasCount(1, composition.WorkspaceService.CreateWorkspace());
    }
}
