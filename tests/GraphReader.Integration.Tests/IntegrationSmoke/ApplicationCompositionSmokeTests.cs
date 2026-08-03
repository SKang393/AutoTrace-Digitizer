// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Localization;
using GraphReader.App.Services;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

[TestClass]
public sealed class ApplicationCompositionSmokeTests
{
    [TestMethod]
    public void ProductionCompositionFailsClosedWhenAdaptersAreUnavailable()
    {
        ApplicationCompositionResult composition =
            ApplicationComposition.Create(WorkflowRuntimeEnvironment.Production);

        Assert.AreEqual(WorkflowRuntimeEnvironment.Production, composition.Environment);
        Assert.IsInstanceOfType<UnavailableWorkspaceService>(composition.WorkspaceService);
        Assert.AreEqual("PRODUCTION_WORKFLOW_UNAVAILABLE", composition.StartupError!.Code);
        Assert.AreEqual(LocalizationKeys.ProductionWorkflowUnavailable, composition.StartupError.UserMessageKey);
        Assert.AreEqual("install_approved_workflow_assets", composition.StartupError.SuggestedAction);
        Assert.IsTrue(composition.StartupError.Recoverable);
        Assert.HasCount(0, composition.WorkspaceService.CreateWorkspace());
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
