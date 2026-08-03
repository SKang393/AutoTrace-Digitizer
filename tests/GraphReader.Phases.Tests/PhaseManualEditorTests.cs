// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Phases.Tests;

[TestClass]
public sealed class PhaseManualEditorTests
{
    private const string DividerOne = "40000000-0000-0000-0000-000000000001";
    private const string DividerTwo = "40000000-0000-0000-0000-000000000002";
    private const string PhaseOne = "50000000-0000-0000-0000-000000000001";

    [TestMethod]
    public void AddMoveDeleteAndRelabelPersistAsImmutableOverrides()
    {
        var editor = new PhaseManualEditor();
        var original = new PhaseManualOverrides();

        PhaseEditResult added = editor.Apply(
            original,
            new AddPhaseDividerCommand(CommandId(1), DividerOne, 140, PhaseDividerStyle.Dotted),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);
        PhaseEditResult moved = editor.Apply(
            added.Overrides,
            new MovePhaseDividerCommand(CommandId(2), DividerOne, 160, PhaseDividerStyle.Dotted, 140),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);
        PhaseEditResult relabeled = editor.Apply(
            moved.Overrides,
            new RelabelPhaseCommand(
                CommandId(3),
                PhaseOne,
                "custom",
                PhaseNormalizedType.Unknown,
                "User phase",
                PhaseTestFixture.PlotBounds.Left,
                260),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);
        PhaseEditResult deleted = editor.Apply(
            relabeled.Overrides,
            new DeletePhaseDividerCommand(CommandId(4), DividerOne, 160),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);

        Assert.IsTrue(added.Succeeded);
        Assert.AreEqual("add_divider", added.Audit?.Action);
        Assert.IsTrue(moved.Succeeded);
        Assert.AreEqual("move_divider", moved.Audit?.Action);
        Assert.AreEqual(160, moved.Overrides.Dividers.Single().OriginalX);
        Assert.AreEqual(PhaseDividerStyle.Dotted, moved.Overrides.Dividers.Single().Style);
        Assert.IsTrue(relabeled.Succeeded);
        Assert.AreEqual("relabel_phase", relabeled.Audit?.Action);
        Assert.AreEqual("custom", relabeled.Overrides.Labels.Single().Code);
        Assert.IsTrue(deleted.Succeeded);
        Assert.AreEqual("delete_divider", deleted.Audit?.Action);
        Assert.AreEqual(0, deleted.Overrides.Dividers.Count);
        CollectionAssert.AreEqual(new[] { DividerOne }, deleted.Overrides.DeletedDividerIds.ToArray());
        Assert.IsNull(deleted.Overrides.DeletedDividers.Single().ReplacedAutomaticOriginalX);
        Assert.AreEqual(1, deleted.Overrides.Labels.Count);

        Assert.AreEqual(0, original.Dividers.Count);
        Assert.AreEqual(140, added.Overrides.Dividers.Single().OriginalX);
        Assert.AreEqual(0, added.Overrides.Labels.Count);
        Assert.AreEqual(0, moved.Overrides.Labels.Count);
    }

    [TestMethod]
    public void MoveDetectedDividerCreatesManualPositionOverride()
    {
        PhaseEditResult result = new PhaseManualEditor().Apply(
            new PhaseManualOverrides(),
            new MovePhaseDividerCommand(CommandId(5), DividerTwo, 225, PhaseDividerStyle.Dashed, 200),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded);
        PhaseManualDivider divider = result.Overrides.Dividers.Single();
        Assert.AreEqual(DividerTwo, divider.DividerId);
        Assert.AreEqual(225, divider.OriginalX);
        Assert.AreEqual(PhaseDividerStyle.Dashed, divider.Style);
        Assert.AreEqual(200, divider.ReplacedAutomaticOriginalX);
    }

    [TestMethod]
    public void InvalidOrConflictingEditReturnsStructuredFailureWithoutMutation()
    {
        var original = new PhaseManualOverrides(
            [new PhaseManualDivider(DividerOne, 140, PhaseDividerStyle.Solid)],
            null,
            null);
        var editor = new PhaseManualEditor();

        PhaseEditResult duplicate = editor.Apply(
            original,
            new AddPhaseDividerCommand(CommandId(6), DividerTwo, 140, PhaseDividerStyle.Dashed),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);
        PhaseEditResult outside = editor.Apply(
            original,
            new MovePhaseDividerCommand(
                CommandId(7),
                DividerOne,
                PhaseTestFixture.PlotBounds.Left,
                PhaseDividerStyle.Solid,
                140),
            PhaseTestFixture.PlotBounds,
            CancellationToken.None);

        Assert.IsFalse(duplicate.Succeeded);
        Assert.AreEqual("PHASE_DUPLICATE_DIVIDER", duplicate.Failure?.Code);
        Assert.IsNull(duplicate.Audit);
        Assert.AreSame(original, duplicate.Overrides);
        Assert.IsFalse(outside.Succeeded);
        Assert.AreEqual("PHASE_INVALID_COMMAND", outside.Failure?.Code);
        Assert.AreSame(original, outside.Overrides);
    }

    [TestMethod]
    public void EditorHonorsCancellation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.ThrowsExactly<OperationCanceledException>(() =>
            new PhaseManualEditor().Apply(
                new PhaseManualOverrides(),
                new AddPhaseDividerCommand(CommandId(8), DividerOne, 140, PhaseDividerStyle.Solid),
                PhaseTestFixture.PlotBounds,
                cancellation.Token));
    }

    private static string CommandId(int value) =>
        $"60000000-0000-0000-0000-{value:000000000000}";
}
