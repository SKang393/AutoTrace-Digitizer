// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Phases;

public sealed class PhaseManualEditor : IPhaseManualEditor
{
    public PhaseEditResult Apply(
        PhaseManualOverrides current,
        PhaseEditCommand command,
        PhaseRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(current);
        ArgumentNullException.ThrowIfNull(command);
        cancellationToken.ThrowIfCancellationRequested();

        PhaseReasoningFailure? stateFailure = ValidateState(current, plotBounds, cancellationToken);
        if (stateFailure is not null)
        {
            return Failure(current, stateFailure);
        }

        if (!IsUuid(command.CommandId))
        {
            return Failure(
                current,
                Error("PHASE_INVALID_COMMAND", "The phase edit command ID must be a non-empty UUID."));
        }

        return command switch
        {
            MovePhaseDividerCommand move => Move(current, move, plotBounds, cancellationToken),
            AddPhaseDividerCommand add => Add(current, add, plotBounds, cancellationToken),
            DeletePhaseDividerCommand delete => Delete(current, delete, plotBounds, cancellationToken),
            RelabelPhaseCommand relabel => Relabel(current, relabel, plotBounds, cancellationToken),
            _ => Failure(
                current,
                Error("PHASE_INVALID_COMMAND", "The phase edit command kind is unsupported.")),
        };
    }

    private static PhaseEditResult Move(
        PhaseManualOverrides current,
        MovePhaseDividerCommand command,
        PhaseRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        if (!IsUuid(command.DividerId) ||
            !IsInteriorX(command.OriginalX, plotBounds) ||
            !IsInteriorX(command.PreviousOriginalX, plotBounds) ||
            !Enum.IsDefined(command.Style))
        {
            return Failure(
                current,
                Error(
                    "PHASE_INVALID_COMMAND",
                    "A moved divider requires a UUID plus previous and replacement coordinates inside the plot bounds."));
        }

        if (ContainsUuid(current.DeletedDividerIds, command.DividerId))
        {
            return Failure(
                current,
                Error("PHASE_DIVIDER_DELETED", "A deleted divider cannot be moved until it is restored."));
        }

        PhaseManualDivider? existing = FindDivider(current.Dividers, command.DividerId);
        if (existing is not null &&
            existing.OriginalX.Equals(command.OriginalX) &&
            existing.Style == command.Style)
        {
            return NoChange(current, "The divider is already at the requested original-pixel coordinate.");
        }

        if (HasOtherDividerAtX(current.Dividers, command.DividerId, command.OriginalX))
        {
            return Failure(
                current,
                Error("PHASE_DUPLICATE_DIVIDER", "Another manual divider already has the requested coordinate."));
        }

        cancellationToken.ThrowIfCancellationRequested();
        double? replacedAutomaticOriginalX = existing is null
            ? command.PreviousOriginalX
            : existing.ReplacedAutomaticOriginalX;
        var moved = new PhaseManualDivider(
            CanonicalUuid(command.DividerId),
            command.OriginalX,
            command.Style,
            1,
            replacedAutomaticOriginalX);
        PhaseManualDivider[] dividers = current.Dividers
            .Where(divider => !UuidEquals(divider.DividerId, command.DividerId))
            .Append(moved)
            .OrderBy(static divider => divider.OriginalX)
            .ThenBy(static divider => divider.DividerId, StringComparer.Ordinal)
            .ToArray();

        return Complete(
            current,
            CanonicalUuid(command.CommandId),
            "move_divider",
            CanonicalUuid(command.DividerId),
            dividers: dividers);
    }

    private static PhaseEditResult Add(
        PhaseManualOverrides current,
        AddPhaseDividerCommand command,
        PhaseRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        if (!IsUuid(command.DividerId) ||
            !IsInteriorX(command.OriginalX, plotBounds) ||
            !Enum.IsDefined(command.Style))
        {
            return Failure(
                current,
                Error(
                    "PHASE_INVALID_COMMAND",
                    "An added divider requires a UUID, a valid style, and a finite x coordinate inside the plot bounds."));
        }

        if (FindDivider(current.Dividers, command.DividerId) is not null ||
            ContainsUuid(current.DeletedDividerIds, command.DividerId))
        {
            return Failure(
                current,
                Error("PHASE_DUPLICATE_DIVIDER", "The divider UUID is already present in the manual override state."));
        }

        if (current.Dividers.Any(divider => divider.OriginalX.Equals(command.OriginalX)))
        {
            return Failure(
                current,
                Error("PHASE_DUPLICATE_DIVIDER", "A manual divider already has the requested coordinate."));
        }

        cancellationToken.ThrowIfCancellationRequested();
        PhaseManualDivider[] dividers = current.Dividers
            .Append(new PhaseManualDivider(command.DividerId, command.OriginalX, command.Style, 1))
            .OrderBy(static divider => divider.OriginalX)
            .ThenBy(static divider => divider.DividerId, StringComparer.Ordinal)
            .ToArray();

        return Complete(
            current,
            CanonicalUuid(command.CommandId),
            "add_divider",
            CanonicalUuid(command.DividerId),
            dividers: dividers);
    }

    private static PhaseEditResult Delete(
        PhaseManualOverrides current,
        DeletePhaseDividerCommand command,
        PhaseRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        if (!IsUuid(command.DividerId) || !IsInteriorX(command.PreviousOriginalX, plotBounds))
        {
            return Failure(
                current,
                Error("PHASE_INVALID_COMMAND", "A deleted divider requires a UUID and its previous original coordinate."));
        }

        if (ContainsUuid(current.DeletedDividerIds, command.DividerId))
        {
            return NoChange(current, "The divider is already deleted.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        PhaseManualDivider? existing = FindDivider(current.Dividers, command.DividerId);
        PhaseManualDivider[] dividers = current.Dividers
            .Where(divider => !UuidEquals(divider.DividerId, command.DividerId))
            .ToArray();
        double? replacedAutomaticOriginalX = existing is null
            ? command.PreviousOriginalX
            : existing.ReplacedAutomaticOriginalX;
        PhaseDeletedDivider[] deletedDividers = current.DeletedDividers
            .Append(new PhaseDeletedDivider(
                CanonicalUuid(command.DividerId),
                replacedAutomaticOriginalX))
            .OrderBy(static divider => divider.DividerId, StringComparer.Ordinal)
            .ToArray();

        return Complete(
            current,
            CanonicalUuid(command.CommandId),
            "delete_divider",
            CanonicalUuid(command.DividerId),
            dividers: dividers,
            deletedDividers: deletedDividers);
    }

    private static PhaseEditResult Relabel(
        PhaseManualOverrides current,
        RelabelPhaseCommand command,
        PhaseRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        if (!IsUuid(command.PhaseId) ||
            string.IsNullOrWhiteSpace(command.Code) ||
            string.IsNullOrWhiteSpace(command.LabelText) ||
            !IsPhaseBounds(command.PreviousOriginalXMinimum, command.PreviousOriginalXMaximum, plotBounds) ||
            !Enum.IsDefined(command.NormalizedType))
        {
            return Failure(
                current,
                Error(
                    "PHASE_INVALID_COMMAND",
                    "A phase relabel requires a UUID, code, label text, valid type, and previous region bounds."));
        }

        string code = command.Code.Trim();
        string labelText = command.LabelText.Trim();
        PhaseLabelOverride? existing = FindLabel(current.Labels, command.PhaseId);
        if (existing is not null &&
            string.Equals(existing.Code, code, StringComparison.Ordinal) &&
            existing.NormalizedType == command.NormalizedType &&
            string.Equals(existing.LabelText, labelText, StringComparison.Ordinal))
        {
            return NoChange(current, "The phase already has the requested label override.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        double originalXMinimum = existing?.OriginalXMinimum ?? command.PreviousOriginalXMinimum;
        double originalXMaximum = existing?.OriginalXMaximum ?? command.PreviousOriginalXMaximum;
        PhaseLabelOverride[] labels = current.Labels
            .Where(label => !UuidEquals(label.PhaseId, command.PhaseId))
            .Append(new PhaseLabelOverride(
                command.PhaseId,
                code,
                command.NormalizedType,
                labelText,
                originalXMinimum,
                originalXMaximum))
            .OrderBy(static label => label.PhaseId, StringComparer.Ordinal)
            .ToArray();

        return Complete(
            current,
            CanonicalUuid(command.CommandId),
            "relabel_phase",
            CanonicalUuid(command.PhaseId),
            labels: labels);
    }

    private static PhaseEditResult Complete(
        PhaseManualOverrides current,
        string commandId,
        string action,
        string targetId,
        IEnumerable<PhaseManualDivider>? dividers = null,
        IEnumerable<PhaseDeletedDivider>? deletedDividers = null,
        IEnumerable<PhaseLabelOverride>? labels = null)
    {
        var next = new PhaseManualOverrides(
            dividers ?? current.Dividers,
            deletedDividerIds: null,
            labels: labels ?? current.Labels,
            deletedDividers: deletedDividers ?? current.DeletedDividers);
        return new PhaseEditResult(next, new PhaseEditAudit(commandId, action, targetId), null);
    }

    private static PhaseReasoningFailure? ValidateState(
        PhaseManualOverrides state,
        PhaseRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        if (!plotBounds.IsValid)
        {
            return Error("PHASE_INVALID_BOUNDS", "Plot bounds must be finite with positive width and height.");
        }

        var dividerIds = new HashSet<Guid>();
        var dividerCoordinates = new HashSet<double>();
        foreach (PhaseManualDivider divider in state.Dividers)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryGetUuid(divider.DividerId, out Guid dividerId) ||
                !dividerIds.Add(dividerId) ||
                !dividerCoordinates.Add(divider.OriginalX) ||
                !IsInteriorX(divider.OriginalX, plotBounds) ||
                (divider.ReplacedAutomaticOriginalX is double replacedX && !IsInteriorX(replacedX, plotBounds)) ||
                !Enum.IsDefined(divider.Style) ||
                !double.IsFinite(divider.Confidence) ||
                divider.Confidence < 0 ||
                divider.Confidence > 1)
            {
                return Error(
                    "PHASE_INVALID_STATE",
                    "Manual dividers must have unique UUIDs and coordinates with valid bounds, styles, and confidence.");
            }
        }

        var deletedIds = new HashSet<Guid>();
        foreach (PhaseDeletedDivider deletedDivider in state.DeletedDividers)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryGetUuid(deletedDivider.DividerId, out Guid deletedId) ||
                !deletedIds.Add(deletedId) ||
                dividerIds.Contains(deletedId) ||
                (deletedDivider.ReplacedAutomaticOriginalX is double replacedX &&
                 !IsInteriorX(replacedX, plotBounds)))
            {
                return Error(
                    "PHASE_INVALID_STATE",
                    "Deleted divider IDs must be unique UUIDs and cannot remain active manual dividers.");
            }
        }

        var phaseIds = new HashSet<Guid>();
        foreach (PhaseLabelOverride label in state.Labels)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryGetUuid(label.PhaseId, out Guid phaseId) ||
                !phaseIds.Add(phaseId) ||
                string.IsNullOrWhiteSpace(label.Code) ||
                string.IsNullOrWhiteSpace(label.LabelText) ||
                !ValidOptionalPhaseBounds(label, plotBounds) ||
                !Enum.IsDefined(label.NormalizedType))
            {
                return Error(
                    "PHASE_INVALID_STATE",
                    "Phase label overrides must have unique UUIDs, valid semantics, codes, and label text.");
            }
        }

        return null;
    }

    private static PhaseManualDivider? FindDivider(
        IEnumerable<PhaseManualDivider> dividers,
        string dividerId) =>
        dividers.SingleOrDefault(divider => UuidEquals(divider.DividerId, dividerId));

    private static PhaseLabelOverride? FindLabel(
        IEnumerable<PhaseLabelOverride> labels,
        string phaseId) =>
        labels.SingleOrDefault(label => UuidEquals(label.PhaseId, phaseId));

    private static bool ContainsUuid(IEnumerable<string> ids, string value) =>
        ids.Any(id => UuidEquals(id, value));

    private static bool HasOtherDividerAtX(
        IEnumerable<PhaseManualDivider> dividers,
        string dividerId,
        double originalX) =>
        dividers.Any(divider =>
            !UuidEquals(divider.DividerId, dividerId) && divider.OriginalX.Equals(originalX));

    private static bool IsInteriorX(double originalX, PhaseRectangle plotBounds) =>
        double.IsFinite(originalX) && originalX > plotBounds.Left && originalX < plotBounds.Right;

    private static bool IsPhaseBounds(double minimum, double maximum, PhaseRectangle plotBounds) =>
        double.IsFinite(minimum) && double.IsFinite(maximum) &&
        minimum >= plotBounds.Left && maximum <= plotBounds.Right && minimum < maximum;

    private static bool ValidOptionalPhaseBounds(PhaseLabelOverride label, PhaseRectangle plotBounds) =>
        label.OriginalXMinimum is null && label.OriginalXMaximum is null ||
        label.OriginalXMinimum is double minimum && label.OriginalXMaximum is double maximum &&
        IsPhaseBounds(minimum, maximum, plotBounds);

    private static bool IsUuid(string? value) =>
        TryGetUuid(value, out _);

    private static bool TryGetUuid(string? value, out Guid id) =>
        Guid.TryParse(value, out id) && id != Guid.Empty;

    private static bool UuidEquals(string left, string right) =>
        Guid.TryParse(left, out Guid leftId) &&
        Guid.TryParse(right, out Guid rightId) &&
        leftId == rightId;

    private static string CanonicalUuid(string value) => Guid.Parse(value).ToString("D");

    private static PhaseEditResult NoChange(PhaseManualOverrides current, string technicalMessage) =>
        Failure(current, Error("PHASE_EDIT_NO_CHANGE", technicalMessage));

    private static PhaseEditResult Failure(
        PhaseManualOverrides current,
        PhaseReasoningFailure failure) =>
        new(current, null, failure);

    private static PhaseReasoningFailure Error(string code, string technicalMessage) =>
        new(
            code,
            "error",
            "Errors." + code,
            technicalMessage,
            true,
            "review_phase");
}
