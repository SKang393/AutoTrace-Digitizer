// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;
using GraphReader.Domain;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Domain.Tests;

[TestClass]
public sealed class RecoveryAndSourceTests
{
    [TestMethod]
    public async Task RecoveryDiscoveryReportsNewerValidAutosaveAndRejectsCorruption()
    {
        using var directory = new TemporaryDirectory();
        string autosaveRoot = Path.Combine(directory.Path, "Autosave");
        string corruptOriginalPath = Path.Combine(directory.Path, "damaged.garproj");
        await File.WriteAllTextAsync(corruptOriginalPath, "{damaged original");
        string originalBytes = await File.ReadAllTextAsync(corruptOriginalPath);
        ProjectDocument original = TestProjectFactory.Create();
        ProjectDocument newer = original with
        {
            ModifiedUtc = original.ModifiedUtc.AddMinutes(10),
            Audit = new AuditTrail(original.Audit.Events, original.ModifiedUtc.AddMinutes(10))
        };
        var snapshotService = new ProjectSnapshotService(autosaveRoot);
        string validAutosavePath = snapshotService.GetSnapshotPath(original.ProjectId);
        DomainResult<ProjectSaveReceipt> save = await new ProjectFileStore().SaveAsync(newer, validAutosavePath);
        Assert.IsTrue(save.IsSuccess, FormatErrors(save.Errors));
        Directory.CreateDirectory(autosaveRoot);
        string corruptAutosavePath = Path.Combine(autosaveRoot, "corrupt.autosave.garproj");
        await File.WriteAllTextAsync(corruptAutosavePath, "{\"schema_version\":1,");
        var recovery = new ProjectRecoveryService();

        DomainResult<RecoveryDiscoveryReport> discovered = await recovery.DiscoverAsync(
            autosaveRoot,
            original,
            corruptOriginalPath);

        Assert.IsTrue(discovered.IsSuccess, FormatErrors(discovered.Errors));
        Assert.AreEqual(1, discovered.Value!.Candidates.Count);
        RecoveryCandidate candidate = discovered.Value.Candidates[0];
        Assert.AreEqual(validAutosavePath, candidate.AutosavePath);
        Assert.IsTrue(candidate.IsNewerThanOriginal);
        Assert.AreEqual(RecoveryRecommendation.RestoreRecommended, candidate.Recommendation);
        Assert.AreEqual(1, discovered.Value.RejectedCandidates.Count);
        Assert.AreEqual("PROJECT_CORRUPT", discovered.Value.RejectedCandidates[0].Errors[0].Code);

        string recoveredPath = Path.Combine(directory.Path, "recovered.garproj");
        DomainResult<ProjectSaveReceipt> recovered = await recovery.RecoverToNewFileAsync(
            candidate.AutosavePath,
            recoveredPath);
        Assert.IsTrue(recovered.IsSuccess, FormatErrors(recovered.Errors));
        Assert.AreEqual(originalBytes, await File.ReadAllTextAsync(corruptOriginalPath));
        DomainResult<ProjectDocument> recoveredProject = await new ProjectFileStore().LoadAsync(recoveredPath);
        Assert.IsTrue(recoveredProject.IsSuccess, FormatErrors(recoveredProject.Errors));
        Assert.AreEqual(newer.ModifiedUtc, recoveredProject.Value!.ModifiedUtc);

        DomainResult<ProjectSaveReceipt> refusedOverwrite = await recovery.RecoverToNewFileAsync(
            candidate.AutosavePath,
            recoveredPath);
        Assert.IsFalse(refusedOverwrite.IsSuccess);
        Assert.AreEqual("PROJECT_TARGET_EXISTS", refusedOverwrite.Errors[0].Code);
    }

    [TestMethod]
    public async Task SourceReferencesStoreOnlyPathAndSha256AndDetectChanges()
    {
        using var directory = new TemporaryDirectory();
        string sourcePath = Path.Combine(directory.Path, "private source.png");
        byte[] originalContent = Encoding.UTF8.GetBytes("private-pixel-fixture");
        await File.WriteAllBytesAsync(sourcePath, originalContent);
        string expectedHash = Convert.ToHexStringLower(SHA256.HashData(originalContent));
        var integrity = new SourceIntegrityService();

        DomainResult<SourceReference> created = await integrity.CreateReferenceAsync(
            sourcePath,
            SourceKind.Image);

        Assert.IsTrue(created.IsSuccess, FormatErrors(created.Errors));
        Assert.AreEqual(sourcePath, created.Value!.LocalPath);
        Assert.AreEqual(expectedHash, created.Value.Sha256);
        DomainResult<SourceIntegrityResult> valid = await integrity.VerifyAsync(created.Value);
        Assert.IsTrue(valid.IsSuccess, FormatErrors(valid.Errors));

        ProjectDocument project = ProjectDocument.Create("0.0.2", TestProjectFactory.CreatedUtc) with
        {
            ProjectId = TestProjectFactory.ProjectId,
            Sources = new[] { created.Value }
        };
        string json = new ProjectJsonSerializer().Serialize(project).Value!;
        Assert.IsTrue(json.Contains(expectedHash, StringComparison.Ordinal));
        Assert.IsTrue(json.Contains(sourcePath.Replace("\\", "\\\\", StringComparison.Ordinal), StringComparison.Ordinal));
        Assert.IsFalse(json.Contains("private-pixel-fixture", StringComparison.Ordinal));

        await File.AppendAllTextAsync(sourcePath, "changed");
        DomainResult<SourceIntegrityResult> changed = await integrity.VerifyAsync(created.Value);
        Assert.IsFalse(changed.IsSuccess);
        Assert.AreEqual("SOURCE_HASH_MISMATCH", changed.Errors[0].Code);
    }

    private static string FormatErrors(IReadOnlyList<DomainError> errors) =>
        string.Join(Environment.NewLine, errors.Select(error => $"{error.Code}: {error.TechnicalMessage}"));
}
