// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Integration.Tests.Validation.Core.Private;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#pragma warning disable CA1861 // Small collection expressions are intentional test fixtures.

namespace GraphReader.Integration.Tests.Validation.Tests;

[TestClass]
public sealed class ValidationPrivateEvaluationTests
{
    [TestMethod]
    public void ValidationPrivateEvaluationRequiresExplicitOptInAndDataDirectory()
    {
        PrivateEvaluationAdapter adapter = CreateAdapter();

        PrivateEvaluationAvailability availability = adapter.CheckAvailability(
            new PrivateEvaluationRequest(ExplicitOptIn: false, ExternalDirectory: null));

        Assert.IsFalse(availability.IsAvailable);
        Assert.IsNull(availability.ExternalDirectory);
        CollectionAssert.AreEquivalent(
            new[]
            {
                PrivateEvaluationUnavailableReason.ExplicitOptInRequired,
                PrivateEvaluationUnavailableReason.ExternalDirectoryRequired,
            },
            availability.Reasons.Select(reason => reason.Code).ToArray());
    }

    [TestMethod]
    public void ValidationPrivateEvaluationCannotRunInContinuousIntegration()
    {
        string externalDirectory = CreateExternalDirectory();
        try
        {
            PrivateEvaluationAdapter adapter = CreateAdapter(
                variable => variable == "CI" ? "true" : null);

            PrivateEvaluationAvailability availability = adapter.CheckAvailability(
                new PrivateEvaluationRequest(ExplicitOptIn: true, externalDirectory));

            Assert.IsFalse(availability.IsAvailable);
            Assert.HasCount(1, availability.Reasons);
            Assert.AreEqual(
                PrivateEvaluationUnavailableReason.ContinuousIntegrationDetected,
                availability.Reasons[0].Code);
        }
        finally
        {
            Directory.Delete(externalDirectory);
        }
    }

    [TestMethod]
    public void ValidationPrivateEvaluationRejectsRepositoryData()
    {
        string repositoryRoot = RepositoryRoot.Find();
        PrivateEvaluationAdapter adapter = CreateAdapter();

        PrivateEvaluationAvailability availability = adapter.CheckAvailability(
            new PrivateEvaluationRequest(ExplicitOptIn: true, repositoryRoot));

        Assert.IsFalse(availability.IsAvailable);
        Assert.HasCount(1, availability.Reasons);
        Assert.AreEqual(
            PrivateEvaluationUnavailableReason.ExternalDirectoryMustBeOutsideRepository,
            availability.Reasons[0].Code);
    }

    [TestMethod]
    public void ValidationPrivateEvaluationAllowsExplicitExternalLocalData()
    {
        string externalDirectory = CreateExternalDirectory();
        try
        {
            PrivateEvaluationAdapter adapter = CreateAdapter();

            PrivateEvaluationAvailability availability = adapter.CheckAvailability(
                new PrivateEvaluationRequest(ExplicitOptIn: true, externalDirectory));

            Assert.IsTrue(availability.IsAvailable);
            Assert.AreEqual(
                Path.TrimEndingDirectorySeparator(Path.GetFullPath(externalDirectory)),
                availability.ExternalDirectory);
            Assert.IsEmpty(availability.Reasons);
        }
        finally
        {
            Directory.Delete(externalDirectory);
        }
    }

    private static PrivateEvaluationAdapter CreateAdapter(
        Func<string, string?>? readEnvironmentVariable = null) =>
        new(RepositoryRoot.Find(), readEnvironmentVariable ?? (_ => null));

    private static string CreateExternalDirectory()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"GraphReader-Validation-{Guid.NewGuid():N}");
        return Directory.CreateDirectory(directory).FullName;
    }
}
