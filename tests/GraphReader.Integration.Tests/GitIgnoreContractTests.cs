// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests;

[TestClass]
public sealed class GitIgnoreContractTests
{
    [TestMethod]
    public void LocalInstructionsAndPrivateDataAreIgnoredByGit()
    {
        string root = RepositoryRoot.Find();
        string[] ignoredPaths =
        [
            "AGENTS.md",
            "CODEX_START_HERE.md",
            "CODEX_GOAL_00_REPOSITORY_FOUNDATION.md",
            ".agents/00_REPOSITORY_FOUNDATION_AND_CONTRACT_FREEZE.md",
            "private/research-graph.png",
            "data/private/human-annotations.csv",
        ];

        foreach (string ignoredPath in ignoredPaths)
        {
            ProcessStartInfo startInfo = new("git")
            {
                WorkingDirectory = root,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                UseShellExecute = false,
            };
            startInfo.ArgumentList.Add("check-ignore");
            startInfo.ArgumentList.Add("--no-index");
            startInfo.ArgumentList.Add("--quiet");
            startInfo.ArgumentList.Add("--");
            startInfo.ArgumentList.Add(ignoredPath);

            using Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Could not start git check-ignore.");
            string standardError = process.StandardError.ReadToEnd();
            process.WaitForExit();

            Assert.AreEqual(
                0,
                process.ExitCode,
                $"Expected '{ignoredPath}' to be ignored. git error: {standardError}");
        }
    }
}
