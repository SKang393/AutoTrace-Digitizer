// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections;
using System.IO;
using System.Resources;
using System.Security.Cryptography;
using System.Text.Json;
using System.Windows.Media;
using System.Xml.Linq;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class NotoSansPackagingTests
{
    private const string ReleaseTag = "noto-monthly-release-2026.05.01";
    private const string ReleaseCommit = "66c4b351c58f99ace5a6265d329080d74b057909";
    private const string LicenseSha256 = "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a";

    private static readonly ExpectedFont[] ExpectedFonts =
    [
        new(
            "noto-sans-regular",
            "NotoSans-Regular.ttf",
            621572,
            400,
            "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823"),
        new(
            "noto-sans-medium",
            "NotoSans-Medium.ttf",
            619976,
            500,
            "635d93d1131d791f2576de90b3bb0f7cdf61929906e8420a61b5f7f8e76420bb"),
        new(
            "noto-sans-semibold",
            "NotoSans-SemiBold.ttf",
            625052,
            600,
            "a4e91fd530ac2b4ef5367240144ff37d7d65d66cf76f2e9a2187b93c676f92d0"),
    ];

    [TestMethod]
    public void SourceFontFilesMatchPinnedReleaseAndEmbeddedMetadata()
    {
        var fontDirectory = Path.Combine(
            RepositoryTestPaths.Root,
            "src",
            "GraphReader.App",
            "Assets",
            "Fonts");
        var bundledNames = Directory
            .EnumerateFiles(fontDirectory, "*.ttf", SearchOption.TopDirectoryOnly)
            .Select(Path.GetFileName)
            .Order(StringComparer.Ordinal)
            .ToArray();

        CollectionAssert.AreEquivalent(
            ExpectedFonts.Select(font => font.FileName).ToArray(),
            bundledNames,
            "Only the three reviewed Noto Sans weights may be bundled.");

        foreach (var expected in ExpectedFonts)
        {
            var path = Path.Combine(fontDirectory, expected.FileName);
            var file = new FileInfo(path);
            Assert.AreEqual(expected.Length, file.Length, $"Unexpected length for {expected.FileName}.");
            Assert.AreEqual(expected.Sha256, ComputeSha256(path), $"Unexpected bytes for {expected.FileName}.");

            var glyph = new GlyphTypeface(new Uri(path, UriKind.Absolute));
            Assert.IsTrue(
                glyph.FamilyNames.Values.Contains("Noto Sans", StringComparer.Ordinal),
                $"{expected.FileName} does not expose the expected preferred family name.");
            Assert.IsTrue(
                glyph.VersionStrings.Values.Any(value => value.Contains("Version 2.015", StringComparison.Ordinal)),
                $"{expected.FileName} has unexpected version metadata.");
            Assert.IsTrue(
                glyph.Copyrights.Values.Any(value => value.Contains("The Noto Project Authors", StringComparison.Ordinal)),
                $"{expected.FileName} has unexpected copyright metadata.");
            Assert.IsTrue(
                glyph.LicenseDescriptions.Values.Any(value => value.Contains("SIL Open Font License", StringComparison.Ordinal)),
                $"{expected.FileName} has unexpected license metadata.");
            Assert.AreEqual(expected.OpenTypeWeight, glyph.Weight.ToOpenTypeWeight());
        }
    }

    [TestMethod]
    public void ApplicationAssemblyEmbedsTheExactApprovedFontBytes()
    {
        var assembly = typeof(GraphReader.App.App).Assembly;
        var generatedResourceName = assembly
            .GetManifestResourceNames()
            .Single(name => name.EndsWith(".g.resources", StringComparison.Ordinal));
        using var generatedResources = assembly.GetManifestResourceStream(generatedResourceName);
        Assert.IsNotNull(generatedResources);
        using var reader = new ResourceReader(generatedResources);
        var embeddedHashes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        IDictionaryEnumerator entries = reader.GetEnumerator();
        while (entries.MoveNext())
        {
            var resourcePath = entries.Key as string;
            if (resourcePath is null ||
                !resourcePath.StartsWith("assets/fonts/", StringComparison.OrdinalIgnoreCase) ||
                !resourcePath.EndsWith(".ttf", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (entries.Value is not Stream fontStream)
            {
                Assert.Fail($"Embedded font resource '{resourcePath}' is not exposed as a stream.");
                return;
            }

            embeddedHashes.Add(Path.GetFileName(resourcePath), ComputeSha256(fontStream));
        }

        CollectionAssert.AreEquivalent(
            ExpectedFonts.Select(font => font.FileName.ToLowerInvariant()).ToArray(),
            embeddedHashes.Keys.ToArray());
        foreach (var expected in ExpectedFonts)
        {
            Assert.AreEqual(expected.Sha256, embeddedHashes[expected.FileName]);
        }
    }

    [TestMethod]
    public void DesignTokensUseOnlyTheEmbeddedNotoSansFamily()
    {
        var appDirectory = Path.Combine(RepositoryTestPaths.Root, "src", "GraphReader.App");
        var designTokensPath = Path.Combine(appDirectory, "Themes", "DesignTokens.xaml");
        XNamespace xaml = "http://schemas.microsoft.com/winfx/2006/xaml";
        var defaultFamily = XDocument.Load(designTokensPath)
            .Descendants()
            .Single(element => string.Equals(
                element.Attribute(xaml + "Key")?.Value,
                "App.FontFamily.Default",
                StringComparison.Ordinal));
        Assert.AreEqual("../Assets/Fonts/#Noto Sans", defaultFamily.Value.Trim());

        var projectPath = Path.Combine(appDirectory, "GraphReader.App.csproj");
        var resources = XDocument.Load(projectPath)
            .Descendants("Resource")
            .Select(element => element.Attribute("Include")?.Value)
            .Where(value => value is not null)
            .Select(value => value!)
            .ToArray();
        CollectionAssert.AreEquivalent(
            ExpectedFonts.Select(font => $"Assets\\Fonts\\{font.FileName}").ToArray(),
            resources.Where(value => value.StartsWith("Assets\\Fonts\\", StringComparison.Ordinal)).ToArray());

        var legacyReferences = Directory
            .EnumerateFiles(appDirectory, "*", SearchOption.AllDirectories)
            .Where(path =>
                !path.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) &&
                !path.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) &&
                Path.GetExtension(path) is ".xaml" or ".cs" or ".csproj")
            .Select(path => new { Path = path, Text = File.ReadAllText(path) })
            .Where(file =>
                file.Text.Contains("IBM Plex", StringComparison.OrdinalIgnoreCase) ||
                file.Text.Contains("Segoe UI", StringComparison.OrdinalIgnoreCase))
            .Select(file => Path.GetRelativePath(RepositoryTestPaths.Root, file.Path))
            .ToArray();
        Assert.IsEmpty(
            legacyReferences,
            $"Production source contains legacy font references:{Environment.NewLine}{string.Join(Environment.NewLine, legacyReferences)}");
    }

    [TestMethod]
    public void LicenseNoticesAndPackagingAuditCoverEveryBundledWeight()
    {
        var root = RepositoryTestPaths.Root;
        var licensePath = Path.Combine(root, "LICENSES", "NotoSans-OFL-1.1.txt");
        Assert.AreEqual(LicenseSha256, ComputeSha256(licensePath));
        var license = File.ReadAllText(licensePath);
        StringAssert.Contains(license, "Copyright 2022 The Noto Project Authors");
        StringAssert.Contains(license, "SIL OPEN FONT LICENSE Version 1.1");

        var notices = File.ReadAllText(Path.Combine(root, "THIRD_PARTY_NOTICES.md"));
        StringAssert.Contains(notices, ReleaseTag);
        StringAssert.Contains(notices, ReleaseCommit);
        StringAssert.Contains(notices, "LICENSES/NotoSans-OFL-1.1.txt");
        foreach (var expected in ExpectedFonts)
        {
            StringAssert.Contains(notices, expected.FileName);
            StringAssert.Contains(notices, expected.Sha256);
        }

        var applicationLegal = File.ReadAllText(Path.Combine(
            root,
            "src",
            "GraphReader.App",
            "Localization",
            "Resources.en-US.xaml"));
        StringAssert.Contains(applicationLegal, ReleaseTag);
        StringAssert.Contains(applicationLegal, ReleaseCommit);
        StringAssert.Contains(applicationLegal, "OFL-1.1.txt");
        foreach (var expected in ExpectedFonts)
        {
            StringAssert.Contains(applicationLegal, expected.FileName);
            StringAssert.Contains(applicationLegal, expected.Sha256);
        }

        using var releaseAudit = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(root, "packaging", "common", "release-audit.json")));
        var components = releaseAudit.RootElement
            .GetProperty("components")
            .EnumerateArray()
            .ToDictionary(component => component.GetProperty("id").GetString()!, StringComparer.Ordinal);
        foreach (var expected in ExpectedFonts)
        {
            var component = components[expected.AuditId];
            Assert.AreEqual("2.015", component.GetProperty("version").GetString());
            Assert.AreEqual("OFL-1.1", component.GetProperty("license").GetString());
            Assert.AreEqual("exact-binary", component.GetProperty("checksumPolicy").GetString());
            Assert.AreEqual(expected.Sha256, component.GetProperty("artifactSha256").GetString());
            StringAssert.Contains(component.GetProperty("sourceRevision").GetString()!, ReleaseTag);
            StringAssert.Contains(component.GetProperty("sourceRevision").GetString()!, ReleaseCommit);
            Assert.IsTrue(component.GetProperty("commercialUse").GetBoolean());
            Assert.IsTrue(component.GetProperty("redistribution").GetBoolean());
            Assert.AreEqual("reviewed", component.GetProperty("reviewStatus").GetString());
        }

        var applicationComponent = components["graph-auto-reader"];
        StringAssert.Contains(applicationComponent.GetProperty("license").GetString()!, "OFL-1.1");
        Assert.IsTrue(applicationComponent
            .GetProperty("noticePaths")
            .EnumerateArray()
            .Any(value => value.GetString() == "LICENSES/NotoSans-OFL-1.1.txt"));

        using var commonPublish = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(root, "packaging", "common", "publish.json")));
        var requiredSources = commonPublish.RootElement
            .GetProperty("requiredContent")
            .EnumerateArray()
            .Select(item => item.GetProperty("source").GetString())
            .ToArray();
        Assert.Contains("LICENSES", requiredSources);
        Assert.Contains("THIRD_PARTY_NOTICES.md", requiredSources);
        Assert.Contains("packaging/common/release-audit.json", requiredSources);
    }

    private static string ComputeSha256(string path)
    {
        using var stream = File.OpenRead(path);
        return ComputeSha256(stream);
    }

    private static string ComputeSha256(Stream stream)
    {
        if (stream.CanSeek)
        {
            stream.Position = 0;
        }

        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private sealed record ExpectedFont(
        string AuditId,
        string FileName,
        long Length,
        int OpenTypeWeight,
        string Sha256);
}
