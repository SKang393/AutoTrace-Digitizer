// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Win32;

namespace GraphReader.Installer;

internal static class Program
{
    private const string PayloadResourceName = "GraphAutoReader.payload.zip";
    private const string ProductDirectoryName = "GraphAutoReader";
    private const string ProductDisplayName = "Graph Auto Reader";
    private const string UninstallRegistryPath =
        @"Software\Microsoft\Windows\CurrentVersion\Uninstall\GraphAutoReader";
    private const int VerificationUsageExitCode = 2;
    private const int VerificationFailureExitCode = 3;
    private const int MaximumPayloadEntries = 100_000;
    private const long MaximumPayloadBytes = 4L * 1024 * 1024 * 1024;

    private static readonly string InstallRoot = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Programs",
        ProductDirectoryName);

    private static readonly string DataRoot = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        ProductDirectoryName);

    private static readonly string StartMenuShortcut = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.StartMenu),
        "Programs",
        ProductDisplayName + ".lnk");

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length > 0 && string.Equals(args[0], "--verify-payload", StringComparison.OrdinalIgnoreCase))
        {
            return VerifyPayload(args);
        }

        bool quiet = args.Contains("--quiet", StringComparer.OrdinalIgnoreCase);
        try
        {
            if (args.Contains("--uninstall", StringComparer.OrdinalIgnoreCase))
            {
                Uninstall();
                ShowMessage(
                    "Graph Auto Reader will finish uninstalling after this message closes. User data will be preserved.",
                    quiet,
                    false);
                return 0;
            }

            bool allowDowngrade = args.Contains("--allow-downgrade", StringComparer.OrdinalIgnoreCase);
            Install(allowDowngrade);
            ShowMessage("Graph Auto Reader was installed for the current user.", quiet, false);
            return 0;
        }
        catch (Exception exception)
        {
            string logPath = Path.Combine(Path.GetTempPath(), "GraphAutoReader-Installer.log");
            File.AppendAllText(
                logPath,
                $"{DateTime.UtcNow:O} {exception}{Environment.NewLine}",
                new UTF8Encoding(false));
            ShowMessage($"Setup could not complete. Details were written to:{Environment.NewLine}{logPath}", quiet, true);
            return 1;
        }
    }

    private static int VerifyPayload(string[] args)
    {
        if (args.Length != 2 || !IsSha256(args[1]))
        {
            Console.Error.WriteLine(
                "Payload verification ERROR: expected --verify-payload followed by one 64-character SHA-256 digest.");
            return VerificationUsageExitCode;
        }

        string expectedDigest = args[1].ToLowerInvariant();
        try
        {
            using Stream payloadStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(PayloadResourceName)
                ?? throw new InvalidDataException("the embedded payload resource is missing");
            using var archive = new ZipArchive(payloadStream, ZipArchiveMode.Read, false);
            string actualDigest = ComputePayloadDigest(archive);
            if (!string.Equals(actualDigest, expectedDigest, StringComparison.Ordinal))
            {
                Console.Error.WriteLine(
                    $"Payload verification FAIL: expected {expectedDigest}, computed {actualDigest}.");
                return VerificationFailureExitCode;
            }

            Console.Out.WriteLine($"Payload verification PASS: {actualDigest}");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Payload verification FAIL: {exception.Message}");
            return VerificationFailureExitCode;
        }
    }

    private static void Install(bool allowDowngrade)
    {
        ValidateInstallRoot();
        string stagingRoot = Path.Combine(Path.GetTempPath(), "GraphAutoReader-Install-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(stagingRoot);

        try
        {
            ExtractPayload(stagingRoot);
            string payloadExecutable = Path.Combine(stagingRoot, "GraphReader.App.exe");
            string payloadMetadata = Path.Combine(stagingRoot, "build-metadata.json");
            if (!File.Exists(payloadExecutable) || !File.Exists(payloadMetadata))
            {
                throw new InvalidDataException("The embedded payload is incomplete.");
            }

            Version payloadVersion = ReadVersion(payloadMetadata);
            string installedMetadata = Path.Combine(InstallRoot, "build-metadata.json");
            if (File.Exists(installedMetadata))
            {
                Version installedVersion = ReadVersion(installedMetadata);
                if (installedVersion > payloadVersion && !allowDowngrade)
                {
                    throw new InvalidOperationException(
                        $"Downgrade from {installedVersion} to {payloadVersion} is blocked. " +
                        "Run setup with --allow-downgrade only after confirming project compatibility.");
                }
            }

            string runningInstaller = Environment.ProcessPath
                ?? throw new InvalidOperationException("The installer executable path is unavailable.");
            if (runningInstaller.StartsWith(InstallRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    "Repair and upgrade must be run from the downloaded setup executable, not Uninstall.exe.");
            }

            if (Directory.Exists(InstallRoot))
            {
                Directory.Delete(InstallRoot, true);
            }

            Directory.CreateDirectory(InstallRoot);
            CopyDirectory(stagingRoot, InstallRoot);

            File.Copy(runningInstaller, Path.Combine(InstallRoot, "Uninstall.exe"), true);

            CreateStartMenuShortcut(Path.Combine(InstallRoot, "GraphReader.App.exe"));
            RegisterUninstaller(payloadVersion);
        }
        finally
        {
            if (Directory.Exists(stagingRoot))
            {
                Directory.Delete(stagingRoot, true);
            }
        }
    }

    private static void Uninstall()
    {
        ValidateInstallRoot();
        if (File.Exists(StartMenuShortcut))
        {
            File.Delete(StartMenuShortcut);
        }

        Registry.CurrentUser.DeleteSubKeyTree(UninstallRegistryPath, false);
        if (!Directory.Exists(InstallRoot))
        {
            return;
        }

        string currentProcessPath = Environment.ProcessPath ?? string.Empty;
        if (!currentProcessPath.StartsWith(InstallRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        {
            Directory.Delete(InstallRoot, true);
            return;
        }

        StartDeferredInstallDirectoryRemoval();
    }

    private static void ExtractPayload(string destination)
    {
        using Stream payloadStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(PayloadResourceName)
            ?? throw new InvalidDataException("The installer payload resource is missing.");
        using var archive = new ZipArchive(payloadStream, ZipArchiveMode.Read, false);
        string destinationPrefix = Path.GetFullPath(destination).TrimEnd(Path.DirectorySeparatorChar) +
            Path.DirectorySeparatorChar;
        var uniquePaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long totalBytes = 0;
        int entryCount = 0;

        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            entryCount++;
            if (entryCount > MaximumPayloadEntries)
            {
                throw new InvalidDataException("The installer payload contains too many entries.");
            }

            string normalizedPath = ValidatePayloadEntry(entry, uniquePaths);
            totalBytes = checked(totalBytes + entry.Length);
            if (totalBytes > MaximumPayloadBytes)
            {
                throw new InvalidDataException("The installer payload exceeds the allowed uncompressed size.");
            }

            string target = Path.GetFullPath(Path.Combine(
                destination,
                normalizedPath.Replace('/', Path.DirectorySeparatorChar)));
            if (!target.StartsWith(destinationPrefix, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"Unsafe path in installer payload: {entry.FullName}");
            }

            if (IsDirectoryEntry(entry))
            {
                Directory.CreateDirectory(target);
                continue;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(target)
                ?? throw new InvalidDataException($"Invalid payload path: {entry.FullName}"));
            entry.ExtractToFile(target, true);
        }
    }

    private static string ComputePayloadDigest(ZipArchive archive)
    {
        var records = new List<PayloadFileRecord>();
        var uniquePaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long totalBytes = 0;
        int entryCount = 0;

        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            entryCount++;
            if (entryCount > MaximumPayloadEntries)
            {
                throw new InvalidDataException("the payload contains too many entries");
            }

            string normalizedPath = ValidatePayloadEntry(entry, uniquePaths);
            totalBytes = checked(totalBytes + entry.Length);
            if (totalBytes > MaximumPayloadBytes)
            {
                throw new InvalidDataException("the payload exceeds the allowed uncompressed size");
            }

            if (IsDirectoryEntry(entry))
            {
                continue;
            }

            using Stream stream = entry.Open();
            string fileDigest = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
            records.Add(new PayloadFileRecord(normalizedPath, fileDigest));
        }

        records.Sort(static (left, right) => StringComparer.Ordinal.Compare(left.Path, right.Path));
        string manifest = string.Join('\n', records.Select(static record => $"{record.Sha256}  {record.Path}"));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(manifest))).ToLowerInvariant();
    }

    private static string ValidatePayloadEntry(ZipArchiveEntry entry, HashSet<string> uniquePaths)
    {
        string normalized = entry.FullName.Replace('\\', '/');
        if (string.IsNullOrWhiteSpace(normalized) || normalized.StartsWith('/'))
        {
            throw new InvalidDataException($"unsafe or empty payload path '{entry.FullName}'");
        }

        string canonical = normalized.TrimEnd('/');
        string[] segments = canonical.Split('/');
        if (segments.Length == 0 || segments.Any(static segment =>
                string.IsNullOrEmpty(segment) ||
                segment is "." or ".." ||
                segment.EndsWith(' ') ||
                segment.EndsWith('.') ||
                segment.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 ||
                IsReservedWindowsName(segment)))
        {
            throw new InvalidDataException($"unsafe payload path '{entry.FullName}'");
        }

        if (!uniquePaths.Add(canonical))
        {
            throw new InvalidDataException($"duplicate payload path '{entry.FullName}'");
        }

        return canonical;
    }

    private static bool IsReservedWindowsName(string segment)
    {
        string name = segment.Split('.')[0].ToUpperInvariant();
        if (name is "CON" or "PRN" or "AUX" or "NUL")
        {
            return true;
        }

        return name.Length == 4 &&
            (name.StartsWith("COM", StringComparison.OrdinalIgnoreCase) ||
             name.StartsWith("LPT", StringComparison.OrdinalIgnoreCase)) &&
            name[3] is >= '1' and <= '9';
    }

    private static bool IsDirectoryEntry(ZipArchiveEntry entry) =>
        entry.FullName.EndsWith('/') || string.IsNullOrEmpty(entry.Name);

    private static bool IsSha256(string value) =>
        value.Length == 64 && value.All(static character =>
            character is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');

    private static Version ReadVersion(string metadataPath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(metadataPath));
        string value = document.RootElement.GetProperty("version").GetString()
            ?? throw new InvalidDataException($"Version is missing from {metadataPath}.");
        return Version.Parse(value);
    }

    private static void CopyDirectory(string source, string destination)
    {
        foreach (string directory in Directory.EnumerateDirectories(source, "*", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(source, directory);
            Directory.CreateDirectory(Path.Combine(destination, relative));
        }

        foreach (string file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(source, file);
            string target = Path.Combine(destination, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(target)
                ?? throw new InvalidOperationException($"Invalid destination path: {target}"));
            File.Copy(file, target, true);
        }
    }

    private static void RegisterUninstaller(Version version)
    {
        using RegistryKey key = Registry.CurrentUser.CreateSubKey(UninstallRegistryPath, true)
            ?? throw new InvalidOperationException("The per-user uninstall registration could not be created.");
        string uninstallPath = Path.Combine(InstallRoot, "Uninstall.exe");
        key.SetValue("DisplayName", ProductDisplayName, RegistryValueKind.String);
        key.SetValue("DisplayVersion", version.ToString(3), RegistryValueKind.String);
        key.SetValue("Publisher", "Sungwoo Kang", RegistryValueKind.String);
        key.SetValue("InstallLocation", InstallRoot, RegistryValueKind.String);
        key.SetValue("DisplayIcon", Path.Combine(InstallRoot, "GraphReader.App.exe"), RegistryValueKind.String);
        key.SetValue("UninstallString", $"\"{uninstallPath}\" --uninstall", RegistryValueKind.String);
        key.SetValue("QuietUninstallString", $"\"{uninstallPath}\" --uninstall --quiet", RegistryValueKind.String);
        key.SetValue("NoModify", 1, RegistryValueKind.DWord);
        key.SetValue("NoRepair", 0, RegistryValueKind.DWord);
    }

    private static void CreateStartMenuShortcut(string targetPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(StartMenuShortcut)
            ?? throw new InvalidOperationException("The Start Menu path is invalid."));
        string script =
            "$shell = New-Object -ComObject WScript.Shell; " +
            $"$shortcut = $shell.CreateShortcut('{EscapePowerShellLiteral(StartMenuShortcut)}'); " +
            $"$shortcut.TargetPath = '{EscapePowerShellLiteral(targetPath)}'; " +
            $"$shortcut.WorkingDirectory = '{EscapePowerShellLiteral(InstallRoot)}'; " +
            "$shortcut.Save()";
        RunPowerShell(script, true);
    }

    private static void StartDeferredInstallDirectoryRemoval()
    {
        string expectedRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Programs",
            ProductDirectoryName);
        if (!string.Equals(Path.GetFullPath(InstallRoot), Path.GetFullPath(expectedRoot), StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Refusing to remove an unexpected installation directory.");
        }

        string script =
            $"Wait-Process -Id {Environment.ProcessId} -ErrorAction SilentlyContinue; " +
            $"Remove-Item -LiteralPath '{EscapePowerShellLiteral(InstallRoot)}' -Recurse -Force";
        RunPowerShell(script, false);
    }

    private static void RunPowerShell(string script, bool waitForExit)
    {
        string encoded = Convert.ToBase64String(Encoding.Unicode.GetBytes(script));
        var startInfo = new ProcessStartInfo
        {
            FileName = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.System),
                "WindowsPowerShell",
                "v1.0",
                "powershell.exe"),
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("-NoLogo");
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-NonInteractive");
        startInfo.ArgumentList.Add("-WindowStyle");
        startInfo.ArgumentList.Add("Hidden");
        startInfo.ArgumentList.Add("-EncodedCommand");
        startInfo.ArgumentList.Add(encoded);

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Windows PowerShell could not be started.");
        if (!waitForExit)
        {
            return;
        }

        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"Windows PowerShell failed with exit code {process.ExitCode}.");
        }
    }

    private static string EscapePowerShellLiteral(string value) => value.Replace("'", "''", StringComparison.Ordinal);

    private static void ValidateInstallRoot()
    {
        string localApplicationData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string expected = Path.Combine(localApplicationData, "Programs", ProductDirectoryName);
        if (string.IsNullOrWhiteSpace(localApplicationData) ||
            !string.Equals(Path.GetFullPath(InstallRoot), Path.GetFullPath(expected), StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("The per-user installation root is unavailable or unsafe.");
        }

        if (InstallRoot.StartsWith(DataRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Application binaries and mutable user data must remain isolated.");
        }
    }

    private static void ShowMessage(string message, bool quiet, bool error)
    {
        if (quiet)
        {
            return;
        }

        uint icon = error ? 0x10u : 0x40u;
        _ = MessageBox(IntPtr.Zero, message, ProductDisplayName + " Setup", icon);
    }

    [DllImport("user32.dll", CharSet = CharSet.Unicode, EntryPoint = "MessageBoxW")]
    private static extern int MessageBox(IntPtr window, string text, string caption, uint type);

    private sealed record PayloadFileRecord(string Path, string Sha256);
}
