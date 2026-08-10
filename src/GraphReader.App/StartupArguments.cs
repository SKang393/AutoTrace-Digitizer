// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;

namespace GraphReader.App;

public sealed record StartupArguments(
    bool PortableSmoke,
    bool ProductionRuntimeSmoke,
    bool RequirePackagedPdfium,
    string? OpenImagePath)
{
    public static StartupArguments Parse(IReadOnlyList<string> arguments)
    {
        ArgumentNullException.ThrowIfNull(arguments);

        bool portableSmoke = false;
        bool productionRuntimeSmoke = false;
        bool requirePackagedPdfium = false;
        string? openImagePath = null;
        for (int index = 0; index < arguments.Count; index++)
        {
            string argument = arguments[index];
            if (string.Equals(argument, "--portable-smoke", StringComparison.OrdinalIgnoreCase))
            {
                portableSmoke = true;
                continue;
            }

            if (string.Equals(argument, "--production-runtime-smoke", StringComparison.OrdinalIgnoreCase))
            {
                productionRuntimeSmoke = true;
                continue;
            }

            if (string.Equals(argument, "--require-packaged-pdfium", StringComparison.OrdinalIgnoreCase))
            {
                requirePackagedPdfium = true;
                continue;
            }

            if (!string.Equals(argument, "--open-image", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (++index >= arguments.Count || string.IsNullOrWhiteSpace(arguments[index]))
            {
                throw new ArgumentException("--open-image requires an image path.", nameof(arguments));
            }

            if (openImagePath is not null)
            {
                throw new ArgumentException("--open-image may be specified only once.", nameof(arguments));
            }

            string requestedPath = arguments[index];
            if (!Path.IsPathFullyQualified(requestedPath))
            {
                throw new ArgumentException("--open-image requires an absolute image path.", nameof(arguments));
            }

            openImagePath = Path.GetFullPath(requestedPath);
        }

        return new StartupArguments(
            portableSmoke,
            productionRuntimeSmoke,
            requirePackagedPdfium,
            openImagePath);
    }
}
