// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;

namespace GraphReader.App.Localization;

public interface ILocalizationService
{
    CultureInfo CurrentCulture { get; }

    IReadOnlyList<CultureInfo> AvailableCultures { get; }

    event EventHandler<LocalizationChangedEventArgs>? CultureChanged;

    string GetString(string key);

    void ApplyCulture(CultureInfo culture);
}
