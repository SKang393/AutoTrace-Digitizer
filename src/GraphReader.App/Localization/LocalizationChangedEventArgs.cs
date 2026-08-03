// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;

namespace GraphReader.App.Localization;

public sealed class LocalizationChangedEventArgs(CultureInfo culture) : EventArgs
{
    public CultureInfo Culture { get; } = culture;
}
