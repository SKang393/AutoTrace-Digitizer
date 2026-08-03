// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Windows;

namespace GraphReader.App.Localization;

public sealed class LocalizationService : ILocalizationService
{
    internal const string EnglishSource =
        "/GraphReader.App;component/Localization/Resources.en-US.xaml";

    private static readonly CultureInfo EnglishCulture = CultureInfo.GetCultureInfo("en-US");
    private static readonly IReadOnlyList<CultureInfo> SupportedCultures =
        Array.AsReadOnly([EnglishCulture]);

    private readonly ResourceDictionary _resources;
    private ResourceDictionary _activeDictionary = null!;

    public LocalizationService(ResourceDictionary resources)
    {
        ArgumentNullException.ThrowIfNull(resources);
        _resources = resources;
        ApplyCulture(ResolveSupportedCulture(CultureInfo.CurrentUICulture));
    }

    public CultureInfo CurrentCulture { get; private set; } = EnglishCulture;

    public IReadOnlyList<CultureInfo> AvailableCultures => SupportedCultures;

    public event EventHandler<LocalizationChangedEventArgs>? CultureChanged;

    public string GetString(string key)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);

        return _activeDictionary[key] as string
            ?? throw new KeyNotFoundException(
                $"Localization key '{key}' does not exist for {CurrentCulture.Name}.");
    }

    public void ApplyCulture(CultureInfo culture)
    {
        ArgumentNullException.ThrowIfNull(culture);

        CultureInfo supportedCulture = ResolveSupportedCulture(culture);
        ResourceDictionary replacement = new()
        {
            Source = new Uri(EnglishSource, UriKind.RelativeOrAbsolute),
        };

        int existingIndex = FindLocalizationDictionaryIndex();
        if (existingIndex >= 0)
        {
            _resources.MergedDictionaries[existingIndex] = replacement;
        }
        else
        {
            _resources.MergedDictionaries.Add(replacement);
        }

        bool changed = !Equals(CurrentCulture, supportedCulture);
        _activeDictionary = replacement;
        CurrentCulture = supportedCulture;
        CultureInfo.CurrentUICulture = supportedCulture;
        CultureInfo.DefaultThreadCurrentUICulture = supportedCulture;

        if (changed)
        {
            CultureChanged?.Invoke(this, new LocalizationChangedEventArgs(CurrentCulture));
        }
    }

    private static CultureInfo ResolveSupportedCulture(CultureInfo requestedCulture)
    {
        if (string.Equals(requestedCulture.Name, EnglishCulture.Name, StringComparison.OrdinalIgnoreCase)
            || string.Equals(
                requestedCulture.TwoLetterISOLanguageName,
                EnglishCulture.TwoLetterISOLanguageName,
                StringComparison.OrdinalIgnoreCase))
        {
            return EnglishCulture;
        }

        return EnglishCulture;
    }

    private int FindLocalizationDictionaryIndex()
    {
        for (int index = 0; index < _resources.MergedDictionaries.Count; index++)
        {
            string? source = _resources.MergedDictionaries[index].Source?.OriginalString;
            if (source is not null
                && source.Contains("Localization/Resources.", StringComparison.OrdinalIgnoreCase)
                && source.EndsWith(".xaml", StringComparison.OrdinalIgnoreCase))
            {
                return index;
            }
        }

        return -1;
    }
}
