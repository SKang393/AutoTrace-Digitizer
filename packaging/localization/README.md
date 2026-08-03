<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Localization audit

`Audit-Localization.ps1` performs an offline, deterministic audit of the WPF
localization contract and resource dictionaries. It reports:

- contract keys declared in `LocalizationKeys.cs`;
- localization keys referenced by XAML or C#;
- contract keys that are present but not referenced by XAML or C#;
- missing, extra, and duplicate keys by culture;
- unresolved WPF static or dynamic resource references.

Run the repository audit:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging/localization/Audit-Localization.ps1 -FailOnExtraKeys -ReportPath artifacts/localization/localization-report.json
```

The command returns `0` for a clean audit, `1` for missing, duplicate, or
unresolved keys, `2` for extra keys when `-FailOnExtraKeys` is supplied, `3`
when the audit cannot run, and `4` for unused contract keys when
`-FailOnUnusedKeys` is supplied. Extra and unused keys are always reported.
They are informational by default because a complete localization contract may
intentionally reserve strings for a later UI surface. Release automation should
use `-FailOnExtraKeys`; maintainers can additionally use `-FailOnUnusedKeys` for
dead-key cleanup gates.

Run the fixture-based self-tests:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging/localization/Test-LocalizationAudit.ps1
```
