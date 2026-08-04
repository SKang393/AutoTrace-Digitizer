<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Portable clean-profile validation

This local harness exercises the implementable portable startup and path gates
without changing the public release audit. It fails when any observed gate is
missing, when a write-trace watcher overflows, or when Graph Auto Reader-owned
persistence escapes the configured portable data or user-selected roots.

Run the bounded self-tests:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File packaging\portable-validation\Test-PortableCleanProfile.Tests.ps1
```

Validate the latest development portable:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File packaging\portable-validation\Test-PortableCleanProfile.ps1
```

Use `-ExecutablePath` to validate a different extracted portable executable.
Use `-KeepSandbox` only when the copied payload is needed for diagnosis.
Generated reports remain under `artifacts\portable-validation\` and are ignored
by Git.

The run copies the portable into a path containing spaces and Korean Unicode,
then checks:

- a live WPF process uses the shared development `Data` root;
- normal portable smoke uses `\.\Data` and not Local AppData;
- a deny-write ACL produces the exact visible read-only corrective message;
- the application-specific HKCU configuration key is absent before and after
  successful startup;
- no process-owned TCP or UDP endpoint is observed during the configured live
  sampling window;
- transient file-system mutations are classified by destination purpose,
  application ownership, live process identity, and loaded component evidence.
- the exact real LocalAppData and RoamingAppData `GraphAutoReader` roots are
  hashed and metadata-snapshotted before and after every scenario, including
  when either root is initially absent.

The classification is not a blanket path allowlist:

- Graph Auto Reader settings, cache, logs, autosave, recovery, or project data
  outside configured `Data` and explicit user-selected writable roots fail;
- attributed Windows, WPF, .NET, font, Direct3D, or GPU-driver cache events are
  retained as warnings with the responsible process, component, path, purpose,
  and evidence;
- external-looking cache events without matching component evidence fail
  closed;
- self-tests create a negative Graph Auto Reader settings write under an
  initially absent `%LOCALAPPDATA%\GraphAutoReader` root and an initially absent
  `%APPDATA%\GraphAutoReader` root and prove both fail.

Evidence boundaries are deliberate. This is a local isolated-profile
simulation, not a clean Windows profile or VM. Network adapters remain enabled,
endpoint polling can miss very short connections, and FileSystemWatcher does
not itself emit a process ID. External-cache attribution therefore combines the
active GraphReader.App process identity, loaded WPF/GPU modules, destination
purpose, and cache-path evidence. Unknown mutations fail closed. Registry
observation is limited to the application configuration key. The JSON report
records these limitations and always sets `cleanVmEvidence` to `false`.
