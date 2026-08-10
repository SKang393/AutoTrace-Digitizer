<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Clean Windows validation

`Invoke-GraphReaderCleanMachineValidation.ps1` runs inside a newly installed,
network-disabled Windows x64 VM. It binds one clean development-portable payload
to exact application and source-built OpenCV checksums, loads the native DLL with
Windows loader search restrictions, runs the real WPF `--portable-smoke` path,
and writes `opencv-clean-machine.json` even when validation fails.

The report passes only when all of these conditions are direct observations:

- the VM provenance binds an official checksum-verified Windows evaluation ISO;
- the Windows installation is less than 24 hours old;
- no developer toolchain command is present on `PATH`;
- no network adapter is up;
- the payload is a clean exact-version build with `portable.mode`;
- the exact reviewed `OpenCvSharpExtern.dll` loads successfully;
- the self-contained application portable smoke exits zero;
- the manual portable contains no `.onnx`, `.param`, or `.bin` model payload.

This script does not edit `packaging/common/release-audit.json` and cannot approve
a public release by itself. The report must be reviewed, copied to the tracked
release-evidence location, checksum-bound in the mandatory gate, and revalidated
by `Install-ReleaseRuntime.ps1` before production packaging may consume it.
