<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Windows portable definition

`portable.json` records the enforceable Goal 00 portable behavior. The build
script creates an empty `portable.mode` beside `GraphReader.App.exe` in
portable staging, which selects `Data` beneath the extracted application
directory for mutable state. Runtime sentinels are generated and remain
ignored by Git.

Goal 00 stages the portable layout but does not emit the final ZIP. A later
release session must verify read-only-folder handling, registry independence,
path isolation, and launch from paths containing spaces and Unicode characters.
