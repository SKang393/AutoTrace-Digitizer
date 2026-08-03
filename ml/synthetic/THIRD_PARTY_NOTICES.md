<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Synthetic generator third-party notices

The synthetic generator installs the following permissively licensed Python
distributions from the exact version and artifact-hash pins in
`requirements.txt` and `requirements-dev.txt`. These dependencies are
downloaded at environment setup time. Wheel audit copies are kept only in the
ignored local provenance cache and are not Git or release content.

| Dependency | Version | License | Authoritative source | Preserved license notice |
| --- | ---: | --- | --- | --- |
| Pillow | 12.3.0 | MIT-CMU | <https://github.com/python-pillow/Pillow> | [`LICENSES/Pillow-12.3.0-LICENSE.txt`](LICENSES/Pillow-12.3.0-LICENSE.txt) |
| jsonschema | 4.26.0 | MIT | <https://github.com/python-jsonschema/jsonschema> | [`LICENSES/jsonschema-4.26.0-COPYING.txt`](LICENSES/jsonschema-4.26.0-COPYING.txt) |
| pytest | 9.1.1 | MIT | <https://github.com/pytest-dev/pytest> | [`LICENSES/pytest-9.1.1-LICENSE.txt`](LICENSES/pytest-9.1.1-LICENSE.txt) |

The source URLs and SPDX license expressions above come from each installed
distribution's Core Metadata `Project-URL: Source` and `License-Expression`
fields. The preserved license files are byte-for-byte copies of the files under
the installed distribution's `.dist-info/licenses/` directory. Pillow's
license artifact contains its complete upstream notices for incorporated
permissive components and is intentionally preserved in full.

## Audited package artifacts

| Dependency | Exact artifact | Direct PyPI artifact URL | Artifact SHA-256 |
| --- | --- | --- | --- |
| Pillow 12.3.0 | `pillow-12.3.0-cp313-cp313-win_amd64.whl` | <https://files.pythonhosted.org/packages/a6/9b/7a58e61d62be561da3a356fe2384d4059a6345fc130e23ef1c36a5b81d24/pillow-12.3.0-cp313-cp313-win_amd64.whl> | `1cca606cd25738df4ed873d5ad46bbdb3d83b5cbca291f6b4ff13a4df6b0bbe8` |
| jsonschema 4.26.0 | `jsonschema-4.26.0-py3-none-any.whl` | <https://files.pythonhosted.org/packages/69/90/f63fb5873511e014207a475e2bb4e8b2e570d655b00ac19a9a0ca0a385ee/jsonschema-4.26.0-py3-none-any.whl> | `d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce` |
| pytest 9.1.1 | `pytest-9.1.1-py3-none-any.whl` | <https://files.pythonhosted.org/packages/24/25/1de2678b631f5a49215c6c96fff41ba892b0a34df68d6d80292b1b48aa7f/pytest-9.1.1-py3-none-any.whl> | `37a86b45efb9a47a61a36449063e8e18d0cab3161329fc099eb21783169c4f0c` |

The artifacts were selected with `pip download --only-binary=:all: --no-deps`
on Windows CPython 3.13 x64. Pillow is the `cp313-cp313-win_amd64` wheel;
jsonschema and pytest are platform-independent `py3-none-any` wheels. Each
artifact checksum is the lowercase SHA-256 digest of the complete downloaded
wheel bytes. The digest was independently matched to the `sha256` value for the
same filename in the PyPI JSON release metadata. The direct artifact URLs above
also come from those exact PyPI release-file records. Downloaded audit artifacts
remain in ignored `ml/synthetic/cache/provenance/` and are not release or Git
content.

## Installed license-file checksums

The copied license artifacts were also checked byte-for-byte against local
installed metadata after copying. Their SHA-256 values are:

| License artifact | SHA-256 |
| --- | --- |
| Pillow 12.3.0 `LICENSE` | `4f7866a74802c6326f81faff59a56546b6aec2b10b91973e0e9308de95e79857` |
| jsonschema 4.26.0 `COPYING` | `4f92a015a13c4d1a040bef018aa13430b4f1bc73b41b16bb846c346766de7439` |
| pytest 9.1.1 `LICENSE` | `ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b` |
