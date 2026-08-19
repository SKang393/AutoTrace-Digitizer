# Versioning and GitHub release policy

Graph Auto Reader uses a custom numeric `x.y.z` sequence. Each component ranges
from 0 through 99. This is not Semantic Versioning.

## Source checkpoints

A completed, verified source checkpoint advances the central version in
`Directory.Build.props` exactly once. Prepare that value before committing:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File packaging/Prepare-CheckpointVersion.ps1
```

The ordinary checkpoint sequence is continuous across component boundaries:

```text
0.0.98 -> 0.0.99 -> 0.1.0 -> 0.1.1
0.99.99 -> 1.0.0 -> 1.0.1
```

Therefore, rollover resets the lower component to `0`, not `1`.

## First stable promotion

The first functional public release is exactly `1.0.0`. Once every mandatory
1.0 gate passes and the maintainer explicitly authorizes the promotion, prepare
it with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File packaging/Prepare-CheckpointVersion.ps1 -PromoteStable
```

This command may move the central version directly from any `0.y.z` checkpoint,
for example `0.23.58`, to `1.0.0`. It is the only permitted nonsequential
checkpoint transition. It does not create intermediate version numbers, rewrite
Git history, tag a release, or publish artifacts. The readiness gates decide
when the maintainer may authorize the promotion; they do not alter normal
pre-1.0 checkpoint increments.

Without `-PromoteStable`, `0.23.58` advances normally to `0.23.59`. After the
stable checkpoint, normal progression resumes with `1.0.1`.

All completed checkpoint commits belong in `main` and `origin/main`. Git commit
history is the track record for internal checkpoints. A GitHub Release is not
created for every commit.

## Development portable previews

`packaging/Build-DevPortable.ps1` and `packaging/Watch-DevPortable.ps1` reuse the
current central version. Rebuilding unchanged or uncommitted source does not
advance `x.y.z`. Preview folders are distinguished by UTC timestamp and commit
identity and remain ignored local output.

Portable previews, training snapshots, generated weights, caches, `bin`, and
`obj` are not committed and are not GitHub Releases. Preserve only evidence that
an active validation or readiness report still references. Use the explicit,
report-first cleanup process in `docs/DEV-PORTABLE-SIZE.md` for obsolete
previews.

## Every-twentieth-checkpoint release cadence

The first release-eligible checkpoint is `0.0.1`, followed by every twentieth
checkpoint. With components limited to 0 through 99, release-eligible `z` values
are always:

```text
1, 21, 41, 61, 81
```

Equivalently, with `ordinal = x * 10000 + y * 100 + z`, an ordinary version is
eligible when `ordinal % 20 == 1`. The one-time `1.0.0` stable promotion is also
release eligible even though it is outside that cadence. This exception does
not move or renumber any scheduled checkpoint.

Examples:

```text
0.0.81  eligible
0.0.99  internal
0.1.0   internal
0.1.1   eligible
1.0.0   eligible stable promotion
```

Release eligibility is only the cadence gate. It does not override failed
tests, dependency or model provenance, clean-machine validation, a dirty tree,
or missing release artifacts.

## GitHub publication

An eligible checkpoint becomes a public GitHub Release only when all of these
conditions are true:

1. The checkpoint is committed and pushed to `origin/main`.
2. Every required build, test, license, provenance, and clean-machine gate
   passes from a clean checkout.
3. The maintainer has authorized that release checkpoint.
4. An annotated tag named exactly `v<version>` points to the checkpoint commit.
5. The installer and portable ZIP are built from one common publish output and
   pass the release artifact validator.

The release operator then pushes the annotated tag:

```powershell
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z
```

The tag workflow validates the tag, rebuilds and revalidates the artifact pair,
and publishes the installer, portable ZIP, checksums, SBOM, release metadata,
release notes, and known limitations. A normal push to `main` never creates a
tag or public release.

## 2026-08-19 versioning regression and correction

The central version remained frozen at `0.0.21` while local portable previews
and later source commits were produced. Preview rebuilds were correct to reuse
the current central version, but completed source checkpoints also failed to
advance it. That was a versioning regression, not evidence that every preview
was the same source checkpoint or that dozens of releases existed.

The correction resumed at `0.0.22` and is enforced by the shared checkpoint
tool and CI transition check. Preview count and historical commit count are not
used to invent skipped versions, and no historical commit or artifact is
renumbered or released retroactively. No `0.0.21` release may be created unless
its original clean commit and every release gate are independently proven.
This verified policy correction advances normally to checkpoint `0.0.23`.

The earlier documentation that named `1.0.1` as the first functional release
and prohibited `1.0.0` was incorrect. The corrected target is `1.0.0`, using
the explicit promotion path above. Accuracy and release gates still control
authorization, but they no longer conflict with the numeric version plan.
