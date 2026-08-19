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

The sequence is continuous across component boundaries:

```text
0.0.98 -> 0.0.99 -> 0.1.0 -> 0.1.1
0.99.99 -> 1.0.0 -> 1.0.1
```

Therefore, rollover resets the lower component to `0`, not `1`.

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

Equivalently, with `ordinal = x * 10000 + y * 100 + z`, a version is eligible
when `ordinal % 20 == 1`.

Examples:

```text
0.0.81  eligible
0.0.99  internal
0.1.0   internal
0.1.1   eligible
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

## Correction from 0.0.21

`0.0.21` remained frozen while many local portable previews and later commits
were produced. Those outputs cannot be assigned reliable versions
retroactively. The correction resumes at `0.0.22`; preview count and historical
commit count are not used to invent skipped version numbers. No `0.0.21`
release may be created unless its original clean commit and every release gate
are independently proven.
