# Marker-classifier confirmation gate

The historical v1 result used the same renderer/degradation families and
templates as the public split with different repeats. It is a failed
same-family repeat, not generalization evidence, and is retired from replay.

The unevaluated v2 split uses `microfilm_dither` and `laser_streak` families
with `upper_left_oblong` and `lower_right_compact` templates. These families
and templates are disjoint from selection and public data. The same shape,
fill, artifact, minority, and packed-parity gates apply. No v2 inference has
been run. Its generated manifest is hash-pinned, but evaluation remains blocked
until the source and split configuration are committed.
