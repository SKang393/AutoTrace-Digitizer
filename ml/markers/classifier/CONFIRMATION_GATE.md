# Marker-classifier confirmation gate

The historical v1 result used the same renderer/degradation families and
templates as the public split with different repeats. It is a failed
same-family repeat, not generalization evidence, and is retired from replay.

The v2 split uses `microfilm_dither` and `laser_streak` families
with `upper_left_oblong` and `lower_right_compact` templates. These families
and templates are disjoint from selection and public data. The same shape,
fill, artifact, minority, and packed-parity gates apply.

After the gate sources were committed, the split was evaluated exactly once on
2026-08-04. Shape macro-F1 was `1.0`, fill macro-F1 was
`0.9624926134837496`, artifact F1 was `1.0`, and every minority-shape F1 was
`1.0`. Packed ONNX maximum absolute error was
`1.1444091796875e-05`, above the `1e-05` limit, so the result is `fail` and
the candidate remains rejected. The ignored report SHA-256 is
`2bb119c5ece6167177e225486c01441e84363c60c5afc4d3e6872aaae99d46b4`.

The separately preregistered v3 split uses the probability-packed runtime and
new procedural families frozen before runtime-repair P1 selection. It was
evaluated exactly once on 2026-08-05. Shape macro-F1 was
`0.9813519813519813`, fill macro-F1 was `0.9440559440559441`, artifact F1 and
every minority-shape F1 were `1.0`, and maximum CPU ONNX error was
`1.0728836059570312e-06` against the `1e-05` limit. The 140-case result passed.
The canonical seal key is
`d0731e45e11f4238de8fc3adce2534c2cd7b922f7f6c944998aa7f5bd62bc48b`,
and the ignored report SHA-256 is
`991e3ec0c41508833791b0b57d1633f9cf6230114dcaa22a4e2eb84feb6c86d1`.
No replay is authorized, and this scientific pass does not approve production
model discovery or packaging.
