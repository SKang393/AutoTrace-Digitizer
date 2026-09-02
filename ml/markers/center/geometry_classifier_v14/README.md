# Marker-center geometry classifier V14

V14 evaluates a fixed deterministic classifier on V13's geometry-filtered
proposal stream. It uses compactness, isotropy, radial ink support, local
connected-line evidence, and text/artifact mask clearance. No learned weights
are trained or selected in this revision.

The synthetic train/dev splits cover 6 to 25 pixel marker diameters. No
private, public, or sealed data is read. The fixed gate is 0.95 precision and
0.95 recall at 5 pixels. The aggregate diagnostic is recorded in
`DEV_DIAGNOSTIC.json`. All five thresholds fail dev: the best dev precision
is `0.74` and the best dev recall is `0.7604166666666666`. Positive scores
begin at `0.3893` while negative scores reach `0.9466`, so this revision stops
without training or public-gate authorization.
