# Marker-center proposal geometry V13 support

V13 supplies the fixed synthetic train/dev scenes and bounded geometry filter
used by the V14 diagnostic. It rejects thin, strongly anisotropic connected
components after text and artifact masking while retaining compact marker-like
ink. The procedural scenes span marker diameters from 6 to 25 pixels.

This support module is not a production candidate or acceptance result. Its
tests establish disjoint synthetic families, private-data exclusion, and full
dev-truth proposal coverage. No private, public, or sealed data is read.
