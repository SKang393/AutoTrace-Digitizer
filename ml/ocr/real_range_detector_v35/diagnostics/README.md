# V35 synthetic-dev diagnosis

`diagnose_v35.py` loads the ignored V35 P1 ONNX checkpoint and the committed
five-scene synthetic `dev` split. It reports only aggregate pixel, proposal,
threshold, connected-component, and tile-overlap statistics. It never opens
real, private, public, or sealed data.

The saved `DIAGNOSTIC.json` records the fixed-pipeline result and threshold,
morphology, and component-area sweeps. The best proposal recall remains below
the 0.95 bar while source coverage is complete, so the isolated responsible
stage is pixel segmentation rather than tiling/overlap mapping or threshold
selection. This evidence supports redesigning V35 supervision/model output
before opening V36.

Observed aggregate values:

- fixed V35 pipeline: 40 true positives, 155 false positives, 46 false
  negatives, precision `0.205128`, recall `0.465116`;
- best tested postprocessing variant: threshold `0.80`, no close operation,
  59 true positives, precision `0.305699`, recall `0.686047`;
- tiling: 204 tiles, covered pixel fraction `1.0`, mean overlap prediction
  standard deviation `0.004185`.
