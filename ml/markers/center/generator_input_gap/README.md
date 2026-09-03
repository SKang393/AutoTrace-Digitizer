# Synthetic marker input-distribution audit

This bounded audit reads only the deterministic V20 synthetic training scenes
and frozen V13 synthetic `dev` scenes. It does not load a model, train, read
the private corpus, create a candidate revision, or emit scene identities,
truth rows, or pixels.

Run from the repository root:

```powershell
python -m ml.markers.center.generator_input_gap.audit
python -m pytest ml/markers/center/generator_input_gap/tests -q
```

The output is aggregate-only. It reports scene and marker counts, tensor
shapes, radius and diameter quantiles, diameter-to-33-pixel proposal-patch
ratios, and whether the resize degradation leaves proposal inputs fixed at
`3x168x224`. The roundtrip flag is evidence about the generator implementation,
not production acceptance evidence.
