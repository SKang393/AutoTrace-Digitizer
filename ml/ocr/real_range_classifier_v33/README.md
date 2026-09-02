# Corrected real-range classifier V33

V33 is a project-owned OCR proposal-classifier architecture. The approved
pretrained detector attempt failed the corrected real-range synthetic
evaluation, and the compatible V32 fine-tune failed two dev attempts. Under
the required model sourcing order, those recorded failures permit a new
project-owned architecture. This revision therefore trains the richer
multiscale detail, context, and geometry fusion classifier from scratch.

The V33 data path retains the committed V32 five-axis family-disjoint
corrected real-range proposal dataset and proposal generator. Proposal
generation, the `0.82` operating threshold, and maximum-cardinality IoU `0.50`
matching remain unchanged. Class-aware training uses the preregistered fixed
weights `[1.0, 2.0]`.

The authorized P1 dev loop completed two unconsumed attempts. Class weights
`[1,2]` reached precision `0.972972972972973` and recall
`0.4186046511627907`; `[1,8]` reached precision `0.8947368421052632` and
recall `0.5930232558139535`. Dynamic CPU ONNX parity passed at proposal counts
1, 7, 64, and 257, but both attempts remained below Tier 1. V33 is retired
without a public, sealed, or private-data read. The aggregate outcome is
`P1_RESULT.json`.

Verification:

```powershell
python -m pytest ml/ocr/real_range_classifier_v33/tests -q
```
