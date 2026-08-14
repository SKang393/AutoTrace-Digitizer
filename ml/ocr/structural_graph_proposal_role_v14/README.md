# OCR structural graph proposal and role V14

V14 is a new OCR-detection defect class opened only from aggregate output of the consumed V13 public gate. V13 produced 223 of 224 exact scenes, 1,792 true regions, one false prohibited structural region, zero misses, zero duplicates, and perfect role classification. It emitted no case-level detail. No V13 public image, truth, scene identity, or fixture byte is inspected or reused, and that public gate cannot rerun.

The P1 preregistration replaces V13's anisotropic morphology mixture with a topology-spectrum residual encoder. It combines tight and contextual ink, fixed horizontal and vertical edge magnitude planes, row and column occupancy spectra, and the unchanged 16 production geometry values. All train, validation, and truth-hidden public renderer and degradation families are new. The production proposal algorithm and `[N,2,32,144]` to `[N,10]` contract remain unchanged.

The candidate budget is three. Before any optimizer execution, the source renderer must prove exactly one production proposal per text truth across every stored split. Selection and the one-time public gate require every scene exact, zero false, missed, duplicate, or prohibited regions, overall role accuracy at least `0.90`, each role at least `0.85`, direct CPU ONNX execution, and maximum absolute parity error at most `1e-5`.

No V14 candidate is currently authorized to execute. No manifest, production model-store entry, package payload, private Chandler automatic run, production approval, or release eligibility exists.
