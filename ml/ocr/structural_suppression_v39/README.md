# OCR V39 structural suppression design

V39 is blocked before candidate creation. V38 attribution identifies
marker/connecting-line ink and OCR text-box margins as major false-positive
sources, but those inputs are not available before OCR inference in the
current production chain. Axis, tick, and phase-divider geometry is available
from the pre-OCR axis stage and is already applied by the checksum-bound
detector derivative.

The binding design decision and evidence hashes are in
`BLOCKED_DESIGN_REPORT.json`. No train, dev, real, private, public, or sealed
evaluation was run for V39.
