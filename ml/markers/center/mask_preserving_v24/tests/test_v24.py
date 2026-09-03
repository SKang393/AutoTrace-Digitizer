# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import math
import json
from pathlib import Path

from ml.markers.center.mask_preserving_v24.mask_preserving import extract_proposals
from ml.markers.center.mask_preserving_v24 import protocol
from ml.markers.center.real_range_generator_v1.generator import build_split

def test_masks_do_not_remove_ink_supported_proposals():
    scene = build_split("dev")[0]
    proposals = extract_proposals(scene.tensor)
    assert proposals.patches.shape[1:] == (3, 33, 33)
    assert proposals.patches.shape[0] > 0
    assert float(proposals.patches[:, 1].sum()) > 0
    assert float(proposals.patches[:, 2].sum()) > 0

def test_mask_crossing_truths_retain_nearby_proposals():
    scene = next(
        item for item in build_split("dev")
        if any(
            float(item.tensor[1:, int(y)-2:int(y)+3, int(x)-2:int(x)+3].max()) >= .35
            for x, y in item.centers
        )
    )
    proposals = extract_proposals(scene.tensor)
    coordinates = proposals.coordinates.tolist()
    masked = [
        (x, y) for x, y in scene.centers
        if float(scene.tensor[1:, int(y)-2:int(y)+3, int(x)-2:int(x)+3].max()) >= .35
    ]
    assert masked
    assert all(any(math.hypot(px-x, py-y) <= 5 for px, py in coordinates) for x, y in masked)

def test_feasibility_result_is_non_consuming_and_startable():
    report = json.loads(Path("ml/markers/center/mask_preserving_v24/FEASIBILITY.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed_frozen_model_training_candidate_startable"
    assert report["scope"]["candidate_consumed"] is False
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["proposal_coverage"]["recall"] == 1.0
    assert report["metrics"]["precision"] < 0.95
    assert report["metrics"]["recall"] < 0.95
    assert report["binding"]["v21_onnx_sha256"] == protocol.V21_ONNX_SHA256
    assert not Path(report["binding"]["v21_onnx_path"]).is_absolute()
