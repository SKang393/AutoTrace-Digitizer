# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from pathlib import Path
import json
from .dataset import REPO_ROOT,write_freeze
if __name__=="__main__":print(json.dumps(write_freeze(REPO_ROOT/Path("ml/ocr/ambiguity_source_group_classifier_v3/artifacts/split-freeze")),indent=2,sort_keys=True))
