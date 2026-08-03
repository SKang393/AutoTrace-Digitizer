# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Asset and font provenance tests."""

from __future__ import annotations

import csv
import hashlib
from importlib.metadata import metadata, version
from pathlib import Path
from urllib.parse import urlparse

from ml.synthetic.fonts import FontResolver


def test_package_contains_no_font_or_private_image_assets() -> None:
    package_root = Path(__file__).resolve().parents[1]
    forbidden_suffixes = {
        ".ttf",
        ".otf",
        ".ttc",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".gif",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".doc",
        ".docx",
    }
    assert not [
        path
        for path in package_root.rglob("*")
        if (
            path.is_file()
            and "datasets" not in path.relative_to(package_root).parts
            and path.suffix.casefold() in forbidden_suffixes
        )
    ]


def test_font_resolver_uses_host_font_and_records_checksum() -> None:
    resolved = FontResolver().resolve("sans", 14)
    assert resolved.path.is_file()
    assert resolved.source == "system"
    assert len(resolved.sha256) == 64
    assert resolved.provenance()["bundled"] is False


def test_dependency_audit_matches_pins_licenses_and_artifact_hashes() -> None:
    package_root = Path(__file__).resolve().parents[1]
    expected = {
        "Pillow": ("12.3.0", "MIT-CMU"),
        "jsonschema": ("4.26.0", "MIT"),
        "pytest": ("9.1.1", "MIT"),
    }
    with (package_root / "DEPENDENCY_PROVENANCE.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["dependency"] for row in rows} == set(expected)

    requirements = (
        (package_root / "requirements.txt").read_text(encoding="utf-8")
        + (package_root / "requirements-dev.txt").read_text(encoding="utf-8")
    )
    for row in rows:
        dependency = row["dependency"]
        expected_version, expected_license = expected[dependency]
        assert row["version"] == expected_version
        assert row["license"] == expected_license
        assert row["bundled_or_downloaded"] == "downloaded"
        assert row["review_status"] == "approved-permissive"
        assert row["checksum_method"] == "SHA-256 of exact downloaded wheel bytes"
        assert row["checksum"].startswith("sha256:")
        assert len(row["checksum"]) == 71
        assert row["artifact_filename"].endswith(".whl")
        parsed_url = urlparse(row["artifact_url"])
        assert parsed_url.scheme == "https"
        assert parsed_url.hostname == "files.pythonhosted.org"
        assert parsed_url.path.endswith(row["artifact_filename"])
        assert (
            f"{dependency}=={expected_version} --hash={row['checksum']}"
            in requirements
        )

        notice_path = package_root / row["notice_path"]
        assert notice_path.is_file()
        notice_sha256 = hashlib.sha256(notice_path.read_bytes()).hexdigest()
        assert row["license_checksum"] == f"sha256:{notice_sha256}"
        assert version(dependency) == expected_version
        installed_license = (
            metadata(dependency).get("License-Expression")
            or metadata(dependency).get("License")
        )
        assert installed_license == expected_license
