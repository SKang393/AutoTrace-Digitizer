# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Verify pinned official PaddleOCR archives and fail closed on legal ambiguity."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any
import unicodedata

PINNED_TAG = "v3.5.0"
PINNED_COMMIT = "33cbdd9deb2e00f61e7966db70669b249c005a37"
PINNED_TREE = "4fd5a734b7a96ce2808c261e89accb90e7299e37"
OFFICIAL_API_ROOT = "https://api.github.com/repos/PaddlePaddle/PaddleOCR"
OFFICIAL_HTML_ROOT = "https://github.com/PaddlePaddle/PaddleOCR"
ALLOWLISTED_LICENSES = frozenset({"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"})
TERMS_FILENAME = "artifact-terms.json"
NOTICE_NAME_PATTERN = re.compile(r"^NOTICE(?:\.(?:txt|md))?$", re.IGNORECASE)
NOTICE_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
TERMS_KEYS = frozenset(
    {
        "model_id",
        "scope",
        "license_spdx",
        "redistribution",
        "commercial_use",
        "notice_path",
        "notice_sha256",
    }
)
WINDOWS_INVALID_SEGMENT_CHARACTERS = frozenset('<>:"|?*')
WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


@dataclass(frozen=True)
class Candidate:
    model_id: str
    archive_name: str
    url: str
    archive_sha256: str
    member_sha256: dict[str, str]


@dataclass(frozen=True)
class OfficialDocument:
    local_name: str
    repository_path: str
    git_blob_sha1: str
    required_url: str


CANDIDATES = (
    Candidate(
        model_id="PP-OCRv5_mobile_det",
        archive_name="PP-OCRv5_mobile_det_infer.tar",
        url=(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
            "official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
        ),
        archive_sha256="50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58",
        member_sha256={
            "PP-OCRv5_mobile_det_infer/inference.json":
                "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
            "PP-OCRv5_mobile_det_infer/inference.pdiparams":
                "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
            "PP-OCRv5_mobile_det_infer/inference.yml":
                "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699",
        },
    ),
    Candidate(
        model_id="en_PP-OCRv5_mobile_rec",
        archive_name="en_PP-OCRv5_mobile_rec_infer.tar",
        url=(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
            "official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar"
        ),
        archive_sha256="e595b4cf2ffad19fbb5a61ba345d63939577a3ab8717b6e5995642590c9101b4",
        member_sha256={
            "en_PP-OCRv5_mobile_rec_infer/inference.json":
                "fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753",
            "en_PP-OCRv5_mobile_rec_infer/inference.pdiparams":
                "3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
            "en_PP-OCRv5_mobile_rec_infer/inference.yml":
                "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067",
        },
    ),
)

OFFICIAL_DOCUMENTS = (
    OfficialDocument(
        local_name="text_detection.en.md",
        repository_path="docs/version3.x/module_usage/text_detection.en.md",
        git_blob_sha1="64546c4a20fddb08a2ec6225cc245c7b180ed97d",
        required_url=CANDIDATES[0].url,
    ),
    OfficialDocument(
        local_name="text_recognition.en.md",
        repository_path="docs/version3.x/module_usage/text_recognition.en.md",
        git_blob_sha1="a52c71a09116d4da09b6a2b4eaff500e1d9849d4",
        required_url=CANDIDATES[1].url,
    ),
)


def _hash_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_hash(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return sha1(header + payload).hexdigest()


class DuplicateJsonKeyError(ValueError):
    """Raised when a reviewed JSON object repeats a key at any depth."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_reviewed_json(payload: bytes) -> Any:
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _review_windows_archive_member_path(name: str) -> tuple[str, str]:
    if not name or name.startswith("/"):
        raise ValueError(f"Archive member path is empty or absolute: {name}")
    if "\\" in name:
        raise ValueError(f"Archive member path contains a backslash: {name}")
    if "//" in name:
        raise ValueError(f"Archive member path contains repeated separators: {name}")

    segments = name.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"Archive member path is not canonical POSIX spelling: {name}")
    canonical_path = PurePosixPath(*segments).as_posix()
    if canonical_path != name:
        raise ValueError(f"Archive member path is not canonical POSIX spelling: {name}")

    for segment in segments:
        if segment.endswith((".", " ")):
            raise ValueError(
                f"Archive member path has a Windows-unsafe trailing dot or space: {name}"
            )
        if any(
            character in WINDOWS_INVALID_SEGMENT_CHARACTERS or ord(character) < 32
            for character in segment
        ):
            raise ValueError(
                f"Archive member path has Windows-unsafe segment characters: {name}"
            )
        device_stem = unicodedata.normalize("NFKC", segment.split(".", 1)[0]).casefold()
        if device_stem in WINDOWS_RESERVED_DEVICE_NAMES:
            raise ValueError(
                f"Archive member path uses a Windows reserved device name: {name}"
            )

    return canonical_path, canonical_path.casefold()


def _review_structured_terms(
    candidate: Candidate,
    payloads: dict[str, bytes],
) -> dict[str, object]:
    term_paths = [path for path in payloads if PurePosixPath(path).name == TERMS_FILENAME]
    blockers: list[str] = []
    if len(term_paths) != 1:
        blockers.append(
            f"Expected exactly one {TERMS_FILENAME}; found {len(term_paths)}."
        )
        return {"valid": False, "terms_path": None, "terms": None, "blockers": blockers}

    terms_path = term_paths[0]
    try:
        terms: Any = _load_reviewed_json(payloads[terms_path])
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
        blockers.append(f"Structured artifact terms are unreadable: {error}")
        return {"valid": False, "terms_path": terms_path, "terms": None, "blockers": blockers}
    if not isinstance(terms, dict) or set(terms) != TERMS_KEYS:
        blockers.append("Structured artifact terms do not match the exact reviewed field set.")
        return {"valid": False, "terms_path": terms_path, "terms": terms, "blockers": blockers}

    if terms["model_id"] != candidate.model_id:
        blockers.append("Artifact terms do not identify the exact model ID.")
    if terms["scope"] != "all-files-in-archive":
        blockers.append("Artifact terms do not affirmatively cover all files in the archive.")
    if terms["license_spdx"] not in ALLOWLISTED_LICENSES:
        blockers.append("Artifact license is absent, ambiguous, proprietary, or not allowlisted.")
    if terms["redistribution"] is not True:
        blockers.append("Artifact redistribution is not explicitly boolean true.")
    if terms["commercial_use"] is not True:
        blockers.append("Artifact commercial use is not explicitly boolean true.")
    notice_path = terms["notice_path"]
    if not isinstance(notice_path, str) or notice_path not in payloads:
        blockers.append("Artifact notice path is missing or does not identify an archive member.")
    elif not NOTICE_NAME_PATTERN.fullmatch(PurePosixPath(notice_path).name):
        blockers.append("Artifact notice path does not identify reviewed NOTICE material.")
    else:
        notice_payload = payloads[notice_path]
        if not notice_payload.strip():
            blockers.append("Artifact notice member is empty.")
        expected_notice_hash = terms["notice_sha256"]
        if (
            not isinstance(expected_notice_hash, str)
            or not NOTICE_SHA256_PATTERN.fullmatch(expected_notice_hash)
            or _hash_bytes(notice_payload) != expected_notice_hash
        ):
            blockers.append("Artifact notice SHA-256 is absent, malformed, or mismatched.")
        try:
            notice_text = notice_payload.decode("utf-8")
        except UnicodeDecodeError:
            blockers.append("Artifact notice is not valid UTF-8 reviewed legal text.")
        else:
            required_lines = frozenset(
                {
                    f"Model-ID: {candidate.model_id}",
                    f"SPDX-License-Identifier: {terms['license_spdx']}",
                    "Redistribution: permitted",
                    "Commercial-Use: permitted",
                }
            )
            actual_lines = [line.strip() for line in notice_text.splitlines() if line.strip()]
            if len(actual_lines) != len(required_lines) or set(actual_lines) != required_lines:
                blockers.append(
                    "Artifact notice nonempty lines must exactly equal the four reviewed "
                    "model, license, redistribution, and commercial-use lines."
                )

    return {
        "valid": not blockers,
        "terms_path": terms_path,
        "terms": terms,
        "blockers": blockers,
    }


def audit_archive(candidate: Candidate, archive: Path) -> dict[str, object]:
    archive_hash = _hash_file(archive)
    archive_hash_matches = archive_hash == candidate.archive_sha256
    if not archive_hash_matches:
        terms_review = {
            "valid": False,
            "terms_path": None,
            "terms": None,
            "blockers": [
                "Archive member parsing was skipped because SHA-256 does not match."
            ],
        }
        return {
            "candidate": asdict(candidate),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_hash,
            "archive_hash_matches": False,
            "members": [],
            "member_inventory_matches": False,
            "artifact_terms_review": terms_review,
            "artifact_level_redistribution_proven": False,
        }

    members = []
    payloads: dict[str, bytes] = {}
    normalized_member_keys: set[str] = set()
    with tarfile.open(archive, "r:") as bundle:
        archive_members = sorted(bundle.getmembers(), key=lambda item: item.name)
        reviewed_members: list[tuple[tarfile.TarInfo, str]] = []
        for member in archive_members:
            if member.sparse:
                raise ValueError(
                    f"Archive member contains sparse metadata: {member.name}"
                )
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
                raise ValueError(
                    "Archive member type is not a regular file or directory: "
                    f"{member.name} ({member.type!r})"
                )
            normalized_path, normalized_key = _review_windows_archive_member_path(
                member.name
            )
            if normalized_key in normalized_member_keys:
                raise ValueError(
                    "Duplicate normalized archive member path under Windows "
                    f"case-insensitive comparison: {normalized_path}"
                )
            normalized_member_keys.add(normalized_key)
            reviewed_members.append((member, normalized_path))

        for member, normalized_path in reviewed_members:
            if member.type == tarfile.DIRTYPE:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            payload = stream.read()
            payloads[normalized_path] = payload
            members.append(
                {"path": normalized_path, "bytes": len(payload), "sha256": _hash_bytes(payload)}
            )

    actual = {item["path"]: item["sha256"] for item in members}
    terms_review = _review_structured_terms(candidate, payloads)
    return {
        "candidate": asdict(candidate),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "archive_hash_matches": archive_hash_matches,
        "members": members,
        "member_inventory_matches": actual == candidate.member_sha256,
        "artifact_terms_review": terms_review,
        "artifact_level_redistribution_proven": terms_review["valid"],
    }


def audit_official_source(source: Path) -> dict[str, object]:
    blockers: list[str] = []
    try:
        tag = _load_reviewed_json((source / "tag-ref.json").read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
        blockers.append(f"Saved tag response is unreadable: {error}")
        tag = {}
    try:
        commit = _load_reviewed_json((source / "commit.json").read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
        blockers.append(f"Saved commit response is unreadable: {error}")
        commit = {}
    expected_ref = f"refs/tags/{PINNED_TAG}"
    expected_ref_url = f"{OFFICIAL_API_ROOT}/git/refs/tags/{PINNED_TAG}"
    expected_object_url = f"{OFFICIAL_API_ROOT}/git/commits/{PINNED_COMMIT}"
    expected_commit_url = f"{OFFICIAL_API_ROOT}/commits/{PINNED_COMMIT}"
    expected_commit_html = f"{OFFICIAL_HTML_ROOT}/commit/{PINNED_COMMIT}"
    if tag.get("ref") != expected_ref:
        blockers.append("Saved tag response does not identify the exact reviewed tag ref.")
    if tag.get("url") != expected_ref_url:
        blockers.append("Saved tag response URL is not the reviewed official repository ref URL.")
    if tag.get("object", {}).get("url") != expected_object_url:
        blockers.append("Saved tag target URL is not the reviewed official repository commit URL.")
    if tag.get("object", {}).get("type") != "commit":
        blockers.append("Pinned tag does not directly reference a commit.")
    if tag.get("object", {}).get("sha") != PINNED_COMMIT:
        blockers.append("Pinned tag target does not match the reviewed commit.")
    if commit.get("sha") != PINNED_COMMIT:
        blockers.append("Official commit response does not match the reviewed commit.")
    if commit.get("url") != expected_commit_url or commit.get("html_url") != expected_commit_html:
        blockers.append("Saved commit response is not bound to the reviewed official repository.")
    if commit.get("commit", {}).get("tree", {}).get("sha") != PINNED_TREE:
        blockers.append("Official commit tree does not match the reviewed tree.")

    documents = []
    for document in OFFICIAL_DOCUMENTS:
        payload = (source / document.local_name).read_bytes()
        blob_hash = _git_blob_hash(payload)
        url_present = document.required_url.encode("utf-8") in payload.replace(b"\\\n", b"")
        if blob_hash != document.git_blob_sha1:
            blockers.append(f"Official document blob mismatch: {document.repository_path}")
        if not url_present:
            blockers.append(f"Official document omits reviewed archive URL: {document.repository_path}")
        documents.append(
            {
                **asdict(document),
                "bytes": len(payload),
                "sha256": _hash_bytes(payload),
                "measured_git_blob_sha1": blob_hash,
                "archive_url_present": url_present,
            }
        )
    return {
        "valid": not blockers,
        "tag": PINNED_TAG,
        "commit": PINNED_COMMIT,
        "tree": PINNED_TREE,
        "documents": documents,
        "blockers": blockers,
    }


def build_decision(
    audits: list[dict[str, object]],
    source_audit: dict[str, object],
) -> dict[str, object]:
    blockers = list(source_audit["blockers"])
    if not audits:
        blockers.append("No model archive audits were supplied.")
    expected_metadata = [asdict(candidate) for candidate in CANDIDATES]
    actual_metadata = [audit.get("candidate") for audit in audits]
    candidate_audits_valid = len(audits) == len(expected_metadata)
    for expected in expected_metadata:
        count = sum(metadata == expected for metadata in actual_metadata)
        if count != 1:
            candidate_audits_valid = False
            blockers.append(
                "Expected exactly one audit for pinned candidate "
                f"{expected['model_id']}; found {count}."
            )
    for index, metadata in enumerate(actual_metadata):
        if metadata not in expected_metadata:
            candidate_audits_valid = False
            blockers.append(
                f"Audit {index} contains unexpected or altered candidate metadata."
            )
    source_valid = source_audit["valid"] is True
    if not source_valid and not blockers:
        blockers.append("Official source provenance is invalid without a specific diagnostic.")
    for audit in audits:
        candidate_metadata = audit.get("candidate")
        model_id = (
            candidate_metadata.get("model_id", "<invalid-candidate>")
            if isinstance(candidate_metadata, dict)
            else "<invalid-candidate>"
        )
        if audit.get("archive_hash_matches") is not True:
            blockers.append(f"{model_id}: archive SHA-256 mismatch.")
        if audit.get("member_inventory_matches") is not True:
            blockers.append(f"{model_id}: extracted member inventory or SHA-256 mismatch.")
        terms_review = audit.get("artifact_terms_review")
        terms_blockers = (
            terms_review.get("blockers", []) if isinstance(terms_review, dict) else []
        )
        if not isinstance(terms_blockers, list):
            terms_blockers = ["Artifact terms review blockers are malformed."]
        for blocker in terms_blockers:
            blockers.append(f"{model_id}: {blocker}")
    hashes_valid = candidate_audits_valid and all(
        audit.get("archive_hash_matches") is True
        and audit.get("member_inventory_matches") is True
        for audit in audits
    )
    redistribution_proven = candidate_audits_valid and all(
        audit.get("artifact_level_redistribution_proven") is True for audit in audits
    )
    conversion_permitted = (
        source_valid and hashes_valid and redistribution_proven and not blockers
    )
    return {
        "source_provenance_valid": source_valid,
        "hashes_valid": hashes_valid,
        "artifact_level_redistribution_proven": redistribution_proven,
        "conversion_permitted": conversion_permitted,
        "status": "eligible_for_conversion" if conversion_permitted else "blocked",
        "blockers": blockers,
    }


def run(archives: Path, source: Path, output: Path) -> dict[str, object]:
    audits = [
        audit_archive(candidate, archives / candidate.archive_name)
        for candidate in CANDIDATES
    ]
    source_audit = audit_official_source(source)
    report = {
        "pinned_tag": PINNED_TAG,
        "pinned_commit": PINNED_COMMIT,
        "official_archives_only": True,
        "source_audit": source_audit,
        "audits": audits,
        **build_decision(audits, source_audit),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.archives, arguments.source, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["conversion_permitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
