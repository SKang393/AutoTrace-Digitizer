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


@dataclass(frozen=True)
class OfficialModelFile:
    name: str
    blob_sha1: str
    size: int
    lfs_sha256: str | None = None
    lfs_pointer_size: int | None = None


@dataclass(frozen=True)
class OfficialModelRepository:
    model_id: str
    repository_id: str
    revision: str
    local_name: str
    readme_sha256: str
    files: tuple[OfficialModelFile, ...]


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

OFFICIAL_MODEL_REPOSITORIES = (
    OfficialModelRepository(
        model_id="PP-OCRv5_mobile_det",
        repository_id="PaddlePaddle/PP-OCRv5_mobile_det",
        revision="0d63e78e2b680928f6b1747d76a08db6e645efb7",
        local_name="det",
        readme_sha256="4cc20ad6d41af86b3ce9885ffb0956e152574a2eb14179aeb07fd2d3956161ca",
        files=(
            OfficialModelFile(".gitattributes", "c48a31d8dce80bfbfe392212dc49792e212a6436", 1575),
            OfficialModelFile("README.md", "6ed97ac58de2375d7010fa6b7562b762d5e804f9", 16243),
            OfficialModelFile("config.json", "4fc190d78e4425094b31a2d3a4744a3623d00b50", 2871),
            OfficialModelFile("inference.json", "6cd678f39460a27372f8fe570e4e12e7a383418f", 229777),
            OfficialModelFile(
                "inference.pdiparams",
                "5e6d602681aa10f3660406dd7ec0ba48268d56e4",
                4692937,
                "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
                132,
            ),
            OfficialModelFile("inference.yml", "579d10695d2dd6e85c5ecba02d151ae5c077aa49", 903),
        ),
    ),
    OfficialModelRepository(
        model_id="en_PP-OCRv5_mobile_rec",
        repository_id="PaddlePaddle/en_PP-OCRv5_mobile_rec",
        revision="267c36e24c331595590fe7bd72bde2436fd286f2",
        local_name="rec",
        readme_sha256="4c1cfd6e103b0966fe97505b5254cfa35a931d47d7effca97a9db47fb57dd699",
        files=(
            OfficialModelFile(".gitattributes", "c48a31d8dce80bfbfe392212dc49792e212a6436", 1575),
            OfficialModelFile("README.md", "c7789c66447d3fbc3a4740b3d2b40e23b1ec6895", 6908),
            OfficialModelFile("config.json", "ee30b810753068b251de393eee66dff31f6d279c", 10455),
            OfficialModelFile("inference.json", "fc20fb935854373220fbeff985cbd310f0450608", 217712),
            OfficialModelFile(
                "inference.pdiparams",
                "81545b8e5cd2f173ddfaeeb9be76ab1574974fca",
                7772315,
                "3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
                132,
            ),
            OfficialModelFile("inference.yml", "91a401a7220881921c249b92852a96f9dbf2132a", 3964),
        ),
    ),
)

MODEL_CARD_LICENSE = "apache-2.0"
MODEL_LICENSE_SPDX = "Apache-2.0"
MODEL_CARD_CONTRADICTIONS = (
    re.compile(r"\bnon[- ]?commercial\b", re.IGNORECASE),
    re.compile(r"\bresearch[- ]?only\b", re.IGNORECASE),
    re.compile(r"\bproprietary\b", re.IGNORECASE),
    re.compile(r"\b(?:redistribution|commercial use)\s+(?:is\s+)?forbidden\b", re.IGNORECASE),
    re.compile(r"\bGPL(?:-[0-9.]+)?\b", re.IGNORECASE),
    re.compile(r"\bAGPL(?:-[0-9.]+)?\b", re.IGNORECASE),
    re.compile(r"\bSSPL(?:-[0-9.]+)?\b", re.IGNORECASE),
    re.compile(r"\bBUSL(?:-[0-9.]+)?\b", re.IGNORECASE),
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


def _git_lfs_pointer(sha256_digest: str, size: int) -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha256_digest}\n"
        f"size {size}\n"
    ).encode("ascii")


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


def audit_official_model_repositories(evidence: Path) -> dict[str, object]:
    """Bind exact official model-card terms to the archived inference bytes."""

    blockers: list[str] = []
    repositories: list[dict[str, object]] = []
    candidates_by_id = {candidate.model_id: candidate for candidate in CANDIDATES}

    if set(candidates_by_id) != {
        repository.model_id for repository in OFFICIAL_MODEL_REPOSITORIES
    }:
        blockers.append(
            "Official model repository definitions do not exactly cover the pinned candidates."
        )

    for repository in OFFICIAL_MODEL_REPOSITORIES:
        repository_blockers: list[str] = []
        repository_root = evidence / repository.local_name
        api_path = repository_root / "model-api.json"
        try:
            api: Any = _load_reviewed_json(api_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
            repository_blockers.append(f"Official model API evidence is unreadable: {error}")
            api = {}

        if not isinstance(api, dict):
            repository_blockers.append("Official model API evidence is not a JSON object.")
            api = {}
        if api.get("id") != repository.repository_id:
            repository_blockers.append(
                "Official model API evidence identifies the wrong repository."
            )
        if api.get("author") != "PaddlePaddle":
            repository_blockers.append("Official model repository is not owned by PaddlePaddle.")
        if api.get("sha") != repository.revision:
            repository_blockers.append("Official model API evidence identifies the wrong revision.")
        if api.get("private") is not False or api.get("gated") is not False:
            repository_blockers.append("Official model repository is private or access-gated.")
        card_data = api.get("cardData")
        if not isinstance(card_data, dict) or card_data.get("license") != MODEL_CARD_LICENSE:
            repository_blockers.append(
                "Official model API metadata does not scope Apache-2.0 to the model repository."
            )
        siblings = api.get("siblings")
        expected_files = {item.name: item for item in repository.files}
        sibling_names: list[str] = []
        if not isinstance(siblings, list):
            repository_blockers.append("Official model API file inventory is not a list.")
            siblings = []
        for index, sibling in enumerate(siblings):
            if not isinstance(sibling, dict):
                repository_blockers.append(
                    f"Official model API file inventory entry {index} is malformed."
                )
                continue
            filename = sibling.get("rfilename")
            if not isinstance(filename, str):
                repository_blockers.append(
                    f"Official model API file inventory entry {index} has no filename."
                )
                continue
            sibling_names.append(filename)
            expected_file = expected_files.get(filename)
            if expected_file is None:
                repository_blockers.append(
                    f"Official model API file inventory contains an unexpected file: {filename}."
                )
                continue
            expected_sibling: dict[str, object] = {
                "rfilename": expected_file.name,
                "blobId": expected_file.blob_sha1,
                "size": expected_file.size,
            }
            if expected_file.lfs_sha256 is not None:
                expected_sibling["lfs"] = {
                    "sha256": expected_file.lfs_sha256,
                    "size": expected_file.size,
                    "pointerSize": expected_file.lfs_pointer_size,
                }
            if sibling != expected_sibling:
                repository_blockers.append(
                    "Official model API file identity does not match the reviewed revision "
                    f"({filename})."
                )
        if len(sibling_names) != len(set(sibling_names)):
            repository_blockers.append("Official model API evidence repeats a repository file.")
        if set(sibling_names) != set(expected_files):
            repository_blockers.append(
                "Official model repository file inventory does not match the reviewed revision."
            )

        expected_local_files = set(expected_files) | {"model-api.json"}
        try:
            actual_local_files = {
                path.name for path in repository_root.iterdir() if path.is_file()
            }
        except OSError as error:
            repository_blockers.append(
                f"Official model evidence directory is unreadable: {error}"
            )
            actual_local_files = set()
        if actual_local_files != expected_local_files:
            repository_blockers.append(
                "Official model evidence directory contains missing or unexpected files."
            )

        local_payloads: dict[str, bytes] = {}
        measured_file_identities: dict[str, dict[str, object]] = {}
        for expected_file in repository.files:
            payload_path = repository_root / expected_file.name
            try:
                payload = payload_path.read_bytes()
            except OSError as error:
                repository_blockers.append(
                    "Official model repository file is unreadable "
                    f"({expected_file.name}): {error}"
                )
                continue
            local_payloads[expected_file.name] = payload
            if len(payload) != expected_file.size:
                repository_blockers.append(
                    "Official model repository file size does not match the reviewed revision "
                    f"({expected_file.name})."
                )
            if expected_file.lfs_sha256 is None:
                measured_blob_sha1 = _git_blob_hash(payload)
                measured_content_sha256 = _hash_bytes(payload)
            else:
                measured_content_sha256 = _hash_bytes(payload)
                pointer = _git_lfs_pointer(expected_file.lfs_sha256, expected_file.size)
                measured_blob_sha1 = _git_blob_hash(pointer)
                if len(pointer) != expected_file.lfs_pointer_size:
                    repository_blockers.append(
                        "Official model LFS pointer size does not match the reviewed revision "
                        f"({expected_file.name})."
                    )
                if measured_content_sha256 != expected_file.lfs_sha256:
                    repository_blockers.append(
                        "Official model LFS content SHA-256 does not match the reviewed revision "
                        f"({expected_file.name})."
                    )
            if measured_blob_sha1 != expected_file.blob_sha1:
                repository_blockers.append(
                    "Official model Git blob identity does not match the reviewed revision "
                    f"({expected_file.name})."
                )
            measured_file_identities[expected_file.name] = {
                "bytes": len(payload),
                "git_blob_sha1": measured_blob_sha1,
                "content_sha256": measured_content_sha256,
                "evidence_path": str(payload_path.resolve()),
                "revision_url": (
                    f"https://huggingface.co/{repository.repository_id}/resolve/"
                    f"{repository.revision}/{expected_file.name}"
                ),
            }

        readme_payload = local_payloads.get("README.md", b"")
        readme_hash = _hash_bytes(readme_payload)
        if readme_hash != repository.readme_sha256:
            repository_blockers.append(
                "Official model card SHA-256 does not match the reviewed revision."
            )
        try:
            readme_text = readme_payload.decode("utf-8")
        except UnicodeDecodeError:
            repository_blockers.append("Official model card is not valid UTF-8.")
            readme_text = ""
        license_lines = [
            line.strip()
            for line in readme_text.splitlines()
            if line.strip().lower().startswith("license:")
        ]
        if license_lines != [f"license: {MODEL_CARD_LICENSE}"]:
            repository_blockers.append(
                "Official model card does not contain exactly one Apache-2.0 license field."
            )
        if f"# {repository.model_id}" not in readme_text:
            repository_blockers.append("Official model card does not identify the exact model ID.")
        if any(pattern.search(readme_text) for pattern in MODEL_CARD_CONTRADICTIONS):
            repository_blockers.append("Official model card contains contradictory license terms.")

        candidate = candidates_by_id.get(repository.model_id)
        expected_payloads = (
            {PurePosixPath(path).name: digest for path, digest in candidate.member_sha256.items()}
            if candidate is not None
            else {}
        )
        measured_payloads: dict[str, str] = {}
        if set(expected_payloads) != {"inference.json", "inference.pdiparams", "inference.yml"}:
            repository_blockers.append(
                "Pinned archive inventory does not expose the exact three inference payloads."
            )
        for filename, expected_hash in expected_payloads.items():
            payload = local_payloads.get(filename)
            if payload is None:
                continue
            measured_hash = _hash_bytes(payload)
            measured_payloads[filename] = measured_hash
            if measured_hash != expected_hash:
                repository_blockers.append(
                    "Official model repository payload does not match the BOS archive "
                    f"({filename})."
                )

        repository_valid = not repository_blockers
        blockers.extend(
            f"{repository.model_id}: {blocker}" for blocker in repository_blockers
        )
        repositories.append(
            {
                "model_id": repository.model_id,
                "repository_id": repository.repository_id,
                "revision": repository.revision,
                "repository_url": f"https://huggingface.co/{repository.repository_id}",
                "revision_url": (
                    f"https://huggingface.co/{repository.repository_id}/tree/"
                    f"{repository.revision}"
                ),
                "revision_api_url": (
                    f"https://huggingface.co/api/models/{repository.repository_id}/"
                    f"revision/{repository.revision}?blobs=true"
                ),
                "api_evidence_path": str(api_path.resolve()),
                "model_card_sha256": readme_hash,
                "license_spdx": MODEL_LICENSE_SPDX if repository_valid else None,
                "redistribution": repository_valid,
                "commercial_use": repository_valid,
                "repository_file_identities": measured_file_identities,
                "archive_payload_sha256": measured_payloads,
                "valid": repository_valid,
                "blockers": repository_blockers,
            }
        )

    covered_model_ids = [
        repository["model_id"] for repository in repositories if repository["valid"] is True
    ]
    expected_model_ids = [candidate.model_id for candidate in CANDIDATES]
    valid = not blockers and sorted(covered_model_ids) == sorted(expected_model_ids)
    return {
        "valid": valid,
        "license_spdx": MODEL_LICENSE_SPDX if valid else None,
        "scope": "exact-model-repository-revision-and-byte-identical-bos-payloads",
        "covered_model_ids": covered_model_ids,
        "repositories": repositories,
        "blockers": blockers,
    }


def build_decision(
    audits: list[dict[str, object]],
    source_audit: dict[str, object],
    model_repository_license_audit: dict[str, object] | None = None,
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
    hashes_valid = candidate_audits_valid and all(
        audit.get("archive_hash_matches") is True
        and audit.get("member_inventory_matches") is True
        for audit in audits
    )
    embedded_redistribution_proven = candidate_audits_valid and all(
        audit.get("artifact_level_redistribution_proven") is True for audit in audits
    )
    expected_model_ids = sorted(candidate.model_id for candidate in CANDIDATES)
    repository_covered_model_ids = (
        model_repository_license_audit.get("covered_model_ids", [])
        if isinstance(model_repository_license_audit, dict)
        else []
    )
    repository_redistribution_proven = (
        isinstance(model_repository_license_audit, dict)
        and model_repository_license_audit.get("valid") is True
        and isinstance(repository_covered_model_ids, list)
        and sorted(repository_covered_model_ids) == expected_model_ids
    )
    redistribution_proven = (
        embedded_redistribution_proven or repository_redistribution_proven
    )
    if not redistribution_proven:
        for audit in audits:
            candidate_metadata = audit.get("candidate")
            model_id = (
                candidate_metadata.get("model_id", "<invalid-candidate>")
                if isinstance(candidate_metadata, dict)
                else "<invalid-candidate>"
            )
            terms_review = audit.get("artifact_terms_review")
            terms_blockers = (
                terms_review.get("blockers", []) if isinstance(terms_review, dict) else []
            )
            if not isinstance(terms_blockers, list):
                terms_blockers = ["Artifact terms review blockers are malformed."]
            for blocker in terms_blockers:
                blockers.append(f"{model_id}: {blocker}")
        if isinstance(model_repository_license_audit, dict):
            repository_blockers = model_repository_license_audit.get("blockers", [])
            if isinstance(repository_blockers, list):
                blockers.extend(
                    f"Official model repository evidence: {blocker}"
                    for blocker in repository_blockers
                )
            else:
                blockers.append("Official model repository evidence blockers are malformed.")
    conversion_permitted = (
        source_valid and hashes_valid and redistribution_proven and not blockers
    )
    return {
        "source_provenance_valid": source_valid,
        "hashes_valid": hashes_valid,
        "artifact_level_redistribution_proven": redistribution_proven,
        "embedded_archive_terms_proven": embedded_redistribution_proven,
        "official_model_repository_terms_proven": repository_redistribution_proven,
        "conversion_permitted": conversion_permitted,
        "status": "eligible_for_conversion" if conversion_permitted else "blocked",
        "blockers": blockers,
    }


def run(
    archives: Path,
    source: Path,
    output: Path,
    model_license_evidence: Path | None = None,
) -> dict[str, object]:
    audits = [
        audit_archive(candidate, archives / candidate.archive_name)
        for candidate in CANDIDATES
    ]
    source_audit = audit_official_source(source)
    model_repository_license_audit = (
        audit_official_model_repositories(model_license_evidence)
        if model_license_evidence is not None
        else None
    )
    report = {
        "pinned_tag": PINNED_TAG,
        "pinned_commit": PINNED_COMMIT,
        "official_archives_only": True,
        "source_audit": source_audit,
        "model_repository_license_audit": model_repository_license_audit,
        "audits": audits,
        **build_decision(audits, source_audit, model_repository_license_audit),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-license-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(
        arguments.archives,
        arguments.source,
        arguments.output,
        arguments.model_license_evidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["conversion_permitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
