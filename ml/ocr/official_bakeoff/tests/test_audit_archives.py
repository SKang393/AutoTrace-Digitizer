# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from dataclasses import asdict, replace
from hashlib import sha1, sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest

from ml.ocr.official_bakeoff.audit_archives import (
    Candidate,
    OfficialDocument,
    audit_archive,
    audit_official_source,
    build_decision,
)
from ml.ocr.official_bakeoff import audit_archives as audit_module

DEFAULT_NOTICE = (
    b"Model-ID: test-model\n"
    b"SPDX-License-Identifier: Apache-2.0\n"
    b"Redistribution: permitted\n"
    b"Commercial-Use: permitted\n"
)


def _write_archive(tmp_path: Path, files: dict[str, bytes], name: str = "model.tar") -> Path:
    return _write_archive_entries(tmp_path, list(files.items()), name)


def _write_archive_entries(
    tmp_path: Path,
    entries: list[tuple[str, bytes]],
    name: str = "model.tar",
) -> Path:
    archive = tmp_path / name
    with tarfile.open(archive, "w") as bundle:
        for member, content in entries:
            info = tarfile.TarInfo(member)
            info.size = len(content)
            bundle.addfile(info, BytesIO(content))
    return archive


def _write_typed_archive(
    tmp_path: Path,
    entries: list[tuple[tarfile.TarInfo, bytes | None]],
    name: str = "typed-model.tar",
) -> Path:
    archive = tmp_path / name
    with tarfile.open(archive, "w") as bundle:
        for info, content in entries:
            if content is not None:
                info.size = len(content)
                bundle.addfile(info, BytesIO(content))
            else:
                bundle.addfile(info)
    return archive


def _candidate(archive: Path, files: dict[str, bytes]) -> Candidate:
    return Candidate(
        model_id="test-model",
        archive_name=archive.name,
        url="https://official.invalid/model.tar",
        archive_sha256=sha256(archive.read_bytes()).hexdigest(),
        member_sha256={path: sha256(content).hexdigest() for path, content in files.items()},
    )


def _terms(notice: bytes = DEFAULT_NOTICE, **overrides: object) -> bytes:
    value: dict[str, object] = {
        "model_id": "test-model",
        "scope": "all-files-in-archive",
        "license_spdx": "Apache-2.0",
        "redistribution": True,
        "commercial_use": True,
        "notice_path": "model/NOTICE.txt",
        "notice_sha256": sha256(notice).hexdigest(),
    }
    value.update(overrides)
    return json.dumps(value).encode("utf-8")


def _files(
    terms: bytes | None = None,
    notice: bytes | None = DEFAULT_NOTICE,
) -> dict[str, bytes]:
    files = {"model/inference.yml": b"Global:\n  model_name: test\n"}
    if terms is not None:
        files["model/artifact-terms.json"] = terms
    if notice is not None:
        files["model/NOTICE.txt"] = notice
    return files


def _audit(tmp_path: Path, files: dict[str, bytes]) -> dict[str, object]:
    archive = _write_archive(tmp_path, files)
    return audit_archive(_candidate(archive, files), archive)


def _assert_rejected_before_payload_read(
    monkeypatch: pytest.MonkeyPatch,
    candidate: Candidate,
    archive: Path,
    message: str,
) -> None:
    extract_calls = 0
    original_extractfile = tarfile.TarFile.extractfile

    def tracked_extractfile(
        bundle: tarfile.TarFile,
        member: tarfile.TarInfo | str,
    ) -> object:
        nonlocal extract_calls
        extract_calls += 1
        return original_extractfile(bundle, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", tracked_extractfile)

    with pytest.raises(ValueError, match=message):
        audit_archive(candidate, archive)
    assert extract_calls == 0


def test_exact_allowlisted_affirmative_terms_with_notice_can_pass(tmp_path: Path) -> None:
    result = _audit(tmp_path, _files(_terms()))

    assert result["archive_hash_matches"] is True
    assert result["member_inventory_matches"] is True
    assert result["artifact_terms_review"]["valid"] is True
    assert result["artifact_level_redistribution_proven"] is True


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            '"license_spdx": "Apache-2.0"',
            '"license_spdx": "GPL-3.0-only", "license_spdx": "Apache-2.0"',
        ),
        (
            '"license_spdx": "Apache-2.0"',
            '"license_spdx": "Apache-2.0", "license_spdx": "GPL-3.0-only"',
        ),
        (
            '"redistribution": true',
            '"redistribution": false, "redistribution": true',
        ),
        (
            '"redistribution": true',
            '"redistribution": true, "redistribution": false',
        ),
    ],
)
def test_duplicate_top_level_terms_keys_fail_in_both_orderings(
    tmp_path: Path,
    needle: str,
    replacement: str,
) -> None:
    duplicated = _terms().decode("utf-8").replace(needle, replacement).encode("utf-8")

    result = _audit(tmp_path, _files(duplicated))

    assert result["artifact_terms_review"]["valid"] is False
    assert any(
        "duplicate JSON object key" in blocker
        for blocker in result["artifact_terms_review"]["blockers"]
    )


def test_duplicate_nested_terms_key_fails_before_field_review(tmp_path: Path) -> None:
    duplicated = _terms().decode("utf-8").replace(
        '"scope": "all-files-in-archive"',
        '"scope": {"value": "all-files-in-archive", "value": "benign"}',
    ).encode("utf-8")

    result = _audit(tmp_path, _files(duplicated))

    assert result["artifact_terms_review"]["valid"] is False
    assert any(
        "duplicate JSON object key: value" in blocker
        for blocker in result["artifact_terms_review"]["blockers"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"license_spdx": "Proprietary"},
        {"license_spdx": "GPL-3.0-only"},
        {"redistribution": False},
        {"commercial_use": False},
        {"redistribution": "permitted"},
        {"commercial_use": "allowed"},
        {"scope": "weights-excluded"},
        {"model_id": "different-model"},
        {"restrictions": "non-commercial and redistribution forbidden"},
    ],
)
def test_negative_prohibited_or_ambiguous_terms_never_count(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    result = _audit(tmp_path, _files(_terms(**overrides)))

    assert result["artifact_terms_review"]["valid"] is False
    assert result["artifact_level_redistribution_proven"] is False


def test_legal_keyword_files_without_structured_grant_never_count(tmp_path: Path) -> None:
    files = _files()
    files["model/LICENSE.txt"] = b"commercial redistribution forbidden; proprietary"
    result = _audit(tmp_path, files)

    assert result["artifact_terms_review"]["valid"] is False
    assert result["artifact_level_redistribution_proven"] is False


def test_missing_terms_remain_blocked_at_run_level(tmp_path: Path) -> None:
    audit = _audit(tmp_path, _files())
    decision = build_decision([audit], {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert decision["status"] == "blocked"
    assert any("artifact-terms.json" in blocker for blocker in decision["blockers"])


@pytest.mark.parametrize("notice", [None, b""])
def test_missing_or_empty_notice_fails(tmp_path: Path, notice: bytes | None) -> None:
    terms_notice = b"" if notice is None else notice
    result = _audit(tmp_path, _files(_terms(notice=terms_notice), notice=notice))

    assert result["artifact_terms_review"]["valid"] is False


@pytest.mark.parametrize(
    "contradiction",
    [
        b"GPL-3.0-only\n",
        b"proprietary\n",
        b"Non-commercial\n",
        b"Commercial use forbidden\n",
        b"Redistribution forbidden\n",
        b"Research-only\n",
        b"Commercial-Use: prohibited\n",
        b"You may not redistribute this artifact\n",
        b"GNU General Public License version 3 applies\n",
        b"Use is limited to research purposes\n",
        b"rEdIsTrIbUtIoN---IS!!!PROHIBITED\n",
        b"COMMERCIAL_use is NOT-permitted\n",
        b"You SHALL-NOT distribute this model\n",
        b"gnu-general-public-license, VERSION-3\n",
        "USE—IS—RESTRICTED—TO—ACADEMIC—PURPOSES\n".encode(),
        b"Evaluation purposes ONLY\n",
        b"NOT FOR COMMERCIAL USE\n",
        b"Redistribution: permitted only after written authorization\n",
        b"No license is granted for this artifact\n",
    ],
)
def test_contradictory_notice_content_is_blocked_at_run_level(
    tmp_path: Path,
    contradiction: bytes,
) -> None:
    notice = DEFAULT_NOTICE + contradiction
    audit = _audit(tmp_path, _files(_terms(notice=notice), notice=notice))
    decision = build_decision([audit], {"valid": True, "blockers": []})

    assert audit["artifact_terms_review"]["valid"] is False
    assert decision["conversion_permitted"] is False
    assert any("must exactly equal" in blocker for blocker in decision["blockers"])


@pytest.mark.parametrize(
    "notice",
    [
        DEFAULT_NOTICE.replace(
            b"Commercial-Use: permitted",
            "Commercial-Usе: permitted".encode("utf-8"),
        ),
        DEFAULT_NOTICE.replace(
            b"Redistribution: permitted",
            b"Redistribution: permitted with written authorization",
        ),
        DEFAULT_NOTICE + b"Redistribution: permitted\n",
    ],
)
def test_notice_contract_rejects_homoglyph_condition_and_duplicate_line(
    tmp_path: Path,
    notice: bytes,
) -> None:
    audit = _audit(tmp_path, _files(_terms(notice=notice), notice=notice))

    assert audit["artifact_terms_review"]["valid"] is False
    assert any(
        "must exactly equal" in blocker
        for blocker in audit["artifact_terms_review"]["blockers"]
    )


@pytest.mark.parametrize(
    "member_path",
    [
        "model/artifact-terms.json",
        "model/NOTICE.txt",
        "model/inference.yml",
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_duplicate_tar_member_path_fails_in_both_orderings(
    tmp_path: Path,
    member_path: str,
    reverse: bool,
) -> None:
    files = _files(_terms())
    first = files[member_path]
    second = b"adversarial duplicate"
    duplicates = [(member_path, first), (member_path, second)]
    if reverse:
        duplicates.reverse()
    entries = [(path, content) for path, content in files.items() if path != member_path]
    entries.extend(duplicates)
    archive = _write_archive_entries(tmp_path, entries)

    with pytest.raises(ValueError, match="(?i)archive member path"):
        audit_archive(_candidate(archive, files), archive)


@pytest.mark.parametrize("equivalent_path", ["model/./NOTICE.txt", "model//NOTICE.txt"])
@pytest.mark.parametrize("reverse", [False, True])
def test_normalization_equivalent_tar_member_paths_fail_closed(
    tmp_path: Path,
    equivalent_path: str,
    reverse: bool,
) -> None:
    files = _files(_terms())
    duplicate_entries = [
        ("model/NOTICE.txt", DEFAULT_NOTICE),
        (equivalent_path, b"adversarial duplicate"),
    ]
    if reverse:
        duplicate_entries.reverse()
    entries = [(path, content) for path, content in files.items() if path != "model/NOTICE.txt"]
    entries.extend(duplicate_entries)
    archive = _write_archive_entries(tmp_path, entries)

    with pytest.raises(ValueError, match="(?i)archive member path"):
        audit_archive(_candidate(archive, files), archive)


def test_duplicate_tar_paths_are_rejected_before_any_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _files(_terms())
    entries = list(files.items()) + [("model/./NOTICE.txt", b"duplicate")]
    archive = _write_archive_entries(tmp_path, entries)
    extract_calls = 0
    original_extractfile = tarfile.TarFile.extractfile

    def tracked_extractfile(
        bundle: tarfile.TarFile,
        member: tarfile.TarInfo | str,
    ) -> object:
        nonlocal extract_calls
        extract_calls += 1
        return original_extractfile(bundle, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", tracked_extractfile)

    with pytest.raises(ValueError, match="(?i)archive member path"):
        audit_archive(_candidate(archive, files), archive)
    assert extract_calls == 0


@pytest.mark.parametrize(
    ("attack_entries", "message"),
    [
        (
            [("model/NOTICE.txt", DEFAULT_NOTICE), ("MODEL/notice.TXT", b"duplicate")],
            "case-insensitive comparison",
        ),
        ([(r"model\NOTICE.txt", DEFAULT_NOTICE)], "backslash"),
        ([("model/NOTICE.txt.", DEFAULT_NOTICE)], "trailing dot or space"),
        ([("model/NOTICE.txt ", DEFAULT_NOTICE)], "trailing dot or space"),
        ([("model/NOTICE.txt:payload", DEFAULT_NOTICE)], "segment characters"),
        ([("model/NUL.txt", DEFAULT_NOTICE)], "reserved device name"),
        ([("MoDeL/cOm1.LoG", DEFAULT_NOTICE)], "reserved device name"),
        ([("model/cOm¹.TxT", DEFAULT_NOTICE)], "reserved device name"),
        ([("MODEL\\aux.TxT::$DATA", DEFAULT_NOTICE)], "backslash"),
        ([("/model/NOTICE.txt", DEFAULT_NOTICE)], "empty or absolute"),
        ([("model/./NOTICE.txt", DEFAULT_NOTICE)], "not canonical POSIX"),
        ([("model/../NOTICE.txt", DEFAULT_NOTICE)], "not canonical POSIX"),
        ([("model//NOTICE.txt", DEFAULT_NOTICE)], "repeated separators"),
    ],
    ids=[
        "case-collision",
        "backslash",
        "trailing-dot",
        "trailing-space",
        "ads-colon",
        "reserved-device",
        "mixed-case-reserved-device",
        "unicode-reserved-device",
        "mixed-backslash-reserved-ads",
        "absolute",
        "dot-segment",
        "dotdot-segment",
        "repeated-separator",
    ],
)
def test_windows_unsafe_paths_are_rejected_before_any_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack_entries: list[tuple[str, bytes]],
    message: str,
) -> None:
    files = _files(_terms())
    entries = [
        (path, content)
        for path, content in files.items()
        if path != "model/NOTICE.txt"
    ]
    entries.extend(attack_entries)
    archive = _write_archive_entries(tmp_path, entries)
    extract_calls = 0
    original_extractfile = tarfile.TarFile.extractfile

    def tracked_extractfile(
        bundle: tarfile.TarFile,
        member: tarfile.TarInfo | str,
    ) -> object:
        nonlocal extract_calls
        extract_calls += 1
        return original_extractfile(bundle, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", tracked_extractfile)

    with pytest.raises(ValueError, match=message):
        audit_archive(_candidate(archive, files), archive)
    assert extract_calls == 0


def test_non_notice_path_is_blocked_even_with_matching_content_and_hash(tmp_path: Path) -> None:
    files = {"model/inference.yml": DEFAULT_NOTICE}
    terms = json.loads(_terms().decode("utf-8"))
    terms["notice_path"] = "model/inference.yml"
    terms["notice_sha256"] = sha256(DEFAULT_NOTICE).hexdigest()
    files["model/artifact-terms.json"] = json.dumps(terms).encode("utf-8")
    audit = _audit(tmp_path, files)
    decision = build_decision([audit], {"valid": True, "blockers": []})

    assert audit["artifact_terms_review"]["valid"] is False
    assert decision["conversion_permitted"] is False
    assert any("NOTICE material" in blocker for blocker in decision["blockers"])


def test_notice_hash_mismatch_fails(tmp_path: Path) -> None:
    terms = json.loads(_terms().decode("utf-8"))
    terms["notice_sha256"] = "0" * 64
    result = _audit(tmp_path, _files(json.dumps(terms).encode("utf-8")))

    assert result["artifact_terms_review"]["valid"] is False
    assert any(
        "notice SHA-256" in blocker
        for blocker in result["artifact_terms_review"]["blockers"]
    )


def test_archive_hash_mismatch_is_reported_at_run_level(tmp_path: Path) -> None:
    files = _files(_terms())
    archive = _write_archive(tmp_path, files)
    candidate = replace(_candidate(archive, files), archive_sha256="0" * 64)
    audit = audit_archive(candidate, archive)
    decision = build_decision([audit], {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert any("archive SHA-256 mismatch" in blocker for blocker in decision["blockers"])


def test_archive_hash_mismatch_never_opens_or_extracts_tar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _files(_terms())
    archive = _write_archive(tmp_path, files)
    candidate = replace(_candidate(archive, files), archive_sha256="0" * 64)
    open_calls = 0
    extract_calls = 0

    def unexpected_open(*args: object, **kwargs: object) -> object:
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("tarfile.open must not run after a hash mismatch")

    def unexpected_extract(*args: object, **kwargs: object) -> object:
        nonlocal extract_calls
        extract_calls += 1
        raise AssertionError("extractfile must not run after a hash mismatch")

    monkeypatch.setattr(tarfile, "open", unexpected_open)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", unexpected_extract)

    result = audit_archive(candidate, archive)

    assert result["archive_hash_matches"] is False
    assert result["member_inventory_matches"] is False
    assert result["artifact_level_redistribution_proven"] is False
    assert open_calls == 0
    assert extract_calls == 0


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize("link_target", ["model/NOTICE.txt", "../../outside.txt"])
def test_internal_and_external_tar_links_are_rejected_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_type: bytes,
    link_target: str,
) -> None:
    info = tarfile.TarInfo("model/link")
    info.type = link_type
    info.linkname = link_target
    archive = _write_typed_archive(tmp_path, [(info, None)])

    _assert_rejected_before_payload_read(
        monkeypatch,
        _candidate(archive, {}),
        archive,
        "not a regular file or directory",
    )


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CONTTYPE,
        b"Z",
    ],
    ids=["character-device", "block-device", "fifo", "contiguous", "unknown"],
)
def test_nonregular_tar_member_types_are_rejected_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_type: bytes,
) -> None:
    info = tarfile.TarInfo("model/special")
    info.type = member_type
    archive = _write_typed_archive(tmp_path, [(info, None)])

    _assert_rejected_before_payload_read(
        monkeypatch,
        _candidate(archive, {}),
        archive,
        "not a regular file or directory",
    )


def test_pax_sparse_metadata_on_regular_member_rejects_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = tarfile.TarInfo("model/pax-sparse.bin")
    info.type = tarfile.REGTYPE
    info.pax_headers = {
        "GNU.sparse.map": "0,1",
        "GNU.sparse.realsize": "1",
    }
    archive = _write_typed_archive(tmp_path, [(info, b"x")])

    _assert_rejected_before_payload_read(
        monkeypatch,
        _candidate(archive, {"model/pax-sparse.bin": b"x"}),
        archive,
        "contains sparse metadata",
    )


def test_gnu_sparse_member_rejects_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = tarfile.TarInfo("model/gnu-sparse.bin")
    info.type = tarfile.GNUTYPE_SPARSE
    archive = _write_typed_archive(tmp_path, [(info, None)])

    _assert_rejected_before_payload_read(
        monkeypatch,
        _candidate(archive, {}),
        archive,
        "sparse metadata|not a regular file or directory",
    )


@pytest.mark.parametrize(
    "directory_name",
    ["unsafe\\directory", "model/../outside", "NUL", "trailing."],
)
def test_directory_paths_receive_windows_safety_review_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_name: str,
) -> None:
    info = tarfile.TarInfo(directory_name)
    info.type = tarfile.DIRTYPE
    archive = _write_typed_archive(tmp_path, [(info, None)])

    _assert_rejected_before_payload_read(
        monkeypatch,
        _candidate(archive, {}),
        archive,
        "Archive member path",
    )


def test_duplicate_directory_file_windows_identity_rejects_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tarfile.TarInfo("Model/Entry")
    directory.type = tarfile.DIRTYPE
    regular = tarfile.TarInfo("model/entry")
    regular.type = tarfile.REGTYPE
    archive = _write_typed_archive(
        tmp_path,
        [(directory, None), (regular, b"payload")],
    )

    _assert_rejected_before_payload_read(
        monkeypatch,
        _candidate(archive, {"model/entry": b"payload"}),
        archive,
        "case-insensitive comparison",
    )


def test_incomplete_member_inventory_is_reported_at_run_level(tmp_path: Path) -> None:
    files = _files(_terms())
    archive = _write_archive(tmp_path, files)
    candidate = _candidate(archive, files)
    candidate = replace(
        candidate,
        member_sha256={**candidate.member_sha256, "model/missing.bin": "1" * 64},
    )
    audit = audit_archive(candidate, archive)
    decision = build_decision([audit], {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert any("member inventory" in blocker for blocker in decision["blockers"])


def test_source_provenance_blocker_prevents_conversion(tmp_path: Path) -> None:
    audit = _audit(tmp_path, _files(_terms()))
    decision = build_decision(
        [audit],
        {"valid": False, "blockers": ["Official document blob mismatch."]},
    )

    assert decision["conversion_permitted"] is False
    assert decision["blockers"][0] == "Official document blob mismatch."


def test_empty_archive_audit_list_fails_closed() -> None:
    decision = build_decision([], {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert decision["hashes_valid"] is False
    assert decision["artifact_level_redistribution_proven"] is False
    assert decision["status"] == "blocked"
    assert decision["blockers"][0] == "No model archive audits were supplied."
    assert all(
        any(candidate.model_id in blocker for blocker in decision["blockers"])
        for candidate in audit_module.CANDIDATES
    )


def _valid_candidate_audit(candidate: Candidate) -> dict[str, object]:
    return {
        "candidate": asdict(candidate),
        "archive_hash_matches": True,
        "member_inventory_matches": True,
        "artifact_terms_review": {"blockers": []},
        "artifact_level_redistribution_proven": True,
    }


def test_exactly_one_complete_audit_per_pinned_candidate_can_pass() -> None:
    audits = [_valid_candidate_audit(candidate) for candidate in audit_module.CANDIDATES]

    decision = build_decision(audits, {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is True
    assert decision["hashes_valid"] is True
    assert decision["artifact_level_redistribution_proven"] is True
    assert decision["blockers"] == []


def test_single_valid_candidate_audit_fails_closed() -> None:
    audits = [_valid_candidate_audit(audit_module.CANDIDATES[0])]

    decision = build_decision(audits, {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert decision["hashes_valid"] is False
    assert decision["artifact_level_redistribution_proven"] is False
    assert any(
        audit_module.CANDIDATES[1].model_id in blocker and "found 0" in blocker
        for blocker in decision["blockers"]
    )


def test_duplicate_valid_candidate_audits_fail_closed() -> None:
    duplicate = _valid_candidate_audit(audit_module.CANDIDATES[0])

    decision = build_decision([duplicate, duplicate], {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert any("found 2" in blocker for blocker in decision["blockers"])
    assert any("found 0" in blocker for blocker in decision["blockers"])


def test_unexpected_extra_candidate_audit_fails_closed() -> None:
    audits = [_valid_candidate_audit(candidate) for candidate in audit_module.CANDIDATES]
    unexpected = replace(audit_module.CANDIDATES[0], model_id="unexpected-model")
    audits.append(_valid_candidate_audit(unexpected))

    decision = build_decision(audits, {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert decision["hashes_valid"] is False
    assert any("unexpected or altered candidate metadata" in blocker for blocker in decision["blockers"])


def test_altered_candidate_metadata_fails_full_binding() -> None:
    altered = replace(audit_module.CANDIDATES[0], url="https://example.invalid/altered.tar")
    audits = [
        _valid_candidate_audit(altered),
        _valid_candidate_audit(audit_module.CANDIDATES[1]),
    ]

    decision = build_decision(audits, {"valid": True, "blockers": []})

    assert decision["conversion_permitted"] is False
    assert any("found 0" in blocker for blocker in decision["blockers"])
    assert any("unexpected or altered candidate metadata" in blocker for blocker in decision["blockers"])


def _write_valid_source_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    commit = "a" * 40
    tree = "b" * 40
    url = "https://official.invalid/model.tar"
    document = b"official archive: " + url.encode("ascii")
    blob_header = f"blob {len(document)}\0".encode("ascii")
    blob = sha1(blob_header + document).hexdigest()
    (tmp_path / "tag-ref.json").write_text(
        json.dumps(
            {
                "ref": "refs/tags/v3.5.0",
                "url": f"{audit_module.OFFICIAL_API_ROOT}/git/refs/tags/v3.5.0",
                "object": {
                    "type": "commit",
                    "sha": commit,
                    "url": f"{audit_module.OFFICIAL_API_ROOT}/git/commits/{commit}",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "commit.json").write_text(
        json.dumps(
            {
                "sha": commit,
                "url": f"{audit_module.OFFICIAL_API_ROOT}/commits/{commit}",
                "html_url": f"{audit_module.OFFICIAL_HTML_ROOT}/commit/{commit}",
                "commit": {"tree": {"sha": tree}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "doc.md").write_bytes(document)
    monkeypatch.setattr(audit_module, "PINNED_COMMIT", commit)
    monkeypatch.setattr(audit_module, "PINNED_TREE", tree)
    monkeypatch.setattr(
        audit_module,
        "OFFICIAL_DOCUMENTS",
        (OfficialDocument("doc.md", "docs/doc.md", blob, url),),
    )
    return document


def test_official_source_audit_binds_tag_tree_document_blob_and_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _write_valid_source_fixture(tmp_path, monkeypatch)

    assert audit_official_source(tmp_path)["valid"] is True
    (tmp_path / "doc.md").write_bytes(document + b" changed")
    result = audit_official_source(tmp_path)
    assert result["valid"] is False
    assert any("blob mismatch" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("ref", [None, "refs/tags/WRONG"])
def test_official_source_rejects_missing_or_wrong_tag_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ref: str | None,
) -> None:
    _write_valid_source_fixture(tmp_path, monkeypatch)
    tag_path = tmp_path / "tag-ref.json"
    tag = json.loads(tag_path.read_text(encoding="utf-8"))
    if ref is None:
        tag.pop("ref")
    else:
        tag["ref"] = ref
    tag_path.write_text(json.dumps(tag), encoding="utf-8")

    result = audit_official_source(tmp_path)
    assert result["valid"] is False
    assert any("exact reviewed tag ref" in blocker for blocker in result["blockers"])


def test_official_source_rejects_wrong_repository_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_source_fixture(tmp_path, monkeypatch)
    tag_path = tmp_path / "tag-ref.json"
    tag = json.loads(tag_path.read_text(encoding="utf-8"))
    tag["url"] = "https://api.github.com/repos/Other/Repo/git/refs/tags/v3.5.0"
    tag_path.write_text(json.dumps(tag), encoding="utf-8")

    result = audit_official_source(tmp_path)
    assert result["valid"] is False
    assert any("official repository ref URL" in blocker for blocker in result["blockers"])


def test_official_source_rejects_duplicate_tag_response_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_source_fixture(tmp_path, monkeypatch)
    tag_path = tmp_path / "tag-ref.json"
    text = tag_path.read_text(encoding="utf-8").replace(
        '"ref": "refs/tags/v3.5.0"',
        '"ref": "refs/tags/WRONG", "ref": "refs/tags/v3.5.0"',
    )
    tag_path.write_text(text, encoding="utf-8")

    result = audit_official_source(tmp_path)

    assert result["valid"] is False
    assert any("duplicate JSON object key: ref" in blocker for blocker in result["blockers"])


def test_official_source_rejects_nested_duplicate_commit_response_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_source_fixture(tmp_path, monkeypatch)
    commit_path = tmp_path / "commit.json"
    text = commit_path.read_text(encoding="utf-8").replace(
        '"tree": {"sha": "' + "b" * 40 + '"}',
        '"tree": {"sha": "WRONG", "sha": "' + "b" * 40 + '"}',
    )
    commit_path.write_text(text, encoding="utf-8")

    result = audit_official_source(tmp_path)

    assert result["valid"] is False
    assert any("duplicate JSON object key: sha" in blocker for blocker in result["blockers"])
