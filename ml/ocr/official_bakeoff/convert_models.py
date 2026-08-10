# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Convert audited PP-OCRv5 Paddle models and prove raw CPU tensor parity."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Sequence

from ml.ocr.official_bakeoff.audit_archives import CANDIDATES, PINNED_COMMIT, PINNED_TAG


OPSET_VERSION = 11
PARITY_CASES = 16
PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE = 1e-4
RNG_SEED = 20260805
EXPECTED_PYTHON = (3, 11, 9)
EXPECTED_PACKAGES = {
    "numpy": "2.4.6",
    "onnx": "1.17.0",
    "onnxruntime": "1.22.0",
    "paddle2onnx": "2.0.2rc3",
    "paddlepaddle": "3.0.0.dev20250426",
}
BOOTSTRAP_PACKAGES = {"pip": "24.0", "setuptools": "65.5.0"}
EXPECTED_PYTHON_INSTALLER_SHA256 = (
    "5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde"
)
EXPECTED_PYTHON_INSTALLER_BYTES = 26_216_840
EXPECTED_VENV_PYTHON_SHA256 = (
    "21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082"
)
EXPECTED_CONVERTER_LAUNCHER_SHA256 = (
    "603f461f7febf50d6f46129f096a9bf55220f7c6e0b11a4b36394859f135004a"
)
REQUIRED_AUDIT_FLAGS = (
    "artifact_level_redistribution_proven",
    "conversion_permitted",
    "hashes_valid",
    "official_archives_only",
    "official_model_repository_terms_proven",
    "source_provenance_valid",
)
FAILURE_LOG_MARKERS = (
    "failed to convert",
    "traceback (most recent call last)",
    "onnx check failed",
)
EXPECTED_WARNING_COUNTS = {
    "PP-OCRv5_mobile_det": {
        "python_no_ccache": 1,
        "python_warning_source": 1,
        "slice_strides_tensor": 0,
        "slice_strides_tensor_list": 0,
    },
    "en_PP-OCRv5_mobile_rec": {
        "python_no_ccache": 1,
        "python_warning_source": 1,
        "slice_strides_tensor": 14,
        "slice_strides_tensor_list": 7,
    },
}
MODEL_INPUT_SHAPES = {
    "PP-OCRv5_mobile_det": (
        (1, 3, 32, 32),
        (1, 3, 32, 64),
        (1, 3, 64, 96),
        (1, 3, 96, 128),
    ),
    "en_PP-OCRv5_mobile_rec": (
        (1, 3, 48, 16),
        (1, 3, 48, 64),
        (1, 3, 48, 160),
        (1, 3, 48, 320),
        (1, 3, 48, 321),
    ),
}
RunnerFactory = Callable[[Path], tuple[str, list[str], Callable[[Any], list[Any]]]]


class ConversionGateError(ValueError):
    """Raised when conversion evidence fails closed."""


class DuplicateJsonKeyError(ConversionGateError):
    """Raised when reviewed JSON repeats an object key."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionGateError(f"Cannot read reviewed JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConversionGateError(f"Reviewed JSON must be an object: {path}")
    return value


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def parse_locked_requirements(path: Path) -> dict[str, dict[str, Any]]:
    try:
        physical_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ConversionGateError(f"Cannot read conversion lock {path}: {error}") from error
    logical_lines: list[str] = []
    current = ""
    for physical in physical_lines:
        stripped = physical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        fragment = stripped[:-1].strip() if continued else stripped
        current = f"{current} {fragment}".strip()
        if not continued:
            logical_lines.append(current)
            current = ""
    if current:
        raise ConversionGateError("Conversion lock ends with an incomplete continuation.")

    result: dict[str, dict[str, Any]] = {}
    requirement_pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)(?:\s+.*)?$")
    hash_pattern = re.compile(r"--hash=sha256:([a-f0-9]{64})(?=\s|$)")
    for line in logical_lines:
        match = requirement_pattern.fullmatch(line)
        if match is None:
            raise ConversionGateError(f"Unsupported conversion-lock entry: {line}")
        name = canonical_package_name(match.group(1))
        if name in result:
            raise ConversionGateError(f"Duplicate conversion-lock distribution: {name}")
        hashes = sorted(set(hash_pattern.findall(line)))
        if not hashes:
            raise ConversionGateError(f"Conversion-lock distribution has no SHA-256: {name}")
        result[name] = {"version": match.group(2), "allowed_sha256": hashes}
    if len(result) != 27:
        raise ConversionGateError(
            f"Conversion lock must contain exactly 27 distributions, got {len(result)}"
        )
    for package, version in EXPECTED_PACKAGES.items():
        entry = result.get(package)
        if entry is None or entry["version"] != version:
            raise ConversionGateError(
                f"Conversion lock must pin {package}=={version}, got {entry}"
            )
    return result


def installed_distribution_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ConversionGateError("Installed distribution is missing its canonical name.")
        name = canonical_package_name(raw_name)
        if name in result:
            raise ConversionGateError(f"Duplicate installed distribution: {name}")
        result[name] = distribution.version
    return result


def verify_installed_distributions(
    package_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for package in sorted(package_names):
        distribution = importlib.metadata.distribution(package)
        files = distribution.files
        if not files:
            raise ConversionGateError(f"Installed distribution has no RECORD inventory: {package}")
        verified = 0
        for entry in files:
            if entry.hash is None:
                continue
            path = Path(entry.locate())
            if not path.is_file():
                raise ConversionGateError(f"Installed distribution file is missing: {path}")
            digest = sha256()
            if entry.hash.mode != "sha256":
                raise ConversionGateError(
                    f"Unsupported RECORD hash mode for {package}/{entry}: {entry.hash.mode}"
                )
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
            if actual != entry.hash.value:
                raise ConversionGateError(
                    f"Installed distribution hash mismatch: {package}/{entry}"
                )
            verified += 1
        if verified == 0:
            raise ConversionGateError(f"Installed distribution has no hashed files: {package}")
        metadata = distribution.metadata
        result[package] = {
            "canonical_name": metadata.get("Name"),
            "version": distribution.version,
            "license_expression": metadata.get("License-Expression"),
            "license": metadata.get("License"),
            "record_files_verified": verified,
        }
    return result


def validate_archive_audit(audit: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if audit.get("pinned_tag") != PINNED_TAG:
        errors.append(f"pinned_tag must be {PINNED_TAG}")
    if audit.get("pinned_commit") != PINNED_COMMIT:
        errors.append(f"pinned_commit must be {PINNED_COMMIT}")
    if audit.get("status") != "eligible_for_conversion":
        errors.append("status must be eligible_for_conversion")
    for flag in dict.fromkeys(REQUIRED_AUDIT_FLAGS):
        if audit.get(flag) is not True:
            errors.append(f"{flag} must be true")
    if audit.get("blockers") != []:
        errors.append("blockers must be an empty array")

    audits = audit.get("audits")
    if not isinstance(audits, list) or len(audits) != len(CANDIDATES):
        errors.append(f"audits must contain exactly {len(CANDIDATES)} entries")
        audits = []

    observed: dict[str, dict[str, Any]] = {}
    for item in audits:
        if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
            errors.append("every audit entry must contain a candidate object")
            continue
        candidate = item["candidate"]
        model_id = candidate.get("model_id")
        if not isinstance(model_id, str) or model_id in observed:
            errors.append("audit candidate model IDs must be unique strings")
            continue
        observed[model_id] = item

    for expected in CANDIDATES:
        item = observed.get(expected.model_id)
        if item is None:
            errors.append(f"missing audit for {expected.model_id}")
            continue
        candidate = item["candidate"]
        expected_candidate = asdict(expected)
        if candidate != expected_candidate:
            errors.append(f"candidate evidence differs for {expected.model_id}")
        if item.get("archive_sha256") != expected.archive_sha256:
            errors.append(f"archive hash differs for {expected.model_id}")
        if item.get("archive_hash_matches") is not True:
            errors.append(f"archive hash is not verified for {expected.model_id}")
        if item.get("member_inventory_matches") is not True:
            errors.append(f"member inventory is not verified for {expected.model_id}")

    if set(observed) != {candidate.model_id for candidate in CANDIDATES}:
        errors.append("audit contains missing or unexpected candidate IDs")
    if errors:
        raise ConversionGateError("Archive audit is not conversion eligible: " + "; ".join(errors))
    return {
        "path_binding": "exact-candidate-evidence",
        "pinned_tag": PINNED_TAG,
        "pinned_commit": PINNED_COMMIT,
        "candidate_count": len(CANDIDATES),
        "status": "eligible_for_conversion",
    }


def _read_authenticode_signature(installer: Path) -> dict[str, str]:
    command = (
        "$ErrorActionPreference='Stop'; "
        "Import-Module -Force -Name $env:GRAPHREADER_SECURITY_MODULE; "
        "$signature=Get-AuthenticodeSignature -LiteralPath "
        "$env:GRAPHREADER_PYTHON_INSTALLER; "
        "if ($null -eq $signature) { throw 'Authenticode signature was not returned.' }; "
        "[PSCustomObject]@{status=[string]$signature.Status; "
        "signer=[string]$signature.SignerCertificate.Subject} | "
        "ConvertTo-Json -Compress"
    )
    encoded_command = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    environment = os.environ.copy()
    environment["GRAPHREADER_PYTHON_INSTALLER"] = str(installer)
    system_root = environment.get("SystemRoot") or environment.get("SYSTEMROOT")
    if not system_root:
        raise ConversionGateError("SystemRoot is unavailable for Authenticode inspection.")
    environment["GRAPHREADER_SECURITY_MODULE"] = str(
        Path(system_root)
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "Modules"
        / "Microsoft.PowerShell.Security"
        / "Microsoft.PowerShell.Security.psd1"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded_command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise ConversionGateError(
            f"Authenticode inspection failed with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(
            completed.stdout,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, DuplicateJsonKeyError) as error:
        raise ConversionGateError(f"Invalid Authenticode inspection output: {error}") from error
    if not isinstance(value, dict):
        raise ConversionGateError("Authenticode inspection output must be an object.")
    return {"status": str(value.get("status", "")), "signer": str(value.get("signer", ""))}


def validate_python_intake(
    intake_path: Path,
    installer: Path,
    executable: Path,
    converter: Path,
    *,
    signature: dict[str, str] | None = None,
) -> dict[str, Any]:
    intake = load_strict_json(intake_path)
    if intake.get("schema") != "graphreader.local-toolchain-intake.v1":
        raise ConversionGateError("Toolchain intake schema is not recognized.")
    if intake.get("status") != "selected_toolchain_import_passed_conversion_pending":
        raise ConversionGateError("Toolchain intake is not ready for controlled conversion.")
    items = intake.get("items")
    if not isinstance(items, list):
        raise ConversionGateError("Toolchain intake items must be an array.")
    python_items = [
        item for item in items
        if isinstance(item, dict) and item.get("name") == "CPython Windows x64 installer"
    ]
    if len(python_items) != 1:
        raise ConversionGateError("Toolchain intake must contain exactly one CPython installer.")
    python_item = python_items[0]
    expected_python_item = {
        "version": "3.11.9",
        "source": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe",
        "source_authority": "Python Software Foundation",
        "license": "PSF License Agreement",
        "expected_sha256": EXPECTED_PYTHON_INSTALLER_SHA256,
        "downloaded_size": EXPECTED_PYTHON_INSTALLER_BYTES,
        "authenticode_status": "Valid",
        "authenticode_signer": "Python Software Foundation",
    }
    for key, expected in expected_python_item.items():
        if python_item.get(key) != expected:
            raise ConversionGateError(
                f"Toolchain intake CPython {key} differs: {python_item.get(key)}"
            )

    selected = intake.get("third_converter_candidate")
    if not isinstance(selected, dict):
        raise ConversionGateError("Toolchain intake lacks the selected converter candidate.")
    selected_expected = {
        "converter_version": EXPECTED_PACKAGES["paddle2onnx"],
        "converter_sha256": (
            "ed678cd40d14efdec30af46c01962a4ccbf8017ebb35ec01b4a9b6e2ceb24077"
        ),
        "paddle_version": EXPECTED_PACKAGES["paddlepaddle"],
        "paddle_expected_sha256": (
            "f62aaab2bd8d3ad4f4f7781bdeed43403546057b7afcdc10a4b33847b2617f1f"
        ),
        "venv_python_bytes": executable.stat().st_size,
        "venv_python_sha256": EXPECTED_VENV_PYTHON_SHA256,
        "converter_launcher_bytes": converter.stat().st_size,
        "converter_launcher_sha256": EXPECTED_CONVERTER_LAUNCHER_SHA256,
        "result": "toolchain import and pip check passed",
    }
    for key, expected in selected_expected.items():
        if selected.get(key) != expected:
            raise ConversionGateError(
                f"Toolchain intake selected candidate {key} differs: {selected.get(key)}"
            )

    installer_path = installer.resolve(strict=True)
    executable_path = executable.resolve(strict=True)
    converter_path = converter.resolve(strict=True)
    if installer_path.stat().st_size != EXPECTED_PYTHON_INSTALLER_BYTES:
        raise ConversionGateError("CPython installer byte length differs from the tracked lock.")
    if hash_file(installer_path) != EXPECTED_PYTHON_INSTALLER_SHA256:
        raise ConversionGateError("CPython installer SHA-256 differs from the tracked lock.")
    if hash_file(executable_path) != EXPECTED_VENV_PYTHON_SHA256:
        raise ConversionGateError("Virtual-environment Python SHA-256 differs from the tracked lock.")
    if hash_file(converter_path) != EXPECTED_CONVERTER_LAUNCHER_SHA256:
        raise ConversionGateError("Converter launcher SHA-256 differs from the tracked lock.")
    actual_signature = signature or _read_authenticode_signature(installer_path)
    if actual_signature.get("status") != "Valid":
        raise ConversionGateError(
            f"CPython installer Authenticode status is {actual_signature.get('status')}"
        )
    signer = actual_signature.get("signer", "")
    if "CN=Python Software Foundation" not in signer or "O=Python Software Foundation" not in signer:
        raise ConversionGateError(f"CPython installer signer is not the PSF: {signer}")
    return {
        "intake": {
            "path": str(intake_path.resolve()),
            "bytes": intake_path.stat().st_size,
            "sha256": hash_file(intake_path),
            "schema": intake["schema"],
            "status": intake["status"],
        },
        "python_installer": {
            "path": str(installer_path),
            "bytes": installer_path.stat().st_size,
            "sha256": hash_file(installer_path),
            "authenticode": actual_signature,
        },
        "venv_python": {
            "path": str(executable_path),
            "bytes": executable_path.stat().st_size,
            "sha256": hash_file(executable_path),
        },
        "converter_launcher": {
            "path": str(converter_path),
            "bytes": converter_path.stat().st_size,
            "sha256": hash_file(converter_path),
        },
    }


def verify_source_models(source_root: Path) -> dict[str, dict[str, Any]]:
    source_root = source_root.resolve(strict=True)
    expected_directories = {f"{candidate.model_id}_infer" for candidate in CANDIDATES}
    observed_directories = {path.name for path in source_root.iterdir() if path.is_dir()}
    observed_root_files = [path.name for path in source_root.iterdir() if path.is_file()]
    if observed_root_files or observed_directories != expected_directories:
        raise ConversionGateError(
            "Extracted source root contains missing or unexpected entries: "
            f"directories={sorted(observed_directories)}, files={sorted(observed_root_files)}"
        )

    result: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        model_root = source_root / f"{candidate.model_id}_infer"
        expected_files = {
            member_path.split("/")[-1]: digest
            for member_path, digest in candidate.member_sha256.items()
        }
        observed = sorted(path.name for path in model_root.iterdir() if path.is_file())
        directories = sorted(path.name for path in model_root.iterdir() if path.is_dir())
        if observed != sorted(expected_files) or directories:
            raise ConversionGateError(
                f"Unexpected extracted inventory for {candidate.model_id}: "
                f"files={observed}, directories={directories}"
            )
        files: dict[str, dict[str, Any]] = {}
        for name, expected_hash in expected_files.items():
            path = model_root / name
            actual_hash = hash_file(path)
            if actual_hash != expected_hash:
                raise ConversionGateError(
                    f"Source hash mismatch for {candidate.model_id}/{name}: {actual_hash}"
                )
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
            }
        result[candidate.model_id] = {
            "directory": str(model_root),
            "files": files,
        }
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def validate_toolchain(
    toolchain_root: Path,
    converter: Path,
    *,
    lock_path: Path | None = None,
    locked_requirements: dict[str, dict[str, Any]] | None = None,
    python_version: tuple[int, int, int] | None = None,
    package_versions: dict[str, str] | None = None,
    providers: Sequence[str] | None = None,
    executable: Path | None = None,
    paddle_runtime_version: str | None = None,
    paddle_module_path: Path | None = None,
    distribution_integrity: dict[str, dict[str, Any]] | None = None,
    pip_check: tuple[int, str, str] | None = None,
) -> dict[str, Any]:
    root = toolchain_root.resolve(strict=True)
    converter_path = converter.resolve(strict=True)
    executable_path = (executable or Path(sys.executable)).resolve(strict=True)
    if not _is_within(converter_path, root):
        raise ConversionGateError("Converter must be inside the selected toolchain root.")
    if not _is_within(executable_path, root):
        raise ConversionGateError(
            "The harness must run with Python inside the selected toolchain root."
        )

    actual_python = python_version or tuple(sys.version_info[:3])
    requirements_path = lock_path or Path(__file__).resolve().with_name(
        "requirements-conversion.txt"
    )
    locked = (
        locked_requirements
        if locked_requirements is not None
        else parse_locked_requirements(requirements_path)
    )
    expected_environment = {
        **{package: str(entry["version"]) for package, entry in locked.items()},
        **BOOTSTRAP_PACKAGES,
    }
    versions = (
        package_versions
        if package_versions is not None
        else installed_distribution_versions()
    )
    errors: list[str] = []
    if actual_python != EXPECTED_PYTHON:
        errors.append(f"Python must be {'.'.join(map(str, EXPECTED_PYTHON))}, got {actual_python}")
    for package, expected in expected_environment.items():
        if versions.get(package) != expected:
            errors.append(f"{package} must be {expected}, got {versions.get(package)}")
    unexpected = sorted(set(versions) - set(expected_environment))
    missing = sorted(set(expected_environment) - set(versions))
    if unexpected:
        errors.append(f"unexpected installed distributions: {unexpected}")
    if missing:
        errors.append(f"missing installed distributions: {missing}")

    if paddle_runtime_version is None:
        import paddle

        actual_paddle_runtime = paddle.__version__
        actual_paddle_module_path = Path(paddle.__file__).resolve(strict=True)
    else:
        actual_paddle_runtime = paddle_runtime_version
        actual_paddle_module_path = (paddle_module_path or executable_path).resolve(strict=True)
    if actual_paddle_runtime != "3.0.0":
        errors.append(f"Paddle runtime must report 3.0.0, got {actual_paddle_runtime}")
    if not _is_within(actual_paddle_module_path, root):
        errors.append("Imported Paddle module must be inside the selected toolchain root")

    if providers is None:
        import onnxruntime as ort

        actual_providers = ort.get_available_providers()
    else:
        actual_providers = list(providers)
    if "CPUExecutionProvider" not in actual_providers:
        errors.append("ONNX Runtime CPUExecutionProvider is required")
    integrity = (
        distribution_integrity
        if distribution_integrity is not None
        else verify_installed_distributions(tuple(expected_environment))
    )
    if set(integrity) != set(expected_environment):
        errors.append("Installed distribution integrity inventory is incomplete")
    if pip_check is None:
        checked = subprocess.run(
            [str(executable_path), "-m", "pip", "check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        pip_result = (checked.returncode, checked.stdout, checked.stderr)
    else:
        pip_result = pip_check
    if pip_result[0] != 0:
        errors.append(f"pip check failed with exit code {pip_result[0]}")
    if errors:
        raise ConversionGateError("Toolchain mismatch: " + "; ".join(errors))

    return {
        "python": {
            "executable": str(executable_path),
            "version": ".".join(map(str, actual_python)),
        },
        "packages": dict(sorted(versions.items())),
        "locked_distributions": locked,
        "bootstrap_distributions": BOOTSTRAP_PACKAGES,
        "installed_distribution_integrity": integrity,
        "paddle_runtime_version": actual_paddle_runtime,
        "paddle_module_path": str(actual_paddle_module_path),
        "pip_check": {
            "return_code": pip_result[0],
            "stdout": pip_result[1].strip(),
            "stderr": pip_result[2].strip(),
        },
        "onnxruntime_available_providers": list(actual_providers),
        "converter": {
            "path": str(converter_path),
            "bytes": converter_path.stat().st_size,
            "sha256": hash_file(converter_path),
        },
    }


def inventory_wheelhouse(
    wheelhouse: Path | None,
    locked_requirements: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if wheelhouse is None:
        return []
    from packaging.utils import parse_wheel_filename

    locked = locked_requirements or parse_locked_requirements(
        Path(__file__).resolve().with_name("requirements-conversion.txt")
    )
    root = wheelhouse.resolve(strict=True)
    wheels = sorted(root.glob("*.whl"), key=lambda path: path.name.casefold())
    if not wheels:
        raise ConversionGateError("The selected wheelhouse contains no wheel files.")
    observed = {path.name: hash_file(path) for path in wheels}
    records: list[dict[str, Any]] = []
    selected_by_package: dict[str, list[dict[str, Any]]] = {}
    for path in wheels:
        try:
            distribution, version, _, _ = parse_wheel_filename(path.name)
        except Exception as error:
            raise ConversionGateError(f"Invalid wheel filename {path.name}: {error}") from error
        package = canonical_package_name(str(distribution))
        version_text = str(version)
        selected = package in locked and locked[package]["version"] == version_text
        record = {
            "name": path.name,
            "distribution": package,
            "version": version_text,
            "bytes": path.stat().st_size,
            "sha256": observed[path.name],
            "selected": selected,
        }
        records.append(record)
        if selected:
            selected_by_package.setdefault(package, []).append(record)
    errors: list[str] = []
    for package, requirement in locked.items():
        selected = selected_by_package.get(package, [])
        if len(selected) != 1:
            errors.append(f"{package}: expected one selected wheel, found {len(selected)}")
            continue
        if selected[0]["sha256"] not in requirement["allowed_sha256"]:
            errors.append(f"{package}: selected wheel hash is not allowed by the lock")
    if errors:
        raise ConversionGateError("Selected wheel evidence differs: " + "; ".join(errors))
    return records


def conversion_command(converter: Path, model_root: Path, output: Path) -> list[str]:
    return [
        str(converter),
        "--model_dir", str(model_root),
        "--model_filename", "inference.json",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(output),
        "--opset_version", str(OPSET_VERSION),
        "--enable_auto_update_opset", "False",
        "--enable_onnx_checker", "True",
        "--optimize_tool", "None",
        "--enable_verbose", "True",
    ]


def remove_generated_onnx(output_root: Path) -> None:
    for root in (output_root, output_root / "repeat"):
        for candidate in CANDIDATES:
            (root / f"{candidate.model_id}.onnx").unlink(missing_ok=True)


def clear_previous_authority(output_root: Path, report_path: Path) -> None:
    report_path.unlink(missing_ok=True)
    remove_generated_onnx(output_root)
    for root in (output_root, output_root / "repeat"):
        for candidate in CANDIDATES:
            (root / f"{candidate.model_id}.conversion.log").unlink(missing_ok=True)


def validate_converter_result(returncode: int, stdout: str, stderr: str, output: Path) -> None:
    combined = f"{stdout}\n{stderr}".casefold()
    markers = [marker for marker in FAILURE_LOG_MARKERS if marker in combined]
    if returncode != 0:
        raise ConversionGateError(f"Converter exited with code {returncode}.")
    if markers:
        raise ConversionGateError(f"Converter log contains failure marker(s): {markers}")
    if not output.is_file() or output.stat().st_size == 0:
        raise ConversionGateError("Converter did not produce a nonempty ONNX file.")


def analyze_converter_warnings(model_id: str, stdout: str, stderr: str) -> dict[str, Any]:
    expected = EXPECTED_WARNING_COUNTS.get(model_id)
    if expected is None:
        raise ConversionGateError(f"No reviewed converter-warning profile for {model_id}.")
    counts = {key: 0 for key in expected}
    unknown: list[str] = []
    for raw_line in f"{stdout}\n{stderr}".splitlines():
        line = raw_line.strip()
        folded = line.casefold()
        if "warning" not in folded:
            continue
        if "userwarning: no ccache found" in folded:
            counts["python_no_ccache"] += 1
        elif line == "warnings.warn(warning_message)":
            counts["python_warning_source"] += 1
        elif (
            "[Paddle2ONNX] [WARNING]" in line
            and "'StridesTensorList'" in line
            and "pd_op.slice" in line
        ):
            counts["slice_strides_tensor_list"] += 1
        elif (
            "[Paddle2ONNX] [WARNING]" in line
            and "'StridesTensor'" in line
            and "pd_op.slice" in line
        ):
            counts["slice_strides_tensor"] += 1
        else:
            unknown.append(line)
    if unknown or counts != expected:
        raise ConversionGateError(
            f"Converter warning profile differs for {model_id}: "
            f"counts={counts}, expected={expected}, unknown={unknown}"
        )
    return {
        "counts": counts,
        "reviewed": True,
        "disposition": (
            "The ccache warning is build-cache-only. Recognition slice metadata "
            "warnings are retained and bounded by exact counts, ONNX full checking, "
            "byte reproducibility, and raw CPU parity across dynamic widths."
        ),
    }


def validate_onnx_model(path: Path) -> dict[str, Any]:
    import onnx

    try:
        model = onnx.load(str(path), load_external_data=False)
        external_tensors: list[str] = []

        def inspect_tensor(tensor: Any, location: str) -> None:
            if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
                external_tensors.append(location)

        def inspect_sparse(sparse: Any, location: str) -> None:
            inspect_tensor(sparse.values, f"{location}.values")
            inspect_tensor(sparse.indices, f"{location}.indices")

        def inspect_graph(graph: Any, location: str) -> None:
            for index, tensor in enumerate(graph.initializer):
                inspect_tensor(tensor, f"{location}.initializer[{index}]")
            for index, sparse in enumerate(graph.sparse_initializer):
                inspect_sparse(sparse, f"{location}.sparse_initializer[{index}]")
            for node_index, node in enumerate(graph.node):
                for attribute_index, attribute in enumerate(node.attribute):
                    attribute_location = (
                        f"{location}.node[{node_index}].attribute[{attribute_index}]"
                    )
                    if attribute.HasField("t"):
                        inspect_tensor(attribute.t, f"{attribute_location}.t")
                    for tensor_index, tensor in enumerate(attribute.tensors):
                        inspect_tensor(tensor, f"{attribute_location}.tensors[{tensor_index}]")
                    if attribute.HasField("sparse_tensor"):
                        inspect_sparse(
                            attribute.sparse_tensor,
                            f"{attribute_location}.sparse_tensor",
                        )
                    for sparse_index, sparse in enumerate(attribute.sparse_tensors):
                        inspect_sparse(
                            sparse,
                            f"{attribute_location}.sparse_tensors[{sparse_index}]",
                        )
                    if attribute.HasField("g"):
                        inspect_graph(attribute.g, f"{attribute_location}.g")
                    for graph_index, nested in enumerate(attribute.graphs):
                        inspect_graph(nested, f"{attribute_location}.graphs[{graph_index}]")

        inspect_graph(model.graph, "graph")
        if external_tensors:
            raise ConversionGateError(
                f"External ONNX tensor data is prohibited: {external_tensors}"
            )
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:
        raise ConversionGateError(f"ONNX checker rejected {path}: {error}") from error

    opsets = {entry.domain: entry.version for entry in model.opset_import}
    if opsets.get("") != OPSET_VERSION:
        raise ConversionGateError(
            f"Default ONNX opset must be {OPSET_VERSION}, got {opsets.get('')}"
        )
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ConversionGateError(
            "Converted OCR model must expose exactly one graph input and one graph output."
        )

    def describe(value: Any) -> dict[str, Any]:
        tensor = value.type.tensor_type
        dimensions: list[int | str | None] = []
        for dimension in tensor.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(str(dimension.dim_param))
            else:
                dimensions.append(None)
        return {
            "name": value.name,
            "element_type": int(tensor.elem_type),
            "shape": dimensions,
        }

    return {
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
        "opsets": dict(sorted(opsets.items())),
        "input": describe(model.graph.input[0]),
        "output": describe(model.graph.output[0]),
        "checker": "passed-full-check",
        "external_tensor_count": 0,
    }


def maximum_absolute_error(reference: Any, candidate: Any) -> float:
    import numpy as np

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ConversionGateError(
            f"Parity output shape mismatch: {reference_array.shape} != {candidate_array.shape}"
        )
    if reference_array.dtype != candidate_array.dtype:
        raise ConversionGateError(
            f"Parity output dtype mismatch: {reference_array.dtype} != {candidate_array.dtype}"
        )
    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise ConversionGateError("Parity outputs must contain only finite values.")
    return float(np.max(np.abs(reference_array - candidate_array)))


def _paddle_runner(model_root: Path) -> tuple[str, list[str], Callable[[Any], list[Any]]]:
    import paddle.inference as paddle_inference

    config = paddle_inference.Config(
        str(model_root / "inference.json"),
        str(model_root / "inference.pdiparams"),
    )
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(1)
    config.disable_glog_info()
    config.disable_mkldnn()
    config.switch_ir_optim(True)
    predictor = paddle_inference.create_predictor(config)
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    if len(input_names) != 1 or len(output_names) != 1:
        raise ConversionGateError("Paddle model must expose exactly one input and one output.")

    def run(value: Any) -> list[Any]:
        input_handle = predictor.get_input_handle(input_names[0])
        input_handle.reshape(value.shape)
        input_handle.copy_from_cpu(value)
        run_result = predictor.run()
        if run_result is False:
            raise ConversionGateError("Paddle CPU inference returned failure.")
        return [predictor.get_output_handle(name).copy_to_cpu() for name in output_names]

    return input_names[0], output_names, run


def _onnx_runner(path: Path) -> tuple[str, list[str], Callable[[Any], list[Any]]]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ConversionGateError(
            f"Parity session is not CPU-only: {session.get_providers()}"
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ConversionGateError("ONNX model must expose exactly one input and one output.")
    input_name = inputs[0].name
    output_names = [value.name for value in outputs]

    def run(value: Any) -> list[Any]:
        return session.run(output_names, {input_name: value})

    return input_name, output_names, run


def run_cpu_parity(
    model_id: str,
    model_root: Path,
    onnx_path: Path,
    *,
    cases: int = PARITY_CASES,
    maximum_allowed: float = PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE,
    paddle_factory: RunnerFactory = _paddle_runner,
    onnx_factory: RunnerFactory = _onnx_runner,
) -> dict[str, Any]:
    import numpy as np

    if cases < PARITY_CASES:
        raise ConversionGateError(f"Parity requires at least {PARITY_CASES} cases per model.")
    shapes = MODEL_INPUT_SHAPES.get(model_id)
    if not shapes:
        raise ConversionGateError(f"No preregistered parity input shape for {model_id}.")
    paddle_input, paddle_outputs, run_paddle = paddle_factory(model_root)
    onnx_input, onnx_outputs, run_onnx = onnx_factory(onnx_path)
    if len(paddle_outputs) != len(onnx_outputs):
        raise ConversionGateError("Paddle and ONNX output counts differ.")

    rng = np.random.default_rng(RNG_SEED)
    records: list[dict[str, Any]] = []
    global_maximum = 0.0
    for index in range(cases):
        shape = shapes[index % len(shapes)]
        sample = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        paddle_values = run_paddle(sample)
        onnx_values = run_onnx(sample)
        if len(paddle_values) != len(onnx_values):
            raise ConversionGateError("Paddle and ONNX runtime output counts differ.")
        errors = [
            maximum_absolute_error(reference, candidate)
            for reference, candidate in zip(paddle_values, onnx_values, strict=True)
        ]
        case_maximum = max(errors, default=0.0)
        global_maximum = max(global_maximum, case_maximum)
        records.append({
            "case": index,
            "input_shape": list(shape),
            "input_sha256": sha256(sample.tobytes(order="C")).hexdigest(),
            "maximum_absolute_difference": case_maximum,
            "output_maximum_absolute_differences": errors,
        })
    if global_maximum > maximum_allowed:
        raise ConversionGateError(
            f"CPU parity failed for {model_id}: {global_maximum} > {maximum_allowed}"
        )
    return {
        "cases": cases,
        "input_shapes": [list(shape) for shape in shapes],
        "input_dtype": "float32",
        "rng_seed": RNG_SEED,
        "paddle_input_name": paddle_input,
        "paddle_output_names": paddle_outputs,
        "onnx_input_name": onnx_input,
        "onnx_output_names": onnx_outputs,
        "provider": "CPUExecutionProvider",
        "maximum_allowed": maximum_allowed,
        "maximum_absolute_difference": global_maximum,
        "passed": True,
        "records": records,
    }


def convert_once(
    candidate: Any,
    source_root: Path,
    output_root: Path,
    converter: Path,
) -> dict[str, Any]:
    model_root = source_root / f"{candidate.model_id}_infer"
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{candidate.model_id}.onnx"
    log_path = output_root / f"{candidate.model_id}.conversion.log"
    output.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    command = conversion_command(converter, model_root, output)
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(output_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration_ms = (perf_counter() - started) * 1000.0
    log = f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}"
    log_path.write_text(log, encoding="utf-8", newline="\n")
    try:
        validate_converter_result(completed.returncode, completed.stdout, completed.stderr, output)
        warning_evidence = analyze_converter_warnings(
            candidate.model_id,
            completed.stdout,
            completed.stderr,
        )
        onnx_evidence = validate_onnx_model(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {
        "model_id": candidate.model_id,
        "source_archive_sha256": candidate.archive_sha256,
        "command": command,
        "return_code": completed.returncode,
        "duration_ms": duration_ms,
        "log": {
            "path": str(log_path),
            "bytes": log_path.stat().st_size,
            "sha256": hash_file(log_path),
        },
        "warnings": warning_evidence,
        "onnx": {"path": str(output), **onnx_evidence},
        "status": "conversion_and_onnx_check_passed",
    }


def convert_one(
    candidate: Any,
    source_root: Path,
    output_root: Path,
    converter: Path,
) -> dict[str, Any]:
    try:
        primary = convert_once(candidate, source_root, output_root, converter)
        repeat = convert_once(candidate, source_root, output_root / "repeat", converter)
        primary_hash = primary["onnx"]["sha256"]
        repeat_hash = repeat["onnx"]["sha256"]
        if primary_hash != repeat_hash:
            raise ConversionGateError(
                f"Conversion is not byte reproducible for {candidate.model_id}: "
                f"{primary_hash} != {repeat_hash}"
            )
        model_root = source_root / f"{candidate.model_id}_infer"
        parity = run_cpu_parity(
            candidate.model_id,
            model_root,
            Path(primary["onnx"]["path"]),
        )
    except Exception:
        (output_root / f"{candidate.model_id}.onnx").unlink(missing_ok=True)
        (output_root / "repeat" / f"{candidate.model_id}.onnx").unlink(missing_ok=True)
        raise
    return {
        **primary,
        "cpu_parity": parity,
        "reproducibility": {
            "independent_conversions": 2,
            "byte_identical": True,
            "primary_sha256": primary_hash,
            "repeat_sha256": repeat_hash,
            "repeat": repeat,
        },
        "status": "reproducible_conversion_and_raw_tensor_parity_passed",
    }


def build_report(
    audit_path: Path,
    source_root: Path,
    output_root: Path,
    toolchain_root: Path,
    converter: Path,
    wheelhouse: Path | None,
    intake_path: Path,
    python_installer: Path,
) -> dict[str, Any]:
    audit = load_strict_json(audit_path)
    audit_evidence = validate_archive_audit(audit)
    source_evidence = verify_source_models(source_root)
    intake_evidence = validate_python_intake(
        intake_path,
        python_installer,
        Path(sys.executable),
        converter,
    )
    requirements_path = Path(__file__).resolve().with_name("requirements-conversion.txt")
    locked_requirements = parse_locked_requirements(requirements_path)
    toolchain = validate_toolchain(
        toolchain_root,
        converter,
        lock_path=requirements_path,
        locked_requirements=locked_requirements,
    )
    wheel_inventory = inventory_wheelhouse(wheelhouse, locked_requirements)
    output_root.mkdir(parents=True, exist_ok=True)
    conversions = [
        convert_one(candidate, source_root.resolve(), output_root.resolve(), converter.resolve())
        for candidate in CANDIDATES
    ]
    harness_root = Path(__file__).resolve().parent
    provenance_files = [
        harness_root / "convert_models.py",
        harness_root / "requirements-conversion.in",
        harness_root / "requirements-conversion.txt",
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "conversion_and_raw_tensor_parity_passed",
        "approval_scope": "conversion-only",
        "production_approved": False,
        "release_ready": False,
        "audit": {
            "path": str(audit_path.resolve()),
            "bytes": audit_path.stat().st_size,
            "sha256": hash_file(audit_path),
            **audit_evidence,
        },
        "conversion_provenance_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in provenance_files
        ],
        "sources": source_evidence,
        "toolchain": toolchain,
        "toolchain_intake": intake_evidence,
        "wheelhouse": wheel_inventory,
        "conversion": {
            "opset": OPSET_VERSION,
            "auto_update_opset": False,
            "onnx_checker": True,
            "optimizer": None,
            "models": conversions,
        },
        "remaining_gates": [
            "public validation and sealed-test OCR metrics",
            "role accuracy and text-to-marker exclusion",
            "artifact notice and approved model manifest",
            "production resolver and packaging discovery",
            "production workflow and clean-machine validation",
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--toolchain-root", required=True, type=Path)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--toolchain-intake", required=True, type=Path)
    parser.add_argument("--python-installer", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        clear_previous_authority(args.output, args.report)
        report = build_report(
            args.audit,
            args.source,
            args.output,
            args.toolchain_root,
            args.converter,
            args.wheelhouse,
            args.toolchain_intake,
            args.python_installer,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (ConversionGateError, OSError) as error:
        try:
            remove_generated_onnx(args.output)
        except OSError as cleanup_error:
            print(f"BLOCKED CLEANUP: {cleanup_error}", file=sys.stderr)
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"PASS: conversion and raw CPU parity evidence written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
