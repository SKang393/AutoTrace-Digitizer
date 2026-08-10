# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ml.ocr.official_bakeoff import convert_models as conversion


@dataclass(frozen=True)
class FakeCandidate:
    model_id: str
    archive_name: str
    url: str
    archive_sha256: str
    member_sha256: dict[str, str]


def _candidate(files: dict[str, bytes] | None = None) -> FakeCandidate:
    payloads = files or {
        "model_infer/inference.json": b"graph",
        "model_infer/inference.pdiparams": b"parameters",
        "model_infer/inference.yml": b"config",
    }
    return FakeCandidate(
        model_id="model",
        archive_name="model.tar",
        url="https://official.invalid/model.tar",
        archive_sha256="a" * 64,
        member_sha256={name: sha256(value).hexdigest() for name, value in payloads.items()},
    )


def _audit(candidate: FakeCandidate) -> dict[str, object]:
    return {
        "pinned_tag": conversion.PINNED_TAG,
        "pinned_commit": conversion.PINNED_COMMIT,
        "status": "eligible_for_conversion",
        "artifact_level_redistribution_proven": True,
        "conversion_permitted": True,
        "hashes_valid": True,
        "official_archives_only": True,
        "official_model_repository_terms_proven": True,
        "source_provenance_valid": True,
        "blockers": [],
        "audits": [
            {
                "candidate": {
                    "model_id": candidate.model_id,
                    "archive_name": candidate.archive_name,
                    "url": candidate.url,
                    "archive_sha256": candidate.archive_sha256,
                    "member_sha256": candidate.member_sha256,
                },
                "archive_sha256": candidate.archive_sha256,
                "archive_hash_matches": True,
                "member_inventory_matches": True,
            }
        ],
    }


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"allowed": true, "allowed": false}', encoding="utf-8")

    with pytest.raises(conversion.DuplicateJsonKeyError, match="allowed"):
        conversion.load_strict_json(path)


def test_conversion_lock_requires_all_hashed_exact_pins(tmp_path: Path) -> None:
    packages = dict(conversion.EXPECTED_PACKAGES)
    packages.update({f"dependency-{index}": f"1.0.{index}" for index in range(22)})
    path = tmp_path / "requirements.txt"
    path.write_text(
        "\n".join(
            f"{name}=={version} \\\n+    --hash=sha256:{index:064x}"
            for index, (name, version) in enumerate(packages.items(), start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    locked = conversion.parse_locked_requirements(path)

    assert len(locked) == 27
    assert locked["paddlepaddle"]["version"] == "3.0.0.dev20250426"
    path.write_text("paddlepaddle==3.0.0.dev20250426\n", encoding="utf-8")
    with pytest.raises(conversion.ConversionGateError, match="no SHA-256"):
        conversion.parse_locked_requirements(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversion_permitted", False),
        ("source_provenance_valid", False),
        ("blockers", ["license unresolved"]),
        ("status", "blocked"),
    ],
)
def test_archive_audit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(conversion, "CANDIDATES", (candidate,))
    evidence = _audit(candidate)
    evidence[field] = value

    with pytest.raises(conversion.ConversionGateError, match="not conversion eligible"):
        conversion.validate_archive_audit(evidence)


def test_archive_audit_rejects_modified_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(conversion, "CANDIDATES", (candidate,))
    evidence = _audit(candidate)
    evidence["audits"][0]["candidate"]["archive_sha256"] = "b" * 64  # type: ignore[index]

    with pytest.raises(conversion.ConversionGateError, match="candidate evidence differs"):
        conversion.validate_archive_audit(evidence)


def test_source_inventory_requires_exact_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    files = {
        "model_infer/inference.json": b"graph",
        "model_infer/inference.pdiparams": b"parameters",
        "model_infer/inference.yml": b"config",
    }
    candidate = _candidate(files)
    monkeypatch.setattr(conversion, "CANDIDATES", (candidate,))
    model = tmp_path / "model_infer"
    model.mkdir()
    for member, payload in files.items():
        (model / member.split("/")[-1]).write_bytes(payload)

    evidence = conversion.verify_source_models(tmp_path)

    assert evidence["model"]["files"]["inference.json"]["sha256"] == sha256(b"graph").hexdigest()
    (model / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(conversion.ConversionGateError, match="Unexpected extracted inventory"):
        conversion.verify_source_models(tmp_path)


def test_source_inventory_rejects_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(conversion, "CANDIDATES", (candidate,))
    model = tmp_path / "model_infer"
    model.mkdir()
    (model / "inference.json").write_bytes(b"changed")
    (model / "inference.pdiparams").write_bytes(b"parameters")
    (model / "inference.yml").write_bytes(b"config")

    with pytest.raises(conversion.ConversionGateError, match="Source hash mismatch"):
        conversion.verify_source_models(tmp_path)


def _toolchain_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "toolchain"
    scripts = root / "Scripts"
    scripts.mkdir(parents=True)
    converter = scripts / "paddle2onnx.exe"
    converter.write_bytes(b"converter")
    executable = root / "python.exe"
    executable.write_bytes(b"python")
    return root, converter, executable


def _distribution_integrity() -> dict[str, dict[str, object]]:
    return {package: {} for package in _environment_versions()}


def _locked_requirements() -> dict[str, dict[str, object]]:
    return {
        package: {"version": version, "allowed_sha256": ["a" * 64]}
        for package, version in conversion.EXPECTED_PACKAGES.items()
    }


def _environment_versions() -> dict[str, str]:
    return {**conversion.EXPECTED_PACKAGES, **conversion.BOOTSTRAP_PACKAGES}


def test_toolchain_requires_exact_versions_and_cpu(tmp_path: Path) -> None:
    root, converter, executable = _toolchain_paths(tmp_path)
    evidence = conversion.validate_toolchain(
        root,
        converter,
        python_version=conversion.EXPECTED_PYTHON,
        package_versions=_environment_versions(),
        locked_requirements=_locked_requirements(),
        providers=["CPUExecutionProvider"],
        executable=executable,
        paddle_runtime_version="3.0.0",
        distribution_integrity=_distribution_integrity(),
        pip_check=(0, "No broken requirements found.", ""),
    )
    assert evidence["packages"] == dict(sorted(_environment_versions().items()))

    wrong = _environment_versions()
    wrong["onnxruntime"] = "0.0.0"
    with pytest.raises(conversion.ConversionGateError, match="onnxruntime"):
        conversion.validate_toolchain(
            root,
            converter,
            python_version=conversion.EXPECTED_PYTHON,
            package_versions=wrong,
            locked_requirements=_locked_requirements(),
            providers=["CPUExecutionProvider"],
            executable=executable,
            paddle_runtime_version="3.0.0",
            distribution_integrity=_distribution_integrity(),
            pip_check=(0, "No broken requirements found.", ""),
        )
    with pytest.raises(conversion.ConversionGateError, match="CPUExecutionProvider"):
        conversion.validate_toolchain(
            root,
            converter,
            python_version=conversion.EXPECTED_PYTHON,
            package_versions=_environment_versions(),
            locked_requirements=_locked_requirements(),
            providers=["AzureExecutionProvider"],
            executable=executable,
            paddle_runtime_version="3.0.0",
            distribution_integrity=_distribution_integrity(),
            pip_check=(0, "No broken requirements found.", ""),
        )
    with pytest.raises(conversion.ConversionGateError, match="Paddle runtime"):
        conversion.validate_toolchain(
            root,
            converter,
            python_version=conversion.EXPECTED_PYTHON,
            package_versions=_environment_versions(),
            locked_requirements=_locked_requirements(),
            providers=["CPUExecutionProvider"],
            executable=executable,
            paddle_runtime_version="3.0.1",
            distribution_integrity=_distribution_integrity(),
            pip_check=(0, "No broken requirements found.", ""),
        )
    with pytest.raises(conversion.ConversionGateError, match="pip check failed"):
        conversion.validate_toolchain(
            root,
            converter,
            python_version=conversion.EXPECTED_PYTHON,
            package_versions=_environment_versions(),
            locked_requirements=_locked_requirements(),
            providers=["CPUExecutionProvider"],
            executable=executable,
            paddle_runtime_version="3.0.0",
            distribution_integrity=_distribution_integrity(),
            pip_check=(1, "", "broken dependency"),
        )
    drifted = _environment_versions()
    drifted["unexpected-runtime"] = "1.0"
    with pytest.raises(conversion.ConversionGateError, match="unexpected installed"):
        conversion.validate_toolchain(
            root,
            converter,
            python_version=conversion.EXPECTED_PYTHON,
            package_versions=drifted,
            locked_requirements=_locked_requirements(),
            providers=["CPUExecutionProvider"],
            executable=executable,
            paddle_runtime_version="3.0.0",
            distribution_integrity=_distribution_integrity(),
            pip_check=(0, "No broken requirements found.", ""),
        )


def test_toolchain_rejects_external_converter(tmp_path: Path) -> None:
    root, _, executable = _toolchain_paths(tmp_path)
    converter = tmp_path / "external.exe"
    converter.write_bytes(b"external")

    with pytest.raises(conversion.ConversionGateError, match="inside"):
        conversion.validate_toolchain(
            root,
            converter,
            python_version=conversion.EXPECTED_PYTHON,
            package_versions=_environment_versions(),
            locked_requirements=_locked_requirements(),
            providers=["CPUExecutionProvider"],
            executable=executable,
            paddle_runtime_version="3.0.0",
            distribution_integrity=_distribution_integrity(),
            pip_check=(0, "No broken requirements found.", ""),
        )


def test_python_intake_binds_installer_executable_and_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "python-installer.exe"
    executable = tmp_path / "python.exe"
    converter = tmp_path / "paddle2onnx.exe"
    installer.write_bytes(b"installer")
    executable.write_bytes(b"python")
    converter.write_bytes(b"converter")
    monkeypatch.setattr(
        conversion,
        "EXPECTED_PYTHON_INSTALLER_SHA256",
        sha256(b"installer").hexdigest(),
    )
    monkeypatch.setattr(conversion, "EXPECTED_PYTHON_INSTALLER_BYTES", len(b"installer"))
    monkeypatch.setattr(
        conversion,
        "EXPECTED_VENV_PYTHON_SHA256",
        sha256(b"python").hexdigest(),
    )
    monkeypatch.setattr(
        conversion,
        "EXPECTED_CONVERTER_LAUNCHER_SHA256",
        sha256(b"converter").hexdigest(),
    )
    intake = tmp_path / "intake.json"
    intake.write_text(
        json.dumps(
            {
                "schema": "graphreader.local-toolchain-intake.v1",
                "status": "selected_toolchain_import_passed_conversion_pending",
                "items": [
                    {
                        "name": "CPython Windows x64 installer",
                        "version": "3.11.9",
                        "source": (
                            "https://www.python.org/ftp/python/3.11.9/"
                            "python-3.11.9-amd64.exe"
                        ),
                        "source_authority": "Python Software Foundation",
                        "license": "PSF License Agreement",
                        "expected_sha256": sha256(b"installer").hexdigest(),
                        "downloaded_size": len(b"installer"),
                        "authenticode_status": "Valid",
                        "authenticode_signer": "Python Software Foundation",
                    }
                ],
                "third_converter_candidate": {
                    "converter_version": "2.0.2rc3",
                    "converter_sha256": (
                        "ed678cd40d14efdec30af46c01962a4ccbf8017ebb35ec01b4a9b6e2ceb24077"
                    ),
                    "paddle_version": "3.0.0.dev20250426",
                    "paddle_expected_sha256": (
                        "f62aaab2bd8d3ad4f4f7781bdeed43403546057b7afcdc10a4b33847b2617f1f"
                    ),
                    "venv_python_bytes": len(b"python"),
                    "venv_python_sha256": sha256(b"python").hexdigest(),
                    "converter_launcher_bytes": len(b"converter"),
                    "converter_launcher_sha256": sha256(b"converter").hexdigest(),
                    "result": "toolchain import and pip check passed",
                },
            }
        ),
        encoding="utf-8",
    )
    signature = {
        "status": "Valid",
        "signer": (
            "CN=Python Software Foundation, O=Python Software Foundation, "
            "L=Beaverton, S=Oregon, C=US"
        ),
    }

    evidence = conversion.validate_python_intake(
        intake,
        installer,
        executable,
        converter,
        signature=signature,
    )

    assert evidence["python_installer"]["sha256"] == sha256(b"installer").hexdigest()
    executable.write_bytes(b"pyth0n")
    with pytest.raises(conversion.ConversionGateError, match="Python SHA-256"):
        conversion.validate_python_intake(
            intake,
            installer,
            executable,
            converter,
            signature=signature,
        )


def test_authenticode_uses_encoded_command_and_explicit_security_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "python-installer.exe"
    installer.write_bytes(b"installer")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"status":"Valid","signer":"CN=Python Software Foundation, '
                'O=Python Software Foundation, C=US"}'
            ),
            stderr="",
        )

    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setattr(conversion.subprocess, "run", fake_run)

    signature = conversion._read_authenticode_signature(installer)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[-2] == "-EncodedCommand"
    decoded = conversion.base64.b64decode(command[-1]).decode("utf-16le")
    assert "Import-Module -Force -Name $env:GRAPHREADER_SECURITY_MODULE" in decoded
    assert "$ErrorActionPreference='Stop'" in decoded
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["GRAPHREADER_PYTHON_INSTALLER"] == str(installer)
    assert environment["GRAPHREADER_SECURITY_MODULE"].endswith(
        "Microsoft.PowerShell.Security.psd1"
    )
    assert signature["status"] == "Valid"


def test_wheelhouse_requires_selected_hashes(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "selected-1.0-py3-none-any.whl"
    wheel.write_bytes(b"selected")
    locked = {
        "selected": {
            "version": "1.0",
            "allowed_sha256": [sha256(b"selected").hexdigest()],
        }
    }

    inventory = conversion.inventory_wheelhouse(tmp_path, locked)

    assert inventory == [
        {
            "name": "selected-1.0-py3-none-any.whl",
            "distribution": "selected",
            "version": "1.0",
            "bytes": 8,
            "sha256": sha256(b"selected").hexdigest(),
            "selected": True,
        }
    ]
    wheel.write_bytes(b"tampered")
    with pytest.raises(conversion.ConversionGateError, match="Selected wheel evidence differs"):
        conversion.inventory_wheelhouse(tmp_path, locked)


def test_converter_result_rejects_nonzero_deceptive_log_and_missing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.onnx"
    with pytest.raises(conversion.ConversionGateError, match="code 1"):
        conversion.validate_converter_result(1, "", "", output)
    output.write_bytes(b"stale")
    with pytest.raises(conversion.ConversionGateError, match="failure marker"):
        conversion.validate_converter_result(0, "Failed to convert model", "", output)
    output.unlink()
    with pytest.raises(conversion.ConversionGateError, match="nonempty ONNX"):
        conversion.validate_converter_result(0, "conversion complete", "", output)


def test_converter_warning_profile_is_exact_and_reviewed() -> None:
    strides = (
        "[Paddle2ONNX] [WARNING] Can not find input/output name "
        "'StridesTensor' in op yaml info of pd_op.slice"
    )
    stride_lists = (
        "[Paddle2ONNX] [WARNING] Can not find input/output name "
        "'StridesTensorList' in op yaml info of pd_op.slice"
    )
    log = "\n".join(
        [
            "module.py:1: UserWarning: No ccache found.",
            "warnings.warn(warning_message)",
            *([strides] * 14),
            *([stride_lists] * 7),
        ]
    )

    evidence = conversion.analyze_converter_warnings(
        "en_PP-OCRv5_mobile_rec",
        log,
        "",
    )

    assert evidence["reviewed"] is True
    assert evidence["counts"]["slice_strides_tensor"] == 14
    with pytest.raises(conversion.ConversionGateError, match="warning profile differs"):
        conversion.analyze_converter_warnings(
            "en_PP-OCRv5_mobile_rec",
            log + "\n[WARNING] new unreviewed warning",
            "",
        )


def test_convert_once_removes_output_rejected_after_converter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = SimpleNamespace(model_id="model", archive_sha256="a" * 64)

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        output = Path(command[command.index("--save_file") + 1])
        output.write_bytes(b"invalid model")
        return SimpleNamespace(returncode=0, stdout="conversion complete", stderr="")

    monkeypatch.setattr(conversion.subprocess, "run", fake_run)
    monkeypatch.setattr(conversion, "analyze_converter_warnings", lambda *_: {})
    monkeypatch.setattr(
        conversion,
        "validate_onnx_model",
        lambda _: (_ for _ in ()).throw(conversion.ConversionGateError("invalid ONNX")),
    )
    output = tmp_path / "output" / "model.onnx"

    with pytest.raises(conversion.ConversionGateError, match="invalid ONNX"):
        conversion.convert_once(candidate, tmp_path, tmp_path / "output", tmp_path / "p2o.exe")

    assert not output.exists()
    assert (tmp_path / "output" / "model.conversion.log").is_file()


def _write_identity_onnx(path: Path, opset: int = 11) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    input_value = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])
    output_value = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [input_value],
        [output_value],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.save(model, path)


def _write_external_onnx(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    input_value = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])
    output_value = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])
    weights = onnx.numpy_helper.from_array(np.ones((1, 3), dtype=np.float32), name="weights")
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "weights"], ["y"])],
        "external",
        [input_value],
        [output_value],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )


def test_onnx_validation_requires_checker_and_exact_opset(tmp_path: Path) -> None:
    valid = tmp_path / "valid.onnx"
    _write_identity_onnx(valid)
    evidence = conversion.validate_onnx_model(valid)
    assert evidence["checker"] == "passed-full-check"
    assert evidence["opsets"][""] == 11

    wrong = tmp_path / "wrong.onnx"
    _write_identity_onnx(wrong, opset=12)
    with pytest.raises(conversion.ConversionGateError, match="opset"):
        conversion.validate_onnx_model(wrong)

    invalid = tmp_path / "invalid.onnx"
    invalid.write_bytes(b"not an ONNX model")
    with pytest.raises(conversion.ConversionGateError, match="ONNX checker rejected"):
        conversion.validate_onnx_model(invalid)

    external = tmp_path / "external.onnx"
    _write_external_onnx(external)
    with pytest.raises(conversion.ConversionGateError, match="External ONNX tensor"):
        conversion.validate_onnx_model(external)


def test_maximum_absolute_error_rejects_shape_dtype_and_nonfinite() -> None:
    reference = np.zeros((1, 2), dtype=np.float32)
    assert conversion.maximum_absolute_error(reference, reference.copy()) == 0.0
    with pytest.raises(conversion.ConversionGateError, match="shape mismatch"):
        conversion.maximum_absolute_error(reference, np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(conversion.ConversionGateError, match="dtype mismatch"):
        conversion.maximum_absolute_error(reference, np.zeros((1, 2), dtype=np.float64))
    with pytest.raises(conversion.ConversionGateError, match="finite"):
        conversion.maximum_absolute_error(reference, np.array([[0.0, np.nan]], dtype=np.float32))


def _runner(offset: float = 0.0):
    def factory(_: Path):
        def run(value: np.ndarray) -> list[np.ndarray]:
            return [value[:, :1, :, :] + np.float32(offset)]

        return "x", ["y"], run

    return factory


def test_cpu_parity_requires_16_cases_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        conversion,
        "MODEL_INPUT_SHAPES",
        {"model": ((1, 3, 32, 32), (1, 3, 32, 64))},
    )
    report = conversion.run_cpu_parity(
        "model",
        tmp_path,
        tmp_path / "model.onnx",
        paddle_factory=_runner(),
        onnx_factory=_runner(1e-5),
    )
    assert report["cases"] == 16
    assert report["passed"] is True
    assert report["maximum_absolute_difference"] <= 1e-4
    assert len({record["input_sha256"] for record in report["records"]}) == 16
    assert {tuple(record["input_shape"]) for record in report["records"]} == {
        (1, 3, 32, 32),
        (1, 3, 32, 64),
    }

    with pytest.raises(conversion.ConversionGateError, match="at least 16"):
        conversion.run_cpu_parity(
            "model",
            tmp_path,
            tmp_path / "model.onnx",
            cases=15,
            paddle_factory=_runner(),
            onnx_factory=_runner(),
        )
    with pytest.raises(conversion.ConversionGateError, match="CPU parity failed"):
        conversion.run_cpu_parity(
            "model",
            tmp_path,
            tmp_path / "model.onnx",
            paddle_factory=_runner(),
            onnx_factory=_runner(2e-4),
        )


def test_cpu_parity_rejects_output_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        conversion,
        "MODEL_INPUT_SHAPES",
        {"model": ((1, 3, 32, 32),)},
    )

    def wrong_factory(_: Path):
        return "x", ["y"], lambda value: [value[:, :1, :, :-1]]

    with pytest.raises(conversion.ConversionGateError, match="shape mismatch"):
        conversion.run_cpu_parity(
            "model",
            tmp_path,
            tmp_path / "model.onnx",
            paddle_factory=_runner(),
            onnx_factory=wrong_factory,
        )


def test_conversion_command_is_fail_closed(tmp_path: Path) -> None:
    command = conversion.conversion_command(
        tmp_path / "paddle2onnx.exe",
        tmp_path / "model",
        tmp_path / "model.onnx",
    )

    assert command[-10:] == [
        "--opset_version", "11",
        "--enable_auto_update_opset", "False",
        "--enable_onnx_checker", "True",
        "--optimize_tool", "None",
        "--enable_verbose", "True",
    ]


def test_conversion_requires_two_byte_identical_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_once(candidate: object, _: Path, output_root: Path, __: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / f"{candidate.model_id}.onnx"  # type: ignore[attr-defined]
        output.write_bytes(str(calls).encode("ascii"))
        return {
            "onnx": {
                "path": str(output),
                "sha256": str(calls) * 64,
            }
        }

    monkeypatch.setattr(conversion, "convert_once", fake_once)
    candidate = SimpleNamespace(model_id="model")

    with pytest.raises(conversion.ConversionGateError, match="not byte reproducible"):
        conversion.convert_one(candidate, tmp_path, tmp_path, tmp_path / "converter.exe")
    assert not (tmp_path / "model.onnx").exists()
    assert not (tmp_path / "repeat" / "model.onnx").exists()


def test_main_removes_stale_authority_when_build_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = SimpleNamespace(model_id="model")
    monkeypatch.setattr(conversion, "CANDIDATES", (candidate,))
    monkeypatch.setattr(
        conversion,
        "build_report",
        lambda *_: (_ for _ in ()).throw(conversion.ConversionGateError("blocked")),
    )
    output = tmp_path / "output"
    repeat = output / "repeat"
    repeat.mkdir(parents=True)
    report = output / "report.json"
    report.write_text("OLD-SUCCESS", encoding="utf-8")
    for root in (output, repeat):
        (root / "model.onnx").write_bytes(b"stale")
        (root / "model.conversion.log").write_text("stale", encoding="utf-8")

    result = conversion.main(
        [
            "--audit", str(tmp_path / "audit.json"),
            "--source", str(tmp_path / "source"),
            "--output", str(output),
            "--report", str(report),
            "--toolchain-root", str(tmp_path / "toolchain"),
            "--converter", str(tmp_path / "converter.exe"),
            "--toolchain-intake", str(tmp_path / "intake.json"),
            "--python-installer", str(tmp_path / "python.exe"),
        ]
    )

    assert result == 2
    assert not report.exists()
    assert not (output / "model.onnx").exists()
    assert not (repeat / "model.onnx").exists()
    assert not (output / "model.conversion.log").exists()
    assert not (repeat / "model.conversion.log").exists()


def test_report_never_approves_conversion_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    converter = toolchain / "paddle2onnx.exe"
    converter.write_bytes(b"converter")
    intake = tmp_path / "intake.json"
    intake.write_text("{}", encoding="utf-8")
    installer = tmp_path / "python-installer.exe"
    installer.write_bytes(b"installer")
    monkeypatch.setattr(conversion, "CANDIDATES", (SimpleNamespace(model_id="model"),))
    monkeypatch.setattr(conversion, "load_strict_json", lambda _: {})
    monkeypatch.setattr(conversion, "validate_archive_audit", lambda _: {"status": "eligible"})
    monkeypatch.setattr(conversion, "verify_source_models", lambda _: {"model": {}})
    monkeypatch.setattr(conversion, "validate_python_intake", lambda *_: {"status": "valid"})
    monkeypatch.setattr(conversion, "parse_locked_requirements", lambda _: {})
    monkeypatch.setattr(
        conversion,
        "validate_toolchain",
        lambda *_, **__: {"packages": {}},
    )
    monkeypatch.setattr(conversion, "inventory_wheelhouse", lambda *_: [])
    monkeypatch.setattr(
        conversion,
        "convert_one",
        lambda *_: {"model_id": "model", "status": "passed"},
    )

    report = conversion.build_report(
        audit,
        source,
        output,
        toolchain,
        converter,
        None,
        intake,
        installer,
    )

    assert report["status"] == "conversion_and_raw_tensor_parity_passed"
    assert report["approval_scope"] == "conversion-only"
    assert report["production_approved"] is False
    assert report["release_ready"] is False
