from __future__ import annotations

import csv
import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from helion.autotuner.aot_kernel import extract_key_features
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.aot_runner import RunConfig
from helion.autotuner.aot_runner import run_evaluate_phase
from helion.autotuner.aot_structural_policy import attach_model_policy
from helion.autotuner.aot_structural_policy import measurement_config_hash
from helion.autotuner.aot_structural_policy import measurement_metadata
from helion.autotuner.aot_structural_policy import policy_path
from helion.autotuner.heuristic_generator import evaluate_heuristic
from helion.autotuner.heuristic_generator import get_backend
from helion.autotuner.heuristic_generator import load_measurements
from helion.runtime.config import Config
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("goal", ["max_slowdown", "geomean_slowdown", "avg_slowdown"])
@pytest.mark.parametrize("slowdown", [1.0, 1.2])
def test_evaluate_requires_both_performance_goal_and_benchmark_success(
    tmp_path: Path, exit_code: int, goal: str, slowdown: float
) -> None:
    config = RunConfig(
        benchmark_cmd=[sys.executable, "-c", f"raise SystemExit({exit_code})"],
        output_dir=tmp_path,
        hardware_id="test_hardware",
        run_id="evaluate",
        goal_type=goal,
        threshold=1.1,
    )
    config.run_log_dir.mkdir(parents=True)
    stats = dict.fromkeys(
        ("max_slowdown", "geomean_slowdown", "avg_slowdown"), slowdown
    )
    results = {"kernel": stats}
    stats["coverage"] = 1.0
    # Isolate the measurement-data boundary; execute the actual benchmark
    # subprocess and retain its exit status and evaluation report.
    with (
        patch(
            "helion.autotuner.aot_runner.load_measurements",
            return_value={"kernel": SimpleNamespace(cute_structural_policy=None)},
        ),
        patch("helion.autotuner.aot_runner.evaluate_heuristic", return_value=results),
    ):
        passed = run_evaluate_phase(config)

    assert passed == (slowdown <= config.threshold and exit_code == 0)
    report = config.run_dir / f"evaluation_{config.hardware_id}.json"
    assert json.loads(report.read_text()) == results
    log = config.run_log_dir / f"evaluate_{config.hardware_id}.log"
    assert f"# Return code: {exit_code}" in log.read_text()


def _measurements(
    path: Path,
    features: list[dict[str, object]],
    policy: CuteStructuralPolicy | None = None,
) -> None:
    metadata = measurement_metadata(policy)
    with path.open("w") as handle:
        fields = [
            "kernel_name",
            "shape_hash",
            "config_hash",
            "config",
            "shape_features",
            "timing_ms",
            *metadata,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, shape in enumerate(features):
            for j, block in enumerate((32, 64)):
                writer.writerow(
                    {
                        "kernel_name": "kernel",
                        "shape_hash": f"s{i}",
                        "config_hash": measurement_config_hash(
                            Config(block_sizes=[block]), policy
                        ),
                        "config": json.dumps({"block_sizes": [block]}),
                        "shape_features": json.dumps(shape),
                        "timing_ms": 1.0 if i == j else 4.0,
                        **metadata,
                    }
                )


@pytest.mark.parametrize("backend", ["decision_tree", "nearest_neighbor"])
@pytest.mark.parametrize(
    "kind",
    ["tensor", "key", "scalar", "mixed_rank", "empty", "constant", "dtype", "batched"],
)
def test_evaluate_current_generated_dispatcher(
    tmp_path: Path, backend: str, kind: str
) -> None:
    if kind == "key":
        features = [extract_key_features((i, "category")) for i in (1, 2)]
    elif kind == "scalar":
        features = [extract_shape_features((None, i)) for i in (1, 2)]
    elif kind == "dtype":
        features = [
            extract_shape_features(
                (
                    torch.empty(
                        (8, 4),
                        dtype=dtype,
                        device="meta",  # @ignore-device-lint: metadata without allocation
                    ),
                )
            )
            for dtype in (torch.int8, torch.int32)
        ]
    else:
        shapes = {
            "tensor": ((8, 4), (16, 8)),
            "batched": ((8, 4), (16, 8)),
            "mixed_rank": ((8,), (4, 8)),
            "empty": ((0, 4), (0, 8)),
            "constant": ((8, 4), (8, 4)),
        }[kind]
        features = [
            extract_shape_features(
                (
                    torch.empty(
                        shape,
                        dtype=torch.bfloat16,
                        device="meta",  # @ignore-device-lint: metadata without allocation
                    ),
                ),
                batched=[[0, None]] if kind == "batched" else None,
            )
            for shape in shapes
        ]
    measured = tmp_path / "measurements.csv"
    _measurements(measured, features)
    data = load_measurements(measured)["kernel"]
    data.selected_config_indices = [0] if kind == "constant" else [0, 1]
    configs = [data.configs[i] for i in data.selected_config_indices]
    names = sorted(
        {
            name
            for row in features
            for name, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    if kind == "constant":
        names = []
    generated = get_backend(backend).generate_heuristic("kernel", data, configs, names)
    (tmp_path / "heuristic_kernel.py").write_text(generated.generated_code)

    stats = evaluate_heuristic(measured, tmp_path)["kernel"]
    assert stats["coverage"] == 1.0
    assert stats["max_slowdown"] == (4.0 if kind == "constant" else 1.0)


@pytest.mark.parametrize(
    "failure",
    ["missing_file", "missing_selector", "partial", "empty_data", "missing_kernel"],
)
def test_runner_rejects_missing_or_incomplete_evaluation(
    tmp_path: Path, failure: str
) -> None:
    config = RunConfig(
        benchmark_cmd=[], output_dir=tmp_path, run_id="evaluate", hardware_id="test"
    )
    config.run_dir.mkdir()
    measured = config.run_dir / "measurements_test.csv"
    _measurements(
        measured, [] if failure == "empty_data" else [{"key_0": 1}, {"key_0": 2}]
    )
    if failure == "partial":
        (config.run_dir / "heuristic_kernel.py").write_text(
            "def select_config_kernel(features):\n"
            "    return {'block_sizes': [32 if features['key_0'] == 1 else 999]}\n"
        )
    elif failure == "missing_kernel":
        with measured.open() as handle:
            rows = list(csv.DictReader(handle))
        with measured.open("a") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            for row in rows:
                writer.writerow({**row, "kernel_name": "other"})
        (config.run_dir / "heuristic_kernel.py").write_text(
            "def select_config_kernel(features):\n"
            "    return {'block_sizes': [32 if features['key_0'] == 1 else 64]}\n"
        )
    elif failure != "missing_file":
        (config.run_dir / "heuristic_kernel.py").write_text("# No selector\n")
    assert not run_evaluate_phase(config)
    report = json.loads((config.run_dir / "evaluation_test.json").read_text())
    if failure == "partial":
        assert report["kernel"]["coverage"] == 0.5


def test_legacy_feature_selector_still_evaluates(tmp_path: Path) -> None:
    measured = tmp_path / "measurements.csv"
    _measurements(measured, [{"key_0": 1}, {"key_0": 2}])
    (tmp_path / "heuristic_kernel.py").write_text(
        "def select_config_kernel(features):\n"
        "    return {'block_sizes': [32 if features['key_0'] == 1 else 64]}\n"
    )
    results = evaluate_heuristic(measured, tmp_path)
    assert results["kernel"]["coverage"] == 1.0
    assert results["kernel"]["max_slowdown"] == 1.0


@pytest.mark.parametrize("backend", ["decision_tree", "nearest_neighbor"])
@pytest.mark.parametrize("mismatched_policy", [False, True])
def test_generated_evaluation_preserves_structural_policy(
    tmp_path: Path, backend: str, mismatched_policy: bool
) -> None:
    policy = CuteStructuralPolicy()
    measured = tmp_path / "measurements.csv"
    _measurements(measured, [{"key_0": 1}, {"key_0": 2}], policy)
    data = load_measurements(measured)["kernel"]
    data.selected_config_indices = [0, 1]
    generated = get_backend(backend).generate_heuristic(
        "kernel", data, data.configs, ["key_0"]
    )
    model_policy = (
        CuteStructuralPolicy(cute_region_fission=True) if mismatched_policy else policy
    )
    code = attach_model_policy(
        generated.generated_code, {"kernel": data.configs}, model_policy
    )
    policy_path(tmp_path / "heuristic_kernel.py", policy).write_text(code)
    if mismatched_policy:
        with pytest.raises(StructuralPolicyError):
            evaluate_heuristic(measured, tmp_path)
    else:
        results = evaluate_heuristic(measured, tmp_path)
        assert results["kernel"]["coverage"] == 1.0
        assert results["kernel"]["max_slowdown"] == 1.0
