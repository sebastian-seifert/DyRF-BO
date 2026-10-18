#!/usr/bin/env python3
"""HPOBench Tabular Verification Suite for Cluster and Local Environments.

Verifies:
1. Task configurations availability (88 ML and 10 NAS CARP-S tasks).
2. Underlying tabular dataset files availability on disk (ML: lr, nn, rf, svm, xgb;
   NAS: NAS-Bench-201, NAS-Bench-101, FCNet).
3. Cluster environment path resolution (CARPS_TASK_DATA_DIR, HPOBENCH_DATA_DIR).
4. Sweep completion status against task manifests and run execution logs.
5. Optional automated downloading of missing tabular dataset bundles.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.carps_realworld_registry import CarpsRealworldRegistry

# Official download URLs for HPOBench tabular bundles
ML_DATASET_URLS = {
    "xgb": "https://ndownloader.figshare.com/files/30469920",
    "svm": "https://ndownloader.figshare.com/files/30379359",
    "lr": "https://ndownloader.figshare.com/files/30379038",
    "rf": "https://ndownloader.figshare.com/files/30469089",
    "nn": "https://ndownloader.figshare.com/files/30379005",
}

NAS_DATASET_URLS = {
    "nasbench_201": "https://www.automl.org/wp-content/uploads/2020/08/nasbench_201_data_v1.3.zip",
    "nasbench_101": "https://storage.googleapis.com/nasbench/nasbench_full.tfrecord",
    "fcnet": "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz",
}


def normalize_suite_name(suite: str) -> str:
    """Normalizes suite string to 'ml', 'nas', or 'all'."""
    s = suite.strip().lower().replace("-", "_")
    if s in ("ml", "hpobench_ml", "tabular_ml"):
        return "ml"
    elif s in ("nas", "hpobench_nas", "tabular_nas"):
        return "nas"
    elif s in ("all", "tabular", "hpobench", "both"):
        return "all"
    else:
        raise ValueError(
            f"Unknown suite '{suite}'. Must be one of 'ml', 'nas', 'all'."
        )


def resolve_hpobench_data_dir(data_dir: Path | str | None = None) -> Path:
    """Resolves HPOBench data directory supporting cluster paths and local fallbacks."""
    if data_dir is not None:
        return Path(data_dir)

    # 1. Environment variable HPOBENCH_DATA_DIR
    env_hpobench = os.environ.get("HPOBENCH_DATA_DIR")
    if env_hpobench:
        return Path(env_hpobench)

    # 2. Environment variable CARPS_TASK_DATA_DIR
    env_carps = os.environ.get("CARPS_TASK_DATA_DIR")
    if env_carps:
        carps_path = Path(env_carps)
        hpobench_sub = carps_path / "hpobench"
        if hpobench_sub.is_dir() or not carps_path.exists():
            return hpobench_sub
        return carps_path

    # 3. Installed hpobench package configuration
    try:
        import hpobench

        if hasattr(hpobench, "config_file") and hasattr(
            hpobench.config_file, "data_dir"
        ):
            return Path(hpobench.config_file.data_dir)
    except Exception:
        pass

    # 4. Fallback to standard user cache directory
    return Path.home() / ".local" / "share" / "hpobench"


def check_task_configs_available(suite: str = "all") -> Dict[str, Any]:
    """Verifies all 88 ML tasks and 10 NAS tasks in CARP-S registry.

    Parameters
    ----------
    suite : str
        One of 'ml', 'nas', or 'all'.

    Returns
    -------
    Dict[str, Any]
        Verification summary including counts, available tasks, and missing tasks.
    """
    suite_norm = normalize_suite_name(suite)

    ml_available: List[str] = []
    ml_missing: List[str] = []
    nas_available: List[str] = []
    nas_missing: List[str] = []

    # Check CARPS package config files on disk if carps is installed
    carps_task_base: Optional[Path] = None
    try:
        import carps

        carps_task_base = (
            Path(carps.__file__).parent / "configs" / "task" / "HPOBench" / "blackbox" / "tabular"
        )
    except Exception:
        pass

    # ML Tasks
    if suite_norm in ("ml", "all"):
        expected_ml_tasks = CarpsRealworldRegistry.get_hpobench_ml_tasks()
        for task_cmd in expected_ml_tasks:
            task_cfg_name = task_cmd.split("/")[-1]
            if carps_task_base and (carps_task_base / "ml").is_dir():
                yaml_file = carps_task_base / "ml" / f"{task_cfg_name}.yaml"
                if yaml_file.is_file():
                    ml_available.append(task_cmd)
                else:
                    ml_missing.append(task_cmd)
            else:
                ml_available.append(task_cmd)

    # NAS Tasks
    if suite_norm in ("nas", "all"):
        expected_nas_tasks = CarpsRealworldRegistry.get_hpobench_nas_tasks()
        for task_cmd in expected_nas_tasks:
            task_cfg_name = task_cmd.split("/")[-1]
            if carps_task_base and (carps_task_base / "nas").is_dir():
                yaml_file = carps_task_base / "nas" / f"{task_cfg_name}.yaml"
                if yaml_file.is_file():
                    nas_available.append(task_cmd)
                else:
                    nas_missing.append(task_cmd)
            else:
                nas_available.append(task_cmd)

    total_expected = 0
    available: List[str] = []
    missing: List[str] = []

    if suite_norm in ("ml", "all"):
        total_expected += 88
        available.extend(ml_available)
        missing.extend(ml_missing)
    if suite_norm in ("nas", "all"):
        total_expected += 10
        available.extend(nas_available)
        missing.extend(nas_missing)

    all_available = len(missing) == 0 and len(available) == total_expected

    result: Dict[str, Any] = {
        "suite": suite_norm,
        "total_expected": total_expected,
        "total_found": len(available),
        "all_available": all_available,
        "available": available,
        "missing": missing,
    }

    if suite_norm in ("ml", "all"):
        result["ml_tasks"] = {
            "expected": 88,
            "found": len(ml_available),
            "available": ml_available,
            "missing": ml_missing,
        }
    if suite_norm in ("nas", "all"):
        result["nas_tasks"] = {
            "expected": 10,
            "found": len(nas_available),
            "available": nas_available,
            "missing": nas_missing,
        }

    return result


def _is_valid_file(p: Path) -> bool:
    """Checks whether a file exists and has size > 0."""
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _check_ml_model_dataset(data_dir: Path, model: str) -> Dict[str, Any]:
    """Checks whether a specific Tabular ML model dataset bundle is available."""
    candidates = [
        data_dir / "TabularData" / model,
        data_dir / model,
    ]
    zip_candidates = [
        data_dir / "TabularData" / f"{model}.zip",
        data_dir / f"{model}.zip",
        data_dir / "TabularData" / model / f"{model}.zip",
        data_dir / model / f"{model}.zip",
    ]

    for zf in zip_candidates:
        if _is_valid_file(zf):
            return {
                "available": True,
                "path": str(zf),
                "type": "zip_archive",
                "size_mb": round(zf.stat().st_size / (1024 * 1024), 2),
            }

    for c in candidates:
        if c.is_dir():
            # Check if directory contains parquet or metadata files or subdirectories
            files = list(c.rglob("*.parquet*")) + list(c.rglob("*.json*"))
            if files:
                return {
                    "available": True,
                    "path": str(c),
                    "type": "extracted_directory",
                    "file_count": len(files),
                }

    return {
        "available": False,
        "path": None,
        "type": "missing",
    }


def _check_nas_dataset_bundle(data_dir: Path, nas_type: str) -> Dict[str, Any]:
    """Checks whether a specific NAS dataset bundle is available."""
    if nas_type == "nasbench_201":
        candidates = [
            data_dir / "nasbench_201",
            data_dir / "NAS-Bench-201",
        ]
        zip_candidates = [
            data_dir / "nasbench_201_data_v1.3.zip",
            data_dir / "nasbench_201" / "nasbench_201_data_v1.3.zip",
        ]
        for zf in zip_candidates:
            if _is_valid_file(zf):
                return {
                    "available": True,
                    "path": str(zf),
                    "type": "zip_archive",
                    "size_mb": round(zf.stat().st_size / (1024 * 1024), 2),
                }
        for c in candidates:
            if c.is_dir():
                jsons = list(c.glob("NAS-Bench-201-*.json")) + list(c.glob("*.json"))
                if jsons:
                    return {
                        "available": True,
                        "path": str(c),
                        "type": "extracted_directory",
                        "file_count": len(jsons),
                    }
    elif nas_type == "nasbench_101":
        candidates = [
            data_dir / "nasbench_101" / "nasbench_full.tfrecord",
            data_dir / "nasbench_101" / "nasbench_only108.tfrecord",
            data_dir / "nasbench_full.tfrecord",
            data_dir / "nasbench_101" / "nasbench_full.tar.gz",
        ]
        for c in candidates:
            if _is_valid_file(c):
                return {
                    "available": True,
                    "path": str(c),
                    "type": "data_file",
                    "size_mb": round(c.stat().st_size / (1024 * 1024), 2),
                }
    elif nas_type == "fcnet":
        candidates = [
            data_dir / "fcnet_tabular_benchmarks",
            data_dir / "fcnet",
        ]
        tar_candidates = [
            data_dir / "fcnet_tabular_benchmarks.tar.gz",
            data_dir / "fcnet_tabular_benchmarks" / "fcnet_tabular_benchmarks.tar.gz",
        ]
        for tf in tar_candidates:
            if _is_valid_file(tf):
                return {
                    "available": True,
                    "path": str(tf),
                    "type": "tar_archive",
                    "size_mb": round(tf.stat().st_size / (1024 * 1024), 2),
                }
        for c in candidates:
            if c.is_dir():
                h5_files = list(c.glob("*.hdf5")) + list(c.glob("*.h5"))
                if h5_files:
                    return {
                        "available": True,
                        "path": str(c),
                        "type": "extracted_directory",
                        "file_count": len(h5_files),
                    }

    return {
        "available": False,
        "path": None,
        "type": "missing",
    }


def check_dataset_files_available(
    suite: str = "all", data_dir: Path | str | None = None
) -> Dict[str, Any]:
    """Checks if underlying tabular datasets are present and loadable on disk.

    Tabular ML: lr, nn, rf, svm, xgb
    NAS: NAS-Bench-201, NAS-Bench-101, FCNet

    Parameters
    ----------
    suite : str
        One of 'ml', 'nas', or 'all'.
    data_dir : Path | str | None
        Target data directory. If None, resolved automatically.

    Returns
    -------
    Dict[str, Any]
        Detailed dataset availability dictionary.
    """
    suite_norm = normalize_suite_name(suite)
    resolved_dir = resolve_hpobench_data_dir(data_dir)

    ml_models = ["lr", "nn", "rf", "svm", "xgb"]
    nas_bundles = ["nasbench_201", "nasbench_101", "fcnet"]

    ml_datasets: Dict[str, Any] = {}
    nas_datasets: Dict[str, Any] = {}
    missing: List[str] = []

    if suite_norm in ("ml", "all"):
        for model in ml_models:
            status = _check_ml_model_dataset(resolved_dir, model)
            ml_datasets[model] = status
            if not status["available"]:
                missing.append(model)

    if suite_norm in ("nas", "all"):
        for nas_type in nas_bundles:
            status = _check_nas_dataset_bundle(resolved_dir, nas_type)
            nas_datasets[nas_type] = status
            if not status["available"]:
                missing.append(nas_type)

    all_available = len(missing) == 0

    result: Dict[str, Any] = {
        "suite": suite_norm,
        "data_dir": str(resolved_dir),
        "all_available": all_available,
        "missing": missing,
    }

    if suite_norm in ("ml", "all"):
        result["ml_datasets"] = ml_datasets
    if suite_norm in ("nas", "all"):
        result["nas_datasets"] = nas_datasets

    return result


def download_missing_datasets(
    missing: Sequence[str], data_dir: Path | str | None = None
) -> Dict[str, bool]:
    """Attempts to download missing tabular dataset bundles."""
    resolved_dir = resolve_hpobench_data_dir(data_dir)
    results: Dict[str, bool] = {}

    for item in missing:
        item_lower = item.lower()
        if item_lower in ML_DATASET_URLS:
            url = ML_DATASET_URLS[item_lower]
            target_dir = resolved_dir / "TabularData" / item_lower
            target_dir.mkdir(parents=True, exist_ok=True)
            zip_dest = target_dir / f"{item_lower}.zip"
            try:
                print(f"Downloading {item_lower} from {url}...")
                urllib.request.urlretrieve(url, zip_dest)
                if zipfile.is_zipfile(zip_dest):
                    with zipfile.ZipFile(zip_dest, "r") as zf:
                        zf.extractall(target_dir)
                results[item] = True
            except Exception as e:
                print(f"Failed to download {item}: {e}")
                results[item] = False
        elif item_lower in NAS_DATASET_URLS:
            url = NAS_DATASET_URLS[item_lower]
            try:
                if item_lower == "nasbench_201":
                    target_dir = resolved_dir / "nasbench_201"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    zip_dest = target_dir / "nasbench_201_data_v1.3.zip"
                    urllib.request.urlretrieve(url, zip_dest)
                    if zipfile.is_zipfile(zip_dest):
                        with zipfile.ZipFile(zip_dest, "r") as zf:
                            zf.extractall(target_dir)
                    results[item] = True
                elif item_lower == "nasbench_101":
                    target_dir = resolved_dir / "nasbench_101"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    tfrecord_dest = target_dir / "nasbench_full.tfrecord"
                    urllib.request.urlretrieve(url, tfrecord_dest)
                    results[item] = True
                elif item_lower == "fcnet":
                    target_dir = resolved_dir / "fcnet_tabular_benchmarks"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    tar_dest = target_dir / "fcnet_tabular_benchmarks.tar.gz"
                    urllib.request.urlretrieve(url, tar_dest)
                    if tarfile.is_tarfile(tar_dest):
                        with tarfile.open(tar_dest, "r:*") as tf:
                            tf.extractall(target_dir)
                    results[item] = True
            except Exception as e:
                print(f"Failed to download {item}: {e}")
                results[item] = False
        else:
            results[item] = False

    return results


def _count_trials_in_file(log_file: Path) -> int:
    """Counts non-empty lines in a JSONL log file."""
    if not log_file.is_file():
        return 0
    cnt = 0
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cnt += 1
    except Exception:
        return 0
    return cnt


def _count_trials_in_telemetry(telemetry_file: Path) -> int:
    """Counts trials in a telemetry JSON file."""
    if not telemetry_file.is_file():
        return 0
    try:
        with open(telemetry_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            history = data.get("history", [])
            return len(history)
    except Exception:
        return 0
    return 0


def check_sweep_completion_status(
    suite: str = "all",
    base_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
    required_fraction: float = 1.0,
) -> Dict[str, Any]:
    """Compares task manifests against completed runs in runs/sweep_hpobench_<suite>.

    Parameters
    ----------
    suite : str
        One of 'ml', 'nas', or 'all'.
    base_dir : Path | str | None
        Directory containing runs (defaults to runs/sweep_hpobench_<suite>).
    manifest_path : Path | str | None
        Path to tasks.txt manifest file. If None, default paths are checked.
    required_fraction : float
        Fraction of n_trials required to mark a task completed (default 1.0).

    Returns
    -------
    Dict[str, Any]
        Summary containing total_tasks, completed_tasks, missing_tasks, completion_rate.
    """
    suite_norm = normalize_suite_name(suite)

    # Resolve manifest path if not provided
    if manifest_path is None:
        if suite_norm == "ml":
            candidate = PROJECT_ROOT / "results" / "sweep_hpobench_ml" / "tasks.txt"
        elif suite_norm == "nas":
            candidate = PROJECT_ROOT / "results" / "sweep_hpobench_nas" / "tasks.txt"
        else:
            candidate = PROJECT_ROOT / "results" / "sweep_hpobench_tabular" / "tasks.txt"
        manifest_file = candidate if candidate.is_file() else None
    else:
        manifest_file = Path(manifest_path)

    # Resolve base run directory
    if base_dir is None:
        if suite_norm == "ml":
            runs_path = PROJECT_ROOT / "runs" / "sweep_hpobench_ml"
        elif suite_norm == "nas":
            runs_path = PROJECT_ROOT / "runs" / "sweep_hpobench_nas"
        else:
            runs_path = PROJECT_ROOT / "runs" / "sweep_hpobench_tabular"
    else:
        runs_path = Path(base_dir)

    total_tasks = 0
    completed_tasks = 0
    missing_tasks = 0
    missing_lines: List[str] = []

    if manifest_file and manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                total_tasks += 1

                # Parse task attributes
                opt_m = re.search(r"optimizer_id=([^\s]+)", line)
                optimizer_id = opt_m.group(1) if opt_m else "unknown"

                task_m = re.search(r"\+task=([^\s]+)", line)
                task_id = task_m.group(1) if task_m else "unknown"

                seed_m = re.search(r"\bseed=(\d+)", line)
                seed = int(seed_m.group(1)) if seed_m else 1

                trials_m = re.search(r"n_trials=(\d+)", line)
                n_trials = int(trials_m.group(1)) if trials_m else 100

                telem_m = re.search(r"\+\+optimizer\.telemetry_path=([^\s]+)", line)
                telemetry_path = Path(telem_m.group(1)) if telem_m else None

                target_trials = int(n_trials * required_fraction)
                is_complete = False

                # Check telemetry file
                if telemetry_path and telemetry_path.is_file():
                    if _count_trials_in_telemetry(telemetry_path) >= target_trials:
                        is_complete = True

                # Check runs directory structure
                if not is_complete and runs_path.is_dir():
                    task_clean = task_id.replace("HPOBench/", "").replace("+task=", "")
                    ml_match = re.search(
                        r"(?:cfg_ml_|ml[_/]|^)(lr|nn|rf|svm|xgboost|xgb)[_/](\d+)",
                        task_id,
                        flags=re.IGNORECASE,
                    )
                    possible_run_dirs: List[Path] = [
                        runs_path / optimizer_id / task_clean / str(seed),
                        runs_path / optimizer_id / "hpobench" / "blackbox" / "tabular" / task_clean / str(seed),
                    ]
                    if ml_match:
                        model_tok = ml_match.group(1).lower()
                        data_tok = ml_match.group(2)
                        possible_run_dirs.extend([
                            runs_path / optimizer_id / "hpobench" / "blackbox" / "tabular" / "ml" / model_tok / data_tok / str(seed),
                            runs_path / optimizer_id / "HPOBench" / "blackbox" / "tabular" / "ml" / model_tok / data_tok / str(seed),
                            runs_path / optimizer_id / model_tok / data_tok / str(seed),
                            runs_path / optimizer_id / f"cfg_ml_{model_tok}_{data_tok}" / str(seed),
                        ])
                    else:
                        nas_clean = task_id.split("/")[-1].replace("cfg_", "")
                        possible_run_dirs.extend([
                            runs_path / optimizer_id / "hpobench" / "blackbox" / "tabular" / "nas" / nas_clean / str(seed),
                            runs_path / optimizer_id / "HPOBench" / "blackbox" / "tabular" / "nas" / nas_clean / str(seed),
                            runs_path / optimizer_id / nas_clean / str(seed),
                        ])

                    for rd in possible_run_dirs:
                        if rd.is_dir():
                            t_log = rd / "trial_logs.jsonl"
                            if _count_trials_in_file(t_log) >= target_trials:
                                is_complete = True
                                break

                    if not is_complete:
                        # Fallback search matching optimizer, seed, AND task identifiers in log path
                        for t_log in runs_path.glob(f"**/{optimizer_id}/**/{seed}/trial_logs.jsonl"):
                            log_str = str(t_log)
                            if ml_match:
                                model_tok = ml_match.group(1).lower()
                                data_tok = ml_match.group(2)
                                if (f"/{model_tok}/" in log_str or f"_{model_tok}_" in log_str) and (f"/{data_tok}/" in log_str or f"_{data_tok}" in log_str):
                                    if _count_trials_in_file(t_log) >= target_trials:
                                        is_complete = True
                                        break
                            else:
                                nas_clean = task_id.split("/")[-1].replace("cfg_", "")
                                if nas_clean in log_str:
                                    if _count_trials_in_file(t_log) >= target_trials:
                                        is_complete = True
                                        break

                if is_complete:
                    completed_tasks += 1
                else:
                    missing_tasks += 1
                    missing_lines.append(line)
    else:
        # If no manifest exists, scan runs directory directly
        if runs_path.is_dir():
            logs = list(runs_path.glob("**/trial_logs.jsonl"))
            for log in logs:
                if _count_trials_in_file(log) >= 10:
                    completed_tasks += 1
            total_tasks = completed_tasks

    rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0

    return {
        "suite": suite_norm,
        "base_dir": str(runs_path),
        "manifest_path": str(manifest_file) if manifest_file else None,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "missing_tasks": missing_tasks,
        "completion_rate": rate,
        "missing_task_lines": missing_lines,
    }


def format_report_text(
    task_res: Dict[str, Any],
    dataset_res: Dict[str, Any],
    sweep_res: Optional[Dict[str, Any]] = None,
) -> str:
    """Formats verification results into a readable terminal report."""
    lines: List[str] = []
    lines.append("==================================================")
    lines.append("   HPOBench Tabular Verification Suite Report")
    lines.append("==================================================")

    # 1. Task Configs
    lines.append("\n[1] CARP-S Task Configurations:")
    lines.append(
        f"  Suite: {task_res['suite']} | Expected: {task_res['total_expected']} | Found: {task_res['total_found']}"
    )
    if "ml_tasks" in task_res:
        lines.append(
            f"  - Tabular ML: {task_res['ml_tasks']['found']}/{task_res['ml_tasks']['expected']} configs available"
        )
    if "nas_tasks" in task_res:
        lines.append(
            f"  - Tabular NAS: {task_res['nas_tasks']['found']}/{task_res['nas_tasks']['expected']} configs available"
        )
    status_icon = "✓" if task_res["all_available"] else "✗"
    lines.append(f"  Status: {status_icon} {'ALL AVAILABLE' if task_res['all_available'] else 'INCOMPLETE'}")

    # 2. Datasets
    lines.append("\n[2] Tabular Datasets on Disk:")
    lines.append(f"  Data Directory: {dataset_res['data_dir']}")
    if "ml_datasets" in dataset_res:
        lines.append("  Tabular ML Models:")
        for model, info in dataset_res["ml_datasets"].items():
            icon = "✓" if info["available"] else "✗"
            desc = info["type"] if info["available"] else "NOT FOUND"
            lines.append(f"    {icon} {model.upper():<5} : {desc} ({info['path'] or 'missing'})")
    if "nas_datasets" in dataset_res:
        lines.append("  Tabular NAS Benchmarks:")
        for nas_type, info in dataset_res["nas_datasets"].items():
            icon = "✓" if info["available"] else "✗"
            desc = info["type"] if info["available"] else "NOT FOUND"
            lines.append(f"    {icon} {nas_type:<15} : {desc} ({info['path'] or 'missing'})")

    ds_icon = "✓" if dataset_res["all_available"] else "✗"
    lines.append(f"  Status: {ds_icon} {'ALL DATASETS AVAILABLE' if dataset_res['all_available'] else 'DATASETS MISSING'}")

    # 3. Sweep Completion (if checked)
    if sweep_res:
        lines.append("\n[3] Sweep Execution Status:")
        lines.append(f"  Runs Directory: {sweep_res['base_dir']}")
        lines.append(
            f"  Progress: {sweep_res['completed_tasks']}/{sweep_res['total_tasks']} completed "
            f"({sweep_res['completion_rate'] * 100:.1f}%)"
        )
        if sweep_res["missing_tasks"] > 0:
            lines.append(f"  Incomplete/Missing: {sweep_res['missing_tasks']} runs")

    lines.append("==================================================")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint for HPOBench tasks and datasets verification."""
    parser = argparse.ArgumentParser(
        description="Verify HPOBench Tabular task configurations and datasets availability."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["ml", "nas", "all"],
        help="Benchmark suite to verify (ml, nas, all). Default: all.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to HPOBench data directory. Defaults to cluster or local cache path.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directory containing runs to check execution completion.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to task manifest (tasks.txt).",
    )
    parser.add_argument(
        "--check-sweep",
        action="store_true",
        help="Audit sweep completion status against runs directory.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Attempt to download missing tabular datasets.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output report format ('text' or 'json'). Default: text.",
    )

    args = parser.parse_args()

    # 1. Check configs
    task_res = check_task_configs_available(suite=args.suite)

    # 2. Check datasets
    dataset_res = check_dataset_files_available(suite=args.suite, data_dir=args.data_dir)

    # 3. Download missing if requested
    if args.download_missing and not dataset_res["all_available"]:
        print(f"Attempting to download missing datasets: {dataset_res['missing']}...")
        dl_res = download_missing_datasets(dataset_res["missing"], data_dir=args.data_dir)
        print("Download results:", dl_res)
        # Re-check datasets
        dataset_res = check_dataset_files_available(suite=args.suite, data_dir=args.data_dir)

    # 4. Check sweep if requested
    sweep_res = None
    if args.check_sweep or args.manifest is not None:
        sweep_res = check_sweep_completion_status(
            suite=args.suite,
            base_dir=args.runs_dir,
            manifest_path=args.manifest,
        )

    # Output report
    if args.format == "json":
        output_payload = {
            "tasks": task_res,
            "datasets": dataset_res,
        }
        if sweep_res:
            output_payload["sweep"] = sweep_res
        print(json.dumps(output_payload, indent=2))
    else:
        report = format_report_text(task_res, dataset_res, sweep_res)
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
