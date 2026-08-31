"""TDD Test Suite for CARP-S BBsubset Dual-Schedule LCB Sweep Task Generator."""

import os
import sys
import pytest

# Ensure DyRF-BO root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry
from scripts.generate_bbsubset_lcb_sweep_tasks import (
    generate_bbsubset_lcb_sweep_tasks,
    get_approach_configs,
    ACTIVE_APPROACHES,
    SCHEDULE_CONFIGS,
)


def test_competitors_matrix_completeness():
    """Test 1: Verify exact 15 competitor configurations (1 baseline, 7 direct, 7 additive)."""
    approaches = get_approach_configs()
    assert len(approaches) == 15

    baseline = [a for a in approaches if a["paradigm"] == "baseline"]
    direct = [a for a in approaches if a["paradigm"] == "direct"]
    additive = [a for a in approaches if a["paradigm"] == "additive"]

    assert len(baseline) == 1
    assert len(direct) == 7
    assert len(additive) == 7

    optimizer_ids = [a["optimizer_id"] for a in approaches]
    assert len(set(optimizer_ids)) == 15  # All 15 optimizer IDs must be unique

    # Check optimizer container IDs
    assert baseline[0]["container_id"] == "SMAC3"
    for d in direct:
        assert d["container_id"] == "SMAC20_CustomUncertainty"
    for a in additive:
        assert a["container_id"] == "CARPSDynamicRF"


def test_zero_chen_occurrences(tmp_path):
    """Test 2: Verify zero occurrences of 'chen' across competitor configs and generated tasks."""
    approaches = get_approach_configs()
    for app in approaches:
        assert "chen" not in app["optimizer_id"].lower()
        if "extractor_name" in app:
            assert "chen" not in app["extractor_name"].lower()

    out_file = str(tmp_path / "tasks.txt")
    tasks = generate_bbsubset_lcb_sweep_tasks(output_file=out_file, seeds=5, schedules="both")
    for idx, line in enumerate(tasks):
        assert "chen" not in line.lower(), f"Found 'chen' in task line {idx}: {line}"


def test_dual_schedule_task_count_default(tmp_path):
    """Test 3: Verify default dual schedules (both) and 5 seeds produce exactly 3,000 tasks."""
    out_file = str(tmp_path / "tasks_default.txt")
    runs_dir = str(tmp_path / "results" / "bbsubset_lcb")
    baserundir = str(tmp_path / "runs" / "bbsubset_runs")
    tasks = generate_bbsubset_lcb_sweep_tasks(
        output_file=out_file,
        runs_dir=runs_dir,
        baserundir=baserundir,
        seeds=5,
        trials=50,
        schedules="both",
    )
    assert len(tasks) == 3000
    assert os.path.exists(out_file)

    with open(out_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 3000

    baseline_count = sum(1 for line in tasks if "+optimizer/smac20=hpo" in line)
    direct_count = sum(1 for line in tasks if "+optimizer=smac20_custom_uncertainty" in line)
    additive_count = sum(1 for line in tasks if "+optimizer=dyrf_additive_epistemic_lcb" in line)

    # 2 schedules * 20 tasks * 1 baseline * 5 seeds = 200
    assert baseline_count == 200
    # 2 schedules * 20 tasks * 7 direct * 5 seeds = 1400
    assert direct_count == 1400
    # 2 schedules * 20 tasks * 7 additive * 5 seeds = 1400
    assert additive_count == 1400
    assert baseline_count + direct_count + additive_count == 3000


def test_single_schedule_filtering(tmp_path):
    """Test 4: Verify single schedule selection produces exactly 1,500 tasks routed to respective subfolders."""
    # Constant schedule only
    out_const = str(tmp_path / "tasks_const.txt")
    tasks_const = generate_bbsubset_lcb_sweep_tasks(
        output_file=out_const,
        seeds=5,
        schedules="constant",
    )
    assert len(tasks_const) == 1500
    for line in tasks_const:
        assert "/constant" in line
        assert "/annealed" not in line

    # Annealed schedule only
    out_anneal = str(tmp_path / "tasks_anneal.txt")
    tasks_anneal = generate_bbsubset_lcb_sweep_tasks(
        output_file=out_anneal,
        seeds=5,
        schedules="annealed",
    )
    assert len(tasks_anneal) == 1500
    for line in tasks_anneal:
        assert "/annealed" in line
        assert "/constant" not in line


def test_custom_seeds_count(tmp_path):
    """Test 5: Verify custom seeds counts (integer and list sequences) across dual schedules."""
    # 10 seeds -> 6,000 tasks
    tasks_10 = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_10.txt"),
        seeds=10,
        schedules="both",
    )
    assert len(tasks_10) == 6000

    # Custom seed list [42, 99] -> 2 schedules * 20 tasks * 15 approaches * 2 seeds = 1,200 tasks
    tasks_custom = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_custom.txt"),
        seeds=[42, 99],
        schedules="both",
    )
    assert len(tasks_custom) == 1200


def test_strict_lcb_acquisition(tmp_path):
    """Test 6: Verify all commands strictly use LCB acquisition (0 EI, 0 PI)."""
    out_file = str(tmp_path / "tasks_lcb.txt")
    tasks = generate_bbsubset_lcb_sweep_tasks(output_file=out_file, seeds=5, schedules="both")

    for idx, line in enumerate(tasks):
        assert "acq_func_name=ei" not in line.lower(), f"Unexpected EI acquisition in line {idx}: {line}"
        assert "acq_func_name=pi" not in line.lower(), f"Unexpected PI acquisition in line {idx}: {line}"
        assert "_ei_" not in line, f"Unexpected _ei_ in line {idx}: {line}"
        assert "_pi_" not in line, f"Unexpected _pi_ in line {idx}: {line}"
        assert "_ei " not in line, f"Unexpected _ei in line {idx}: {line}"
        assert "_pi " not in line, f"Unexpected _pi in line {idx}: {line}"
        assert "_lcb" in line, f"Expected LCB identifier missing in line {idx}: {line}"


def test_schedule_parameter_injection(tmp_path):
    """Test 7: Verify exact hyperparameter injection for constant vs annealed schedules."""
    tasks = generate_bbsubset_lcb_sweep_tasks(output_file=str(tmp_path / "tasks.txt"), seeds=5, schedules="both")

    const_additive = [l for l in tasks if "/constant" in l and "+optimizer=dyrf_additive_epistemic_lcb" in l]
    anneal_additive = [l for l in tasks if "/annealed" in l and "+optimizer=dyrf_additive_epistemic_lcb" in l]

    assert len(const_additive) == 700
    assert len(anneal_additive) == 700

    # Constant schedule parameters: beta_max=1.0, beta_min=1.0, warmup_ratio=1.0
    for line in const_additive:
        assert "++optimizer.beta_max=1.0" in line
        assert "++optimizer.beta_min=1.0" in line
        assert "++optimizer.warmup_ratio=1.0" in line

    # Annealed schedule parameters: beta_max=1.0, beta_min=0.0, warmup_ratio=0.2 (or 0.20)
    for line in anneal_additive:
        assert "++optimizer.beta_max=1.0" in line
        assert "++optimizer.beta_min=0.0" in line
        assert "++optimizer.warmup_ratio=0.2" in line or "++optimizer.warmup_ratio=0.20" in line


def test_seed_uniformity(tmp_path):
    """Test 8: Verify exact seed uniformity across 1..5 seeds (600 tasks/seed: 300 constant, 300 annealed)."""
    tasks = generate_bbsubset_lcb_sweep_tasks(output_file=str(tmp_path / "tasks.txt"), seeds=5, schedules="both")

    for s in range(1, 6):
        count = sum(1 for line in tasks if f"seed={s} " in line or line.endswith(f"seed={s}"))
        assert count == 600, f"Seed {s} total count is {count}, expected 600"

        const_count = sum(1 for line in tasks if "/constant" in line and (f"seed={s} " in line or line.endswith(f"seed={s}")))
        anneal_count = sum(1 for line in tasks if "/annealed" in line and (f"seed={s} " in line or line.endswith(f"seed={s}")))
        assert const_count == 300, f"Seed {s} constant count is {const_count}, expected 300"
        assert anneal_count == 300, f"Seed {s} annealed count is {anneal_count}, expected 300"


def test_dev_tasks_coverage(tmp_path):
    """Test 9: Verify all 20 dev tasks from CarpsBBSubsetRegistry are covered (150 tasks each for 5 seeds dual-schedule)."""
    tasks = generate_bbsubset_lcb_sweep_tasks(output_file=str(tmp_path / "tasks.txt"), seeds=5, schedules="both")
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    assert len(dev_tasks) == 20

    for dev_task in dev_tasks:
        task_name = dev_task.split("/")[-1]
        matching = [l for l in tasks if task_name in l]
        assert len(matching) == 150, f"Dev task {task_name} has {len(matching)} tasks, expected 150"


def test_collision_free_telemetry_paths(tmp_path):
    """Test 10: Verify 3,000 distinct collision-free telemetry paths located in {runs_dir}/{schedule}/."""
    runs_dir = str(tmp_path / "results" / "bbsubset_lcb")
    tasks = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks.txt"),
        runs_dir=runs_dir,
        seeds=5,
        schedules="both",
    )
    telemetry_paths = []
    for line in tasks:
        assert "++optimizer.telemetry_path=" in line
        path = line.split("++optimizer.telemetry_path=")[1].split()[0]
        assert path.startswith(f"{runs_dir}/constant/") or path.startswith(f"{runs_dir}/annealed/")
        telemetry_paths.append(path)

    assert len(telemetry_paths) == 3000
    assert len(set(telemetry_paths)) == 3000


def test_baserundir_subfolder_routing(tmp_path):
    """Test 11: Verify baserundir is correctly routed to {baserundir}/constant and {baserundir}/annealed."""
    # Default baserundir
    tasks_default = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_default.txt"),
        seeds=5,
        schedules="both",
    )
    for line in tasks_default:
        if "/constant" in line.split("baserundir=")[1]:
            assert "baserundir=runs/bbsubset_runs/constant" in line
        else:
            assert "baserundir=runs/bbsubset_runs/annealed" in line

    # Custom baserundir
    custom_baserundir = "runs/custom_bbsubset_lcb_runs"
    tasks_custom = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_custom.txt"),
        baserundir=custom_baserundir,
        seeds=5,
        schedules="both",
    )
    for line in tasks_custom:
        if "/constant" in line.split("baserundir=")[1]:
            assert f"baserundir={custom_baserundir}/constant" in line
        else:
            assert f"baserundir={custom_baserundir}/annealed" in line


def test_paradigm_filtering_and_invalid_schedule(tmp_path):
    """Test 12: Verify paradigm filtering (baseline, direct, additive) and error handling for invalid schedule."""
    # Baseline only -> 200 tasks (100 const + 100 anneal)
    tasks_base = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_base.txt"),
        paradigm="baseline",
        seeds=5,
        schedules="both",
    )
    assert len(tasks_base) == 200

    # Direct only -> 1,400 tasks (700 const + 700 anneal)
    tasks_direct = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_direct.txt"),
        paradigm="direct",
        seeds=5,
        schedules="both",
    )
    assert len(tasks_direct) == 1400

    # Additive only -> 1,400 tasks (700 const + 700 anneal)
    tasks_additive = generate_bbsubset_lcb_sweep_tasks(
        output_file=str(tmp_path / "tasks_additive.txt"),
        paradigm="additive",
        seeds=5,
        schedules="both",
    )
    assert len(tasks_additive) == 1400

    # Invalid schedule error check
    with pytest.raises(ValueError, match="Unknown schedule"):
        generate_bbsubset_lcb_sweep_tasks(schedules="invalid_schedule")
