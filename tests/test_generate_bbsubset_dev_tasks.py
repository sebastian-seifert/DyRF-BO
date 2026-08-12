import os
import pytest
from scripts.generate_bbsubset_dev_tasks import generate_bbsubset_dev_tasks
from ep_extractors import UQExtractorRegistry

def test_generate_bbsubset_dev_tasks(tmp_path):
    output_file = os.path.join(tmp_path, "bbsubset_dev_tasks.txt")
    tasks = generate_bbsubset_dev_tasks(output_path=output_file)
    
    assert os.path.exists(output_file)
    assert len(tasks) > 0

    # 20 tasks * 9 optimizers (8 extractors + 1 baseline) * 3 acqs * 5 seeds = 2,700 tasks
    assert len(tasks) == 2700

    with open(output_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2700

    extractors = UQExtractorRegistry.list_registered()
    acquisitions = ["ei", "pi", "lcb"]

    for acq in acquisitions:
        for ext in extractors:
            matching = [l for l in lines if f"acq_func_name={acq}" in l and f"uncertainty_func={ext}" in l]
            assert len(matching) > 0, f"Missing task for acq={acq}, extractor={ext}"

    for line in lines:
        assert "--config-dir carps_integration/configs" in line, f"Missing --config-dir in line: {line}"
        assert "+task=subselection/blackbox/dev/" in line, f"Missing dev subselection prefix: {line}"
        assert "/test/" not in line, f"Test set task generated in dev task file: {line}"
