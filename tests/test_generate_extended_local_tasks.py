import os
import pytest
from scripts.generate_extended_local_tasks import generate_extended_local_tasks
from ep_extractors import UQExtractorRegistry

def test_generate_extended_local_tasks(tmp_path):
    output_file = os.path.join(tmp_path, "extended_smoke_tasks.txt")
    tasks = generate_extended_local_tasks(output_path=output_file)
    
    assert os.path.exists(output_file)
    assert len(tasks) > 0

    # Read back generated lines
    with open(output_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == len(tasks)
    # 6 benchmarks * 9 optimizers * 3 acqs = 162 tasks
    assert len(lines) == 162

    # Verify all 8 extractors are included across acquisition functions
    extractors = UQExtractorRegistry.list_registered()
    acquisitions = ["ei", "pi", "lcb"]

    for acq in acquisitions:
        for ext in extractors:
            matching = [l for l in lines if f"acq_func_name={acq}" in l and f"uncertainty_func={ext}" in l]
            assert len(matching) > 0, f"Missing task for acq={acq}, extractor={ext}"

    # Verify --config-dir and +task/ prefix exist in generated lines
    for line in lines:
        assert "--config-dir carps_integration/configs" in line, f"Missing --config-dir in line: {line}"
        assert " +task/" in line or line.startswith("+task/"), f"Missing +task/ prefix in line: {line}"
        assert "seed=2" in line, f"Missing seed=2 in line: {line}"
