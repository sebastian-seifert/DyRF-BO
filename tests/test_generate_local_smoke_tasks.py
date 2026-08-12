import os
import pytest
from scripts.generate_local_smoke_tasks import generate_local_smoke_tasks
from ep_extractors import UQExtractorRegistry

def test_generate_local_smoke_tasks(tmp_path):
    output_file = os.path.join(tmp_path, "smoke_tasks.txt")
    tasks = generate_local_smoke_tasks(output_path=output_file)
    
    assert os.path.exists(output_file)
    assert len(tasks) > 0

    # Read back generated lines
    with open(output_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == len(tasks)

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

    # Verify baseline runs exist
    for acq in acquisitions:
        matching_base = [l for l in lines if f"acq_func_name={acq}" in l and "optimizer/smac20=hpo" in l]
        assert len(matching_base) > 0, f"Missing baseline task for acq={acq}"
