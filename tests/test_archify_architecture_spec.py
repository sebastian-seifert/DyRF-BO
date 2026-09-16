import json
import os
import subprocess
import sys

SPEC_PATH = "/home/sebastians/Projects/university/bachelorthesis/architecture/dyrf_bo_architecture.json"
ARCHIFY_BIN = "/home/sebastians/Projects/university/bachelorthesis/tools/archify/archify/bin/archify.mjs"


def test_archify_spec_exists_and_valid_json():
    assert os.path.isfile(SPEC_PATH), f"Specification file missing: {SPEC_PATH}"
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema_version") == 1
    assert data.get("diagram_type") == "architecture"
    assert "meta" in data
    assert data["meta"].get("title")
    assert "components" in data and len(data["components"]) >= 5


def test_archify_spec_component_references_consistent():
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    comp_ids = {c["id"] for c in data["components"]}

    # Check boundaries
    for b in data.get("boundaries", []):
        for member in b["wraps"]:
            assert member in comp_ids, f"Boundary member '{member}' not found in components"

    # Check connections
    for conn in data.get("connections", []):
        assert conn["from"] in comp_ids, f"Connection 'from' {conn['from']} not found in components"
        assert conn["to"] in comp_ids, f"Connection 'to' {conn['to']} not found in components"


def test_archify_cli_validation():
    assert os.path.isfile(ARCHIFY_BIN), f"Archify CLI binary not found: {ARCHIFY_BIN}"
    cmd = [
        "node",
        ARCHIFY_BIN,
        "validate",
        "architecture",
        SPEC_PATH,
        "--quality",
        "showcase",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Archify validation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    out = json.loads(res.stdout)
    assert out.get("ok") is True


if __name__ == "__main__":
    print("Running Archify architecture spec verification tests...")
    test_archify_spec_exists_and_valid_json()
    test_archify_spec_component_references_consistent()
    test_archify_cli_validation()
    print("All Archify architecture tests PASSED successfully!")
