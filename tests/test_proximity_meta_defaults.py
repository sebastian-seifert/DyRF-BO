"""Unit tests for verifying meta-optimal default configurations for Proximity approaches.

Verifies that the YAML configs for Proximity A, B, AC, BC (and their CV counterparts)
correctly specify the optimal hyperparameters found via meta-optimization:
- Proximity A / A CV: k=28, lambda=0.205, eps=0.081, level=0.95
- Proximity B / B CV: lambda=1.273, eps=0.156, level=0.95
- Proximity AC / AC CV: alpha=0.863, lambda=0.328, eps=0.093, k=7, level=0.95
- Proximity BC / BC CV: alpha=1.474, lambda=1.494, eps=0.143, level=0.95
"""

import os
import yaml
import pytest

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "carps_integration",
    "configs",
    "optimizer",
)


def load_yaml(filename: str) -> dict:
    path = os.path.join(CONFIG_DIR, filename)
    assert os.path.isfile(path), f"Config file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("suffix", ["", "_cv"])
def test_proximity_a_defaults(suffix: str):
    filename = f"smac20_proximity_a_lcb{suffix}.yaml"
    cfg = load_yaml(filename)
    opt = cfg["optimizer"]
    acq_kwargs = opt["acq_func_kwargs"]
    model_kwargs = opt["smac_cfg"]["model_kwargs"]
    ext_kwargs = model_kwargs["extractor_kwargs"]

    assert acq_kwargs["k"] == 28
    assert acq_kwargs["k_warmup"] == 28
    assert pytest.approx(acq_kwargs["eps"], abs=1e-3) == 0.081
    assert pytest.approx(acq_kwargs["level"], abs=1e-3) == 0.95
    assert pytest.approx(ext_kwargs["decay_lambda"], abs=1e-3) == 0.205
    if suffix == "":
        assert model_kwargs["uncertainty_func"] == "standard_proximity"
    else:
        assert model_kwargs["uncertainty_func"] == "standard_proximity_cv"


@pytest.mark.parametrize("suffix", ["", "_cv"])
def test_proximity_b_defaults(suffix: str):
    filename = f"smac20_proximity_b_lcb{suffix}.yaml"
    cfg = load_yaml(filename)
    opt = cfg["optimizer"]
    acq_kwargs = opt["acq_func_kwargs"]
    model_kwargs = opt["smac_cfg"]["model_kwargs"]
    ext_kwargs = model_kwargs["extractor_kwargs"]

    assert pytest.approx(acq_kwargs["eps"], abs=1e-3) == 0.156
    assert pytest.approx(acq_kwargs["level"], abs=1e-3) == 0.95
    assert pytest.approx(ext_kwargs["decay_lambda"], abs=1e-3) == 1.273
    if suffix == "":
        assert model_kwargs["uncertainty_func"] == "proximity_b"
    else:
        assert model_kwargs["uncertainty_func"] == "proximity_b_cv"


@pytest.mark.parametrize("suffix", ["", "_cv"])
def test_proximity_ac_defaults(suffix: str):
    filename = f"smac20_proximity_ac_lcb{suffix}.yaml"
    cfg = load_yaml(filename)
    opt = cfg["optimizer"]
    acq_kwargs = opt["acq_func_kwargs"]
    model_kwargs = opt["smac_cfg"]["model_kwargs"]
    ext_kwargs = model_kwargs["extractor_kwargs"]

    assert acq_kwargs["k"] == 7
    assert acq_kwargs["k_warmup"] == 7
    assert pytest.approx(acq_kwargs["eps"], abs=1e-3) == 0.093
    assert pytest.approx(acq_kwargs["level"], abs=1e-3) == 0.95
    assert pytest.approx(ext_kwargs["decay_lambda"], abs=1e-3) == 0.328
    assert pytest.approx(ext_kwargs["alpha"], abs=1e-3) == 0.863
    if suffix == "":
        assert model_kwargs["uncertainty_func"] == "proximity_ac"
    else:
        assert model_kwargs["uncertainty_func"] == "proximity_ac_cv"


@pytest.mark.parametrize("suffix", ["", "_cv"])
def test_proximity_bc_defaults(suffix: str):
    filename = f"smac20_proximity_bc_lcb{suffix}.yaml"
    cfg = load_yaml(filename)
    opt = cfg["optimizer"]
    acq_kwargs = opt["acq_func_kwargs"]
    model_kwargs = opt["smac_cfg"]["model_kwargs"]
    ext_kwargs = model_kwargs["extractor_kwargs"]

    assert pytest.approx(acq_kwargs["eps"], abs=1e-3) == 0.143
    assert pytest.approx(acq_kwargs["level"], abs=1e-3) == 0.95
    assert pytest.approx(ext_kwargs["decay_lambda"], abs=1e-3) == 1.494
    assert pytest.approx(ext_kwargs["alpha"], abs=1e-3) == 1.474
    if suffix == "":
        assert model_kwargs["uncertainty_func"] == "proximity_bc"
    else:
        assert model_kwargs["uncertainty_func"] == "proximity_bc_cv"
