"""TDD Unit Test for CARP-S BBSubset Big Comparison Optimizer Configurations.

Asserts that all 11 required optimizers exist and specify valid configurations:
1. Baselines (2):
   - SMAC3_HPOFacade_lcb
   - SMAC3_HPOFacade_ei
2. Entropy (1):
   - SMAC20_Entropy_LCB (uses shaker_entropy with beta=3.8416 / kappa=1.96)
3. Proximity A family (2):
   - SMAC20_ProximityA_LCB
   - SMAC20_ProximityA_LCB_CV
4. Proximity B family (2):
   - SMAC20_ProximityB_LCB
   - SMAC20_ProximityB_LCB_CV
5. Proximity AC family (2):
   - SMAC20_ProximityAC_LCB
   - SMAC20_ProximityAC_LCB_CV
6. Proximity BC family (2):
   - SMAC20_ProximityBC_LCB
   - SMAC20_ProximityBC_LCB_CV
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


EXPECTED_OPTIMIZERS = [
    ("smac3_hpo_facade_lcb.yaml", "SMAC3_HPOFacade_lcb"),
    ("smac3_hpo_facade_ei.yaml", "SMAC3_HPOFacade_ei"),
    ("smac20_entropy_lcb.yaml", "SMAC20_Entropy_LCB"),
    ("smac20_proximity_a_lcb.yaml", "SMAC20_ProximityA_LCB"),
    ("smac20_proximity_a_lcb_cv.yaml", "SMAC20_ProximityA_LCB_CV"),
    ("smac20_proximity_b_lcb.yaml", "SMAC20_ProximityB_LCB"),
    ("smac20_proximity_b_lcb_cv.yaml", "SMAC20_ProximityB_LCB_CV"),
    ("smac20_proximity_ac_lcb.yaml", "SMAC20_ProximityAC_LCB"),
    ("smac20_proximity_ac_lcb_cv.yaml", "SMAC20_ProximityAC_LCB_CV"),
    ("smac20_proximity_bc_lcb.yaml", "SMAC20_ProximityBC_LCB"),
    ("smac20_proximity_bc_lcb_cv.yaml", "SMAC20_ProximityBC_LCB_CV"),
]


@pytest.mark.parametrize("filename, expected_opt_id", EXPECTED_OPTIMIZERS)
def test_all_11_optimizers_exist_and_valid(filename: str, expected_opt_id: str):
    cfg = load_yaml(filename)
    assert cfg.get("optimizer_id") == expected_opt_id
    assert "optimizer" in cfg
    opt = cfg["optimizer"]
    assert isinstance(opt, dict)


def test_entropy_lcb_configuration_details():
    cfg = load_yaml("smac20_entropy_lcb.yaml")
    opt = cfg["optimizer"]
    assert opt.get("acq_func_name") == "lcb"
    acq_kwargs = opt.get("acq_func_kwargs", {})
    assert pytest.approx(acq_kwargs.get("beta", 0.0), abs=1e-3) == 3.8416
    assert acq_kwargs.get("update_beta") is False

    smac_cfg = opt.get("smac_cfg", {})
    assert smac_cfg.get("model_class") == "carps_integration.custom_uncertainty_model.CustomUncertaintyRandomForest"
    model_kwargs = smac_cfg.get("model_kwargs", {})
    assert model_kwargs.get("uncertainty_func") == "shaker_entropy"
