#!/usr/bin/env python3
"""Centralized Benchmark Registry for CARP-S & YAHPO Task Definitions."""

from typing import Dict, List

class BenchmarkRegistry:
    LOW_DIM = [
        "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3",
        "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_12",
        "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_31",
        "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_3",
        "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_12",
        "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_31",
        "+task/YAHPO/SO=cfg_rbv2_glmnet_375",
        "+task/YAHPO/SO=cfg_rbv2_glmnet_458",
        "+task/YAHPO/SO=cfg_rbv2_rpart_14",
        "+task/YAHPO/SO=cfg_rbv2_rpart_40499",
    ]

    MID_DIM = [
        "+task/YAHPO/SO=cfg_lcbench_167168",
        "+task/YAHPO/SO=cfg_lcbench_189873",
        "+task/YAHPO/SO=cfg_lcbench_189906",
        "+task/YAHPO/SO=cfg_rbv2_ranger_16",
        "+task/YAHPO/SO=cfg_rbv2_ranger_42",
        "+task/YAHPO/SO=cfg_rbv2_xgboost_12",
        "+task/YAHPO/SO=cfg_rbv2_xgboost_1501",
        "+task/YAHPO/SO=cfg_rbv2_xgboost_16",
        "+task/YAHPO/SO=cfg_rbv2_xgboost_40499",
    ]

    HIGH_DIM_NAS = [
        "+task/YAHPO/SO=cfg_nb301_CIFAR10",
        "+task/YAHPO/SO=cfg_rbv2_super_1040",
        "+task/YAHPO/SO=cfg_rbv2_super_1049",
        "+task/YAHPO/SO=cfg_rbv2_super_1050",
        "+task/YAHPO/SO=cfg_rbv2_super_1053",
        "+task/YAHPO/SO=cfg_rbv2_super_1056",
        "+task/YAHPO/SO=cfg_rbv2_super_1063",
        "+task/YAHPO/SO=cfg_rbv2_super_1067",
        "+task/YAHPO/SO=cfg_rbv2_super_1068",
        "+task/YAHPO/SO=cfg_rbv2_super_1111",
        "+task/YAHPO/SO=cfg_rbv2_super_1220",
        "+task/YAHPO/SO=cfg_rbv2_super_1457",
        "+task/YAHPO/SO=cfg_rbv2_super_1461",
        "+task/YAHPO/SO=cfg_rbv2_super_1462",
        "+task/YAHPO/SO=cfg_rbv2_super_1464",
        "+task/YAHPO/SO=cfg_rbv2_super_1468",
        "+task/YAHPO/SO=cfg_rbv2_super_1479",
        "+task/YAHPO/SO=cfg_rbv2_super_15",
    ]

    @classmethod
    def get_all_tasks(cls) -> List[str]:
        return cls.LOW_DIM + cls.MID_DIM + cls.HIGH_DIM_NAS

    @classmethod
    def get_tasks_by_category(cls) -> Dict[str, List[str]]:
        return {
            "Low-Dim (<=6D)": cls.LOW_DIM,
            "Mid-Dim (7-20D)": cls.MID_DIM,
            "High-Dim & NAS (>20D)": cls.HIGH_DIM_NAS,
        }

    @classmethod
    def get_task_category(cls, task_str: str) -> str:
        if any(t in task_str for t in cls.LOW_DIM):
            return "Low-Dim (<=6D)"
        elif any(t in task_str for t in cls.MID_DIM):
            return "Mid-Dim (7-20D)"
        elif any(t in task_str for t in cls.HIGH_DIM_NAS):
            return "High-Dim & NAS (>20D)"
        return "High-Dim & NAS (>20D)"
