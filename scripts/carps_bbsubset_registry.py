#!/usr/bin/env python3
"""CARP-S BBsubset (Blackbox Single-Objective) Task Registry.

Provides the dev set definitions (20 tasks) for active research and parameter calibration.
The held-out test set remains strictly un-exposed and isolated.
"""

from typing import List

class CarpsBBSubsetRegistry:
    # Official CARP-S Blackbox Dev Subset (20 Tasks)
    DEV_TASKS = [
        "+task=subselection/blackbox/dev/subset_bbob_2_12_0",
        "+task=subselection/blackbox/dev/subset_bbob_2_12_1",
        "+task=subselection/blackbox/dev/subset_bbob_2_20_0",
        "+task=subselection/blackbox/dev/subset_bbob_4_6_1",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_lr_146818",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_rf_146212",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_xgboost_146212",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_nas_NavalPropulsionBenchmark",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_nas_SliceLocalizationBenchmark",
        "+task=subselection/blackbox/dev/subset_yahpo_lcbench_168335_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_1462_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_312_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_40498_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_458_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_glmnet_41157_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_ranger_40927_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_svm_182_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_svm_24_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_xgboost_23512_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_xgboost_42_None",
    ]

    @classmethod
    def get_dev_tasks(cls) -> List[str]:
        """Returns the 20 CARP-S Blackbox Dev Tasks."""
        return list(cls.DEV_TASKS)

    # 14 Real-World ML Tasks (3 HPOBench ML + 11 YAHPO)
    REALWORLD_ML_DEV_TASKS = [
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_lr_146818",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_rf_146212",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_xgboost_146212",
        "+task=subselection/blackbox/dev/subset_yahpo_lcbench_168335_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_1462_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_312_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_40498_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_458_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_glmnet_41157_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_ranger_40927_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_svm_182_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_svm_24_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_xgboost_23512_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_xgboost_42_None",
    ]

    # 2 HPOBench Tabular NAS Tasks
    REALWORLD_NAS_DEV_TASKS = [
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_nas_NavalPropulsionBenchmark",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_nas_SliceLocalizationBenchmark",
    ]

    @classmethod
    def get_realworld_dev_tasks(cls, include_nas: bool = False) -> List[str]:
        """Returns the real-world dev tasks (14 ML tasks by default, or 16 if include_nas=True)."""
        tasks = list(cls.REALWORLD_ML_DEV_TASKS)
        if include_nas:
            tasks.extend(cls.REALWORLD_NAS_DEV_TASKS)
        return tasks

    @classmethod
    def get_working_dev_tasks(cls, exclude_nas: bool = True) -> List[str]:
        """Returns all working tasks from the CARP-S BBsubset (18 tasks when excluding broken NAS tasks)."""
        if exclude_nas:
            return [t for t in cls.DEV_TASKS if "tabular_nas" not in t]
        return list(cls.DEV_TASKS)
