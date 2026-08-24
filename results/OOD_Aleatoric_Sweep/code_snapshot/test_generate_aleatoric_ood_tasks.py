import os
import pytest

def test_generate_aleatoric_ood_tasks(tmp_path):
    from scripts.generate_aleatoric_ood_tasks import generate_aleatoric_ood_tasks
    
    target_file = str(tmp_path / "aleatoric_ood_tasks.txt")
    tasks = generate_aleatoric_ood_tasks(output_file=target_file)
    
    assert os.path.exists(target_file)
    assert len(tasks) > 0
    
    # 15 functions x 7 noises x 5 RF configs x 5 seeds = 2625 tasks
    assert len(tasks) == 2625
    
    # Assert formatting in task lines
    first_task = tasks[0]
    assert "run_single_aleatoric_ood_experiment" in first_task
    assert "results/OOD_Aleatoric_Sweep/json" in first_task
    assert "hetero_ood_step_double" in "\n".join(tasks)
