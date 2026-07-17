import os
import re
import json
import numpy as np

def parse_telemetry_directory(results_dir="results"):
    """
    Parses all telemetry JSON files in results_dir and aggregates the best cost per task, approach, and seed.
    Returns:
        dict: {task_name: {approach_name: [list_of_best_costs_across_seeds]}}
    """
    results = {}
    if not os.path.exists(results_dir):
        return results

    # Pattern for telemetry filenames: telemetry_{approach}_{task_name}_seed{seed}.json
    # Note: task_name can contain underscores (e.g., cfg_ml_svm_12, cfg_lcbench_167168)
    filename_pattern = re.compile(
        r"^telemetry_(?P<approach>.+?)_(?P<task>cfg_.+?)_seed(?P<seed>\d+)\.json$"
    )


    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        match = filename_pattern.match(filename)
        if not match:
            # Fall back to checking content task_name if format differs slightly
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    content = json.load(f)
                task = content.get("task_name", "unknown").replace("/", "_")
                approach = content.get("extractor_name", "unknown")
                trials = content.get("trials", [])
                if not trials:
                    continue
                best_cost = min(t["cost"] for t in trials)
                
                results.setdefault(task, {}).setdefault(approach, []).append(best_cost)
            except Exception:
                pass
            continue

        meta = match.groupdict()
        task = meta["task"]
        approach = meta["approach"]
        filepath = os.path.join(results_dir, filename)

        try:
            with open(filepath, "r") as f:
                content = json.load(f)
            trials = content.get("trials", [])
            if not trials:
                continue
            best_cost = min(t["cost"] for t in trials)
            results.setdefault(task, {}).setdefault(approach, []).append(best_cost)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    return results

def main():
    results = parse_telemetry_directory("results")
    if not results:
        print("No CARP-S telemetry results found in results/")
        return

    report_lines = []
    report_lines.append("# CARP-S Optimization Sweep Summary Report\n")
    report_lines.append("This report summarizes the performance (minimum cost reached in 50 trials) for 7 dynamic Random Forest UQ approaches across 6 HPOBench tasks, aggregated across 5 seeds.\n")

    for task_name in sorted(results.keys()):
        report_lines.append(f"## Task: {task_name}\n")
        report_lines.append("| Approach | Mean Best Cost | Std Dev | Finished Seeds | Best Individual Cost |")
        report_lines.append("| --- | --- | --- | --- | --- |")

        # Sort approaches by mean best cost (lower cost is better)
        approaches_perf = []
        for approach, costs in results[task_name].items():
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            best_cost = np.min(costs)
            approaches_perf.append((approach, mean_cost, std_cost, len(costs), best_cost))

        approaches_perf.sort(key=lambda x: x[1])

        for app, mean, std, n_seeds, best in approaches_perf:
            report_lines.append(f"| {app} | {mean:.5f} | {std:.5f} | {n_seeds}/5 | {best:.5f} |")
        report_lines.append("")

    report_content = "\n".join(report_lines)
    print(report_content)

    # Save to summary_report.md
    report_path = "results/summary_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Summary report written to {report_path}")

if __name__ == "__main__":
    main()
