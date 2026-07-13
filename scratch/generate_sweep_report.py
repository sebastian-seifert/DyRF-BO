import os
import json
import sys
import numpy as np

def generate_report(results_dir):
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f != "summary_report.json"]
    if not json_files:
        print(f"No result JSON files found in '{results_dir}'.")
        return

    # First pass: identify all approaches and metrics present
    all_approaches = set()
    all_metrics = set()
    all_data = []

    for jf in json_files:
        filepath = os.path.join(results_dir, jf)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "results_all" in data:
                    res_all = data["results_all"]
                    all_data.append(res_all)
                    for app, metrics_dict in res_all.items():
                        all_approaches.add(app)
                        for metric in metrics_dict.keys():
                            all_metrics.add(metric)
        except Exception as e:
            print(f"Warning: Failed to read {jf}: {e}")

    if not all_data:
        print("No valid 'results_all' data blocks found.")
        return

    # Sort approaches and metrics for clean presentation
    sorted_approaches = sorted(list(all_approaches))
    # Standard ordering for known metrics, others appended
    preferred_metric_order = ["auroc", "fpr95", "aupr", "spearman", "brier", "mi", "jsd", "naurc", "nlpd"]
    sorted_metrics = [m for m in preferred_metric_order if m in all_metrics]
    sorted_metrics += sorted([m for m in all_metrics if m not in preferred_metric_order])

    # Build the markdown report content
    report_lines = []
    report_lines.append(f"# Parameter Sweep Summary Report")
    report_lines.append(f"**Directory**: `{results_dir}`  ")
    report_lines.append(f"**Processed Files**: {len(json_files)} runs  \n")
    report_lines.append(f"This report presents the consolidated mean and standard deviation of all evaluated metrics across all dataset configurations and seeds present in this folder.\n")
    
    # Table Header
    header = "| Metric | " + " | ".join(sorted_approaches) + " |"
    separator = "| --- | " + " | ".join("---" for _ in sorted_approaches) + " |"
    report_lines.append(header)
    report_lines.append(separator)

    # Table Rows
    for metric in sorted_metrics:
        row = f"| **{metric.upper()}** |"
        for app in sorted_approaches:
            vals = []
            for d in all_data:
                if app in d and metric in d[app]:
                    vals.extend([v for v in d[app][metric] if v is not None and not np.isnan(v)])
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row += f" {mean_val:.4f} &plusmn; {std_val:.4f} |"
            else:
                row += " N/A |"
        report_lines.append(row)

    # Write out report
    report_path = os.path.join(results_dir, "summary_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Successfully generated summary report at: {report_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_sweep_report.py <results_directory>")
        sys.exit(1)
    generate_report(sys.argv[1])

if __name__ == "__main__":
    main()
