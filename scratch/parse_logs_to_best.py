import os
import re
import glob

def main():
    # Dictionary to hold the data: metric -> dimension -> list of dicts
    data = {}

    log_files = glob.glob("results/**/logs/run_*.log", recursive=True)
    if not log_files:
        log_files = glob.glob("local_results/**/logs/run_*.log", recursive=True)
    print(f"Found {len(log_files)} log files to parse.")

    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract config
        config_match = re.search(r"Config: (.*?)\n", content)
        if not config_match:
            continue
        config_str = config_match.group(1)
        
        # Parse config details
        config_dict = {}
        for item in config_str.split(", "):
            if "=" in item:
                k, v = item.split("=")
                config_dict[k.strip()] = v.strip()
                
        # Find metric blocks
        metric_blocks = re.findall(
            r"-{40,}\nMETRIC:\s*([A-Z0-9@_]+)\n-{40,}\n(.*?)(?=-{40,}\nMETRIC:|\Z)",
            content,
            re.DOTALL
        )
        
        for metric_name, block in metric_blocks:
            metric_name = metric_name.strip()
            # Normalize names
            if metric_name == "FPR95":
                metric_name = "FPR@95TPR"
            
            # Find DESCRIPTIVE STATISTICS section
            stats_match = re.search(
                r"DESCRIPTIVE STATISTICS\s*\n\s*(.*?)\n-{40,}\n(.*?)\n\n",
                block,
                re.DOTALL
            )
            if not stats_match:
                continue
            
            headers_str = stats_match.group(1)
            approaches = re.split(r"\s+", headers_str.strip())
            
            rows_str = stats_match.group(2)
            for line in rows_str.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                row_match = re.match(r"^(\d+D)\s+Functions\s+(.*)$", line)
                if not row_match:
                    continue
                dim = row_match.group(1)
                values_str = row_match.group(2).strip()
                values = re.split(r"\s+", values_str)
                
                for i, val in enumerate(values):
                    if i >= len(approaches):
                        break
                    app = approaches[i]
                    # Parse mean+/-std
                    val_match = re.match(r"^([\-\d\.]+)\+/-(.*)$", val)
                    if val_match:
                        mean = float(val_match.group(1))
                        std = float(val_match.group(2))
                        
                        if metric_name not in data:
                            data[metric_name] = {}
                        if dim not in data[metric_name]:
                            data[metric_name][dim] = []
                        
                        data[metric_name][dim].append({
                            "mean": mean,
                            "std": std,
                            "approach": app,
                            "config": config_dict
                        })

    # Define best metric check
    lower_better_metrics = {"FPR@95TPR", "BRIER", "NAURC"}
    
    # Map approaches to their mathematically relevant parameters to prune unused defaults
    proximity_methods = {"Proximity_Baseline", "Proximity_Method_A", "Proximity_Method_B", "Proximity_Method_C", "Proximity_Method_B_C"}

    sorted_metrics = sorted(list(data.keys()))
    
    # Generate tables
    for metric in sorted_metrics:
        print(f"\n### Metric: {metric}")
        print("| Dimension | Best Approach | Value (Mean ± Std) | Best Configuration |")
        print("| :--- | :--- | :--- | :--- |")
        
        # Dimensions sorted logically: 1D to 10D
        dims = sorted(list(data[metric].keys()), key=lambda x: int(x[:-1]))
        for dim in dims:
            runs = data[metric][dim]
            if not runs:
                continue
            
            # Find best run
            if metric in lower_better_metrics:
                best_run = min(runs, key=lambda x: x["mean"])
            else:
                best_run = max(runs, key=lambda x: x["mean"])
                
            app = best_run["approach"]
            val_str = f"{best_run['mean']:.4f} ± {best_run['std']:.4f}"
            
            # Format config
            cfg = best_run["config"]
            cfg_parts = []
            if "RF Config" in cfg:
                cfg_parts.append(f"RF={cfg['RF Config']}")
            
            # Only append neighborhood details if relevant to proximity methods
            if app in proximity_methods:
                if "K Neighbors" in cfg and cfg["K Neighbors"] != "None":
                    cfg_parts.append(f"K={cfg['K Neighbors']}")
                if "density_scaling_alpha" in cfg:
                    cfg_parts.append(f"alpha={cfg['density_scaling_alpha']}")
                if "Topological Decay Lambda" in cfg and cfg["Topological Decay Lambda"] not in ("None", "null"):
                    cfg_parts.append(f"lambda={cfg['Topological Decay Lambda']}")
            
            if "Gap Type" in cfg:
                cfg_parts.append(f"Gap={cfg['Gap Type']}")
            if cfg.get("Gap Type") == "sparse" and "Multiplier" in cfg and cfg["Multiplier"] != "12":
                cfg_parts.append(f"M={cfg['Multiplier']}")
            if cfg.get("Gap Type") == "sparse" and "Scaling Law" in cfg:
                cfg_parts.append(f"Law={cfg['Scaling Law']}")
                
            cfg_str = ", ".join(cfg_parts)
            print(f"| {dim} | {app} | {val_str} | {cfg_str} |")

if __name__ == "__main__":
    main()
