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
                
        # Find all dimension blocks: [DIMENSION] 11D Functions
        # We split by [DIMENSION] to isolate each dimension's outputs
        dim_sections = re.split(r"\[DIMENSION\]\s*(\d+D)\s+Functions", content)
        if len(dim_sections) < 2:
            continue
            
        # The split returns: [prefix, dim1, content1, dim2, content2, ...]
        for i in range(1, len(dim_sections), 2):
            dim = dim_sections[i]
            section_content = dim_sections[i+1]
            
            # Find metric blocks in this section
            metric_blocks = re.findall(
                r"---\s*([A-Z0-9@_]+)\s*---\n(.*?)(?=\n---\s*[A-Z0-9@_]+\s*---|\Z)",
                section_content,
                re.DOTALL
            )
            
            for metric_name, block in metric_blocks:
                metric_name = metric_name.strip()
                # Normalize names
                if metric_name == "FPR95":
                    metric_name = "FPR@95TPR"
                
                # Parse lines like "Standard    : Mean = 0.6769, Std = 0.0554"
                for line in block.strip().split("\n"):
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    
                    line_match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*Mean\s*=\s*([\-\d\.]+),\s*Std\s*=\s*([\-\d\.]+)$", line)
                    if not line_match:
                        continue
                        
                    app = line_match.group(1)
                    mean = float(line_match.group(2))
                    std = float(line_match.group(3))
                    
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
    proximity_methods = {"Proximity_Baseline", "Proximity_Method_A", "Proximity_Method_B", "Proximity_Method_C", "Proximity_Method_B_C"}

    sorted_metrics = sorted(list(data.keys()))
    
    # Generate tables
    for metric in sorted_metrics:
        print(f"\n### Metric: {metric}")
        print("| Dimension | Best Approach | Value (Mean ± Std) | Best Configuration |")
        print("| :--- | :--- | :--- | :--- |")
        
        # Dimensions sorted logically: 1D to 15D
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
