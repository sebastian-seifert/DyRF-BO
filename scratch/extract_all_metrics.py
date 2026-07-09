import os
import re
import json
import numpy as np

def parse_filename(filename):
    pattern = r"uncertainty_quantification_results_rf(\d+)_k([a-zA-Z0-9]+)_([a-z]+)_m(\d+)_([a-z]+)_(\d{8})_(\d{6})\.json"
    match = re.search(pattern, filename)
    if match:
        rf_config = int(match.group(1))
        k = match.group(2)
        gap_type = match.group(3)
        mult = int(match.group(4))
        law = match.group(5)
        return rf_config, k, gap_type, mult, law
    return None

def main():
    results_dir = "results"
    all_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and "2026070" in f]
    
    metrics_list = ["auroc", "spearman", "brier", "mi", "jsd", "naurc"]
    data = {m: {} for m in metrics_list}
    
    for filename in all_files:
        config_info = parse_filename(filename)
        if not config_info:
            continue
        rf_config, k, gap_type, mult, law = config_info
        
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "r") as f:
                d = json.load(f)
        except Exception as e:
            continue
            
        approaches = d.get("approaches", [])
        results_by_dim = d.get("results_by_dim", {})
        
        for app in approaches:
            for m in metrics_list:
                if app not in data[m]:
                    data[m][app] = {}
                if gap_type not in data[m][app]:
                    data[m][app][gap_type] = {}
                    
            for dim, app_data in results_by_dim.items():
                for app_name, metrics_dict in app_data.items():
                    if app_name != app:
                        continue
                    for m in metrics_list:
                        if dim not in data[m][app][gap_type]:
                            data[m][app][gap_type][dim] = {}
                        
                        vals = metrics_dict.get(m, [])
                        if not vals:
                            continue
                            
                        config_key = (rf_config, k, mult, law)
                        if config_key not in data[m][app][gap_type][dim]:
                            data[m][app][gap_type][dim][config_key] = []
                        data[m][app][gap_type][dim][config_key].extend(vals)

    # Process likelihood variants
    likelihood_variants = [
        "Shaker_Likelihood_GL_Bisect",
        "Shaker_Likelihood_GL_Newton",
        "Shaker_Likelihood_Trapz_Bisect",
        "Shaker_Likelihood_Trapz_Newton"
    ]
    for m in metrics_list:
        data[m]["Shaker_Likelihood"] = {}
        for gap_type in ["empty", "sparse"]:
            data[m]["Shaker_Likelihood"][gap_type] = {}
            for dim in ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D"]:
                data[m]["Shaker_Likelihood"][gap_type][dim] = {}
                for variant in likelihood_variants:
                    if variant in data[m] and gap_type in data[m][variant] and dim in data[m][variant][gap_type]:
                        for cfg, vals in data[m][variant][gap_type][dim].items():
                            if cfg not in data[m]["Shaker_Likelihood"][gap_type][dim]:
                                data[m]["Shaker_Likelihood"][gap_type][dim][cfg] = []
                            data[m]["Shaker_Likelihood"][gap_type][dim][cfg].extend(vals)

    app_mapping = {
        "Standard": "Standard RF",
        "Chen": "Chen Paired",
        "Shaker_GMM_Entropy": "Shaker GMM",
        "Shaker_Likelihood": "Credal Likelihood",
        "Proximity_Baseline": "Std Prox",
        "Proximity_Method_A": "Method A",
        "Proximity_Method_B": "Method B",
        "Proximity_Method_C": "Method C",
        "Proximity_Method_B_C": "Method B+C"
    }
    
    dimensions = ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D"]
    gap_types = ["empty", "sparse"]
    
    def get_best_stats(m_name, app_name, gap_type, dim):
        if app_name not in data[m_name] or gap_type not in data[m_name][app_name] or dim not in data[m_name][app_name][gap_type]:
            return None, None
            
        configs = data[m_name][app_name][gap_type][dim]
        maximize = m_name in ["auroc", "spearman", "mi"]
        best_val = float("-inf") if maximize else float("inf")
        best_std = 0.0
        best_cfg = None
        
        for cfg, vals in configs.items():
            if not vals:
                continue
            mean = np.mean(vals)
            if maximize:
                if mean > best_val:
                    best_val = mean
                    best_std = np.std(vals)
                    best_cfg = cfg
            else:
                if mean < best_val:
                    best_val = mean
                    best_std = np.std(vals)
                    best_cfg = cfg
                    
        if best_cfg is None:
            return None, None
        return best_val, best_std

    # Generate Beamer Slides
    latex_slides = []
    latex_slides.append("\n\\appendix\n")
    
    metric_labels = {
        "auroc": "AUROC (ID vs. OOD Classification)",
        "spearman": "Spearman Rank Correlation ($r_s$)",
        "brier": "Brier Score (Calibration)",
        "mi": "Mutual Information (MI)",
        "jsd": "Jensen-Shannon Divergence (JSD)"
    }
    
    for m in ["auroc", "spearman", "brier", "mi", "jsd"]:
        label = metric_labels[m]
        maximize = m in ["auroc", "spearman", "mi"]
        opt_dir = "Higher is better" if maximize else "Lower is better"
        
        slide = f"\\begin{{frame}}{{Appendix: Global {label}}}\n"
        slide += f"    \\centering \\scriptsize ({opt_dir})\n\n"
        slide += "    \\begin{columns}\n"
        
        for gap_type in gap_types:
            slide += "        \\begin{column}{0.5\\textwidth}\n"
            slide += f"            \\centering \\textbf{{{gap_type.capitalize()} Gap}}\n\n"
            slide += "            \\begin{table}\n"
            slide += "                \\tiny\\resizebox{\\columnwidth}{!}{\n"
            slide += "                \\begin{tabular}{lccccccccc}\n"
            slide += "                    \\toprule\n"
            slide += "                    \\textbf{Dim} & \\textbf{Std RF} & \\textbf{Chen} & \\textbf{GMM} & \\textbf{Credal} & \\textbf{Std Prox} & \\textbf{A} & \\textbf{B} & \\textbf{C} & \\textbf{B+C} \\\\\n"
            slide += "                    \\midrule\n"
            
            for dim in dimensions:
                row_vals = []
                # Collect values to find the best on this row to bold it
                val_mean_list = []
                for app in app_mapping.keys():
                    mean, std = get_best_stats(m, app, gap_type, dim)
                    val_mean_list.append(mean)
                
                # Filter out None
                valid_means = [v for v in val_mean_list if v is not None]
                best_idx = None
                if valid_means:
                    best_val = max(valid_means) if maximize else min(valid_means)
                    best_idx = val_mean_list.index(best_val)
                
                for idx, app in enumerate(app_mapping.keys()):
                    mean, std = get_best_stats(m, app, gap_type, dim)
                    if mean is not None:
                        val_str = f"{mean:.3f}"
                        if idx == best_idx:
                            row_vals.append(f"\\textbf{{{val_str}}}")
                        else:
                            row_vals.append(val_str)
                    else:
                        row_vals.append("N/A")
                slide += f"                    {dim} & " + " & ".join(row_vals) + " \\\\\n"
                
            slide += "                    \\bottomrule\n"
            slide += "                \\end{tabular}\n"
            slide += "                }\n"
            slide += "            \\end{table}\n"
            slide += "        \\end{column}\n"
            
        slide += "    \\end{columns}\n"
        slide += "\\end{frame}\n\n"
        latex_slides.append(slide)

    # Insert into the presentation TeX file
    tex_path = "../presentation_progress_2026_07_07.tex"
    if os.path.exists(tex_path):
        with open(tex_path, "r") as f:
            tex_content = f.read()
            
        # Find where to insert: right before \end{document}
        end_doc_idx = tex_content.rfind("\\end{document}")
        if end_doc_idx != -1:
            # Let's strip any existing \appendix and following slides up to \end{document}
            appendix_idx = tex_content.find("\\appendix")
            if appendix_idx != -1:
                new_tex = tex_content[:appendix_idx] + "".join(latex_slides) + "\\end{document}\n"
            else:
                new_tex = tex_content[:end_doc_idx] + "".join(latex_slides) + "\\end{document}\n"
                
            with open(tex_path, "w") as f:
                f.write(new_tex)
            print(f"[Presentation] Updated presentation with appendix slides: {tex_path}")

    # Also print markdown tables to stdout
    for m in metrics_list:
        print(f"\n================================================================================")
        print(f"METRIC: {m.upper()} ({'Maximize' if m in ['auroc', 'spearman', 'mi'] else 'Minimize'})")
        print(f"================================================================================")
        for gap_type in gap_types:
            print(f"\n### GAP TYPE: {gap_type.upper()}")
            header = "| Dim | " + " | ".join(app_mapping.values()) + " |"
            sep = "|---|" + "|".join(["---" for _ in app_mapping]) + "|"
            print(header)
            print(sep)
            for dim in dimensions:
                row_vals = []
                for app in app_mapping.keys():
                    mean, std = get_best_stats(m, app, gap_type, dim)
                    if mean is not None:
                        row_vals.append(f"{mean:.4f}±{std:.4f}")
                    else:
                        row_vals.append("N/A")
                print(f"| {dim} | " + " | ".join(row_vals) + " |")

if __name__ == "__main__":
    main()
