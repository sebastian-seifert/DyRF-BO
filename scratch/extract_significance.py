import os
import re

def parse_report_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
        
    # Split content by "METRIC: " sections
    sections = content.split("METRIC: ")
    naurc_section = None
    for sec in sections:
        if sec.startswith("NAURC"):
            naurc_section = sec
            break
            
    if not naurc_section:
        return None
        
    results = {}
    
    # 1. Parse Friedman tests
    # Format: Friedman: chi2 =  11.4000, p = 2.2418e-02 *
    # Format: Friedman: chi2 =   4.9600, p = 2.9143e-01 ns
    friedman_matches = re.finditer(r"> (All Functions|\d+D Functions)\n\s+Friedman: chi2 =\s+([0-9\.]+),\s+p =\s+([0-9e\.\-]+)\s+(\*|ns|\*\*|\*\*\*)", naurc_section)
    for m in friedman_matches:
        group_name = m.group(1)
        chi2 = float(m.group(2))
        p_val = float(m.group(3))
        sig = m.group(4)
        results[group_name] = {"friedman_p": p_val, "significant": (sig != "ns"), "pairwise": []}
        
        # Look for pairwise comparisons within this group
        group_start = naurc_section.find(f"> {group_name}")
        # Find next ">" or boundary
        group_end = naurc_section.find(">", group_start + 1)
        if group_end == -1:
            group_end = len(naurc_section)
        group_text = naurc_section[group_start:group_end]
        
        # Parse comparisons
        # Format: Proximity_Baseline vs Proximity_Method_A     6.5809e-01            [NS]
        comp_matches = re.finditer(r"([A-Za-z0-9_]+) vs ([A-Za-z0-9_]+)\s+([0-9e\.\-]+)\s+\[(Significant|\*)\]", group_text)
        for cm in comp_matches:
            results[group_name]["pairwise"].append((cm.group(1), cm.group(2), float(cm.group(3))))
            
    return results

def main():
    results_dir = "results"
    txt_files = [f for f in os.listdir(results_dir) if f.endswith(".txt") and "2026070" in f]
    
    significant_groups = []
    
    for filename in txt_files:
        filepath = os.path.join(results_dir, filename)
        sig_data = parse_report_file(filepath)
        if not sig_data:
            continue
            
        for group, stats in sig_data.items():
            if stats["significant"] or stats["pairwise"]:
                significant_groups.append({
                    "file": filename,
                    "group": group,
                    "friedman_p": stats["friedman_p"],
                    "pairwise": stats["pairwise"]
                })
                
    if not significant_groups:
        print("No statistically significant differences were found in any sweep report for NAURC.")
        return
        
    print(f"Found {len(significant_groups)} significant configurations/groups:")
    for sg in significant_groups:
        print(f"\nReport: {sg['file']}")
        print(f"Group: {sg['group']} (Friedman p-value = {sg['friedman_p']:.6f})")
        if sg['pairwise']:
            print("Significant Pairwise Comparisons:")
            for c1, c2, p in sg['pairwise']:
                print(f"  - {c1} vs {c2} (p = {p:.6f})")
        else:
            print("  No individual pairwise comparison passed the Bonferroni-corrected significance threshold.")

if __name__ == "__main__":
    main()
