import os
import shutil
import subprocess

def main():
    results_dir = "results"
    archive_dir = "local_results"
    
    if not os.path.exists(results_dir):
        print("results/ directory not found.")
        return
        
    os.makedirs(archive_dir, exist_ok=True)
    
    # List items in results/
    items = os.listdir(results_dir)
    archived = []
    
    for item in items:
        # Ignore the baseline logs directory or any git files
        if item in ["logs", ".gitignore", "density_scaling"]:
            continue
            
        src_path = os.path.join(results_dir, item)
        dst_path = os.path.join(archive_dir, item)
        
        print(f"Archiving {item}...")
        
        # 1. Copy to local_results/
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
            
        # 2. Stage deletion in git
        try:
            # Check if tracked by git
            res = subprocess.run(["git", "ls-files", "--error-unmatch", src_path], capture_output=True, text=True)
            if res.returncode == 0:
                subprocess.run(["git", "rm", "-r", "-f", src_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # If untracked, just delete from results/ locally
                if os.path.isdir(src_path):
                    shutil.rmtree(src_path)
                else:
                    os.remove(src_path)
        except Exception as e:
            print(f"Warning: Could not remove {src_path} via git: {e}")
            if os.path.isdir(src_path):
                shutil.rmtree(src_path)
            elif os.path.exists(src_path):
                os.remove(src_path)
                
        archived.append(item)
        
    if archived:
        print("\n🎉 Archiving complete!")
        print("The following items were copied to local_results/ (gitignored) and removed from results/:")
        for item in archived:
            print(f"  - {item}")
        print("\nTo push these deletions and clean up GitHub, execute:")
        print("  git commit -m \"Archive sweep results to local_results/\" && git push\n")
    else:
        print("No new result directories or files found to archive.")

if __name__ == "__main__":
    main()
