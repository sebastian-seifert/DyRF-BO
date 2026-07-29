#!/usr/bin/env python3
import os
import sys
import glob
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_single_test_file(test_file):
    t_start = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo_root)
    # Execute the python test script as a separate process
    res = subprocess.run([sys.executable, test_file], capture_output=True, text=True, env=env)
    duration = time.time() - t_start
    return test_file, res.returncode, duration, res.stdout, res.stderr

def main():
    t0 = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    
    test_files = sorted(glob.glob("tests/test_*.py"))
    num_workers = min(12, os.cpu_count() or 4)
    print(f"Executing {len(test_files)} test modules in parallel across {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(run_single_test_file, test_files))
        
    duration_total = time.time() - t0
    failed = [r for r in results if r[1] != 0]
    
    print("\n" + "=" * 60)
    print("PARALLEL TEST EXECUTION SUMMARY")
    print("=" * 60)
    for f, code, dur, out, err in results:
        status = "PASSED" if code == 0 else "FAILED"
        print(f"  [{status}] {f:<40} ({dur:5.2f}s)")
        
    print("=" * 60)
    if failed:
        print(f"\nFAILURE DETAILS ({len(failed)} files failed):")
        for f, code, dur, out, err in failed:
            print(f"\n--- Output from {f} ---")
            if out.strip():
                print(out.strip())
            if err.strip():
                print(err.strip())
        print(f"\nFAILED: {len(failed)}/{len(results)} test modules failed in {duration_total:.2f}s.")
        sys.exit(1)
    else:
        print(f"\nSUCCESS: All {len(results)} test modules passed in {duration_total:.2f}s!")
        sys.exit(0)

if __name__ == "__main__":
    main()
