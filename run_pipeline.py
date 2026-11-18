<<<<<<< Updated upstream
#!/usr/bin/env python3
import os
import argparse
from stage1._version import get_version_info
from stage4_stylized_avatar import run_stylized_avatar_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline Orchestrator for AI Video Dubbing & Avatars")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input source video (e.g., sample_clip.mp4)")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="Root directory for outputs")
    return parser.parse_args()
=======
import sys
import subprocess
from pathlib import Path
>>>>>>> Stashed changes

def main():
    root_dir = Path(__file__).parent.resolve()
    target_script = root_dir / "batch_pipeline.py"
    
    if not target_script.exists():
        print(f"[!] Error: Target script '{target_script.name}' not found in {root_dir}")
        sys.exit(1)
        
    print(f"[*] Delegating execution to core script: {target_script.name}")
    
    # Use the virtual environment python interpreter if available, otherwise system python
    venv_python = root_dir / "animegan2-pytorch" / "venv_win" / "Scripts" / "python.exe"
    python_exec = str(venv_python) if venv_python.exists() else sys.executable
    
    # Forward all arguments and execute batch_pipeline.py dynamically
    cmd = [python_exec, "-u", str(target_script)] + sys.argv[1:]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
