with open("inference_1080p.py", "r") as f:
    code = f.read()

path_injection = """import sys
import os
# Ensure animegan2-pytorch directory is in sys.path for model.py import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'animegan2-pytorch')))
"""

if "sys.path.insert" not in code:
    code = path_injection + code
    with open("inference_1080p.py", "w") as f:
        f.write(code)
    print("[+] Successfully patched inference_1080p.py with automatic sys.path injection!")
else:
    print("[*] Inference script already patched.")
