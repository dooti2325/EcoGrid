import os
import re
import yaml
import subprocess
import sys

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(check, passed, details=""):
    if passed:
        print(f"{Colors.GREEN}[PASS]{Colors.RESET} {check}")
        if details: print(f"  -> {details}")
    else:
        print(f"{Colors.RED}[FAIL]{Colors.RESET} {check}")
        if details: print(f"  -> {Colors.RED}{details}{Colors.RESET}")

def run_checks():
    print(f"\n{Colors.BOLD}=== ECOGRID-OPENENV HACKATHON COMPLIANCE AUDIT ==={Colors.RESET}\n")
    all_passed = True
    
    # 1. OpenEnv Compliance
    try:
        with open("openenv.yaml", "r") as f:
            data = yaml.safe_load(f)
            has_obs = "observation_space" in data
            has_act = "action_space" in data
            has_tasks = "tasks" in data
            passed = has_obs and has_act and has_tasks
            print_status("OpenEnv Schema Compliance", passed, "Checked openenv.yaml for mandatory fields.")
            if not passed: all_passed = False
    except FileNotFoundError:
        print_status("OpenEnv Schema Compliance", False, "openenv.yaml not found.")
        all_passed = False

    # 2. Training Script (wandb)
    try:
        with open("train_unsloth.py", "r") as f:
            content = f.read()
            passed = "wandb" in content and "report_to=\"wandb\"" in content.replace(" ", "")
            print_status("Training Script (W&B)", passed, "Verified Weights & Biases integration.")
            if not passed: all_passed = False
    except FileNotFoundError:
        print_status("Training Script (W&B)", False, "train_unsloth.py not found.")
        all_passed = False

    # 3. Proof of Training
    has_reward = os.path.exists("docs/reward_curve.png")
    has_loss = os.path.exists("docs/loss_curve.png")
    passed = has_reward and has_loss
    print_status("Proof of Training Plots", passed, "Verified reward and loss PNGs exist.")
    if not passed: all_passed = False

    # 4. README Completeness
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
            has_space = "huggingface.co/spaces/" in content
            has_blog = "BLOG.md" in content or "blog" in content.lower()
            has_img = "docs/reward_curve.png" in content
            passed = has_space and has_blog and has_img
            print_status("README Completeness", passed, "Verified HF Space links, Blog links, and embedded images.")
            if not passed: all_passed = False
    except FileNotFoundError:
        print_status("README Completeness", False, "README.md not found.")
        all_passed = False

    # 5. Environment Health
    try:
        cmd = [sys.executable, "scripts/benchmark.py", "--seeds", "1"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
        passed = result.returncode == 0
        details = "Ran benchmark.py to test environment." if passed else result.stderr.strip().split('\n')[-1]
        print_status("Environment Run Test", passed, details)
        if not passed: all_passed = False
    except Exception as e:
        print_status("Environment Run Test", False, str(e))
        all_passed = False

    print(f"\n{Colors.BOLD}=== FINAL VERDICT ==={Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}FULLY COMPLIANT & COMPETITIVE{Colors.RESET}")
        print("All hackathon requirements are satisfied. The project is ready for submission.")
        sys.exit(0)
    else:
        print(f"{Colors.RED}AT RISK{Colors.RESET}")
        print("One or more mandatory hackathon checks failed. Do not submit until fixed.")
        sys.exit(1)

if __name__ == "__main__":
    run_checks()
