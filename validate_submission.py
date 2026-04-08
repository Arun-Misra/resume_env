#!/usr/bin/env python3
"""
validate_submission.py — OpenEnv Submission Validator (resume_env)
===================================================================

Mirrors the official validate-submission.sh from the contest spec.
Works on Windows, Linux and macOS.

Checks (in order — hard-stops on failure):
  Step 1/3 — HF Space is live: POST /reset must return HTTP 200
  Step 2/3 — Docker builds: dockerfile found and `docker build` succeeds
  Step 3/3 — openenv validate passes

Prerequisites:
  pip install openenv-core requests
  Docker Desktop (or Docker Engine) installed and running

Usage:
  python validate_submission.py <ping_url> [repo_dir]

  ping_url   Your HuggingFace Space URL  (e.g. https://arun-misra-resume-env.hf.space)
  repo_dir   Path to your repo           (default: current directory)

Examples:
  python validate_submission.py https://arun-misra-resume-env.hf.space
  python validate_submission.py https://arun-misra-resume-env.hf.space .
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DOCKER_BUILD_TIMEOUT = 600  # seconds

# ANSI colours (disabled on Windows CI without colour support)
_use_colour = sys.stdout.isatty() and os.name != "nt"
RED    = "\033[0;31m"  if _use_colour else ""
GREEN  = "\033[0;32m"  if _use_colour else ""
YELLOW = "\033[1;33m"  if _use_colour else ""
BOLD   = "\033[1m"     if _use_colour else ""
NC     = "\033[0m"     if _use_colour else ""

passed = 0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}")

def ok(msg: str):
    global passed
    passed += 1
    log(f"{GREEN}PASSED{NC} -- {msg}")

def fail(msg: str):
    log(f"{RED}FAILED{NC} -- {msg}")

def hint(msg: str):
    print(f"  {YELLOW}Hint:{NC} {msg}")

def stop_at(step: str):
    print()
    print(f"{RED}{BOLD}Validation stopped at {step}.{NC} Fix the above before continuing.")
    sys.exit(1)

def divider():
    print(f"{BOLD}{'=' * 44}{NC}")


# ─────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <ping_url> [repo_dir]")
    print()
    print("  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)")
    print("  repo_dir   Path to your repo (default: current directory)")
    sys.exit(1)

PING_URL = sys.argv[1].rstrip("/")
REPO_DIR = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else ".")

if not os.path.isdir(REPO_DIR):
    print(f"Error: directory '{REPO_DIR}' not found")
    sys.exit(1)

print()
divider()
print(f"{BOLD}  OpenEnv Submission Validator — resume_env{NC}")
divider()
log(f"Repo:     {REPO_DIR}")
log(f"Ping URL: {PING_URL}")
print()


# ─────────────────────────────────────────────
# Step 1/3 — Ping HF Space /reset
# ─────────────────────────────────────────────
log(f"{BOLD}Step 1/3: Pinging HF Space{NC} ({PING_URL}/reset) ...")

try:
    resp = requests.post(
        f"{PING_URL}/reset",
        json={},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    http_code = resp.status_code
except requests.exceptions.ConnectionError:
    fail("HF Space not reachable (connection failed or timed out)")
    hint("Check your network and that the Space is Running (not sleeping).")
    hint(f"Try opening {PING_URL} in your browser.")
    stop_at("Step 1")
except requests.exceptions.Timeout:
    fail("HF Space /reset timed out after 30 seconds")
    hint("The Space may be cold-starting. Wait 60 seconds and retry.")
    stop_at("Step 1")
except Exception as exc:
    fail(f"Unexpected error pinging Space: {exc}")
    stop_at("Step 1")

if http_code == 200:
    ok("HF Space is live and responds to /reset")
else:
    fail(f"HF Space /reset returned HTTP {http_code} (expected 200)")
    hint("Make sure your Space is running and the URL is correct.")
    hint(f"Try opening {PING_URL} in your browser first.")
    stop_at("Step 1")


# ─────────────────────────────────────────────
# Step 2/3 — Docker build
# ─────────────────────────────────────────────
log(f"{BOLD}Step 2/3: Running docker build{NC} ...")

# Locate Dockerfile (repo root first, then server/)
dockerfile_dir = None
if os.path.isfile(os.path.join(REPO_DIR, "Dockerfile")):
    dockerfile_dir = REPO_DIR
elif os.path.isfile(os.path.join(REPO_DIR, "server", "Dockerfile")):
    dockerfile_dir = os.path.join(REPO_DIR, "server")

if dockerfile_dir is None:
    fail("No Dockerfile found in repo root or server/ directory")
    stop_at("Step 2")

log(f"  Found Dockerfile in {dockerfile_dir}")

# Check Docker is installed
try:
    subprocess.run(
        ["docker", "info"],
        capture_output=True, timeout=10, check=True,
    )
except FileNotFoundError:
    fail("docker command not found")
    hint("Install Docker Desktop: https://docs.docker.com/get-docker/")
    stop_at("Step 2")
except subprocess.CalledProcessError:
    fail("Docker is installed but not running")
    hint("Start Docker Desktop and try again.")
    stop_at("Step 2")

# Run docker build
try:
    result = subprocess.run(
        ["docker", "build", dockerfile_dir],
        capture_output=True, text=True,
        timeout=DOCKER_BUILD_TIMEOUT,
    )
    if result.returncode == 0:
        ok("Docker build succeeded")
    else:
        fail(f"Docker build failed (exit code {result.returncode})")
        # Show last 20 lines of build output
        build_lines = (result.stdout + result.stderr).strip().splitlines()
        for line in build_lines[-20:]:
            print(f"    {line}")
        stop_at("Step 2")
except subprocess.TimeoutExpired:
    fail(f"Docker build timed out after {DOCKER_BUILD_TIMEOUT}s")
    stop_at("Step 2")


# ─────────────────────────────────────────────
# Step 3/3 — openenv validate
# ─────────────────────────────────────────────
log(f"{BOLD}Step 3/3: Running openenv validate{NC} ...")

# Check openenv is installed
try:
    subprocess.run(
        ["openenv", "--version"],
        capture_output=True, timeout=10,
    )
except FileNotFoundError:
    fail("openenv command not found")
    hint("Install it: pip install openenv-core")
    stop_at("Step 3")

# Run openenv validate
try:
    result = subprocess.run(
        ["openenv", "validate"],
        cwd=REPO_DIR,
        capture_output=True, text=True,
        timeout=120,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        ok("openenv validate passed")
        if output:
            log(f"  {output}")
    else:
        fail("openenv validate failed")
        print(output)
        stop_at("Step 3")
except subprocess.TimeoutExpired:
    fail("openenv validate timed out after 120 seconds")
    stop_at("Step 3")


# ─────────────────────────────────────────────
# All checks passed
# ─────────────────────────────────────────────
print()
divider()
print(f"{GREEN}{BOLD}  All 3/3 checks passed!{NC}")
print(f"{GREEN}{BOLD}  Your submission is ready to submit.{NC}")
divider()
print()
sys.exit(0)
