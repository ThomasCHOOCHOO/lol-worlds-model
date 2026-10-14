"""Post-game model refresh: pull the latest Oracle's Elixir 2026 data, and if it
contains games newer than the current snapshot, rebuild features, retrain
(same v1.2.0 pipeline), re-export phone_artifacts.json with a patch-bumped
model_version, validate, and copy the bundle to the iPhone app.

Idempotent and safe:
  * If the download fails (Drive quota, network) or no games are newer than
    the snapshot, it logs and exits 0 without touching any artifacts.
  * On any pipeline failure after replacing data/raw/2026.csv, the previous
    CSV is restored from backup and the run exits 1.

Usage:
  .venv/bin/python src/refresh.py            # normal run (from project root or anywhere)
  .venv/bin/python src/refresh.py --check    # only report whether new games exist

Exit codes: 0 = ok (refreshed or clean no-op), 1 = failed (data restored).
Stdout ends with a single machine-readable line:  REFRESH_STATUS=<refreshed|noop|failed>
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
RAW_2026 = os.path.join(BASE, "data", "raw", "2026.csv")
ARTIFACTS = os.path.join(BASE, "model", "phone_artifacts.json")
APP_DATA = os.path.expanduser("~/workspace/lol-predictor-app/src/data/phone_artifacts.json")
LOG_PATH = os.path.join(BASE, "reports", "refresh_log.md")
VENV_PY = os.path.join(BASE, ".venv", "bin", "python")

# Oracle's Elixir 2026 CSV on public Google Drive (linked from
# oracleselixir.com/tools/downloads; the old S3 bucket is dead).
DRIVE_FILE_ID_2026 = "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}


def log(msg):
    print(f"[refresh] {msg}", flush=True)


def append_run_log(lines):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(LOG_PATH, "a") as f:
        f.write(f"\n## {ts}\n")
        for ln in lines:
            f.write(f"- {ln}\n")


def current_snapshot_date():
    """Snapshot date from the exported artifacts; fallback to raw CSV max date."""
    try:
        with open(ARTIFACTS) as f:
            return json.load(f)["snapshot_date"]
    except (OSError, KeyError, ValueError):
        return max_game_date(RAW_2026)


def max_game_date(csv_path):
    latest = None
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not header.startswith("gameid"):
            raise ValueError(f"{csv_path}: bad header, not an Oracle's Elixir CSV")
        didx = header.rstrip("\n").split(",").index("date")
        for line in f:
            # date is field didx; gameid never contains a comma, and date comes
            # before any quoted fields, so a bounded split is safe.
            parts = line.split(",", didx + 1)
            if len(parts) > didx and parts[didx]:
                d = parts[didx].strip()[:10]
                if not latest or d > latest:
                    latest = d
    return latest


def download_latest_2026(dest):
    """Download the current 2026 CSV from Google Drive. Returns (ok, reason)."""
    import requests
    s = requests.Session()
    s.headers.update(UA)
    s.headers["Accept"] = "text/html,application/xhtml+xml,*/*"
    patterns = [
        ("https://drive.usercontent.google.com/download",
         {"id": DRIVE_FILE_ID_2026, "export": "download", "confirm": "t"}),
        ("https://drive.google.com/uc",
         {"export": "download", "id": DRIVE_FILE_ID_2026, "confirm": "t"}),
    ]
    for base, params in patterns:
        try:
            with s.get(base, params=params, timeout=300, stream=True) as r:
                ctype = r.headers.get("content-type", "")
                if "text/html" in ctype:
                    body = r.text[:400] if hasattr(r, "text") else ""
                    if "Quota exceeded" in body or "can't view or download" in body:
                        return False, "Google Drive quota exceeded for the 2026 file (retries later)"
                    continue  # interstitial we can't follow; try next pattern
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            with open(dest, encoding="utf-8", errors="replace") as f:
                if f.read(6) == "gameid":
                    return True, "downloaded"
            os.remove(dest)
        except Exception as e:  # noqa: BLE001
            log(f"download attempt failed ({base[:45]}...): {e}")
    return False, "all download patterns failed (network or Drive block)"


def bump_patch(version):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not m:
        raise ValueError(f"cannot bump non-semver version {version!r}")
    major, minor, patch = map(int, m.groups())
    return f"{major}.{minor}.{patch + 1}"


def run_step(name, args, env_extra=None):
    log(f"running: {name}")
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([VENV_PY] + args, cwd=BASE, env=env,
                       capture_output=True, text=True, timeout=3600)
    tail = (p.stdout or "")[-800:] + (p.stderr or "")[-800:]
    if p.returncode != 0:
        log(f"{name} FAILED (exit {p.returncode}):\n{tail}")
        return False
    log(f"{name} ok")
    return True


def main():
    check_only = "--check" in sys.argv
    run_lines = []
    try:
        snap = current_snapshot_date()
    except Exception as e:  # noqa: BLE001
        log(f"cannot determine current snapshot: {e}")
        print("REFRESH_STATUS=failed")
        return 1
    log(f"current snapshot date: {snap}")
    run_lines.append(f"snapshot before run: {snap}")

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    try:
        ok, reason = download_latest_2026(tmp.name)
    finally:
        pass
    if not ok:
        msg = f"no refresh: download unavailable ({reason}); artifacts untouched"
        log(msg)
        run_lines.append(msg)
        append_run_log(run_lines)
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        print("REFRESH_STATUS=noop")
        return 0

    try:
        latest = max_game_date(tmp.name)
    except ValueError as e:
        log(f"downloaded file invalid: {e}")
        append_run_log(run_lines + ["downloaded file failed validation; artifacts untouched"])
        print("REFRESH_STATUS=noop")
        return 0
    log(f"latest game in download: {latest}")
    run_lines.append(f"latest game in Oracle's Elixir 2026 feed: {latest}")

    if not latest or latest <= snap:
        msg = "no refresh: no games newer than snapshot; artifacts untouched"
        log(msg)
        run_lines.append(msg)
        append_run_log(run_lines)
        print("REFRESH_STATUS=noop")
        return 0

    if check_only:
        log(f"CHECK: {latest} > {snap}: refresh would run")
        print("REFRESH_STATUS=noop")
        return 0

    # --- new games: back up, replace, rebuild ---
    with open(ARTIFACTS) as f:
        old_version = json.load(f)["model_version"]
    new_version = bump_patch(old_version)
    backup = RAW_2026 + f".bak.{snap}"
    shutil.copy2(RAW_2026, backup)
    shutil.copy2(tmp.name, RAW_2026)
    log(f"installed new 2026.csv (backup: {os.path.basename(backup)}); "
        f"version {old_version} -> {new_version}")
    run_lines.append(f"new games {snap} -> {latest}; version {old_version} -> {new_version}")

    steps = [
        ("build_features", ["src/build_features.py"], None),
        ("train", ["src/train.py"], None),
        ("export_artifacts", ["src/export_artifacts.py"],
         {"MODEL_VERSION_OVERRIDE": new_version}),
        ("validate_export", ["src/validate_export.py"], None),
    ]
    for name, args, env_extra in steps:
        if not run_step(name, args, env_extra):
            shutil.copy2(backup, RAW_2026)
            msg = f"{name} failed; restored previous 2026.csv from backup"
            log(msg)
            append_run_log(run_lines + [msg])
            print("REFRESH_STATUS=failed")
            return 1

    # fresh bundle -> iPhone app
    shutil.copy2(ARTIFACTS, APP_DATA)
    run_lines.append(f"copied phone_artifacts.json to app ({os.path.getsize(APP_DATA)//1024} KB)")
    log("app bundle updated")

    with open(ARTIFACTS) as f:
        art = json.load(f)
    m = art["metrics"]
    run_lines.append(
        f"done: snapshot {art['snapshot_date']}, version {art['model_version']}, "
        f"holdout n={m['holdout_n']} acc={m['holdout_accuracy']} ece={m['holdout_ece']}")
    append_run_log(run_lines)
    log(f"refresh complete: {art['model_version']} @ {art['snapshot_date']}")
    print("REFRESH_STATUS=refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
