"""Download the Oracle's Elixir match-data CSVs (2024/2025/2026) into data/raw/.
Skips any file that already exists and has a valid 'gameid' header.
Works on any OS with Python + requests (no Unix-only code).
"""
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install -r requirements.txt")

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

FILES = {
    # 2026 is shipped with the project (already in data/raw/2026.csv); listed here for reference.
    "2024": "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    "2025": "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
}


def _valid_csv(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(6) == "gameid"
    except OSError:
        return False


def download(fid, dest):
    urls = [
        f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={fid}&confirm=t",
    ]
    last_err = None
    for url in urls:
        try:
            print(f"  trying {url[:70]}...")
            with requests.get(url, timeout=300, stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            if _valid_csv(dest):
                print(f"  OK: {dest}")
                return True
            print("  downloaded file is not a CSV (bad header), trying next URL...")
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"Could not download a valid CSV for {dest}. Last error: {last_err}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    for year, fid in FILES.items():
        dest = os.path.join(RAW_DIR, f"{year}.csv")
        if _valid_csv(dest):
            print(f"{dest}: already present and valid, skipping")
            continue
        print(f"downloading {year}.csv ...")
        download(fid, dest)
    print("done.")


if __name__ == "__main__":
    main()
