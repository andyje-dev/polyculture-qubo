"""Download raw datasets for the polyculture QUBO project.

Datasets:
  1. Paut et al. (2024) - Global intercropping/agroforestry horticulture dataset
     Source: Recherche Data Gouv (INRAE), DOI: 10.57745/HV33V1
     License: Open access

  2. Companion Plants network (Kaggle, aramacus)
     Source: https://www.kaggle.com/datasets/aramacus/companion-plants
     License: ODbL 1.0
     Note: Requires manual download or Kaggle CLI authentication.

  3. Bischoff et al. (2024) - PNAS systems agroecology SI appendix
     Source: DOI 10.1073/pnas.2415315121
     Note: Data is embedded in SI PDF tables, not a standalone spreadsheet.
     Downloaded for reference but requires manual extraction.
"""

import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DOWNLOADS = {
    "paut_database.csv": {
        "url": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/667421",
        "description": "Paut et al. intercropping database (1,544 experiments, 45 variables)",
    },
    "paut_database.xlsx": {
        "url": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/667422",
        "description": "Paut et al. intercropping database (Excel version)",
    },
    "paut_summary.csv": {
        "url": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/667423",
        "description": "Paut et al. variable summary / data dictionary",
    },
    "paut_unique_intercrops.xlsx": {
        "url": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/667428",
        "description": "Paut et al. list of unique intercropping systems",
    },
}


def download_file(url: str, dest: Path, description: str) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    print(f"  Downloading {dest.name}: {description}")
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  [ok] {dest.name} ({size_kb:.1f} KB)")
        return True
    except requests.RequestException as e:
        print(f"  [FAIL] {dest.name}: {e}")
        return False


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Downloading Paut et al. dataset ===")
    all_ok = True
    for filename, info in DOWNLOADS.items():
        dest = DATA_DIR / filename
        if not download_file(info["url"], dest, info["description"]):
            all_ok = False

    print()
    companion_path = DATA_DIR / "companion_plants.csv"
    if companion_path.exists():
        print("=== Companion Plants dataset already present ===")
    else:
        print("=== Companion Plants dataset (manual step) ===")
        print(
            "  Download from: https://www.kaggle.com/datasets/aramacus/companion-plants"
        )
        print(f"  Extract CSV to: {companion_path}")
        print("  Or use Kaggle CLI:")
        print(
            f"    kaggle datasets download -d aramacus/companion-plants -p {DATA_DIR} --unzip"
        )

    if all_ok:
        print("\nDone. Run preprocessing next.")
    else:
        print("\nSome downloads failed. Check errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
