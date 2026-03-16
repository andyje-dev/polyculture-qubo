"""Extract Table S10 from Bischoff et al. PNAS SI appendix PDF.

Table S10: "References for Intercropping combinations and number of
experiments included in the dataset."

Columns: row_num, SpeciesM (main), SpeciesC (companion), References, N.Exps

Uses pdftotext (poppler) with -layout flag for structured text extraction.

Usage:
    python scripts/extract_bischoff.py [path_to_pdf]

Output:
    data/raw/bischoff_species_pairs.csv
"""

import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PDF = DATA_DIR / "external" / "pnas.2415315121.sapp.pdf"
OUTPUT_CSV = DATA_DIR / "raw" / "bischoff_species_pairs.csv"

# Regex for Table S10 data rows:
# Leading whitespace, row number, two species names (genus + species), references in parens, experiment count
ROW_PATTERN = re.compile(
    r"^\s+(\d+)\s+"  # row number
    r"([A-Z][a-z]+ [a-z-]+)\s+"  # SpeciesM (genus species, allows hyphens)
    r"([A-Z][a-z]+ [a-z-]+)\s+"  # SpeciesC (genus species, allows hyphens)
    r"\(([^)]+)\)\s+"  # References in parentheses
    r"(\d+)\s*$"  # N.Exps
)


def extract_table_s10(pdf_path: Path) -> list[dict]:
    """Extract Table S10 from the Bischoff SI appendix PDF."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )

    text = result.stdout

    # Find table boundaries
    start_marker = "Table S10."
    end_marker = "Table S11."
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx == -1:
        raise ValueError("Could not find Table S10 in PDF")
    if end_idx == -1:
        raise ValueError("Could not find Table S11 (end boundary) in PDF")

    table_text = text[start_idx:end_idx]
    rows = []

    for line in table_text.splitlines():
        match = ROW_PATTERN.match(line)
        if match:
            rows.append(
                {
                    "row_num": int(match.group(1)),
                    "species_m": match.group(2),
                    "species_c": match.group(3),
                    "references": match.group(4),
                    "n_exps": int(match.group(5)),
                }
            )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write extracted rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("row_num,species_m,species_c,references,n_exps\n")
        for row in rows:
            # Quote references field since it contains commas
            refs = row["references"].replace('"', '""')
            f.write(
                f'{row["row_num"]},{row["species_m"]},{row["species_c"]},"{refs}",{row["n_exps"]}\n'
            )


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        print(
            "Download from: https://www.pnas.org/doi/suppl/10.1073/pnas.2415315121/suppl_file/pnas.2415315121.sapp.pdf"
        )
        sys.exit(1)

    print(f"Extracting Table S10 from {pdf_path.name}...")
    rows = extract_table_s10(pdf_path)
    print(f"  Extracted {len(rows)} species pair rows")

    # Summary stats
    unique_species = set()
    for row in rows:
        unique_species.add(row["species_m"])
        unique_species.add(row["species_c"])

    # Deduplicate directed pairs to undirected
    undirected_pairs = set()
    for row in rows:
        pair = tuple(sorted([row["species_m"], row["species_c"]]))
        undirected_pairs.add(pair)

    total_exps = sum(r["n_exps"] for r in rows)

    print(f"  Unique species: {len(unique_species)}")
    print(f"  Unique undirected pairs: {len(undirected_pairs)}")
    print(f"  Total experiments: {total_exps}")

    write_csv(rows, OUTPUT_CSV)
    print(f"  Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
