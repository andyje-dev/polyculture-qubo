"""Load and clean raw datasets into standardized DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd

from regen_qubo.species import normalize_species_name

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_paut_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load and clean the Paut et al. intercropping dataset.

    Returns a DataFrame with columns:
        crop_1: canonical species key (or None if not in candidate pool)
        crop_2: canonical species key (or None if not in candidate pool)
        crop_1_raw: original species name from dataset
        crop_2_raw: original species name from dataset
        ler_total: Land Equivalent Ratio (best available: reported or calculated)
        ler_crop1: partial LER for crop 1
        ler_crop2: partial LER for crop 2
        n_fixer_combo: nitrogen fixation combination (e.g., 'no-yes')
        country: country where experiment was conducted
        climate_zone: Köppen climate zone
        latitude: latitude of experiment site
        longitude: longitude of experiment site
    """
    if path is None:
        path = RAW_DIR / "paut_database.csv"

    df = pd.read_csv(path, sep=";")

    # Use reported LER if available, fall back to calculated
    ler_total = pd.to_numeric(df["LER_tot"], errors="coerce")
    ler_total_calc = pd.to_numeric(df["LER_tot_calc"], errors="coerce")
    ler_crop1 = pd.to_numeric(df["LER_crop1"], errors="coerce")
    ler_crop2 = pd.to_numeric(df["LER_crop2"], errors="coerce")
    ler_crop1_calc = pd.to_numeric(df["LER_crop_1_calc"], errors="coerce")
    ler_crop2_calc = pd.to_numeric(df["LER_crop_2_calc"], errors="coerce")

    result = pd.DataFrame(
        {
            "crop_1_raw": df["Crop_1_Common_Name"].str.strip(),
            "crop_2_raw": df["Crop_2_Common_Name"].str.strip(),
            "crop_1": df["Crop_1_Common_Name"].map(normalize_species_name),
            "crop_2": df["Crop_2_Common_Name"].map(normalize_species_name),
            "ler_total": ler_total.fillna(ler_total_calc),
            "ler_crop1": ler_crop1.fillna(ler_crop1_calc),
            "ler_crop2": ler_crop2.fillna(ler_crop2_calc),
            "n_fixer_combo": df["Comb_Nf"],
            "country": df["Country"],
            "climate_zone": df["Climate_zone"],
            "latitude": pd.to_numeric(df["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["Longitude"], errors="coerce"),
            "article_id": df["Id_article"],
        }
    )

    return result


def load_companion_plants(path: Path | None = None) -> pd.DataFrame:
    """Load the companion planting network dataset.

    Returns a DataFrame with columns:
        crop_1: canonical species key (or None)
        crop_2: canonical species key (or None)
        crop_1_raw: raw source plant name
        crop_2_raw: raw destination plant name
        interaction: +1 for helps/helped_by, -1 for avoid
    """
    if path is None:
        path = RAW_DIR / "companion_plants.csv"

    df = pd.read_csv(path)

    # Normalize column names (dataset uses 'Source Node', 'Destination Node', 'Link')
    source_col = [c for c in df.columns if "source" in c.lower() and "prop" not in c.lower()][0]
    dest_col = [c for c in df.columns if "dest" in c.lower()][0]
    link_col = [c for c in df.columns if "link" in c.lower()][0]

    interaction_map = {"helps": 1, "helped_by": 1, "avoid": -1}

    result = pd.DataFrame(
        {
            "crop_1_raw": df[source_col].str.strip(),
            "crop_2_raw": df[dest_col].str.strip(),
            "crop_1": df[source_col].map(normalize_species_name),
            "crop_2": df[dest_col].map(normalize_species_name),
            "interaction": df[link_col].str.strip().str.lower().map(interaction_map),
        }
    )

    return result


def load_bischoff_pairs(path: Path | None = None) -> pd.DataFrame:
    """Load Bischoff et al. species pair data extracted from PNAS SI appendix.

    Returns a DataFrame with columns:
        crop_1: canonical species key (or None)
        crop_2: canonical species key (or None)
        crop_1_raw: scientific name of main species
        crop_2_raw: scientific name of companion species
        n_exps: number of experiments for this directed pair
    """
    if path is None:
        path = RAW_DIR / "bischoff_species_pairs.csv"

    df = pd.read_csv(path)

    result = pd.DataFrame(
        {
            "crop_1_raw": df["species_m"],
            "crop_2_raw": df["species_c"],
            "crop_1": df["species_m"].map(normalize_species_name),
            "crop_2": df["species_c"].map(normalize_species_name),
            "n_exps": df["n_exps"],
        }
    )

    return result


def aggregate_bischoff_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate directed Bischoff pairs into undirected pair experiment counts.

    Takes filtered output of load_bischoff_pairs().
    Returns DataFrame with columns:
        species_a, species_b: canonical keys (alphabetically ordered)
        n_exps_total: total experiments across both directions
    """
    pairs = pd.DataFrame(
        {
            "species_a": np.minimum(df["crop_1"], df["crop_2"]),
            "species_b": np.maximum(df["crop_1"], df["crop_2"]),
            "n_exps": df["n_exps"].values,
        }
    )

    return (
        pairs.groupby(["species_a", "species_b"])
        .agg(n_exps_total=("n_exps", "sum"))
        .reset_index()
        .sort_values("n_exps_total", ascending=False)
        .reset_index(drop=True)
    )


def filter_to_candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a DataFrame to only rows where both species are in our candidate pool."""
    mask = df["crop_1"].notna() & df["crop_2"].notna()
    # Exclude self-pairs
    mask = mask & (df["crop_1"] != df["crop_2"])
    return df[mask].copy()


def compute_pairwise_ler_stats(
    df: pd.DataFrame,
    ler_cap: float = 3.0,
) -> pd.DataFrame:
    """Compute pairwise LER statistics from filtered Paut data.

    Takes output of filter_to_candidate_pairs(load_paut_dataset()).

    Args:
        df: Filtered DataFrame with ler_total, crop_1, crop_2, article_id columns.
        ler_cap: Maximum credible LER value. Observations above this are
            winsorized (capped) to reduce influence of outliers from extreme
            experimental conditions. Default 3.0 based on meta-analysis norms.

    Returns DataFrame with one row per species pair and columns:
        species_a, species_b: canonical keys (alphabetically ordered)
        ler_mean: mean LER across observations
        ler_std: standard deviation
        ler_median: median LER
        n_obs: number of observations
        n_articles: number of distinct articles/studies
        pct_above_1: fraction of observations with LER > 1.0
    """
    filtered = df.dropna(subset=["ler_total"]).copy()

    # Winsorize extreme LER values
    filtered["ler_total"] = filtered["ler_total"].clip(upper=ler_cap)

    # Normalize pair ordering (alphabetical) so (A,B) == (B,A)
    pairs = pd.DataFrame(
        {
            "species_a": np.minimum(filtered["crop_1"], filtered["crop_2"]),
            "species_b": np.maximum(filtered["crop_1"], filtered["crop_2"]),
            "ler_total": filtered["ler_total"].values,
            "article_id": filtered["article_id"].values,
        }
    )

    stats = (
        pairs.groupby(["species_a", "species_b"])
        .agg(
            ler_mean=("ler_total", "mean"),
            ler_std=("ler_total", "std"),
            ler_median=("ler_total", "median"),
            n_obs=("ler_total", "size"),
            n_articles=("article_id", "nunique"),
            pct_above_1=("ler_total", lambda x: (x > 1.0).mean()),
        )
        .reset_index()
    )

    # Fill NaN std (single observation) with a high-uncertainty value
    stats["ler_std"] = stats["ler_std"].fillna(stats["ler_std"].median())

    return stats.sort_values("n_obs", ascending=False).reset_index(drop=True)
