"""Species definitions, properties, and name normalization.

Candidate species pool for cereal-legume-cover crop polyculture system
in temperate climate (USDA Zones 5-7).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    common_name: str
    scientific_name: str
    functional_group: str
    is_nitrogen_fixer: bool
    is_cash_crop: bool
    # Typical mature height in meters (midpoint of common range)
    # Source: FAO 56, USDA PLANTS Plant Guides, extension publications
    height_m: float
    # Maximum effective rooting depth in meters
    # Source: FAO Irrigation & Drainage Paper 56, Table 22 (Allen et al., 1998)
    root_depth_m: float


# Candidate species pool — selected based on CLAUDE.md recommendations
# and data availability in Paut et al. dataset.
#
# Trait sources:
#   height_m: Midpoint of typical field cultivar range.
#     FAO "Save and Grow"; USDA PLANTS Plant Guides; extension publications.
#   root_depth_m: Maximum effective rooting depth.
#     FAO Irrigation & Drainage Paper 56, Table 22 (Allen et al., 1998).
#     Supplemented by Fan et al. (2016) Field Crops Res. 189:68-74;
#     Thorup-Kristensen (2006) Soil Use Mgmt. 22:29-38;
#     USDA NRCS cover crop fact sheets.
CANDIDATE_SPECIES: dict[str, Species] = {
    "maize": Species(
        common_name="Maize",
        scientific_name="Zea mays",
        functional_group="cereal",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=2.2,
        root_depth_m=1.4,
    ),
    "wheat": Species(
        common_name="Wheat",
        scientific_name="Triticum aestivum",
        functional_group="cereal",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.9,
        root_depth_m=1.2,
    ),
    "barley": Species(
        common_name="Barley",
        scientific_name="Hordeum vulgare",
        functional_group="cereal",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.9,
        root_depth_m=1.2,
    ),
    "oat": Species(
        common_name="Oat",
        scientific_name="Avena sativa",
        functional_group="cereal",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.8,
        root_depth_m=1.2,
    ),
    "sorghum": Species(
        common_name="Sorghum",
        scientific_name="Sorghum bicolor",
        functional_group="cereal",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=1.5,
        root_depth_m=1.5,
    ),
    "fava_bean": Species(
        common_name="Fava bean",
        scientific_name="Vicia faba",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=True,
        height_m=1.0,
        root_depth_m=0.75,
    ),
    "pea": Species(
        common_name="Pea",
        scientific_name="Pisum sativum",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=True,
        height_m=0.6,
        root_depth_m=0.8,
    ),
    "soybean": Species(
        common_name="Soybean",
        scientific_name="Glycine max",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=True,
        height_m=0.7,
        root_depth_m=1.0,
    ),
    "cowpea": Species(
        common_name="Cowpea",
        scientific_name="Vigna unguiculata",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=True,
        height_m=0.5,
        root_depth_m=0.8,
    ),
    "common_bean": Species(
        common_name="Bean",
        scientific_name="Phaseolus vulgaris",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=True,
        height_m=0.4,
        root_depth_m=0.75,
    ),
    "lentil": Species(
        common_name="Lentil",
        scientific_name="Lens culinaris",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=True,
        height_m=0.4,
        root_depth_m=0.7,
    ),
    "crimson_clover": Species(
        common_name="Crimson clover",
        scientific_name="Trifolium incarnatum",
        functional_group="cover_crop",
        is_nitrogen_fixer=True,
        is_cash_crop=False,
        height_m=0.35,
        root_depth_m=0.75,
    ),
    "radish": Species(
        common_name="Radish",
        scientific_name="Raphanus sativus",
        functional_group="cover_crop",
        is_nitrogen_fixer=False,
        is_cash_crop=False,
        height_m=0.45,
        root_depth_m=1.0,
    ),
    "sunflower": Species(
        common_name="Sunflower",
        scientific_name="Helianthus annuus",
        functional_group="oilseed",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=2.0,
        root_depth_m=1.2,
    ),
    "lupine": Species(
        common_name="Lupine",
        scientific_name="Lupinus spp.",
        functional_group="legume",
        is_nitrogen_fixer=True,
        is_cash_crop=False,
        height_m=0.8,
        root_depth_m=0.8,
    ),
    "cabbage": Species(
        common_name="Cabbage",
        scientific_name="Brassica oleracea",
        functional_group="vegetable",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.4,
        root_depth_m=0.65,
    ),
    "tomato": Species(
        common_name="Tomato",
        scientific_name="Solanum lycopersicum",
        functional_group="vegetable",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=1.0,
        root_depth_m=1.1,
    ),
    "lettuce": Species(
        common_name="Lettuce",
        scientific_name="Lactuca sativa",
        functional_group="vegetable",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.25,
        root_depth_m=0.4,
    ),
    "onion": Species(
        common_name="Onion",
        scientific_name="Allium cepa",
        functional_group="vegetable",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.3,
        root_depth_m=0.45,
    ),
    "pepper": Species(
        common_name="Pepper",
        scientific_name="Capsicum annuum",
        functional_group="vegetable",
        is_nitrogen_fixer=False,
        is_cash_crop=True,
        height_m=0.65,
        root_depth_m=0.75,
    ),
}


# Name normalization: map various dataset names to our canonical species keys
_NAME_ALIASES: dict[str, str] = {
    # Exact matches (lowercase)
    "maize": "maize",
    "corn": "maize",
    "zea mays": "maize",
    "wheat": "wheat",
    "triticum aestivum": "wheat",
    "barley": "barley",
    "hordeum vulgare": "barley",
    "oat": "oat",
    "avena sativa": "oat",
    "sorghum": "sorghum",
    "sorghum bicolor": "sorghum",
    "fava bean": "fava_bean",
    "faba bean": "fava_bean",
    "broad bean": "fava_bean",
    "vicia faba": "fava_bean",
    "pea": "pea",
    "pisum sativum": "pea",
    "soybean": "soybean",
    "soya bean": "soybean",
    "glycine max": "soybean",
    "cowpea": "cowpea",
    "vigna unguiculata": "cowpea",
    "bean": "common_bean",
    "common bean": "common_bean",
    "phaseolus vulgaris": "common_bean",
    "french bean": "common_bean",
    "lentil": "lentil",
    "lens culinaris": "lentil",
    "crimson clover": "crimson_clover",
    "trifolium incarnatum": "crimson_clover",
    "radish": "radish",
    "raphanus sativus": "radish",
    "sunflower": "sunflower",
    "helianthus annuus": "sunflower",
    "lupine": "lupine",
    "lupin": "lupine",
    "white lupine": "lupine",
    "lupinus": "lupine",
    "cabbage": "cabbage",
    "brassica oleracea": "cabbage",
    "tomato": "tomato",
    "solanum lycopersicum": "tomato",
    "lettuce": "lettuce",
    "lactuca sativa": "lettuce",
    "onion": "onion",
    "allium cepa": "onion",
    "pepper": "pepper",
    "capsicum annuum": "pepper",
    "sweet pepper": "pepper",
    "bell pepper": "pepper",
    "chili pepper": "pepper",
    "chili peppers": "pepper",
    "peppers": "pepper",
    "capsicum": "pepper",
    # Plurals and variants found in companion planting dataset
    "beans": "common_bean",
    "beans, bush": "common_bean",
    "beans, pole": "common_bean",
    "beans, fava": "fava_bean",
    "dry beans": "common_bean",
    "green beans": "common_bean",
    "snap beans": "common_bean",
    "soybeans": "soybean",
    "peas": "pea",
    "lentils": "lentil",
    "clover": "crimson_clover",
    "radishes": "radish",
    "sunflowers": "sunflower",
    "cabbages": "cabbage",
    "brassicas": "cabbage",
    "tomatoes": "tomato",
    "onions": "onion",
    "green onions": "onion",
    "malting barley": "barley",
    "grain sorghum": "sorghum",
    # Scientific name variants from Bischoff et al. dataset
    "lupinus albus": "lupine",
    "lupinus angustifolius": "lupine",
    "triticum durum": "wheat",
    # NOTE: Brassica napus (canola), B. campestris (field mustard), and
    # B. juncea (mustard greens) are intentionally excluded — they are
    # distinct species from B. oleracea (cabbage) with different growth
    # habits, heights, and root systems.
}


def normalize_species_name(name: str) -> str | None:
    """Map a raw species name to a canonical species key.

    Returns None if the species is not in our candidate pool.
    """
    return _NAME_ALIASES.get(name.strip().lower())


def get_species(key: str) -> Species:
    """Get species definition by canonical key."""
    return CANDIDATE_SPECIES[key]
