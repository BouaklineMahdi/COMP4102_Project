
"""
Shared configuration for SIFT and ORB experiments.

Both algorithms must use the same datasets and
evaluation parameters for a fair comparison.
"""

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "hpatches-sequences-release"

RESULTS_ROOT = PROJECT_ROOT / "results"


# HPatches scenes used for comparison
SCENES = [
    # Illumination changes
    "i_ajuntament",
    "i_books",
    "i_leuven",
    "i_fog",
    "i_crownnight",

    # Viewpoint changes
    "v_graffiti",
    "v_boat",
    "v_bricks",
    "v_wall",
    "v_laptop",
]


# Shared evaluation parameters
REPROJECTION_THRESHOLD = 3.0  # pixels

REFERENCE_IMAGE = 1

TARGET_IMAGES = range(2, 7)
