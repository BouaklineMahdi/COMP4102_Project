
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
# True  -> evaluate every scene found in DATASET_ROOT
# False -> evaluate only the scenes listed in SCENES
USE_ALL_SCENES = True

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

# Scenes used for the artificial perturbation experiments.
# Kept to a subset because every scene is tested at every level in
# PERTURBATION_LEVELS (25 image pairs per scene per algorithm).
PERTURBATION_SCENES = SCENES

# Shared detector and matching parameters
# Passed to BOTH algorithms by common/runner.py so they cannot drift apart.
N_FEATURES = 1000  # keypoint budget per image

# Lowe's ratio test threshold. 0.8 is the value recommended by Lowe (2004):
# it rejects about 90% of false matches while discarding under 5% of correct ones.
LOWE_RATIO = 0.8
 
# The ratio test alone allows many reference keypoints to match the SAME
# target keypoint (e.g. a blurred target with only a few keypoints).
# True -> keep only the closest match for each target keypoint (one-to-one).
ONE_TO_ONE_MATCHES = True

# Shared evaluation parameters
REPROJECTION_THRESHOLD = 3.0  # pixels

REFERENCE_IMAGE = 1

TARGET_IMAGES = range(2, 7)

# Artificial image changes
PERTURBATION_LEVELS = {
    "rotation": [0, 15, 30, 45, 60, 90, 135, 180],  # degrees
    "scale": [1.0, 0.25, 0.5, 0.75, 1.5, 2.0],      # zoom factor
    "blur": [0, 1, 2, 4, 8],                         # Gaussian sigma (px)
    "darken": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],       # intensity multiplier
}

SAVE_VISUALIZATIONS = True

VISUALIZE_SCENES = SCENES
 
SAVE_PERTURBATION_VISUALIZATIONS = True