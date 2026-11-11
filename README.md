# COMP4102_Project

**SIFT vs ORB: Feature Detection and Matching on HPatches**

COMP 4102 (Computer Vision), Carleton University, Fall 2026

**Group:** Mahdi Bouakline, Liam Bennet, Samy Kaddour

---

## Overview

This project compares two local feature detection and matching algorithms:

- **SIFT (Scale-Invariant Feature Transform):** floating-point descriptors (128 values), matched with L2 distance.
- **ORB (Oriented FAST and Rotated BRIEF):** binary descriptors (256 bits), matched with Hamming distance.

Both are evaluated for accuracy and speed under:

- **Viewpoint and illumination changes:** the HPatches image sequences.
- **Rotation, scale, blur, and darkening:** artificial changes applied to image 1 of a scene, each with a known ground-truth homography.

Both algorithms run through one shared pipeline (dataset loading, matching filters, evaluation, timing, visualizations, CSV output). The keypoint budget and ratio-test threshold are defined once in `common/config.py` and passed to both algorithms, so the settings cannot differ between them.

## Metrics

Each image pair produces one CSV row with these metrics:

| Metric | Description |
|---|---|
| Reference Keypoints | Features detected in the reference image (image 1). |
| Target Keypoints | Features detected in the target image (images 2–6, or the perturbed image). |
| Raw Matches | Matches that passed Lowe's ratio test, before the one-to-one filter. |
| Accepted Matches | Matches kept after the one-to-one filter (see [Matching](#matching)). |
| Correct Matches | Accepted matches whose reprojection error is at most the threshold **and** whose projected point lies inside the target image. |
| Incorrect Matches | Accepted matches that are not correct. |
| Match Precision (%) | Correct Matches / Accepted Matches × 100 (0% when no matches are accepted). |
| Mean Reprojection Error (px) | Average reprojection error over matches whose projection lands inside the target image, correct or not. |
| Reference Detection Time (ms) | Time to detect and describe features in the reference image. |
| Target Detection Time (ms) | Time to detect and describe features in the target image. |
| Matching Time (ms) | Time to match the two descriptor sets, including the ratio test and one-to-one filter. |

**Match correctness:** a match is correct when the reference keypoint, projected by the ground-truth homography, lands within **3 pixels** of its matched target keypoint. The homography maps image 1 coordinates to the target image (HPatches `H_1_k`, or the perturbation's H).

**Precision in plots is pooled:** for a group of image pairs, the plots use (total correct) / (total accepted) × 100, not the average of each pair's percentage. A pair with 19 matches would otherwise count as much as one with 900, which makes heavily degraded levels noisy. The CSVs still store each pair's own precision.

Timing excludes image loading, visualization, evaluation, and CSV writing.

## Matching

Both algorithms use the same matching steps:

1. **Nearest neighbours:** brute-force matcher, k = 2 (L2 for SIFT, Hamming for ORB).
2. **Lowe's ratio test:** keep a match only if `best.distance < LOWE_RATIO × second_best.distance`. `LOWE_RATIO = 0.8`, the value recommended by Lowe (2004), which rejects about 90% of false matches while discarding fewer than 5% of correct ones.
3. **One-to-one filter** (`ONE_TO_ONE_MATCHES = True`): matching runs from every reference descriptor, so several reference keypoints can match the *same* target keypoint. When the target image has few keypoints (e.g. strong blur), that inflates the match counts: in testing, ORB once reported 187 matches against a target image with only 4 keypoints. The filter keeps only the closest match for each target keypoint. The raw count before filtering is kept in the `raw_matches` column.

Steps 2 and 3 are applied in `common/runner.py`, so they are identical for both algorithms.

## Repository Structure

```text
COMP4102_Project/
├── README.md
├── requirements.txt
├── main.py                    # Command-line entry point
│
├── algorithms/                # Algorithm-specific code
│   ├── sift_features.py       # SIFT detection, description, L2 matching
│   └── orb_features.py        # ORB detection, description, Hamming matching
│
├── common/                    # Shared pipeline
│   ├── config.py              # Paths, scenes, shared parameters, perturbation levels
│   ├── dataset.py             # HPatches image and homography loading
│   ├── runner.py              # Runs experiments, matching filters, timing, CSV export
│   ├── evaluate.py            # Ground-truth match evaluation
│   ├── perturb.py             # Rotation, scale, blur, darkening + ground-truth H
│   ├── visualize.py           # Match visualization images
│   └── plots.py               # SIFT vs ORB comparison plots
│
├── data/
│   └── hpatches-sequences-release/   # Not tracked by git
│
├── results/
│   ├── sift/
│   │   ├── sift_results.csv          # HPatches pairs
│   │   ├── sift_perturbations.csv    # Perturbation pairs (with -perturb)
│   │   ├── <scene>/                  # Visualizations (VISUALIZE_SCENES only)
│   │   │   ├── all_matches/
│   │   │   ├── correct_matches/
│   │   │   └── incorrect_matches/
│   │   └── perturbations/<scene>/    # Perturbation visualizations
│   ├── orb/                          # Same layout as sift/
│   └── plots/                        # Comparison figures
│
└── report/                    # Final project report
```

| File | Responsibility |
|---|---|
| `main.py` | Parses command-line flags and selects the algorithms and experiments to run. |
| `common/config.py` | Dataset paths, scene selection, shared detector/matching parameters, perturbation levels, visualization settings. |
| `common/dataset.py` | Lists HPatches scenes; loads images (as grayscale) and ground-truth homographies. |
| `common/runner.py` | Runs the HPatches and perturbation experiments, applies the shared matching filters, measures time, saves visualizations and CSVs. |
| `common/evaluate.py` | Classifies matches as correct or incorrect using a ground-truth homography. |
| `common/perturb.py` | Applies rotation, scale, blur, and darkening, and returns the ground-truth homography for each. |
| `common/visualize.py` | Draws all, correct, and incorrect matches. |
| `common/plots.py` | Reads the CSVs and generates the comparison figures. |

The runner takes each algorithm's functions as arguments, so both methods run through the same code.

## Setup

### 1. Clone the repository and install dependencies

Python 3.9+ is recommended.

```bash
git clone https://github.com/BouaklineMahdi/COMP4102_Project.git
cd COMP4102_Project
pip install -r requirements.txt
```

### 2. Download the HPatches dataset

Download the **HPatches sequences release** (`hpatches-sequences-release.tar.gz`, approximately 1.3 GB) from the [HPatches dataset repository](https://github.com/hpatches/hpatches-dataset). Use the **sequences** release, not the **patches** release.

Extract it into:

```text
data/hpatches-sequences-release/
├── i_ajuntament/       # Illumination changes
├── i_books/
├── v_graffiti/         # Viewpoint changes
├── v_boat/
└── ...
```

The dataset contains 116 scenes: 57 with illumination changes (`i_`) and 59 with viewpoint changes (`v_`). Each scene has six images (`1.ppm`–`6.ppm`) and five ground-truth homographies (`H_1_2`–`H_1_6`) mapping image 1 to each other image.

## Running Experiments

Run everything from the project root.

| Command | What it runs |
|---|---|
| `python main.py -sift` | SIFT on HPatches |
| `python main.py -orb` | ORB on HPatches |
| `python main.py -sift -orb` | Both on HPatches (SIFT first) |
| `python main.py -sift -orb -perturb` | Both on HPatches **and** the perturbation experiments |

At least one algorithm must be selected. `-perturb` adds the perturbation experiments to the selected algorithms; it does not run on its own.

### Configuration (`common/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `USE_ALL_SCENES` | `True` | Evaluate all 116 HPatches scenes. Set to `False` to use only `SCENES`. |
| `SCENES` | 10 scenes (5 `i_`, 5 `v_`) | Scene list used when `USE_ALL_SCENES = False`; also the default for the two settings below. |
| `PERTURBATION_SCENES` | `SCENES` | Scenes used for the perturbation experiments. Each scene is tested at all 25 perturbation levels. |
| `VISUALIZE_SCENES` | `SCENES` | Match images are saved only for these scenes (saving all 116 would write thousands of images). |
| `N_FEATURES` | `1000` | Keypoint budget, passed to both algorithms. |
| `LOWE_RATIO` | `0.8` | Ratio-test threshold, passed to both algorithms. |
| `ONE_TO_ONE_MATCHES` | `True` | Keep only the closest match per target keypoint. |
| `REPROJECTION_THRESHOLD` | `3.0` | Pixel threshold for a correct match. |
| `PERTURBATION_LEVELS` | see file | Rotation (degrees), scale (factor), blur (Gaussian sigma), darken (intensity multiplier). |
| `SAVE_VISUALIZATIONS` / `SAVE_PERTURBATION_VISUALIZATIONS` | `True` | Turn match images on or off. |

With the defaults, each algorithm is evaluated on 116 × 5 = **580 HPatches pairs**, plus 10 × 25 = **250 perturbation pairs** with `-perturb`.

### Perturbation experiments

For each scene in `PERTURBATION_SCENES`, image 1 is changed at every level in `PERTURBATION_LEVELS` and matched against the unchanged image 1:

| Change | Ground-truth homography |
|---|---|
| Rotation | Rotation about the image centre; the canvas is enlarged so nothing is cut off. |
| Scale | Scaling matrix, with OpenCV's pixel-centre convention. |
| Blur, darken | Identity (no pixel moves). |

Level 0 rotation, scale 1.0, blur 0 and darken 1.0 leave the image unchanged, so they should give 100% precision. They serve as a check that the ground-truth homographies are correct.

### Automatic results cleanup

Before running an algorithm, its results folder (`results/sift/` or `results/orb/`) is deleted and recreated. Running one algorithm never deletes the other's results. The selected scenes are checked **before** anything is deleted, so a missing scene stops the run without losing previous results. Close any open CSV files first (particularly in Excel on Windows); if a CSV can't be overwritten, a timestamped copy is saved instead.

## Results

### CSV files

| File | Contents |
|---|---|
| `results/<method>/<method>_results.csv` | One row per HPatches pair. |
| `results/<method>/<method>_perturbations.csv` | One row per perturbation pair, with `perturbation` and `level` columns. |

SIFT and ORB files have identical columns.

### Visualizations

For each scene in `VISUALIZE_SCENES`:

| Folder | Contents |
|---|---|
| `all_matches/` | All accepted matches (cyan). |
| `correct_matches/` | Correct matches (green). |
| `incorrect_matches/` | Incorrect matches (red). |

Files are named `matches_1_vs_<k>.jpg` for HPatches pairs and `matches_1_vs_<kind>_<level>.jpg` for perturbations.

### Comparison plots

After both algorithms have run:

```bash
python -m common.plots
```

Figures are saved to `results/plots/`:

| File | Shows |
|---|---|
| `precision_comparison.png` | Pooled precision, illumination vs viewpoint. |
| `correct_matches_comparison.png` | Average correct matches, illumination vs viewpoint. |
| `runtime_comparison.png` | Average target detection + matching time. |
| `<kind>_match_precision_percent.png` | Pooled precision vs perturbation level (one per change type). |
| `<kind>_correct_matches.png` | Average correct matches vs perturbation level. |

Perturbation plots are generated only when both `*_perturbations.csv` files exist.

## Algorithm Interface

Each algorithm module in `algorithms/` provides three functions, which `main.py` passes to `common.runner.run_experiments()`:

```python
def create_detector(n_features):
    """Create and return the OpenCV detector with the given keypoint budget."""

def detect_and_describe(detector, gray):
    """Return (keypoints, descriptors). descriptors is None if no keypoints are found."""

def match_descriptors(desc1, desc2, ratio):
    """Return the list of cv2.DMatch objects passing Lowe's ratio test."""
```

The runner supplies `n_features` and `ratio` from `config.py`.

### SIFT (`algorithms/sift_features.py`)

| Parameter | Value |
|---|---|
| `nfeatures` | `N_FEATURES` (1000) |
| `contrastThreshold` | 0.04 (OpenCV default) |
| `edgeThreshold` | 10 (OpenCV default) |
| `sigma` | 1.6 (OpenCV default) |
| Matcher | `cv2.BFMatcher(cv2.NORM_L2)`, k = 2 |

### ORB (`algorithms/orb_features.py`)

| Parameter | Value |
|---|---|
| `nfeatures` | `N_FEATURES` (1000; OpenCV default is 500) |
| `scaleFactor` | 1.2 (OpenCV default) |
| `nlevels` | 8 (OpenCV default) |
| `edgeThreshold` | 31 (OpenCV default) |
| `fastThreshold` | 20 (OpenCV default) |
| `patchSize` | 31 (OpenCV default) |
| `WTA_K` | 2 (OpenCV default: each test compares a pixel pair, giving 1 bit) |
| `scoreType` | `ORB_HARRIS_SCORE` (OpenCV default: FAST keypoints ranked by Harris response) |
| Matcher | `cv2.BFMatcher(cv2.NORM_HAMMING)`, k = 2. ORB descriptors are binary, so Hamming distance (number of differing bits) is used; FLANN KD-trees don't accept binary descriptors. |

## Fair Comparison

- Both algorithms use the same image pairs and ground-truth homographies.
- `N_FEATURES`, `LOWE_RATIO`, the one-to-one filter, and the reprojection threshold come from `config.py` and are identical for both.
- Timing is measured the same way, on the same machine, in the same run.
- Accepted-match counts differ between the algorithms, so precision should always be read alongside the number of correct matches.
- SIFT can return slightly more than `N_FEATURES` keypoints, because keypoints with several dominant orientations are duplicated.

## Future Work

- Plot precision by HPatches pair difficulty (1–2 through 1–6).
- Estimate homographies with RANSAC from the accepted matches, and measure inlier counts and corner error.
- Vary the keypoint budget (`N_FEATURES`) to study the accuracy/runtime trade-off.

## References

### Methods and dataset

- D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *International Journal of Computer Vision*, 60(2), 2004. (SIFT, ratio test)
- E. Rublee, V. Rabaud, K. Konolige, G. Bradski, "ORB: An Efficient Alternative to SIFT or SURF," *ICCV*, 2011.
- E. Rosten, T. Drummond, "Machine Learning for High-Speed Corner Detection," *ECCV*, 2006. (FAST, used by ORB)
- M. Calonder, V. Lepetit, C. Strecha, P. Fua, "BRIEF: Binary Robust Independent Elementary Features," *ECCV*, 2010. (used by ORB)
- C. Harris, M. Stephens, "A Combined Corner and Edge Detector," *Alvey Vision Conference*, 1988. (ORB keypoint ranking)
- V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk, "HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors," *CVPR*, 2017. https://github.com/hpatches/hpatches-dataset

### Libraries and documentation

- G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000. https://opencv.org
  - ORB tutorial: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
  - Feature matching (`BFMatcher`, `knnMatch`, `drawMatches`): https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
  - `perspectiveTransform`: https://docs.opencv.org/4.x/d2/de8/group__core__array.html
  - Geometric transformations (`warpPerspective`, `resize`): https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
  - Image filtering (`GaussianBlur`): https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- C. R. Harris et al., "Array programming with NumPy," *Nature*, 585, 2020. https://numpy.org (`loadtxt`, `linalg.norm`)
- J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*, 9(3), 2007. https://matplotlib.org
- Python standard library: `argparse`, `csv`, `pathlib`, `shutil`, `time`, `collections`. https://docs.python.org/3/library/