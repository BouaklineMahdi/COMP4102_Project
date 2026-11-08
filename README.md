# COMP4102_Project

**SIFT vs ORB: Feature Detection and Matching on HPatches**

COMP 4102 (Computer Vision), Carleton University, Fall 2026

**Group:** Mahdi Bouakline, Liam Bennet, Samy Kaddour

---

## Overview

This project compares two local feature detection and matching algorithms:

- **SIFT (Scale-Invariant Feature Transform):** Uses floating-point descriptors and L2 distance matching with Lowe's ratio test.
- **ORB (Oriented FAST and Rotated BRIEF):** Uses binary descriptors and Hamming distance matching.

The goal is to evaluate the accuracy and computational performance of both algorithms under different image conditions:

- **Viewpoint and illumination changes:** Using the HPatches image sequences.
- **Rotation, scale, blur, and darkening:** Using additional image transformations applied to the original images (planned).

Both algorithms use a shared experimental pipeline to ensure consistent dataset loading, evaluation, timing measurements, visualizations, and CSV output.

The number of detected features and descriptor matching parameters are configurable. Any differences in these settings must be documented when comparing the algorithms.

## Metrics

The current evaluation pipeline measures the following metrics for each image pair:

| Metric | Description |
|---|---|
| Reference Keypoints | Number of features detected in the reference image (image 1). |
| Target Keypoints | Number of features detected in the target image (images 2–6). |
| Accepted Matches | Number of descriptor matches accepted by the algorithm's matching function. |
| Correct Matches | Number of accepted matches whose homography reprojection error is within the specified threshold and whose projected point lies inside the target image. |
| Incorrect Matches | Number of accepted matches that do not satisfy the correctness criteria. |
| Match Precision (%) | Percentage of accepted matches classified as correct. |
| Mean Reprojection Error (px) | Average reprojection error across matches with valid projections inside the target image, including incorrect matches. |
| Reference Detection Time (ms) | Time required to detect and describe features in the reference image. |
| Target Detection Time (ms) | Time required to detect and describe features in the target image. |
| Matching Time (ms) | Time required to match the two sets of descriptors. |

**Match correctness:** A match is considered correct when its reprojection error is at most **3 pixels**, using the HPatches ground-truth homography.

The homography maps coordinates from reference image 1 to the corresponding target image.

Match precision is calculated as:

**Precision (%) = (Correct Matches / Accepted Matches) × 100**

If no matches are accepted, precision is reported as 0%.

The timing measurements exclude visualization generation, homography evaluation, and CSV writing.

## Repository Structure

```text
COMP4102_Project/
├── README.md
├── requirements.txt
├── main.py                    # Main command-line entry point
│
├── algorithms/                # Algorithm-specific implementations
│   ├── sift_features.py       # SIFT detection and descriptor matching
│   └── orb_features.py        # ORB detection and descriptor matching
│
├── common/                    # Shared experimental pipeline
│   ├── config.py              # Dataset paths, scenes, evaluation settings
│   ├── dataset.py             # HPatches image and homography loading
│   ├── evaluate.py            # Ground-truth match evaluation
│   ├── runner.py              # Shared experiment execution and CSV export
│   ├── visualize.py           # Match visualization generation
│   ├── perturb.py             # Additional image transformations
│   └── plots.py               # SIFT and ORB comparison plots
│
├── data/
│   └── hpatches-sequences-release/
│
├── results/
│   ├── sift/
│   │   ├── i_ajuntament/
│   │   │   ├── all_matches/
│   │   │   ├── correct_matches/
│   │   │   └── incorrect_matches/
│   │   ├── ...
│   │   └── sift_results.csv
│   │
│   ├── orb/
│   │   ├── i_ajuntament/
│   │   │   ├── all_matches/
│   │   │   ├── correct_matches/
│   │   │   └── incorrect_matches/
│   │   ├── ...
│   │   └── orb_results.csv
│   │
│   └── plots/                 # Generated comparison graphs
│       ├── precision_comparison.png
│       ├── correct_matches_comparison.png
│       └── runtime_comparison.png
│
└── report/                    # Final project report
```

### Shared Pipeline

The project has been refactored to separate algorithm-specific implementations from shared experimental functionality.

| File | Responsibility |
|---|---|
| `main.py` | Parses command-line arguments and selects which algorithms to run. |
| `common/config.py` | Defines shared dataset paths, selected scenes, reference/target images, and reprojection threshold. |
| `common/dataset.py` | Loads HPatches images and ground-truth homographies. |
| `common/runner.py` | Executes experiments, measures runtime, evaluates matches, saves visualizations, and exports CSV results. |
| `common/evaluate.py` | Determines correct and incorrect matches using ground-truth homographies. |
| `common/visualize.py` | Generates visualizations of all, correct, and incorrect matches. |
| `algorithms/sift_features.py` | Implements SIFT feature detection, description, and matching. |
| `algorithms/orb_features.py` | Implements ORB feature detection, description, and matching. |

The shared runner accepts algorithm-specific functions as arguments, allowing both methods to use the same evaluation pipeline without duplicating code.

## Setup

### 1. Clone the repository and install dependencies

Python 3.9+ is recommended.

```bash
git clone https://github.com/BouaklineMahdi/COMP4102_Project.git
cd COMP4102_Project
pip install -r requirements.txt
```

### 2. Download the HPatches dataset

Download the **HPatches sequences release** (`hpatches-sequences-release.tar.gz`, approximately 1.3 GB) from the [HPatches dataset repository](https://github.com/hpatches/hpatches-dataset).

Use the **sequences** release, not the **patches** release.

Extract the dataset into:

```text
data/hpatches-sequences-release/
├── i_ajuntament/       # Illumination changes
├── i_books/
├── v_graffiti/         # Viewpoint changes
├── v_boat/
└── ...
```

The complete dataset contains 116 scene sequences.

Each scene contains:

- Six images (`1.ppm` through `6.ppm`).
- Five ground-truth homographies (`H_1_2` through `H_1_6`).

Each homography maps coordinates from reference image 1 to the corresponding target image.

## Running Experiments

All experiments are executed from the project root using `main.py`.

**Run SIFT only:**

```bash
python main.py -sift
```

**Run ORB only:**

```bash
python main.py -orb
```

**Run both algorithms:**

```bash
python main.py -sift -orb
```

Running both algorithms executes SIFT followed by ORB.

At least one algorithm must be selected.

### Configuring Experiments

Experiment settings are defined in `common/config.py`.

The current selected scenes are:

```python
SCENES = [
    "i_ajuntament",
    "i_books",
    "i_leuven",
    "i_fog",
    "i_crownnight",
    "v_graffiti",
    "v_boat",
    "v_bricks",
    "v_wall",
    "v_laptop",
]
```

The remaining shared settings are:

```python
REPROJECTION_THRESHOLD = 3.0
REFERENCE_IMAGE = 1
TARGET_IMAGES = range(2, 7)
```

Each selected scene compares image 1 against images 2 through 6.

With the current configuration:

- 10 scenes are evaluated.
- 5 image pairs are evaluated per scene.
- 50 image pairs are evaluated per algorithm.
- 5 scenes contain illumination changes.
- 5 scenes contain viewpoint changes.

Both algorithms use the same scene selection and reprojection threshold.

### Automatic Results Cleanup

Before running an algorithm, `common/runner.py` clears its previous results directory and creates a new one.

For example:

- `python main.py -sift` clears and regenerates `results/sift/`.
- `python main.py -orb` clears and regenerates `results/orb/`.
- `python main.py -sift -orb` regenerates both directories.

**Important:** Running one algorithm does not delete the other algorithm's results.

This ensures that each output directory contains only the results from its most recent run, using the currently selected scenes.

Previous results for the selected algorithm are permanently deleted when its run begins. If an experiment fails, its previous results will already have been removed.

Close any open CSV files before running the experiments, particularly when using Excel on Windows.

## Results

Results are organized by algorithm, followed by HPatches scene.

Each scene contains three visualization directories:

| Directory | Contents |
|---|---|
| `all_matches/` | All accepted feature matches, displayed in cyan. |
| `correct_matches/` | Matches satisfying the ground-truth reprojection threshold, displayed in green. |
| `incorrect_matches/` | Matches failing the correctness criteria, displayed in red. |

For each image pair, the corresponding visualization is saved as:

```text
matches_1_vs_2.jpg
matches_1_vs_3.jpg
matches_1_vs_4.jpg
matches_1_vs_5.jpg
matches_1_vs_6.jpg
```

A CSV file is also generated for each algorithm:

- `results/sift/sift_results.csv`
- `results/orb/orb_results.csv`

Both CSV files use the same column structure, allowing direct comparison between algorithms.

### Generating Comparison Plots

The `common/plots.py` file was implemented to compare the results of SIFT and ORB using their generated CSV files.

To generate the plots, run the following command from the project root:

```bash
python3 -m common.plots
```

The script reads:

- `results/sift/sift_results.csv`
- `results/orb/orb_results.csv`

Both files must exist before running the comparison.

The script generates three baseline comparison plots:

1. **Matching precision:** Compares the average percentage of correct matches under illumination and viewpoint changes.
2. **Correct matches:** Compares the average number of geometrically correct matches found by each algorithm.
3. **Runtime:** Compares the average time used for target feature detection and descriptor matching, measured in milliseconds.

All generated plots are saved in:

```text
results/plots/
```

The generated files are:

```text
precision_comparison.png
correct_matches_comparison.png
runtime_comparison.png
```

The plotting file contains functions for comparing rotation, scaling, blur, and darkening at different transformation levels. 
- These functions require perturbation experiment results

If a CSV cannot be overwritten because of a permissions error, the runner attempts to save a timestamped CSV instead.

## Implementing an Algorithm

The shared pipeline is designed so that SIFT and ORB require only their own feature detection and matching implementations.

### Required Functions

Each algorithm module must expose the following functions:

```python
def create_detector():
    """Create and return the OpenCV feature detector."""
```

```python
def detect_and_describe(detector, gray):
    """
    Detect keypoints and compute descriptors.

    Returns:
        keypoints, descriptors
    """
```

```python
def match_descriptors(desc1, desc2):
    """
    Match two descriptor sets.

    Returns:
        List of accepted cv2.DMatch objects.
    """
```

These functions are passed into `common.runner.run_experiments()` by `main.py`.

The shared runner handles everything else.

### SIFT Implementation

Located in `algorithms/sift_features.py`.

Current SIFT configuration:

```python
DEFAULTS = {
    "n_features": 1000,
    "contrast_threshold": 0.04,
    "edge_threshold": 10,
    "sigma": 1.6,
}
```

SIFT uses:

- `cv2.SIFT_create()` for feature detection and description.
- `cv2.BFMatcher(cv2.NORM_L2)` for descriptor matching.
- K-nearest-neighbor matching with `k=2`.
- Lowe's ratio test with a threshold of `0.8`.

The SIFT module returns only the accepted matches to the shared runner.

### ORB Implementation

Located in `algorithms/orb_features.py`.

The ORB implementation should expose the same three function names and compatible return values.

ORB should use:

- `cv2.ORB_create()` for feature detection and description.
- `cv2.BFMatcher(cv2.NORM_HAMMING)` for standard ORB descriptor matching.
- A documented descriptor match filtering strategy.

**ORB integration requirements:**

1. Implement `create_detector()`, `detect_and_describe()`, and `match_descriptors()` in `algorithms/orb_features.py`.
2. Ensure `detect_and_describe()` returns OpenCV keypoints and descriptors.
3. Ensure `match_descriptors()` returns a list of accepted `cv2.DMatch` objects.
4. Handle images with no descriptors or insufficient descriptors without crashing.
5. Document ORB detector parameters and matching thresholds.
6. Verify that `python main.py -orb` runs successfully.
7. Compare the generated `orb_results.csv` with `sift_results.csv`.

ORB descriptor matching function has been implemented
- function uses cv2.BFmatcher with cv2.NORM_HAMMING --> since ORD uses binary descriptors
- Matching is done using K-nearest-neighbor (K=2), along with lowes ratio test with a .8 threshold

No separate ORB experiment runner, evaluation function, visualization function, or CSV exporter is required.

The shared pipeline already provides those features.

### Notes on Fair Comparison

To ensure meaningful results:

- Both algorithms must use the same image pairs and ground-truth homographies.
- Both algorithms must use the same reprojection threshold.
- Feature count settings should be controlled and documented.
- Descriptor matching methods and filtering thresholds should be documented.
- Detection and matching times should be measured under comparable conditions.
- Differences in accepted match counts should be considered when interpreting precision.

The shared runner ensures consistent evaluation and output formatting, but algorithm-specific parameters must still be selected carefully.

## Future Work

The following functionality is planned:

- Integrate existing image transformations from `common/perturb.py` into the experiment runner.
- Evaluate robustness under rotation, scaling, blur, and darkening.
- Generate comparison plots using `common/plots.py`.
- Compare SIFT and ORB accuracy and runtime across different conditions.
- Analyze differences between illumination and viewpoint performance.
- Summarize experimental findings in the final report.

## References

- D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *IJCV*, 2004.
- E. Rublee, V. Rabaud, K. Konolige, G. Bradski, "ORB: An Efficient Alternative to SIFT or SURF," *ICCV*, 2011.
- V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk, "HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors," *CVPR*, 2017.
- OpenCV: https://opencv.org