# COMP4102_Project

**SIFT vs ORB: Feature Detection and Matching on HPatches**

COMP 4102 (Computer Vision), Carleton University, Fall 2026

**Group:** Mahdi Bouakline, Liam Bennet, Samy Kaddour

## Overview

This project compares SIFT and ORB using OpenCV. SIFT uses 128-dimensional floating-point descriptors and L2 distance, while ORB uses 256-bit binary descriptors and Hamming distance.

Both methods use the same experiment runner, images, feature budget, matching settings and evaluation criteria. We test all 116 HPatches scenes (580 image pairs per method) and four controlled image transformations on ten selected scenes (250 additional pairs per method).

## Setup

From the project root, install the dependencies:

```bash
pip install -r requirements.txt
```

To obtain the project from GitHub, if needed:

```bash
git clone https://github.com/BouaklineMahdi/COMP4102_Project.git
cd COMP4102_Project
```

Download hpatches-sequences-release.zip (about 1.3 GB) from https://huggingface.co/datasets/vbalnt/hpatches and extract it into data/ so the structure looks like this:

```text
data/hpatches-sequences-release/
├── i_ajuntament/
├── i_books/
├── v_graffiti/
├── v_boat/
└── ...
```

The dataset contains 57 illumination scenes and 59 viewpoint scenes. Each scene contains six images (`1.ppm`–`6.ppm`) and five homographies (`H_1_2`–`H_1_6`) mapping image 1 to the other images. The dataset is not included in the repository.

## Running the experiments

Run these commands from the project root:

| Command | Description |
|---|---|
| `python main.py -sift` | Run SIFT on HPatches. |
| `python main.py -orb` | Run ORB on HPatches. |
| `python main.py -sift -orb` | Run both methods on HPatches. |
| `python main.py -sift -orb -perturb` | Run both methods on HPatches and the controlled transformations. |
| `python -m common.plots` | Generate comparison plots from the saved CSVs. |

At least one algorithm must be selected when running `main.py`. The `-perturb` flag adds the controlled experiments to the HPatches evaluation.

**Note:** Running an algorithm deletes and recreates its own results folder, including previous perturbation results. Back up any results you want to keep. Selected scene folders are checked before cleanup. Close open CSV files before running the experiments.

The included CSVs can be used to regenerate the plots without downloading HPatches or rerunning the experiments.

## Configuration

Experiment settings are stored in `common/config.py`. The settings used for the reported results are:

| Setting | Value |
|---|---|
| `USE_ALL_SCENES` | `True` (all 116 scenes) |
| `SCENES` | 10 selected scenes (5 illumination, 5 viewpoint) |
| `PERTURBATION_SCENES` | `SCENES` |
| `VISUALIZE_SCENES` | `SCENES` |
| `N_FEATURES` | 1000 |
| `LOWE_RATIO` | 0.8 |
| `ONE_TO_ONE_MATCHES` | `True` |
| `REPROJECTION_THRESHOLD` | 3.0 pixels |

Set `USE_ALL_SCENES = False` to run only the scenes listed in `SCENES`. The `SAVE_VISUALIZATIONS` and `SAVE_PERTURBATION_VISUALIZATIONS` settings control whether match images are saved.

### Controlled transformations

| Transformation | Levels |
|---|---|
| Rotation (degrees) | 0, 15, 30, 45, 60, 90, 135, 180 |
| Scale | 0.25, 0.5, 0.75, 1, 1.5, 2 |
| Gaussian blur (sigma) | 0, 1, 2, 4, 8 |
| Brightness multiplier | 0.1, 0.2, 0.4, 0.6, 0.8, 1 |

Each transformed image is matched against the original image 1. Rotation and scaling return the corresponding homography; blur and darkening use the identity matrix. Rotation expands the canvas to avoid cropping. The unchanged-image conditions serve as basic checks of the pipeline.

## Matching and evaluation

1. Detect keypoints and compute descriptors using OpenCV.
2. Find the two nearest descriptors using `BFMatcher` (L2 for SIFT, Hamming for ORB).
3. Apply Lowe's ratio test (`d1 < 0.8 * d2`) in each algorithm module.
4. In the shared runner, keep only the lowest-distance match for each target keypoint. This prevents duplicate target matches but is not a bidirectional matching check.
5. Project reference keypoints using the known homography. A match is correct if the projected point is inside the target image and within 3 pixels of the matched target keypoint.

The feature budget is shared, but the detectors may return different numbers of keypoints. SIFT can also return slightly more than the requested budget due to multiple orientations.

### Metrics

| Metric | Description |
|---|---|
| Reference / target keypoints | Number of detected features in each image. |
| Raw matches | Matches passing the ratio test, before one-to-one filtering. |
| Accepted matches | Matches remaining after one-to-one filtering. |
| Correct / incorrect matches | Accepted matches classified using the ground-truth homography. |
| Match precision (%) | Correct matches divided by accepted matches, multiplied by 100. |
| Mean reprojection error (px) | Average error for matches whose projected points are inside the target image. |
| Reference detection time (ms) | Reference feature detection and descriptor computation. |
| Target detection time (ms) | Target feature detection and descriptor computation. |
| Matching time (ms) | Descriptor matching, ratio test and one-to-one filtering. |

Precision is mathematically undefined when no matches are accepted. The per-pair CSV stores 0 in this case, but comparison plots calculate pooled precision from total correct and accepted matches. If the total accepted count is zero, the plotted value is omitted. This occurs for SIFT at brightness 0.1.

Runtime comparisons use target detection and descriptor computation plus matching time. Reference features are computed separately and reused for the target pairs. Timing excludes image loading, evaluation, visualization, plotting and CSV writing.

## Repository structure

```text
COMP4102_Project/
├── algorithms/
│   ├── __init__.py
│   ├── orb_features.py
│   └── sift_features.py
├── common/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── perturb.py
│   ├── plots.py
│   ├── runner.py
│   └── visualize.py
├── report/                    # Final project report
├── results/
│   ├── orb/
│   │   ├── orb_results.csv
│   │   └── orb_perturbations.csv
│   ├── sift/
│   │   ├── sift_results.csv
│   │   └── sift_perturbations.csv
│   └── plots/                 # 11 comparison plots
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

`main.py` selects the algorithms. `common/runner.py` runs the experiments and saves results, `dataset.py` loads images and homographies, `evaluate.py` checks match correctness, `perturb.py` applies transformations, and `plots.py` generates the comparison figures. `visualize.py` draws match images.

The dataset folder is added locally during setup. Match visualization folders can also be generated during experiments; they are not shown in the submitted folder structure above.

## Results

Each algorithm has 580 HPatches CSV rows and 250 perturbation CSV rows. The perturbation files include the transformation type and level.

The 11 plots in `results/plots/` include:

- `precision_comparison.png`: pooled HPatches precision.
- `correct_matches_comparison.png`: average correct matches on HPatches.
- `runtime_comparison.png`: average target detection and matching time.
- `<kind>_match_precision_percent.png`: pooled precision for each transformation.
- `<kind>_correct_matches.png`: average correct matches for each transformation.

Here, `<kind>` is `rotation`, `scale`, `blur` or `darken`. Perturbation plots require both algorithms' perturbation CSVs. SIFT uses dashed lines with square markers and ORB uses solid lines with circular markers in the updated perturbation plots.

When enabled, match visualizations are saved in scene folders under each algorithm's results directory (under `perturbations/` for controlled transformations). They include `all_matches/` (cyan), `correct_matches/` (green) and `incorrect_matches/` (red).

### Main findings

- SIFT produces about 15% more correct matches under illumination changes and 44% more under viewpoint changes.
- ORB is about 6–7 times faster in the HPatches runtime comparison.
- SIFT performs better under heavy blur, large scale changes and most tested rotations.
- ORB produces more correct matches under moderate blur and darkening.
- High precision does not necessarily mean there are enough correct matches to be useful.

The report in `report/` contains the detailed results, analysis, limitations and examples of successful and failed matches.

### Benchmark environment

Windows, Intel Core i5-10400F (6 cores, 12 threads, 2.90 GHz), 12 GB DDR4 RAM, NVIDIA GeForce GTX 1650 SUPER (4 GB VRAM), Python 3.13.1 and OpenCV 5.0.0. Timings are single-run measurements and may differ on other machines.

## References

- Lowe, D. G. (2004). [Distinctive Image Features from Scale-Invariant Keypoints](https://doi.org/10.1023/B:VISI.0000029664.99615.94). IJCV.
- Rublee, E., et al. (2011). [ORB: An Efficient Alternative to SIFT or SURF](https://doi.org/10.1109/ICCV.2011.6126544). ICCV.
- Rosten, E., and Drummond, T. (2006). [Machine Learning for High-Speed Corner Detection](https://doi.org/10.1007/11744023_34). ECCV.
- Calonder, M., et al. (2010). [BRIEF: Binary Robust Independent Elementary Features](https://doi.org/10.1007/978-3-642-15561-1_56). ECCV.
- Harris, C., and Stephens, M. (1988). A Combined Corner and Edge Detector. Alvey Vision Conference.
- Balntas, V., et al. (2017). HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors. CVPR. [Dataset](https://github.com/hpatches/hpatches-dataset).
- OpenCV documentation: [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html), [feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html), [geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html), [image filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html), [perspectiveTransform](https://docs.opencv.org/4.x/d2/de8/group__core__array.html).
- [NumPy documentation](https://numpy.org/doc/).
- [Matplotlib documentation](https://matplotlib.org/stable/).
