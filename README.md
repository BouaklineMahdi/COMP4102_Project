# COMP4102_Project

**SIFT vs ORB: Feature Detection and Matching on HPatches**

COMP 4102 (Computer Vision), Carleton University, Fall 2026

Group: Mahdi Bouakline, Liam Bennet, Samy Kaddour

---

## Overview

This project compares two local feature methods, **SIFT** (float descriptor, L2 matching) and **ORB** (binary descriptor, Hamming matching), on how accurately and how quickly they detect and match features between images under changes in:

- **Viewpoint** and **illumination**, using the HPatches sequences
- **Rotation**, **scale**, **blur**, and **darkening**, using image transformations that we apply ourselves.

Both methods use the same evaluation process and the same number of detected features, ensuring a fair comparison.

## Metrics

| Metric | What it measures |
|---|---|
Not sure yet

## Repository Structure

```
COMP4102_Project/
├── README.md
├── requirements.txt
├── data/                  # HPatches will be placed here
├── src/
├── run_experiments.py     # Main File
├── results/
│   ├── figures/
│   └── tables/
└── report/                # Final report
```

## Setup

**1. Clone the repo and install dependencies** (Python 3.9+)

```bash
git clone https://github.com/BouaklineMahdi/COMP4102_Project.git
cd COMP4102_Project
pip install -r requirements.txt
```

**2. Download the dataset**

Download the **HPatches sequences release** (`hpatches-sequences-release.tar.gz`, about 1.3 GB) from the [HPatches dataset repository](https://github.com/hpatches/hpatches-dataset). Use the *sequences* release, not the *patches* release.

Extract it so the folder layout looks like this:

```
data/hpatches-sequences-release/
├── i_ajuntament/      # i_ = illumination changes
├── v_abstract/        # v_ = viewpoint changes
└── ...
```

Each of the 116 scene folders contains six images (`1.ppm` to `6.ppm`) and ground-truth homographies `H_1_2` to `H_1_6`, which map image 1 to each other image.

## Running

```bash
python run_experiments.py
```

Results are saved to `results/tables/` and figures to `results/figures/`.

*(Command-line options will be documented here as they are added.)*

## Results

*To be completed.*

## References

- D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *IJCV*, 2004.
- E. Rublee, V. Rabaud, K. Konolige, G. Bradski, "ORB: An Efficient Alternative to SIFT or SURF," *ICCV*, 2011.
- V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk, "HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors," *CVPR*, 2017.
- OpenCV: https://opencv.org