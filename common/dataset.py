"""
Load HPatches sequences: images and ground-truth homographies.

HPatches sequences release (Balntas et al., CVPR 2017):
    116 scene folders, 57 starting with "i_" (illumination change only) and
    59 starting with "v_" (viewpoint change only).
    Each folder holds 1.ppm ... 6.ppm and text files H_1_2 ... H_1_6.
    H_1_k is a 3x3 homography (Lecture 12) mapping pixel coordinates in
    image 1 to image k, in homogeneous coordinates.

Shared by ORB and SIFT so both methods see exactly the same data.

note: I tested this locally but the dataset is never commited, we'll see how to work with that
"""
import os

import cv2
import numpy as np


def list_scenes(root, kind="all"):
    """
    Return the sorted scene folder names under root.

    kind: "all", "i" (illumination scenes only) or "v" (viewpoint scenes only).
    """
    names = []
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        if kind == "all" or name.startswith(kind + "_"):
            names.append(name)
    return names


def load_gray(path):
    """
    Load an image as 8-bit grayscale. ORB and SIFT both work on intensity only.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_homography(path):
    """
    Read a 3x3 homography stored as plain text (3 rows of 3 numbers).
    """
    return np.loadtxt(path)


def load_scene(root, name):
    """
    Load one scene.

    Returns:
        images: dict {1: img1, ..., 6: img6} of grayscale uint8 images
        homographies: dict {2: H_1_2, ..., 6: H_1_6}
    """
    folder = os.path.join(root, name)
    images = {}
    for k in range(1, 7):
        images[k] = load_gray(os.path.join(folder, f"{k}.ppm"))
    homographies = {}
    for k in range(2, 7):
        homographies[k] = load_homography(os.path.join(folder, f"H_1_{k}"))
    return images, homographies
