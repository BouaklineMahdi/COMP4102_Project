"""
Artificial image changes with a known ground-truth homography.

Each function returns (changed_image, H), where H is the 3x3 homography
mapping pixel coordinates in the ORIGINAL image to the CHANGED image.
This is the same convention as HPatches' H_1_k (image 1 -> image k),
so evaluate_matches() scores these pairs exactly like HPatches pairs.

    rotation, scale  -> geometric: H built from 2D transformation
                        matrices (translation, rotation, scaling), Lecture 12
    blur, darken     -> photometric: no pixel moves, so H = identity

Levels for each change are set in common/config.py (PERTURBATION_LEVELS).

Sources:
    OpenCV geometric transformations (resize pixel-centre convention):
    https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
    OpenCV GaussianBlur:
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
"""

import cv2
import numpy as np


def translation_matrix(tx, ty):
    """
    3x3 homogeneous translation matrix.
    """

    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ])


def image_corners(width, height):
    """
    Corner coordinates (x, y) of a width x height image, as a 4x2 array."""

    return np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float64)


def apply_homography(H, points):
    """
    Map an Nx2 array of (x, y) points through H
    using homogeneous coordinates.
    """

    ones = np.ones((points.shape[0], 1))
    homogeneous = np.concatenate([points, ones], axis=1) @ H.T

    # Divide by w to return to pixel coordinates
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def rotate(image, angle_deg):
    """
    Rotate the image about its centre.

    The output canvas is enlarged so no part of the image is cut off.
    Positive angles rotate clockwise on screen (the image y axis points down).
    """

    height, width = image.shape[:2]

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    rotation = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Move the image centre to the origin, then rotate
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    H = rotation @ translation_matrix(-cx, -cy)

    # Shift so the rotated image's bounding box starts at (0, 0)
    rotated_corners = apply_homography(H, image_corners(width, height))
    min_xy = rotated_corners.min(axis=0)
    max_xy = rotated_corners.max(axis=0)

    H = translation_matrix(-min_xy[0], -min_xy[1]) @ H

    output_size = (
        int(np.ceil(max_xy[0] - min_xy[0])) + 1,
        int(np.ceil(max_xy[1] - min_xy[1])) + 1
    )

    # Inverse warping with bilinear interpolation, as in A2
    rotated = cv2.warpPerspective(image, H, output_size)

    return rotated, H


def scale(image, factor):
    """
    Resize the image by 'factor' (< 1 zooms out, > 1 zooms in).
    """

    height, width = image.shape[:2]

    new_width = max(1, int(round(width * factor)))
    new_height = max(1, int(round(height * factor)))

    # INTER_AREA averages pixels when shrinking, which avoids aliasing
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # cv2.resize aligns pixel centres: x' = sx * (x + 0.5) - 0.5
    sx = new_width / width
    sy = new_height / height

    H = np.array([
        [sx, 0.0, 0.5 * sx - 0.5],
        [0.0, sy, 0.5 * sy - 0.5],
        [0.0, 0.0, 1.0]
    ])

    return resized, H


def blur(image, sigma):
    """
    Gaussian blur with standard deviation 'sigma' pixels.
    No pixel moves, so H is the identity.
    """

    if sigma <= 0:
        return image.copy(), np.eye(3)

    # Kernel size (0, 0): OpenCV picks the size from sigma
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    return blurred, np.eye(3)


def darken(image, factor):
    """
    Multiply every intensity by 'factor' (<= 1).

    This also scales every intensity difference (contrast) by 'factor'.
    No pixel moves, so H is the identity.
    """

    darker = (image.astype(np.float32) * factor).astype(np.uint8)

    return darker, np.eye(3)


PERTURBATIONS = {
    "rotation": rotate,
    "scale": scale,
    "blur": blur,
    "darken": darken,
}


def perturb(image, kind, level):
    """
    Apply perturbation 'kind' ("rotation", "scale", "blur" or "darken")
    at 'level'.

    Returns:
        changed_image, H (original -> changed)
    """

    if kind not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {kind}")

    return PERTURBATIONS[kind](image, level)