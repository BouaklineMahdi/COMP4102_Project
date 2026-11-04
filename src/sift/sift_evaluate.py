
"""
Evaluate SIFT matches against HPatches homographies.
"""

import cv2
import numpy as np

from common.config import REPROJECTION_THRESHOLD


def evaluate_matches(
    kp1,
    kp2,
    matches,
    H,
    image_shape,
    threshold=REPROJECTION_THRESHOLD
):
    """
    Evaluate accepted descriptor matches using the
    ground-truth homography mapping image 1 to image 2.

    Returns:
        correct_count
        precision
        mean_error
        errors
        correct_mask
    """

    if not matches:
        return 0, 0.0, None, np.array([]), np.array([], dtype=bool)

    # Extract coordinates of matched keypoints
    pts1 = np.float32([
        kp1[m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt for m in matches
    ]).reshape(-1, 2)

    # Project reference keypoints into image 2
    projected = cv2.perspectiveTransform(
        pts1,
        np.asarray(H, dtype=np.float64)
    ).reshape(-1, 2)

    # Compute Euclidean reprojection errors
    errors = np.linalg.norm(projected - pts2, axis=1)

    height, width = image_shape[:2]

    # Check whether projected points are inside image 2
    valid = (
        np.isfinite(errors)
        & np.all(np.isfinite(projected), axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )

    # A match is correct if its error is within threshold
    correct_mask = valid & (errors <= threshold)

    correct_count = int(np.sum(correct_mask))

    precision = correct_count / len(matches) * 100

    mean_error = (
        float(np.mean(errors[valid]))
        if np.any(valid) else None
    )

    return (
        correct_count,
        precision,
        mean_error,
        errors,
        correct_mask
    )
