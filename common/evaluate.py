
"""
Shared feature-matching evaluation for SIFT and ORB.

Uses HPatches ground-truth homographies to evaluate
the geometric accuracy of descriptor matches.
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
    Evaluate descriptor matches against a ground-truth homography.

    Parameters:
        kp1: Keypoints detected in the reference image.
        kp2: Keypoints detected in the target image.
        matches: List of accepted cv2.DMatch objects.
        H: Homography mapping reference to target image.
        image_shape: Shape of the target image.
        threshold: Maximum reprojection error in pixels.

    Returns:
        correct_count: Number of geometrically correct matches.
        precision: Percentage of accepted matches that are correct.
        mean_error: Mean error of valid projected matches.
        errors: Reprojection error for each match.
        correct_mask: Boolean mask identifying correct matches.
    """

    if not matches:
        return (
            0,
            0.0,
            None,
            np.array([]),
            np.array([], dtype=bool)
        )

    # Extract coordinates of matched keypoints
    pts1 = np.float32([
        kp1[m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt for m in matches
    ]).reshape(-1, 2)

    # Project reference keypoints into target image
    projected = cv2.perspectiveTransform(
        pts1,
        np.asarray(H, dtype=np.float64)
    ).reshape(-1, 2)

    # Calculate Euclidean reprojection errors
    errors = np.linalg.norm(projected - pts2, axis=1)

    height, width = image_shape[:2]

    # Check whether projected points are inside target image
    valid = (
        np.isfinite(errors)
        & np.all(np.isfinite(projected), axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )

    # Identify geometrically correct matches
    correct_mask = valid & (errors <= threshold)

    correct_count = int(np.sum(correct_mask))

    # Precision relative to all accepted matches
    precision = correct_count / len(matches) * 100

    # Average reprojection error of valid projected matches
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
