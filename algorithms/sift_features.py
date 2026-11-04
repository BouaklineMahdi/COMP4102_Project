
"""
SIFT feature detection, description, and matching.
"""

import cv2

NAME = "SIFT"

DEFAULTS = {
    "n_features": 1000,
    "contrast_threshold": 0.04,
    "edge_threshold": 10,
    "sigma": 1.6,
}


def create_detector(
    n_features=DEFAULTS["n_features"],
    contrast_threshold=DEFAULTS["contrast_threshold"],
    edge_threshold=DEFAULTS["edge_threshold"],
    sigma=DEFAULTS["sigma"]
):
    """Create an OpenCV SIFT detector."""

    return cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )


def detect_and_describe(detector, gray):
    """
    Detect keypoints and compute 128-dimensional
    SIFT descriptors from a grayscale image.
    """

    return detector.detectAndCompute(gray, None)


def match_descriptors(desc1, desc2, ratio=0.8):
    """
    Match SIFT descriptors using L2 distance
    and Lowe's ratio test.
    """

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches
