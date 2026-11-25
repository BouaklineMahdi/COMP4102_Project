"""
ORB: Oriented FAST and Rotated BRIEF (Rublee, Rabaud, Konolige, Bradski, ICCV 2011).

External sources: OpenCV ORB tutorial https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html,
OpenCV feature matching tutorial https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html,
ORB class reference https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html.
"""
import cv2

NAME = "ORB"

# OpenCV defaults 
DEFAULTS = {
    "n_features": 1000,      
    "scale_factor": 1.2,
    "n_levels": 8,
    "edge_threshold": 31,
    "fast_threshold": 20,
    "patch_size": 31,
    "wta_k": 2,             
}


def create_detector(n_features=DEFAULTS["n_features"], scale_factor=DEFAULTS["scale_factor"], n_levels=DEFAULTS["n_levels"], fast_threshold=DEFAULTS["fast_threshold"]):
    """
    Create an OpenCV ORB detector/descriptor.
    """
    return cv2.ORB_create(nfeatures=n_features,
                          scaleFactor=scale_factor,
                          nlevels=n_levels,
                          edgeThreshold=DEFAULTS["edge_threshold"],
                          firstLevel=0,
                          WTA_K=DEFAULTS["wta_k"],
                          # rank FAST points with Harris
                          scoreType=cv2.ORB_HARRIS_SCORE,  
                          patchSize=DEFAULTS["patch_size"],
                          fastThreshold=fast_threshold)


def detect_and_describe(detector, gray):
    """
    Detect keypoints and compute descriptors on a grayscale uint8 image.

    Returns (keypoints, descriptors). descriptors is an N x 32 uint8 array, or None when no
    keypoint was found (e.g. a very dark image).
    """
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio=0.8):
    """
    Match image-1 descriptors to image-2 descriptors with Hamming distance + Lowe's ratio test.

    For each image-1 descriptor we take its 2 nearest image-2 descriptors and keep the best one only 
    if best.distance < ratio * second.distance. ratio >= 1 disables the test (keeps every nearest neighbour).
    """
    if desc1 is None or desc2 is None: #cheching for missing desc
        return []

    if len(desc1) == 0 or len(desc2) < 2: #Check the desc counts
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING) #creating the matcher and we use hamming distance as this is where ORB is different form SIFT

    matches = matcher.knnMatch(desc1, desc2, k=2) #Finding two nearest matches

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair

        if ratio >= 1 or m.distance < (ratio *n.distance): #applying Lowes ratio test to see if good match
            good_matches.append(m)

    return good_matches

