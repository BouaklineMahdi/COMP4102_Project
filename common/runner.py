
"""
Shared HPatches experiment runner for SIFT and ORB.

Both algorithms use the same:
    - Dataset and image pairs
    - Ground-truth homography evaluation
    - Timing measurements
    - Match visualizations
    - CSV output format
"""

import csv
import time
import shutil

from common.config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    RESULTS_ROOT,
    SCENES,
    REPROJECTION_THRESHOLD,
    REFERENCE_IMAGE,
    TARGET_IMAGES
)
from common.dataset import load_scene
from common.evaluate import evaluate_matches
from common.visualize import save_match_visualizations


def run_scene(scene_name, detector, detect_and_describe, match_descriptors, method):
    """
    Evaluate one HPatches scene using SIFT or ORB.

    The detector and matching functions are supplied
    by the algorithm-specific implementation.
    """

    images, homographies = load_scene(
        str(DATASET_ROOT), scene_name
    )

    method = method.lower()
    output_path = RESULTS_ROOT / method / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    img1 = images[REFERENCE_IMAGE]

    # Detect features in reference image
    start = time.perf_counter()
    kp1, desc1 = detect_and_describe(detector, img1)
    ref_time_ms = (time.perf_counter() - start) * 1000

    results = []

    for i in TARGET_IMAGES:
        img2 = images[i]
        H = homographies[i]

        # Detect target image features
        start = time.perf_counter()
        kp2, desc2 = detect_and_describe(detector, img2)
        detect_time_ms = (time.perf_counter() - start) * 1000

        # Match descriptors
        start = time.perf_counter()
        matches = match_descriptors(desc1, desc2)
        match_time_ms = (time.perf_counter() - start) * 1000

        # Evaluate matches using ground-truth homography
        (
            correct_count,
            precision,
            mean_error,
            errors,
            correct_mask
        ) = evaluate_matches(
            kp1,
            kp2,
            matches,
            H,
            img2.shape,
            REPROJECTION_THRESHOLD
        )

        # Save visualizations
        save_match_visualizations(
            img1,
            kp1,
            img2,
            kp2,
            matches,
            correct_mask,
            scene_name,
            i,
            output_path,
            method=method
        )

        # Record experiment metrics
        results.append({
            "method": method.upper(),
            "scene": scene_name,
            "change_type": (
                "illumination"
                if scene_name.startswith("i_")
                else "viewpoint"
            ),
            "image_pair": f"{REFERENCE_IMAGE}-{i}",
            "reference_keypoints": len(kp1),
            "target_keypoints": len(kp2),
            "accepted_matches": len(matches),
            "correct_matches": correct_count,
            "incorrect_matches": len(matches) - correct_count,
            "match_precision_percent": round(precision, 2),
            "mean_reprojection_error_px": (
                round(mean_error, 3)
                if mean_error is not None else ""
            ),
            "reference_detection_ms": round(ref_time_ms, 3),
            "target_detection_ms": round(detect_time_ms, 3),
            "matching_ms": round(match_time_ms, 3)
        })

        error_text = (
            f"{mean_error:.2f} px"
            if mean_error is not None
            else "N/A"
        )

        print(
            f"  {REFERENCE_IMAGE} vs {i} | "
            f"Accepted: {len(matches)} | "
            f"Correct: {correct_count} | "
            f"Precision: {precision:.2f}% | "
            f"Mean error: {error_text}"
        )

    return results


def save_results_csv(all_results, method):
    """Save experiment results to the algorithm's folder."""

    if not all_results:
        print("No results to export.")
        return

    method = method.lower()
    output_dir = RESULTS_ROOT / method
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{method}_results.csv"

    try:
        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=all_results[0].keys()
            )
            writer.writeheader()
            writer.writerows(all_results)

    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{method}_results_{timestamp}.csv"

        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=all_results[0].keys()
            )
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nCSV saved: {csv_path}")



def clear_results(method):
    """
    Remove previous results for the selected algorithm.

    Only results/sift or results/orb may be cleared.
    Other algorithm results are preserved.
    """

    method = method.lower()

    if method not in ("sift", "orb"):
        raise ValueError(
            "Cannot clear results: invalid algorithm"
        )

    output_dir = RESULTS_ROOT / method

    # Verify the directory is exactly where expected
    expected_dir = (PROJECT_ROOT / "results" / method).resolve()

    if output_dir.resolve() != expected_dir:
        raise ValueError(
            f"Unsafe results directory: {output_dir}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cleared previous {method.upper()} results.")


def run_experiments(method, create_detector, detect_and_describe, match_descriptors):
    """
    Run HPatches experiments for SIFT or ORB.

    Clears previous results for the selected algorithm,
    then executes the shared evaluation pipeline.
    """

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"HPatches dataset not found: {DATASET_ROOT}"
        )

    method = method.lower()

    if method not in ("sift", "orb"):
        raise ValueError("Method must be 'sift' or 'orb'")

    # Validate selected scenes before deleting previous results
    if not SCENES:
        raise ValueError("No HPatches scenes selected")

    missing_scenes = [
        scene for scene in SCENES
        if not (DATASET_ROOT / scene).is_dir()
    ]

    if missing_scenes:
        raise FileNotFoundError(
            f"Missing HPatches scenes: {missing_scenes}"
        )

    detector = create_detector()

    # Clear only the selected algorithm's previous results
    clear_results(method)

    all_results = []

    print(f"\n========== {method.upper()} EVALUATION ==========")

    for scene in SCENES:
        print(f"\nScene: {scene}")

        scene_results = run_scene(
            scene,
            detector,
            detect_and_describe,
            match_descriptors,
            method
        )

        all_results.extend(scene_results)

    save_results_csv(all_results, method)

    print("\n========== COMPLETE ==========")
    print(f"Results saved to: {RESULTS_ROOT / method}")

    return all_results
