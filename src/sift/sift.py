
"""
Run SIFT experiments on HPatches image sequences.

For each scene:
    1. Detect SIFT features.
    2. Match image 1 against images 2 through 6.
    3. Evaluate matches using ground-truth homographies.
    4. Save matching visualizations.
    5. Export experimental results to CSV.
"""

import csv
import time

from common.dataset import load_scene
from common.evaluate import evaluate_matches
from common.config import (
    DATASET_ROOT,
    RESULTS_ROOT,
    SCENES,
    REPROJECTION_THRESHOLD,
    REFERENCE_IMAGE,
    TARGET_IMAGES
)

from .sift_features import (
    create_detector,
    detect_and_describe,
    match_descriptors
)

from common.visualize import save_match_visualizations


# SIFT-specific output directory
SIFT_RESULTS_ROOT = RESULTS_ROOT / "sift"


def run_scene(scene_name, detector):
    """
    Evaluate SIFT on one HPatches scene.

    Compare reference image 1 against images 2-6.

    Returns:
        List of dictionaries containing experimental metrics.
    """

    # Load images and ground-truth homographies
    images, homographies = load_scene(
        str(DATASET_ROOT),
        scene_name
    )

    # Create scene-specific output directory
    output_path = SIFT_RESULTS_ROOT / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect features in reference image
    img1 = images[REFERENCE_IMAGE]

    start = time.perf_counter()
    kp1, desc1 = detect_and_describe(detector, img1)
    ref_time_ms = (time.perf_counter() - start) * 1000

    results = []

    # Compare reference image against target images
    for i in TARGET_IMAGES:

        img2 = images[i]
        H = homographies[i]

        # Detect target image features
        start = time.perf_counter()
        kp2, desc2 = detect_and_describe(detector, img2)
        detect_time_ms = (time.perf_counter() - start) * 1000

        # Match SIFT descriptors
        start = time.perf_counter()
        matches = match_descriptors(desc1, desc2)
        match_time_ms = (time.perf_counter() - start) * 1000

        # Evaluate matches against ground-truth homography
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

        # Save all, correct, and incorrect match images
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
            method="SIFT"
        )

        # Record evaluation metrics
        results.append({
            "method": "SIFT",
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

        # Display progress
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


def save_results_csv(all_results):
    """
    Save experiment metrics to CSV.

    If the primary CSV is locked, save to a new
    timestamped filename instead.
    """

    if not all_results:
        print("No results to export.")
        return

    csv_path = SIFT_RESULTS_ROOT / "sift_results.csv"

    try:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=all_results[0].keys()
            )

            writer.writeheader()
            writer.writerows(all_results)

    except PermissionError:
        # Generate a unique filename if the CSV is locked
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        csv_path = (
            SIFT_RESULTS_ROOT /
            f"sift_results_{timestamp}.csv"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=all_results[0].keys()
            )

            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nCSV saved: {csv_path}")


def main():
    """Run SIFT experiments and export results."""

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"HPatches dataset not found: {DATASET_ROOT}"
        )

    SIFT_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    detector = create_detector()
    all_results = []

    print("\n========== SIFT EVALUATION ==========")

    # Run all selected scenes
    for scene in SCENES:

        print(f"\nScene: {scene}")

        scene_results = run_scene(scene, detector)
        all_results.extend(scene_results)

    # Export results
    save_results_csv(all_results)

    print("\n========== COMPLETE ==========")
    print(f"Results saved to: {SIFT_RESULTS_ROOT}")


if __name__ == "__main__":
    main()
