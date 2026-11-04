
"""
Shared feature-matching visualizations for SIFT and ORB.

Saves all, correct, and incorrect matches in
separate folders for each HPatches scene.
"""

from pathlib import Path

import cv2


def save_match_visualizations(
    img1,
    kp1,
    img2,
    kp2,
    matches,
    correct_mask,
    scene_name,
    image_number,
    output_path,
    method
):
    """
    Save feature-matching visualizations.

    Parameters:
        img1, img2: Grayscale input images.
        kp1, kp2: Detected OpenCV keypoints.
        matches: Accepted descriptor matches.
        correct_mask: Boolean mask from evaluate_matches().
        scene_name: HPatches scene name.
        image_number: Target image number (2-6).
        output_path: Output directory for the scene.
        method: Algorithm name ("SIFT" or "ORB").
    """

    output_path = Path(output_path)

    if len(matches) != len(correct_mask):
        raise ValueError(
            "correct_mask length must equal number of matches"
        )

    # Separate correct and incorrect matches
    correct_matches = [
        match
        for match, is_correct in zip(matches, correct_mask)
        if is_correct
    ]

    incorrect_matches = [
        match
        for match, is_correct in zip(matches, correct_mask)
        if not is_correct
    ]

    # Categories: folder, matches, OpenCV BGR color
    categories = [
        ("all_matches", matches, (255, 255, 0)),
        ("correct_matches", correct_matches, (0, 255, 0)),
        ("incorrect_matches", incorrect_matches, (0, 0, 255))
    ]

    for folder_name, selected_matches, color in categories:

        folder = output_path / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        # Draw selected feature correspondences
        visualization = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            selected_matches,
            None,
            matchColor=color,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Add descriptive label
        label = (
            f"{method.upper()} | {scene_name} | "
            f"Image 1 vs {image_number} | "
            f"{folder_name}: {len(selected_matches)}"
        )

        cv2.putText(
            visualization,
            label,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Save visualization
        filename = f"matches_1_vs_{image_number}.jpg"
        file_path = folder / filename

        success = cv2.imwrite(
            str(file_path),
            visualization
        )

        if not success:
            raise OSError(
                f"Failed to save visualization: {file_path}"
            )
