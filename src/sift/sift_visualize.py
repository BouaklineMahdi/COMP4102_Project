
"""
Generate and save SIFT feature-matching visualizations.
"""

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
    output_path
):
    """
    Save all, correct, and incorrect matches
    in separate folders for each HPatches scene.
    """

    # Separate correct and incorrect matches
    correct_matches = [
        m for m, is_correct in zip(matches, correct_mask)
        if is_correct
    ]

    incorrect_matches = [
        m for m, is_correct in zip(matches, correct_mask)
        if not is_correct
    ]

    # Visualization categories and OpenCV BGR colors
    categories = [
        ("all_matches", matches, (255, 255, 0)),
        ("correct_matches", correct_matches, (0, 255, 0)),
        ("incorrect_matches", incorrect_matches, (0, 0, 255))
    ]

    for folder_name, selected_matches, color in categories:

        folder = output_path / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        # Draw feature correspondences
        visualization = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            selected_matches,
            None,
            matchColor=color,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Add descriptive image label
        label = (
            f"SIFT | {scene_name} | "
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
            2
        )

        # Save image
        filename = f"matches_1_vs_{image_number}.jpg"

        success = cv2.imwrite(
            str(folder / filename),
            visualization
        )

        if not success:
            raise OSError(
                f"Could not save {folder / filename}"
            )
