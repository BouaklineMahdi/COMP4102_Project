"""
Main entry point for SIFT vs ORB experiments.

Usage:
    python main.py -sift
    python main.py -orb
    python main.py -sift -orb
    python main.py -sift -orb -perturb
"""

import argparse

from common.runner import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run SIFT and ORB experiments on HPatches"
    )

    parser.add_argument(
        "-sift",
        action="store_true",
        help="Run SIFT experiments"
    )

    parser.add_argument(
        "-orb",
        action="store_true",
        help="Run ORB experiments"
    )

    args = parser.parse_args()

    if not args.sift and not args.orb:
        parser.error(
            "Select at least one algorithm: -sift or -orb"
        )

    if args.sift:
        from algorithms.sift_features import (
            create_detector,
            detect_and_describe,
            match_descriptors
        )

        run_experiments(
            method="SIFT",
            create_detector=create_detector,
            detect_and_describe=detect_and_describe,
            match_descriptors=match_descriptors,
            run_perturbations=args.perturb
        )

    if args.orb:
        from algorithms.orb_features import (
            create_detector,
            detect_and_describe,
            match_descriptors
        )

        run_experiments(
            method="ORB",
            create_detector=create_detector,
            detect_and_describe=detect_and_describe,
            match_descriptors=match_descriptors,
            run_perturbations=args.perturb
        )


if __name__ == "__main__":
    main()
