"""
Report figures (static PNGs) and CSV helpers. Shared so ORB and SIFT plots look identical.
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

from common.config import RESULTS_ROOT, PERTURBATION_LEVELS


plots_dir = RESULTS_ROOT / "plots"


def load_results(method):
    csv_path = RESULTS_ROOT / method / f"{method}_results.csv"

    if not csv_path.exists(): #checking if csv exists
        print(f"Missing csv: {csv_path}")
        return []

    with open(csv_path, "r") as file:
        rows = list(csv.DictReader(file)) #reading and storing our results

    return rows


def get_average(rows, column):
    values = []

    for row in rows:
        if row.get(column) not in (None, ""):
            values.append(float(row[column]))

    if len(values)== 0: #checking for empty results
        return None

    return sum(values) /len(values) #getting the average


def save_plot(filename):
    plots_dir.mkdir(parents=True, exist_ok=True) #creating folder if needed

    plt.tight_layout()
    plt.savefig(plots_dir / filename)
    plt.close()

    print(f"Saved {filename}")

def get_pooled_precision(rows):
    """
    Precision over a group of image pairs, pooled:
        100 * (total correct matches) / (total accepted matches)
 
    Averaging per-pair percentages lets a pair with 19 matches count as
    much as one with 900, which makes the curves noisy when an image is
    heavily degraded. Pooling weights every match equally.
    """
 
    if len(rows) == 0:
        return None
 
    correct = 0
    accepted = 0
 
    for row in rows:
        correct += float(row["correct_matches"])
        accepted += float(row["accepted_matches"])

    # Fix: This was skewing graph results
    if accepted == 0: # no matches at all -> precision is undefined, not 0%
        return None
 
    return 100 * correct / accepted
 
 
def summarize(rows, column):
    """Pooled precision for the precision column, plain average otherwise."""
 
    if column == "match_precision_percent":
        return get_pooled_precision(rows)
 
    return get_average(rows, column)

def plot_comparison(sift, orb, column, ylabel, filename):
    categories = ["illumination", "viewpoint"]

    sift_values = []
    orb_values = []

    for category in categories:
        sift_rows = []
        orb_rows = []

        #getting results for each type of change
        for row in sift:
            if row.get("change_type") == category and not row.get("perturbation"):
                sift_rows.append(row)

        for row in orb:
            if row.get("change_type") == category and not row.get("perturbation"):
                orb_rows.append(row)

        sift_values.append(summarize(sift_rows, column))
        orb_values.append(summarize(orb_rows, column))

    if None in sift_values or None in orb_values: #checking for missing results
        print(f"Missing data for {filename}")
        return

    x = [0, 1]

    plt.figure(figsize=(8,5))

    #putting the sift and orb bars beside each other
    plt.bar([i - 0.2 for i in x], sift_values, width=0.4, label="SIFT")
    plt.bar([i + 0.2 for i in x], orb_values, width = 0.4, label="ORB")

    plt.xticks(x, ["Illumination", "Viewpoint"])
    plt.ylabel(ylabel)
    plt.title("SIFT vs ORB - " + ylabel)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    if column == "match_precision_percent":
        plt.ylim(0,100)

    save_plot(filename)


def plot_runtime(sift, orb):
    categories = ["illumination", "viewpoint"]

    plt.figure(figsize=(8, 5))

    for method, rows in [("SIFT", sift), ("ORB", orb)]:
        times = []

        for category in categories:
            values = []

            for row in rows:
                if row.get("change_type")== category and not row.get("perturbation"):
                    detection = float(row["target_detection_ms"])
                    matching = float(row["matching_ms"])

                    values.append(detection + matching) #total time for detection and matching

            if len(values) == 0:
                print(f"Missing runtime data for {method}")
                plt.close()
                return

            times.append(sum(values)/len(values))

        plt.plot(categories, times, marker="o", label=method)

    plt.title("SIFT vs ORB - Runtime")
    plt.ylabel("Average runtime (ms)")
    plt.legend()
    plt.grid(alpha = 0.3)

    save_plot("runtime_comparison.png")


def load_perturbations(method):
    rows = load_results(method)

    csv_path = RESULTS_ROOT / method / f"{method}_perturbations.csv"

    if csv_path.exists(): #checking for separate perturbation results
        with open(csv_path, "r") as file:
            rows.extend(list(csv.DictReader(file)))

    perturb_rows = []

    for row in rows:
        if row.get("perturbation"):
            perturb_rows.append(row)

    return perturb_rows


def get_level_averages(rows, kind, column):
    grouped = defaultdict(list)
 
    for row in rows:
        if row.get("perturbation") != kind:
            continue
 
        level = row.get("level")
 
        if level in (None, "") or row.get(column) in (None, ""):
            continue
 
        grouped[float(level)].append(row) #grouping results by transformation level
 
    averages = {}
 
    for level, level_rows in grouped.items():
        averages[level] = summarize(level_rows, column) #pooled precision or plain average
 
    return averages


def plot_perturbation(sift, orb, kind, column, ylabel):
    sift_values = get_level_averages(sift,kind,column)
    orb_values = get_level_averages(orb, kind, column)

    if not sift_values or not orb_values: #checking if we have results for both
        print(f"Missing perturbation data for {kind}")
        return

    plt.figure(figsize=(8, 5))

    for method, values in [("SIFT", sift_values), ("ORB", orb_values)]:
        levels = []
        averages = []

        for level in sorted(values.keys()):
            if values[level] is None: #no accepted matches -> leave the point out
                continue
            levels.append(level)
            averages.append(values[level])

        #SIFT dashed with square markers and drawn on top, so it stays visible where the lines overlap
        if method == "SIFT":
            plt.plot(levels, averages, marker="s", linestyle="--", markersize=7, zorder=3, label=method)
        else:
            plt.plot(levels, averages, marker="o", zorder=2, label=method)

    labels = {
        "rotation": "Rotation angle (degrees)",
        "scale": "Scale factor",
        "blur": "Blur sigma",
        "darken": "Brightness multiplier"
    }

    plt.xlabel(labels.get(kind, kind))
    plt.ylabel(ylabel)
    plt.title(f"SIFT vs ORB - {kind.capitalize()}")
    plt.legend()
    plt.grid(alpha=0.3)

    if column == "match_precision_percent":
        plt.ylim(0, 100)

    save_plot(f"{kind}_{column}.png")


def generate_plots():
    sift = load_results("sift") #loading sift results
    orb = load_results("orb") #loading orb results

    if not sift or not orb:
        print("Missing SIFT or ORB results")
        return

    #comparing matching precision
    plot_comparison(
        sift, orb,
        "match_precision_percent",
        "Match precision (%, pooled)",
        "precision_comparison.png"
    )

    #comparing number of correct matches
    plot_comparison(
        sift, orb,
        "correct_matches",
        "Correct matches",
        "correct_matches_comparison.png"
    )

    #comparing detection and matching runtime
    plot_runtime(sift, orb)

    sift_perturb = load_perturbations("sift")
    orb_perturb = load_perturbations("orb")

    if not sift_perturb or not orb_perturb: #checking if perturbation results exist
        print("No perturbation results found")
        return

    for kind in PERTURBATION_LEVELS:
        #comparing each transformation at different levels
        plot_perturbation(
            sift_perturb, orb_perturb,
            kind,
            "match_precision_percent",
            "Match precision (%, pooled)"
        )

        plot_perturbation(
            sift_perturb, orb_perturb,
            kind,
            "correct_matches",
            "Correct matches"
        )

    print("All plots generated")


if __name__ == "__main__":
    generate_plots()