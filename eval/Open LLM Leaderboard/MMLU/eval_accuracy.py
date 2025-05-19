import json
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-subject and overall accuracy from a JSON file of model predictions"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=r"path/json_file",
        help="Path to the JSON file containing prediction entries"
    )
    args = parser.parse_args()

    # Load entries from the JSON file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Initialize counters
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_entries = 0

    # Process each entry and update statistics
    for entry in entries:
        predicted = entry.get("predicted_option", "").strip().lower()
        true_answer = entry.get("answer", "").strip().lower()
        subject = entry.get("subject", "Unknown")

        subject_stats[subject]["total"] += 1
        total_entries += 1

        if predicted == true_answer:
            subject_stats[subject]["correct"] += 1
            total_correct += 1

    # Print accuracy for each subject
    print("Per-subject accuracy:")
    sum_subject_rates = 0.0
    for subj, stats in subject_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        rate = (correct / total * 100) if total > 0 else 0.0
        sum_subject_rates += rate
        print(f"  {subj}: {rate:.2f}% ({correct}/{total})")

    # Compute average of subject accuracies and overall accuracy
    num_subjects = len(subject_stats)
    avg_subject_rate = (sum_subject_rates / num_subjects) if num_subjects > 0 else 0.0
    overall_rate = (total_correct / total_entries * 100) if total_entries > 0 else 0.0

    print(f"\nAverage subject accuracy: {avg_subject_rate:.2f}%")
    print(f"Overall accuracy: {overall_rate:.2f}% ({total_correct}/{total_entries})")


if __name__ == "__main__":
    main()
