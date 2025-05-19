import os
import json
import argparse

def analyze_json_file(file_path):
    """
    Analyze a JSON file containing entries with forward and reverse scores.

    Each entry is expected to have:
      - answer_1_score, answer_2_score
      - answer_1_score_reverse, answer_2_score_reverse

    Returns a dict with:
      win_count, tie_count, lose_count, accuracy (percent), total_entries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as err:
        print(f"Error reading {file_path}: {err}")
        return None

    # Support either a list at top level or a dict with 'data1' key
    if isinstance(data, dict):
        entries = data.get('data1', [])
        if not isinstance(entries, list):
            print(f"Warning: 'data1' in {file_path} is not a list, skipping.")
            return None
    elif isinstance(data, list):
        entries = data
    else:
        print(f"Unexpected JSON structure in {file_path}, skipping.")
        return None

    win = tie = loss = 0
    for entry in entries:
        f1 = entry.get('answer_1_score', 0)
        f2 = entry.get('answer_2_score', 0)
        r1 = entry.get('answer_1_score_reverse', 0)
        r2 = entry.get('answer_2_score_reverse', 0)

        # forward AND reverse both favor answer 1 ⇒ win
        if f1 > f2 and r1 > r2:
            win += 1
        # forward AND reverse both favor answer 2 ⇒ lose
        elif f1 < f2 and r1 < r2:
            loss += 1
        # all other combinations ⇒ tie
        else:
            tie += 1

    total = win + tie + loss
    accuracy = (win / total * 100) if total else 0.0

    return {
        'win_count': win,
        'tie_count': tie,
        'lose_count': loss,
        'accuracy': accuracy,
        'total': total
    }

def main():
    """
    Parse command-line argument for folder path, then analyze each JSON file in it.
    """
    parser = argparse.ArgumentParser(
        description="Analyze JSON files for forward vs. reverse answer scores in a single folder."
    )
    parser.add_argument(
        "folder",
        nargs="?", 
        default=r"V:\ry\train2.0\eval\jsons",#Point to the folder address where the accuracy needs to be calculated
        help="Path to the folder containing your JSON files (default: %(default)s)."
    )

    args = parser.parse_args()
    folder = args.folder

    if not os.path.isdir(folder):
        print(f"Invalid folder path: {folder}")
        return

    json_files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.json'))
    if not json_files:
        print("No JSON files found in the specified folder.")
        return

    for filename in json_files:
        full_path = os.path.join(folder, filename)
        result = analyze_json_file(full_path)
        if result:
            print(f"File: {filename}")
            print(f"  Wins:       {result['win_count']}")
            print(f"  Ties:       {result['tie_count']}")
            print(f"  Losses:     {result['lose_count']}")
            print(f"  Accuracy:   {result['accuracy']:.2f}%")
            print(f"  Total:      {result['total']}\n")

if __name__ == "__main__":
    main()
