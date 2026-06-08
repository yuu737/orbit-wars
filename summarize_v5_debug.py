import argparse
import json
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize OW_V5_LOG JSONL output.")
    parser.add_argument("path", help="Path to the debug JSONL file.")
    parser.add_argument("--top", type=int, default=12, help="Number of top reasons to print.")
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    args = parse_args()
    rows = list(load_rows(args.path))
    if not rows:
        print("No rows.")
        return

    reasons = Counter()
    counts = Counter()
    zero_action_reasons = Counter()
    step_buckets = defaultdict(Counter)

    for row in rows:
        row_reasons = row.get("reasons", {})
        row_counts = row.get("counts", {})
        notes = row.get("notes", {})
        final_action_count = int(notes.get("final_action_count", 0))
        step = int(row.get("step", 0))
        bucket = f"{(step // 20) * 20:03d}-{(step // 20) * 20 + 19:03d}"

        reasons.update(row_reasons)
        counts.update(row_counts)
        step_buckets[bucket].update(row_reasons)

        if final_action_count == 0:
            zero_action_reasons.update(row_reasons)

    print(f"Rows: {len(rows)}")
    print()
    print("Top reasons:")
    for key, value in reasons.most_common(args.top):
        print(f"  {key}: {value}")

    print()
    print("Generated counts:")
    for key, value in counts.most_common():
        print(f"  {key}: {value}")

    print()
    print("Top reasons on zero-action turns:")
    for key, value in zero_action_reasons.most_common(args.top):
        print(f"  {key}: {value}")

    print()
    print("Reasons by step bucket:")
    for bucket in sorted(step_buckets):
        common = ", ".join(
            f"{key}={value}" for key, value in step_buckets[bucket].most_common(5)
        )
        print(f"  {bucket}: {common}")


if __name__ == "__main__":
    main()
