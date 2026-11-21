
import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("datasets/FaceForensics++")
INPUT = ROOT / "deepfake_manifest.csv"
OUTPUT = ROOT / "split_manifest.csv"

SEED = 42
random.seed(SEED)

# Read original manifest
with open(INPUT, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Keep only successfully downloaded videos
rows = [r for r in rows if r["status"] in ("downloaded", "skipped")]

# Group videos by pair_id
groups = defaultdict(list)

for row in rows:
    groups[row["pair_id"]].append(row)

pair_ids = sorted(groups.keys())

if len(pair_ids) != 100:
    raise ValueError(f"Expected 100 pairs, found {len(pair_ids)}")

# Every pair must contain 2 REAL and 2 FAKE videos
for pair_id, videos in groups.items():
    counts = Counter(v["label"] for v in videos)

    if len(videos) != 4 or counts["REAL"] != 2 or counts["FAKE"] != 2:
        raise ValueError(f"Invalid pair group: {pair_id}, {counts}")

# Deterministic shuffle
random.shuffle(pair_ids)

train_ids = set(pair_ids[:70])
val_ids = set(pair_ids[70:85])
test_ids = set(pair_ids[85:100])

# Assign split to every video
output_rows = []

for pair_id, videos in groups.items():
    if pair_id in train_ids:
        split = "train"
    elif pair_id in val_ids:
        split = "validation"
    else:
        split = "test"

    for video in videos:
        output_rows.append({
            **video,
            "split": split
        })

# Save manifest
fieldnames = list(output_rows[0].keys())

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

# Print summary
print("\n========== DATASET SPLIT SUMMARY ==========")

for split in ["train", "validation", "test"]:
    selected = [r for r in output_rows if r["split"] == split]
    counts = Counter(r["label"] for r in selected)
    pairs = len(set(r["pair_id"] for r in selected))

    print(f"\n{split.upper()}")
    print("Pairs:", pairs)
    print("REAL:", counts["REAL"])
    print("FAKE:", counts["FAKE"])
    print("Total:", len(selected))

print("\nTotal videos:", len(output_rows))
print("Split manifest:", OUTPUT)
print("Random seed:", SEED)
print("\nSPLIT COMPLETED SUCCESSFULLY!")
