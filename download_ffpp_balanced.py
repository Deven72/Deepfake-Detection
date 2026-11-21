
import argparse
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2

BASE_URL = "http://kaldir.vc.in.tum.de/faceforensics/v3/"
MANIFEST_FIELDS = [
    "path", "label", "source_1", "source_2",
    "pair_id", "url", "size_bytes", "status"
]


def get_filelist():
    url = BASE_URL + "misc/filelist.json"
    request = urllib.request.Request(
        url, headers={"User-Agent": "DeepShield/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        pairs = json.loads(response.read().decode("utf-8"))

    if not isinstance(pairs, list):
        raise RuntimeError("Unexpected official filelist format")

    return [
        (str(pair[0]), str(pair[1]))
        for pair in pairs
        if isinstance(pair, list)
        and len(pair) == 2
        and str(pair[0]) != str(pair[1])
    ]


def select_disjoint_pairs(pairs):
    selected = []
    used_sources = set()
    seen = set()

    for a, b in pairs:
        key = tuple(sorted((a, b)))

        if key in seen:
            continue
        seen.add(key)

        if a in used_sources or b in used_sources:
            continue

        selected.append((a, b))
        used_sources.update([a, b])

    return selected


def valid_video(path):
    if not path.exists() or path.stat().st_size < 1024:
        return False

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False

    ok, frame = cap.read()
    cap.release()

    return bool(ok and frame is not None and frame.size > 0)


def download_video(url, destination, retries=3):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if valid_video(destination):
        print("SKIP:", destination.name)
        return destination.stat().st_size, "skipped"

    temp_path = destination.with_suffix(".mp4.part")

    for attempt in range(1, retries + 1):
        try:
            if temp_path.exists():
                temp_path.unlink()

            request = urllib.request.Request(
                url, headers={"User-Agent": "DeepShield/1.0"}
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected else None

                total = 0
                with open(temp_path, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        total += len(chunk)

            if total < 1024:
                raise RuntimeError("File too small")

            if expected is not None and total != expected:
                raise RuntimeError("Incomplete download")

            if not valid_video(temp_path):
                raise RuntimeError("Video decode verification failed")

            os.replace(temp_path, destination)
            print(f"DOWNLOADED: {destination.name} ({total} bytes)")
            return total, "downloaded"

        except Exception as error:
            print(f"Attempt {attempt}/{retries} failed: {error}")
            if temp_path.exists():
                temp_path.unlink()

            if attempt < retries:
                time.sleep(attempt * 3)

    raise RuntimeError(f"Download failed: {url}")


def save_manifest(manifest, rows):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists() or manifest.stat().st_size == 0

    with open(manifest, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--output",
        default="datasets/FaceForensics++"
    )
    args = parser.parse_args()

    if args.pairs <= 0 or args.offset < 0:
        parser.error("Invalid pairs or offset")

    output = Path(args.output).resolve()

    print("FaceForensics++ C23 Downloader")
    print("Server: EU2")
    print("Dataset: REAL + Deepfakes")
    print("Compression: C23")
    print("Output:", output)

    print("\nTerms of Use:")
    print(BASE_URL + "webpage/FaceForensics_TOS.pdf")
    input("Confirm you have read and agree to the terms. Press Enter to continue: ")

    pairs = get_filelist()
    selected = select_disjoint_pairs(pairs)

    print("Official pairs:", len(pairs))
    print("Source-disjoint pairs:", len(selected))

    batch = selected[args.offset:args.offset + args.pairs]

    if len(batch) != args.pairs:
        raise RuntimeError(
            f"Only {len(batch)} pairs available for this batch. "
            "No downloads started."
        )

    real_dir = (
        output / "original_sequences" / "youtube" / "c23" / "videos"
    )
    fake_dir = (
        output / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    )

    manifest = output / "deepfake_manifest.csv"

    success = 0
    failures = 0

    for index, (a, b) in enumerate(batch, start=args.offset):
        print(f"\nPAIR {index + 1}: {a}, {b}")

        jobs = [
            (real_dir / f"{a}.mp4",
             f"original_sequences/youtube/c23/videos/{a}.mp4",
             "REAL"),
            (real_dir / f"{b}.mp4",
             f"original_sequences/youtube/c23/videos/{b}.mp4",
             "REAL"),
            (fake_dir / f"{a}_{b}.mp4",
             f"manipulated_sequences/Deepfakes/c23/videos/{a}_{b}.mp4",
             "FAKE"),
            (fake_dir / f"{b}_{a}.mp4",
             f"manipulated_sequences/Deepfakes/c23/videos/{b}_{a}.mp4",
             "FAKE")
        ]

        rows = []

        for destination, relative, label in jobs:
            url = BASE_URL + relative

            try:
                size, status = download_video(url, destination)

                rows.append({
                    "path": relative,
                    "label": label,
                    "source_1": a,
                    "source_2": b,
                    "pair_id": f"{a}_{b}",
                    "url": url,
                    "size_bytes": size,
                    "status": status
                })
                success += 1

            except Exception as error:
                print("ERROR:", error)
                rows.append({
                    "path": relative,
                    "label": label,
                    "source_1": a,
                    "source_2": b,
                    "pair_id": f"{a}_{b}",
                    "url": url,
                    "size_bytes": 0,
                    "status": "failed"
                })
                failures += 1

        save_manifest(manifest, rows)

    print("\n========== SUMMARY ==========")
    print("Expected videos:", len(batch) * 4)
    print("Successful:", success)
    print("Failed:", failures)
    print("Manifest:", manifest)

    if failures:
        print("Rerun the same command to retry failed downloads.")
    else:
        print("BATCH COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
