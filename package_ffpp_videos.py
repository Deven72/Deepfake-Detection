import zipfile
from pathlib import Path

root = Path("datasets/FaceForensics++")

real_dir = root / "original_sequences/youtube/c23/videos"
fake_dir = root / "manipulated_sequences/Deepfakes/c23/videos"

real_files = sorted(real_dir.glob("*.mp4"))
fake_files = sorted(fake_dir.glob("*.mp4"))

manifest = root / "split_manifest.csv"

assert len(real_files) == 200, "REAL video count incorrect"
assert len(fake_files) == 200, "FAKE video count incorrect"
assert manifest.exists(), "Split manifest missing"

output = Path("ffpp_original_videos.zip")

with zipfile.ZipFile(
    output, "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=1
) as z:

    for file in real_files + fake_files:
        z.write(file, file.relative_to(root).as_posix())

    z.write(manifest, "split_manifest.csv")

print("ZIP CREATED SUCCESSFULLY")
print("REAL VIDEOS:", len(real_files))
print("FAKE VIDEOS:", len(fake_files))
print("TOTAL VIDEOS:", len(real_files) + len(fake_files))
print("ZIP SIZE (GB):", round(output.stat().st_size / (1024**3), 3))
print("ZIP PATH:", output.resolve())
