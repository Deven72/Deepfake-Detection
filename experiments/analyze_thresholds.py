
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "models" / "uadfv_video_evaluation.json"

with open(REPORT, "r") as file:
    data = json.load(file)

videos = data["video_predictions"]

y_true = [item["actual"] for item in videos]
scores = [item["fake_probability"] for item in videos]

print("\nTHRESHOLD COMPARISON")
print("-" * 75)
print(
    f"{'Threshold':<12}{'Accuracy':<12}"
    f"{'Precision':<12}{'Recall':<12}{'F1':<12}"
)
print("-" * 75)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    predictions = [
        int(score >= threshold) for score in scores
    ]

    print(
        f"{threshold:<12.2f}"
        f"{accuracy_score(y_true, predictions):<12.3f}"
        f"{precision_score(y_true, predictions, zero_division=0):<12.3f}"
        f"{recall_score(y_true, predictions, zero_division=0):<12.3f}"
        f"{f1_score(y_true, predictions, zero_division=0):<12.3f}"
    )

    print("  Confusion matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
