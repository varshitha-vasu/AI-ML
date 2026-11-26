from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "best.pt"
IMAGES = ROOT / "dataset" / "images"
OUTPUT = ROOT / "results"
CLASSES = ("circle", "square", "triangle", "rectangle")


def classify_defect(area):
    if area < 5000:
        return "less_defect"
    if area <= 15000:
        return "moderate_defect"
    return "severe_defect"


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL}")
    paths = sorted(
        path for path in IMAGES.glob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"No source images found in {IMAGES}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    fields = (
        ["image_name"]
        + [f"{name}_count" for name in CLASSES]
        + [f"{name}_area_px2" for name in CLASSES]
        + [f"{name}_center" for name in CLASSES]
        + [f"{name}_defect" for name in CLASSES]
    )

    with (OUTPUT / "final_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"Could not read {path}")
            height, width = image.shape[:2]

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]], axis=0)
            background = np.median(border, axis=0).astype(np.float32)
            mask = (np.linalg.norm(lab.astype(np.float32) - background, axis=2) > 16).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            prediction = model.predict(str(path), imgsz=640, conf=0.25, verbose=False)[0]
            row = {"image_name": path.name}
            values = {name: {"areas": [], "centers": [], "defects": []} for name in CLASSES}
            annotated = image.copy()
            detections = sorted(list(prediction.boxes), key=lambda box: (int(box.xyxy[0][1]), int(box.xyxy[0][0])))

            for box in detections:
                class_id = int(box.cls[0])
                if class_id not in range(len(CLASSES)):
                    raise ValueError(f"Unknown class id {class_id} in {path.name}")
                name = CLASSES[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                contours, _ = cv2.findContours(mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(contour))
                moments = cv2.moments(contour)
                if area < 20 or moments["m00"] == 0:
                    continue
                center_x = x1 + moments["m10"] / moments["m00"]
                center_y = y1 + moments["m01"] / moments["m00"]
                defect = classify_defect(area)
                values[name]["areas"].append(round(area, 2))
                values[name]["centers"].append((round(center_x, 2), round(center_y, 2)))
                values[name]["defects"].append(defect)

                contour = contour.copy()
                contour += np.array([x1, y1], dtype=np.int32)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.drawContours(annotated, [contour], -1, (0, 0, 255), 3)
                cv2.circle(annotated, (round(center_x), round(center_y)), 6, (255, 0, 0), -1)
                cv2.putText(
                    annotated, f"{name} {confidence:.2f} {area:.0f}px2 {defect}",
                    (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )

            for name in CLASSES:
                row[f"{name}_count"] = [len(values[name]["areas"])]
                row[f"{name}_area_px2"] = values[name]["areas"]
                row[f"{name}_center"] = values[name]["centers"]
                row[f"{name}_defect"] = values[name]["defects"]
            writer.writerow(row)
            if not cv2.imwrite(str(annotated_dir / path.name), annotated):
                raise OSError(f"Could not save annotated image {path.name}")

    print(f"Processed {len(paths)} images. CSV: {OUTPUT / 'final_results.csv'}")


if __name__ == "__main__":
    main()
