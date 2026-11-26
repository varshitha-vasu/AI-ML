from pathlib import Path
import csv
import time

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "best.onnx"
TEST_IMAGES = ROOT / "dataset" / "images" / "test"
OUTPUT = ROOT / "results" / "onnx"


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL}")
    images = sorted(TEST_IMAGES.glob("*"))
    images = [path for path in images if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not images:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGES}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL), task="detect")
    model.predict(str(images[0]), imgsz=640, conf=0.25, verbose=False)
    rows, elapsed = [], []
    for path in images:
        start = time.perf_counter()
        result = model.predict(str(path), imgsz=640, conf=0.25, verbose=False)[0]
        elapsed.append((time.perf_counter() - start) * 1000)
        if not cv2.imwrite(str(OUTPUT / path.name), result.plot()):
            raise OSError(f"Could not save annotated image for {path.name}")
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            rows.append([
                path.name, class_id, model.names[class_id], float(box.conf[0]),
                x1, y1, x2, y2,
            ])

    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    average = sum(elapsed) / len(elapsed)
    (OUTPUT / "timing.txt").write_text(f"Average inference time: {average:.2f} ms/image\n", encoding="utf-8")
    print(f"Processed {len(images)} images. Results: {OUTPUT}")


if __name__ == "__main__":
    main()
