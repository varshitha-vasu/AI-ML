from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "data.yaml"
MODEL = ROOT / "models"
LOGS = ROOT / "training" / "training_logs"


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA}. Prepare a YOLO dataset with train, val, test and names entries."
        )
    MODEL.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    result = model.train(
        data=str(DATA),
        epochs=30,
        imgsz=640,
        batch=4,
        workers=0,
        device="cpu",
        patience=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        project=str(LOGS),
        name="shape_yolov8n",
        exist_ok=True,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training completed but best checkpoint is missing: {best}")
    shutil.copy2(best, MODEL / "best.pt")
    trained = YOLO(str(MODEL / "best.pt"))
    trained.export(format="onnx", imgsz=640, opset=12, dynamic=False)
    exported = MODEL / "best.onnx"
    generated_onnx = best.with_suffix(".onnx")
    if generated_onnx.exists():
        shutil.copy2(generated_onnx, exported)

    print("Best PyTorch model:", MODEL / "best.pt")
    print("ONNX model:", exported)
    print("Training logs:", result.save_dir)


if __name__ == "__main__":
    main()
