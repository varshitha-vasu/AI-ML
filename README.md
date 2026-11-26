# Shape Detection, Area Measurement & Defect Classification

YOLOv8 detects circles, squares, triangles and rectangles. OpenCV estimates visible contour area and centroid, then maps area to the assignment's defect categories.

## Repository layout

```text
models/
  best.pt
  best.onnx
results/
  final_results.csv
  pytorch/
  onnx/
  annotated_images/
  loss_curves/
  confusion_matrix/
training/
  train.py
  training_logs/
README.md
README_scripts.txt
inference_onnx.py
inference_pytorch.py
opencv_measurement.py
train.py
```

The 200 original images and labels are intentionally not committed to keep the repository small. Put the dataset locally under `dataset/` with a YOLO config at `dataset/data.yaml`; preserve the train/validation/test split.

## Setup

Use Python 3.11 and install the project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy pyyaml onnx onnxruntime
```

Expected dataset layout:

```text
dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

`data.yaml` must define `train`, `val`, `test`, and class names in this order: `circle`, `square`, `triangle`, `rectangle`.

## Run

```powershell
python train.py
python inference_pytorch.py
python inference_onnx.py
python opencv_measurement.py
```

Training writes `models/best.pt`, exports `models/best.onnx`, and records Ultralytics logs under `training/training_logs/`. Inference reads `dataset/images/test/` and writes predictions and annotated images under `results/`. OpenCV measurement reads `dataset/images/` and writes `results/final_results.csv` plus annotated images in `results/annotated_images/`.

## Measurement definitions

- Area uses `cv2.findContours` and `cv2.contourArea` inside each predicted box; it is not bounding-box area.
- Centroid uses `cv2.moments` and is expressed in original-image pixel coordinates.
- `less_defect`: area `< 5000 px²`
- `moderate_defect`: `5000 <= area <= 15000 px²`
- `severe_defect`: area `> 15000 px²`

## Annotation and metric limitations

The bundled checkpoint and evaluation summaries were produced using the supplied automatic/corrected labels. Human verification is incomplete, so reported metrics are agreement with those labels—not independently verified academic accuracy. Retrain and evaluate only after providing validated labels and a fixed held-out test set.
