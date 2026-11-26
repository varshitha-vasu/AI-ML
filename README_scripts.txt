SHAPEVISION SCRIPT GUIDE

1. Provide the dataset locally under dataset/ and configure dataset/data.yaml.
2. Train and export:
       python train.py
3. PyTorch test inference:
       python inference_pytorch.py
4. ONNX test inference:
       python inference_onnx.py
5. OpenCV contour areas, centroids, defect categories and 200-image CSV:
       python opencv_measurement.py

The training script expects YOLO train/val/test splits. The original split and
corrected annotation files are not included in this compact GitHub submission.
The checked-in labels and existing metrics are not fully human-verified.

Output locations:
- Models: models/best.pt and models/best.onnx
- Training history: training/training_logs/
- PT results: results/pytorch/
- ONNX results: results/onnx/
- Measurements: results/final_results.csv
- Annotated images: results/annotated_images/
