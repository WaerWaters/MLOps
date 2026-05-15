import os
import time
import torch
import mlflow
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"
ONNX_FP32_PATH = "/tmp/model_fp32.onnx"
ONNX_INT8_PATH = "/tmp/model_int8.onnx"


class _LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values).logits


def export_to_onnx(model, path):
    wrapper = _LogitsWrapper(model).eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported ONNX model to {path}")


class _CalibReader(CalibrationDataReader):
    def __init__(self, data_loader):
        self.data = [
            {"pixel_values": batch["pixel_values"].numpy().astype(np.float32)}
            for batch in data_loader
        ]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        item = self.data[self.idx]
        self.idx += 1
        return item


def get_size_mb(path):
    return os.path.getsize(path) / 1024**2


def run_onnx_inference(session, test_loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    start = time.time()
    for batch in test_loader:
        pixel_values = batch["pixel_values"].numpy().astype(np.float32)
        labels = batch["labels"]

        logits = session.run(["logits"], {"pixel_values": pixel_values})[0]
        logits_tensor = torch.tensor(logits)

        total_loss += loss_fn(logits_tensor, labels).item()
        preds = logits_tensor.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    duration = time.time() - start

    return total_loss / len(test_loader), correct / total, duration


def quantize(config):
    data = Data(config["data_path"])
    splits = data.get_train_val_test_sets(
        [config["train_split"], config["val_split"], config["test_split"]]
    )

    test_data = splits["test"]
    if config.get("for_testing", False):
        test_data = test_data.select(range(int(len(test_data) * 0.01)))

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data.collate_fn,
    )

    print("Loading production model from MLflow...")
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location="cpu"
    )
    model.eval()

    print("Exporting to ONNX...")
    export_to_onnx(model, ONNX_FP32_PATH)
    fp32_size = get_size_mb(ONNX_FP32_PATH)

    print("Quantizing to INT8 (static)...")
    quantize_static(
        ONNX_FP32_PATH,
        ONNX_INT8_PATH,
        calibration_data_reader=_CalibReader(test_loader),
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QOperator,
    )
    int8_size = get_size_mb(ONNX_INT8_PATH)

    print("Running FP32 inference...")
    fp32_session = ort.InferenceSession(ONNX_FP32_PATH)
    fp32_loss, fp32_accuracy, fp32_duration = run_onnx_inference(
        fp32_session, test_loader
    )

    with mlflow.start_run(run_name="quantized-fp32-onnx-inference"):
        mlflow.set_tag("quantization", "fp32_onnx")
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.log_metric("inference_accuracy", fp32_accuracy)
        mlflow.log_metric("inference_loss", fp32_loss)
        mlflow.log_metric("inference_duration_seconds", fp32_duration)
        mlflow.log_metric("model_size_mb", fp32_size)

    print(
        f"FP32 — accuracy: {fp32_accuracy:.4f}, duration: {fp32_duration:.2f}s, size: {fp32_size:.2f} MB"
    )

    print("Running INT8 inference...")
    int8_session = ort.InferenceSession(ONNX_INT8_PATH)
    int8_loss, int8_accuracy, int8_duration = run_onnx_inference(
        int8_session, test_loader
    )

    with mlflow.start_run(run_name="quantized-int8-onnx-inference"):
        mlflow.set_tag("quantization", "static_int8_onnx")
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.log_metric("inference_accuracy", int8_accuracy)
        mlflow.log_metric("inference_loss", int8_loss)
        mlflow.log_metric("inference_duration_seconds", int8_duration)
        mlflow.log_metric("model_size_mb", int8_size)
        mlflow.log_metric("model_size_fp32_mb", fp32_size)
        mlflow.log_metric("size_reduction_pct", (1 - int8_size / fp32_size) * 100)

    print(
        f"INT8 — accuracy: {int8_accuracy:.4f}, duration: {int8_duration:.2f}s, size: {int8_size:.2f} MB"
    )

    print("\n--- Comparison ---")
    print(f"Size reduction:    {(1 - int8_size / fp32_size) * 100:.1f}%")
    print(f"Speedup:           {fp32_duration / int8_duration:.2f}x")
    print(f"Accuracy delta:    {int8_accuracy - fp32_accuracy:+.4f}")
