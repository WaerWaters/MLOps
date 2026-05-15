import os
import copy
import time
import torch
import mlflow
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"


def get_size_mb(model):
    path = "/tmp/model_size_check.pth"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024**2
    os.remove(path)
    return size


def run_inference(model, test_loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            total_loss += loss_fn(outputs.logits, batch["labels"]).item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
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
    fp32_size = get_size_mb(model)

    print("Running FP32 inference...")
    fp32_loss, fp32_accuracy, fp32_duration = run_inference(model, test_loader)

    with mlflow.start_run(run_name="quantized-fp32-inference"):
        mlflow.set_tag("quantization", "fp32")
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.log_metric("inference_accuracy", fp32_accuracy)
        mlflow.log_metric("inference_loss", fp32_loss)
        mlflow.log_metric("inference_duration_seconds", fp32_duration)
        mlflow.log_metric("model_size_mb", fp32_size)

    print(
        f"FP32 — accuracy: {fp32_accuracy:.4f}, duration: {fp32_duration:.2f}s, size: {fp32_size:.2f} MB"
    )

    print("Preparing model for static quantization...")
    model_int8 = copy.deepcopy(model)
    model_int8.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model_int8, inplace=True)

    print("Calibrating...")
    model_int8.eval()
    with torch.no_grad():
        for batch in test_loader:
            model_int8(**batch)

    torch.quantization.convert(model_int8, inplace=True)
    int8_size = get_size_mb(model_int8)

    print("Running INT8 inference...")
    int8_loss, int8_accuracy, int8_duration = run_inference(model_int8, test_loader)

    with mlflow.start_run(run_name="quantized-int8-inference"):
        mlflow.set_tag("quantization", "static_int8_pytorch")
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
