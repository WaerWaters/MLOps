import os
import time
import torch
import mlflow
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"


def get_model_size_mb(model):
    torch.save(model.state_dict(), "_tmp.pt")
    size = os.path.getsize("_tmp.pt") / 1024**2
    os.remove("_tmp.pt")
    return size


def run_inference(model, test_loader, device="cpu"):
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
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

    test_loader = DataLoader(
        splits["test"],
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data.collate_fn,
    )

    # Load production model from MLflow
    print("Loading production model from MLflow...")
    model_fp32 = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location="cpu"
    )
    model_fp32.eval()
    fp32_size = get_model_size_mb(model_fp32)

    # --- Baseline: FP32 on CPU ---
    print("Running FP32 baseline inference on CPU...")
    fp32_loss, fp32_accuracy, fp32_duration = run_inference(model_fp32, test_loader)

    with mlflow.start_run(run_name="baseline-fp32-cpu-inference"):
        mlflow.set_tag("quantization", "none")
        mlflow.set_tag("device", "cpu")
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.log_metric("inference_accuracy", fp32_accuracy)
        mlflow.log_metric("inference_loss", fp32_loss)
        mlflow.log_metric("inference_duration_seconds", fp32_duration)
        mlflow.log_metric("model_size_mb", fp32_size)

    print(
        f"FP32 — accuracy: {fp32_accuracy:.4f}, duration: {fp32_duration:.2f}s, size: {fp32_size:.2f} MB"
    )

    # Apply dynamic quantization
    print("Applying dynamic quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    int8_size = get_model_size_mb(model_int8)

    # --- Quantized: INT8 on CPU ---
    print("Running INT8 quantized inference on CPU...")
    int8_loss, int8_accuracy, int8_duration = run_inference(model_int8, test_loader)

    with mlflow.start_run(run_name="quantized-int8-cpu-inference"):
        mlflow.set_tag("quantization", "dynamic_int8")
        mlflow.set_tag("device", "cpu")
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
