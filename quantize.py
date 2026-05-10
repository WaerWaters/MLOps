import os
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


def quantize(config):
    data = Data(config["data_path"])
    splits = data.get_train_val_test_sets(
        [config["train_split"], config["val_split"], config["test_split"]]
    )

    # 1. Load production model from MLflow
    print("Loading production model from MLflow...")
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location="cpu"
    )
    model.eval()

    fp32_size = get_model_size_mb(model)
    print(f"FP32 model size: {fp32_size:.2f} MB")

    # 2. Set quantization config (fbgemm = optimised for x86 CPUs)
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # 3. Prepare model — inserts observers to record activation ranges
    model_prepared = torch.quantization.prepare(model, inplace=False)

    # 4. Calibrate — run representative data through to determine scale/zero-point
    print("Calibrating on 200 training samples...")
    cal_loader = DataLoader(
        splits["train"].select(range(200)),
        batch_size=32,
        shuffle=False,
        collate_fn=data.collate_fn,
    )
    with torch.no_grad():
        for batch in cal_loader:
            batch = {k: v.cpu() for k, v in batch.items()}
            model_prepared(**batch)

    # 5. Convert — replace float ops with quantized INT8 ops
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)

    int8_size = get_model_size_mb(model_quantized)
    print(f"INT8 model size: {int8_size:.2f} MB")
    print(f"Size reduction:  {(1 - int8_size / fp32_size) * 100:.1f}%")

    # 6. Run inference and log everything to MLflow
    print("Running inference with quantized model...")
    test_loader = DataLoader(
        splits["test"],
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data.collate_fn,
    )

    with mlflow.start_run(run_name="quantized-int8-inference"):
        mlflow.set_tag("quantization", "static_int8")
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")

        total_loss = 0
        correct = 0
        total = 0

        model_quantized.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.cpu() for k, v in batch.items()}
                outputs = model_quantized(**batch)
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        mlflow.log_metric("inference_accuracy", accuracy)
        mlflow.log_metric("inference_loss", avg_loss)
        mlflow.log_metric("model_size_fp32_mb", fp32_size)
        mlflow.log_metric("model_size_int8_mb", int8_size)
        mlflow.log_metric("size_reduction_pct", (1 - int8_size / fp32_size) * 100)

        print("\n--- Results ---")
        print(f"Accuracy:       {accuracy:.4f}")
        print(f"Loss:           {avg_loss:.4f}")
        print(f"FP32 size:      {fp32_size:.2f} MB")
        print(f"INT8 size:      {int8_size:.2f} MB")
        print(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")
