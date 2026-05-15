import time
import torch
import mlflow
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NUM_WARMUP_BATCHES = 3


def benchmark_batch_size(model, dataset, batch_size, device, collate_fn):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Warmup — GPU needs a few forward passes before timings stabilise
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= NUM_WARMUP_BATCHES:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_images = 0
    total_time = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            actual_batch_size = batch["pixel_values"].size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            model(**batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - t0
            total_images += actual_batch_size

    latency = total_time / len(loader)
    throughput = total_images / total_time
    return latency, throughput


def batch_benchmark(config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Running batch benchmark on: {device}")

    data = Data(config["data_path"])
    splits = data.get_train_val_test_sets(
        [config["train_split"], config["val_split"], config["test_split"]]
    )
    test_dataset = splits["test"]

    print("Loading production model from MLflow...")
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location=device
    )
    model.to(device)
    model.eval()

    with mlflow.start_run(run_name="batch-size-benchmark"):
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.set_tag("device", str(device))

        print(f"\n{'Batch size':>12} {'Latency (s)':>14} {'Throughput (img/s)':>20}")
        print("-" * 50)

        for batch_size in BATCH_SIZES:
            try:
                latency, throughput = benchmark_batch_size(
                    model, test_dataset, batch_size, device, data.collate_fn
                )
                mlflow.log_metric("latency_seconds", latency, step=batch_size)
                mlflow.log_metric("throughput_img_per_sec", throughput, step=batch_size)
                mlflow.log_metric("batch_size", batch_size, step=batch_size)
                print(f"{batch_size:>12} {latency:>14.4f} {throughput:>20.1f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{batch_size:>12} {'OOM':>14} {'OOM':>20}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise
