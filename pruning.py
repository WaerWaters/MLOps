import copy
import os
import time
import torch
import torch.nn.utils.prune as prune
import mlflow
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"
PRUNING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _get_prunable_params(model):
    params = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            params.append((module, "weight"))
    return params


def _apply_pruning(model, amount):
    if amount == 0.0:
        return model
    params = _get_prunable_params(model)
    prune.global_unstructured(
        params, pruning_method=prune.L1Unstructured, amount=amount
    )
    for module, param in params:
        prune.remove(module, param)
    return model


def _get_size_mb(model):
    path = "/tmp/pruned_model_size.pth"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024**2
    os.remove(path)
    return size


def _run_inference(model, test_loader):
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


def pruning_experiment(config):
    data = Data(config["data_path"])
    splits = data.get_train_val_test_sets(
        [config["train_split"], config["val_split"], config["test_split"]]
    )

    test_data = splits["test"]
    if config.get("for_testing", False):
        test_data = test_data.select(range(min(1000, len(test_data))))

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data.collate_fn,
    )

    print("Loading production model from MLflow...")
    base_model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location="cpu"
    )
    base_model.eval()

    with mlflow.start_run(run_name="pruning-accuracy-tradeoff"):
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.set_tag("pruning_method", "global_unstructured_l1")
        mlflow.set_tag("pruning_scope", "Conv2d+Linear weights")

        print(
            f"\n{'Pruning %':>10} {'Accuracy':>10} {'Loss':>10} {'Size (MB)':>12} {'Duration (s)':>14}"
        )
        print("-" * 60)

        for ratio in PRUNING_RATIOS:
            model = _apply_pruning(copy.deepcopy(base_model), ratio)
            loss, accuracy, duration = _run_inference(model, test_loader)
            size_mb = _get_size_mb(model)

            pct = int(ratio * 100)
            mlflow.log_metric("accuracy", accuracy, step=pct)
            mlflow.log_metric("loss", loss, step=pct)
            mlflow.log_metric("pruning_ratio", ratio, step=pct)
            mlflow.log_metric("model_size_mb", size_mb, step=pct)
            mlflow.log_metric("inference_duration_seconds", duration, step=pct)

            print(
                f"{pct:>9}% {accuracy:>10.4f} {loss:>10.4f} {size_mb:>12.2f} {duration:>14.2f}"
            )
