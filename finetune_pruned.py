import torch
import torch.nn.utils.prune as prune
import mlflow
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"
PRUNING_RATIO = 0.30
FINETUNE_EPOCHS = 3
FINETUNE_LR = 0.0001


def _get_prunable_params(model):
    params = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            params.append((module, "weight"))
    return params


def _apply_pruning(model, amount):
    params = _get_prunable_params(model)
    prune.global_unstructured(
        params, pruning_method=prune.L1Unstructured, amount=amount
    )
    for module, param in params:
        prune.remove(module, param)
    return model


def _evaluate(model, loader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += loss_fn(outputs.logits, batch["labels"]).item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    return total_loss / len(loader), correct / total


def finetune_pruned(config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Fine-tuning on: {device}")

    data = Data(config["data_path"])
    splits = data.get_train_val_test_sets(
        [config["train_split"], config["val_split"], config["test_split"]]
    )

    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]

    if config.get("for_testing", False):
        train_data = train_data.select(
            range(min(int(len(train_data) * 0.10), len(train_data)))
        )
        val_data = val_data.select(range(min(1000, len(val_data))))
        test_data = test_data.select(range(min(1000, len(test_data))))

    bs = config["batch_size"]
    train_loader = DataLoader(
        train_data, batch_size=bs, shuffle=True, collate_fn=data.collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=bs, shuffle=False, collate_fn=data.collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=bs, shuffle=False, collate_fn=data.collate_fn
    )

    print("Loading production model from MLflow...")
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location=device
    )

    print(f"Applying {int(PRUNING_RATIO * 100)}% pruning...")
    model = _apply_pruning(model, PRUNING_RATIO)
    model.to(device)

    _, pre_val_accuracy = _evaluate(model, val_loader, device)
    print(f"Accuracy before fine-tuning (val): {pre_val_accuracy:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="pruning-finetune"):
        mlflow.set_tag("source_model", f"{MODEL_NAME}@{ALIAS}")
        mlflow.set_tag("pruning_ratio", str(PRUNING_RATIO))
        mlflow.log_param("pruning_ratio", PRUNING_RATIO)
        mlflow.log_param("finetune_epochs", FINETUNE_EPOCHS)
        mlflow.log_param("finetune_lr", FINETUNE_LR)
        mlflow.log_metric("pre_finetune_val_accuracy", pre_val_accuracy)

        print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Accuracy':>14}")
        print("-" * 36)

        for epoch in range(1, FINETUNE_EPOCHS + 1):
            model.train()
            total_train_loss = 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch["labels"])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            _, val_accuracy = _evaluate(model, val_loader, device)

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"{epoch:>6} {avg_train_loss:>12.4f} {val_accuracy:>14.4f}")

        _, post_test_accuracy = _evaluate(model, test_loader, device)
        mlflow.log_metric("post_finetune_test_accuracy", post_test_accuracy)

        print(f"\nAccuracy before fine-tuning (val):  {pre_val_accuracy:.4f}")
        print(f"Accuracy after fine-tuning (test):  {post_test_accuracy:.4f}")
        print(
            f"Recovery:                           {post_test_accuracy - pre_val_accuracy:+.4f}"
        )
