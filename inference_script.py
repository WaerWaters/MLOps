import os
import torch
from torch.utils.data import DataLoader
from data.get_data import Data
from models.image_classifier import ImageModel
import mlflow
from model_registry import promote_if_best


def inference(config, git_hash="unknown"):
    data_path = config["data_path"]
    data_splits = [config["train_split"], config["val_split"], config["test_split"]]
    bs = config["batch_size"]
    model_path = config["save_path"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Running inference on: {device}")

    data = Data(data_path)
    splits = data.get_train_val_test_sets(data_splits)

    test_data = splits["test"]
    if config.get("for_testing", False):
        test_data = test_data.select(range(int(len(test_data) * 0.01)))

    test_loader = DataLoader(
        test_data, batch_size=bs, shuffle=False, collate_fn=data.collate_fn
    )

    model = ImageModel().get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")

    total_loss = 0
    correct = 0
    total = 0
    with mlflow.start_run(run_name=f"{git_hash}-inference"):
        mlflow.set_tag("git_commit", git_hash)
        mlflow.set_tag(
            "jenkins_build_number", os.environ.get("BUILD_NUMBER", "unknown")
        )
        mlflow.set_tag("docker_image", f"dvml_gruppe1:{git_hash}")
        mlflow.log_param("data_minio_version", "3500cb7f2b0b1b58ab0c70345cf40596.dir")
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        mlflow.log_metric("inference-loss", avg_loss)
        mlflow.log_metric("inference accuracy", accuracy)

        mlflow.log_artifact("model_card.md")

        performance_criteria = 0.0001
        if accuracy > performance_criteria:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="dvml_gruppe1",
            )
            mlflow.set_tag("model_registered", "true")
            new_version = (
                mlflow.MlflowClient()
                .get_latest_versions("dvml_gruppe1", stages=["None"])[0]
                .version
            )
            promote_if_best("dvml_gruppe1", new_version, accuracy, git_hash)
            print(
                f"Model registered to MLflow registry (accuracy{accuracy:.4f} threshold: {performance_criteria})"
            )
        else:
            mlflow.set_tag("model_registered", "false")
            print(
                f"Model did not meet performance criteria (accuracy{accuracy:.4f} threshold: {performance_criteria})"
            )

        print("\nTest Results:")
        print(f"  Loss:     {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    return avg_loss, accuracy
