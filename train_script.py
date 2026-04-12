import os
import time
import torch
from torch.utils.data import DataLoader
from data.get_data import Data
from models.image_classifier import ImageModel
import mlflow


def train(config, git_hash="unknown"):
    data_path = config["data_path"]
    data_splits = [config["train_split"], config["val_split"], config["test_split"]]
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    bs = config["batch_size"]
    epochs = config["num_epochs"]
    save_path = config["save_path"]

    with mlflow.start_run(run_name=f"{git_hash}-train"):
        mlflow.set_tag("git_commit", git_hash)
        mlflow.set_tag(
            "jenkins_build_number", os.environ.get("BUILD_NUMBER", "unknown")
        )
        mlflow.set_tag("docker_image", f"dvml_gruppe1:{git_hash}")
        # log params for mlflow run
        mlflow.log_params(config)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"Running training on: {device}")

        data = Data(data_path)
        splits = data.get_train_val_test_sets(data_splits)

        model = ImageModel().get_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        train_data = splits["train"]
        if config.get("for_testing", False):
            train_data = train_data.select(range(int(len(train_data) * 0.01)))

        train_loader = DataLoader(
            train_data, batch_size=bs, shuffle=True, collate_fn=data.collate_fn
        )

        model.train()
        train_start = time.time()
        for epoch in range(epochs):
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss

                # log the loss of current batch in epoch
                mlflow.log_metric("train_loss", loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Loss: {loss.item():.4f}")

        training_duration = time.time() - train_start
        mlflow.log_metric("training_duration_seconds", training_duration)
        print(f"Training duration: {training_duration:.2f}s")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
