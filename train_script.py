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
            # we can enable this to log the model aswell, but implement signature before doing it.
            # mlflow.pytorch.log_model(pytorch_model=model, name = f"checkpoint-epoch-{epoch}", step=epoch)

        torch.save(model.state_dict(), save_path)
