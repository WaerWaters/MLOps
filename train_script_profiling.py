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
        parallel_time = 0.0  # forward + backward (parallelizable across GPUs)
        sequential_time = (
            0.0  # data transfer, zero_grad, optimizer step (sequential overhead)
        )

        for epoch in range(epochs):
            for batch in train_loader:
                # Sequential: data transfer to device
                t0 = time.time()
                batch = {k: v.to(device) for k, v in batch.items()}
                sequential_time += time.time() - t0

                # Parallel: forward pass
                t0 = time.time()
                outputs = model(**batch)
                loss = outputs.loss
                if device.type == "cuda":
                    torch.cuda.synchronize()
                parallel_time += time.time() - t0

                mlflow.log_metric("train_loss", loss.item())

                # Parallel: zero_grad (each GPU clears its own gradients independently)
                t0 = time.time()
                optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                parallel_time += time.time() - t0

                # Parallel: backward pass
                t0 = time.time()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                parallel_time += time.time() - t0

                # Sequential: optimizer step
                t0 = time.time()
                optimizer.step()
                sequential_time += time.time() - t0

                print(f"Loss: {loss.item():.4f}")

        training_duration = time.time() - train_start
        sequential_fraction = sequential_time / training_duration

        mlflow.log_metric("training_duration_seconds", training_duration)

        print("\n--- Parallelization Profiling ---")
        print(f"Total training time:             {training_duration:.2f}s")
        print(f"Parallel time (fwd + bwd):       {parallel_time:.2f}s")
        print(f"Sequential time (overhead):      {sequential_time:.2f}s")
        print(f"Sequential fraction (s):         {sequential_fraction:.4f}")
        print("")
        print("  Amdahl's Law (fixed problem size):")
        print(f"  Max speedup (inf N):            {1 / sequential_fraction:.2f}x")
        print("")
        print("  Gustafson's Law (scaled problem size):")
        for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            gustafson_speedup = n - sequential_fraction * (n - 1)
            print(f"  Speedup with {n:>2} GPUs:          {gustafson_speedup:.2f}x")
        print("------------------------------\n")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
