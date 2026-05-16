import os
import re
import glob
import time
import tempfile
import torch
from torch.utils.data import DataLoader
from data.get_data import Data
from models.image_classifier import ImageModel
import mlflow
from carbontracker.tracker import CarbonTracker


def _parse_carbontracker_log(log_dir):
    logs = sorted(f for f in glob.glob(f"{log_dir}/*.log") if "_detail" not in f)
    if not logs:
        return None, None
    with open(logs[-1]) as f:
        content = f.read()
    energy_kwh = None
    co2_g = None
    in_actual = False
    for line in content.splitlines():
        if "Actual consumption" in line:
            in_actual = True
        elif "Predicted consumption" in line:
            in_actual = False
        elif in_actual:
            if "Energy:" in line:
                m = re.search(r"Energy:\s+([\d.e+-]+)\s+kWh", line)
                if m:
                    energy_kwh = float(m.group(1))
            elif "CO2" in line:
                m = re.search(r"CO2[a-z]*:\s+([\d.e+-]+)\s+g", line)
                if m:
                    co2_g = float(m.group(1))
    return energy_kwh, co2_g


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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        ct_log_dir = tempfile.mkdtemp(prefix="carbontracker_")
        tracker = CarbonTracker(epochs=epochs, log_dir=ct_log_dir, verbose=2)

        train_start = time.time()
        for epoch in range(epochs):
            tracker.epoch_start()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                mlflow.log_metric("train_loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Loss: {loss.item():.4f}")
            tracker.epoch_end()

        tracker.stop()

        training_duration = time.time() - train_start
        mlflow.log_metric("training_duration_seconds", training_duration)
        print(f"Training duration: {training_duration:.2f}s")

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2
            mlflow.log_metric("peak_vram_mb", peak_vram)
            print(f"Peak VRAM usage:  {peak_vram:.1f} MB")

        energy_kwh, co2_g = _parse_carbontracker_log(ct_log_dir)
        if energy_kwh is not None:
            mlflow.log_metric("training_energy_kwh", energy_kwh)
            print(f"Training energy:  {energy_kwh:.6f} kWh")
        if co2_g is not None:
            mlflow.log_metric("training_co2_g", co2_g)
            print(f"Training CO2eq:   {co2_g:.4f} g")

        for lf in glob.glob(f"{ct_log_dir}/*.log"):
            mlflow.log_artifact(lf, artifact_path="carbontracker")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
