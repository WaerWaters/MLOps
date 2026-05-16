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

# A100 GPU TDP (Watt) — fallback when NVML is unavailable
_A100_TDP_W = 400
# Denmark average carbon intensity gCO2eq/kWh (2024)
_DK_CARBON_INTENSITY = 170
# Typical university data-center PUE
_PUE = 1.2


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


def _estimate_carbon_manual(duration_seconds):
    """Fallback: estimate energy from A100 TDP + Denmark carbon intensity."""
    energy_kwh = (_A100_TDP_W / 1000) * (duration_seconds / 3600) * _PUE
    co2_g = energy_kwh * _DK_CARBON_INTENSITY
    return energy_kwh, co2_g


def train_with_carbon_tracking(config):
    data_path = config["data_path"]
    data_splits = [config["train_split"], config["val_split"], config["test_split"]]
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    bs = config["batch_size"]
    epochs = config["num_epochs"]
    save_path = config["save_path"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Running carbon-tracked training on: {device}")

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

    ct_log_dir = tempfile.mkdtemp(prefix="carbontracker_")
    # components="gpu" skips Intel RAPL which is unavailable in Docker
    tracker = CarbonTracker(
        epochs=epochs, log_dir=ct_log_dir, verbose=2, components="gpu"
    )
    tracker_ok = True

    with mlflow.start_run(run_name="carbon-tracking-train"):
        mlflow.log_params(config)

        train_start = time.time()
        for epoch in range(epochs):
            if tracker_ok:
                try:
                    tracker.epoch_start()
                except Exception as e:
                    print(
                        f"CarbonTracker epoch_start failed: {e}. Continuing without tracking."
                    )
                    tracker_ok = False

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                mlflow.log_metric("train_loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Loss: {loss.item():.4f}")

            if tracker_ok:
                try:
                    tracker.epoch_end()
                except Exception as e:
                    print(
                        f"CarbonTracker epoch_end failed: {e}. Continuing without tracking."
                    )
                    tracker_ok = False

        training_duration = time.time() - train_start
        mlflow.log_metric("training_duration_seconds", training_duration)
        print(f"Training duration: {training_duration:.2f}s")

        energy_kwh, co2_g = None, None

        if tracker_ok:
            try:
                tracker.stop()
                energy_kwh, co2_g = _parse_carbontracker_log(ct_log_dir)
                mlflow.set_tag("carbon_source", "carbontracker_gpu")
                for lf in glob.glob(f"{ct_log_dir}/*.log"):
                    mlflow.log_artifact(lf, artifact_path="carbontracker")
            except Exception as e:
                print(
                    f"CarbonTracker stop/parse failed: {e}. Falling back to manual estimate."
                )

        if energy_kwh is None:
            energy_kwh, co2_g = _estimate_carbon_manual(training_duration)
            mlflow.set_tag(
                "carbon_source",
                f"manual_estimate_A100_{_A100_TDP_W}W_DK_{_DK_CARBON_INTENSITY}gCO2/kWh",
            )
            print(
                "Using manual carbon estimate (carbontracker unavailable in this environment)."
            )

        mlflow.log_metric("training_energy_kwh", energy_kwh)
        mlflow.log_metric("training_co2_g", co2_g)
        print(f"Training energy:  {energy_kwh:.6f} kWh")
        print(f"Training CO2eq:   {co2_g:.4f} g")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
