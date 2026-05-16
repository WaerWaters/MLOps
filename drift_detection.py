import torch
import torchdrift
import mlflow
from torch.utils.data import DataLoader
from data.get_data import Data

MODEL_NAME = "dvml_gruppe1"
ALIAS = "production"
REFERENCE_SAMPLES = 5000
TEST_SAMPLES = 1000
DRIFT_P_VALUE_THRESHOLD = 0.05
NOISE_STD = 0.3


def _get_feature_extractor(model, device):
    backbone = model.mobilenet_v2
    backbone.eval()
    backbone.to(device)

    def extract(pixel_values):
        with torch.no_grad():
            out = backbone(pixel_values)
            return out.last_hidden_state.mean(dim=[2, 3])

    return extract


def _collect_embeddings(extractor, loader, n_samples, device):
    embeddings = []
    collected = 0
    for batch in loader:
        if collected >= n_samples:
            break
        pixel_values = batch["pixel_values"].to(device)
        emb = extractor(pixel_values)
        embeddings.append(emb.cpu())
        collected += emb.shape[0]
    return torch.cat(embeddings)[:n_samples]


def _collect_embeddings_noisy(extractor, loader, n_samples, device, noise_std):
    embeddings = []
    collected = 0
    for batch in loader:
        if collected >= n_samples:
            break
        pixel_values = batch["pixel_values"].to(device)
        pixel_values = pixel_values + torch.randn_like(pixel_values) * noise_std
        emb = extractor(pixel_values)
        embeddings.append(emb.cpu())
        collected += emb.shape[0]
    return torch.cat(embeddings)[:n_samples]


def _run_detection(detector, embeddings, label):
    score = detector(embeddings)
    p_val = detector.compute_p_value(embeddings)
    drifted = p_val < DRIFT_P_VALUE_THRESHOLD
    print(
        f"  [{label}] MMD score: {score:.4f} | p-value: {p_val:.4f} | drifted: {drifted}"
    )
    return float(score), float(p_val), bool(drifted)


def drift_detection(config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Running drift detection on: {device}")

    data = Data(config["data_path"])
    splits = data.get_train_val_test_sets(
        [config["train_split"], config["val_split"], config["test_split"]]
    )

    train_loader = DataLoader(
        splits["train"].select(range(REFERENCE_SAMPLES)),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data.collate_fn,
    )
    test_loader = DataLoader(
        splits["test"].select(range(TEST_SAMPLES)),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data.collate_fn,
    )

    print("Loading production model from MLflow...")
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}", map_location=device
    )
    model.eval()

    extractor = _get_feature_extractor(model, device)

    print(f"Collecting reference embeddings ({REFERENCE_SAMPLES} samples)...")
    reference_embeddings = _collect_embeddings(
        extractor, train_loader, REFERENCE_SAMPLES, device
    )

    print("Fitting KernelMMD drift detector on reference distribution...")
    detector = torchdrift.detectors.KernelMMDDriftDetector()
    detector.fit(reference_embeddings)

    print(f"\nCollecting test embeddings ({TEST_SAMPLES} samples)...")
    clean_embeddings = _collect_embeddings(extractor, test_loader, TEST_SAMPLES, device)
    noisy_embeddings = _collect_embeddings_noisy(
        extractor, test_loader, TEST_SAMPLES, device, NOISE_STD
    )

    print("\nRunning drift detection:")
    clean_score, clean_p, clean_drifted = _run_detection(
        detector, clean_embeddings, "clean"
    )
    noisy_score, noisy_p, noisy_drifted = _run_detection(
        detector, noisy_embeddings, f"noisy (std={NOISE_STD})"
    )

    shared_params = {
        "reference_samples": REFERENCE_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "drift_threshold": DRIFT_P_VALUE_THRESHOLD,
        "detector": "KernelMMDDriftDetector",
    }

    with mlflow.start_run(run_name="drift-detection-clean"):
        mlflow.log_params(shared_params)
        mlflow.log_param("noise_std", 0.0)
        mlflow.log_metric("mmd_score", clean_score)
        mlflow.log_metric("p_value", clean_p)
        mlflow.log_metric("drift_detected", int(clean_drifted))

    with mlflow.start_run(run_name="drift-detection-noisy"):
        mlflow.log_params(shared_params)
        mlflow.log_param("noise_std", NOISE_STD)
        mlflow.log_metric("mmd_score", noisy_score)
        mlflow.log_metric("p_value", noisy_p)
        mlflow.log_metric("drift_detected", int(noisy_drifted))

    print("\nSummary:")
    print(f"  Clean test data  — drift detected: {clean_drifted}")
    print(f"  Noisy test data  — drift detected: {noisy_drifted}")
