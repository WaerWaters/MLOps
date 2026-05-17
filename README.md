# MLOps — dvml_gruppe1

MobileNetV2 image classifier on Tiny ImageNet (200 classes), built as part of the DVML8 MLOps course at Aalborg University (Spring 2026).

## Stack

| Component | Detail |
|---|---|
| Model | MobileNetV2 (HuggingFace Transformers) |
| Dataset | Tiny ImageNet — 200 classes |
| Experiment tracking | MLflow|
| CI/CD | Jenkins → Docker → university GPU cluster |

## Pipeline

Push to `dev` triggers Jenkins, which runs `pipeline.sh`:

1. Builds the Docker image (pytest runs inside the build)
2. Pushes the image to the private registry
3. Runs the container with GPU access
4. Merges `dev` → `main` on success

The active experiment is selected in `main.py` by uncommenting the relevant function call.

## Experiments

| Script | Description |
|---|---|
| `train_script.py` | Standard training |
| `inference_script.py` | Standard inference |
| `quantize.py` | INT8/FP32 ONNX export and benchmarking |
| `pruning.py` | Unstructured magnitude pruning |
| `finetune_pruned.py` | Fine-tune after pruning |
| `batch_benchmark.py` | Latency across batch sizes |
| `carbon_tracking.py` | Training energy and CO₂ tracking |
| `drift_detection.py` | KernelMMD data drift detection |

## Configuration

Experiments are configured via `experiment_configs/test_config.yaml`.

## Monitoring (local)

A Prometheus + Grafana stack for inference monitoring (The predictions are simulated):

```bash
cd monitoring/
docker compose up -d
./generate_traffic.sh
```

## MM7 — Continual Learning & Unlearning

Standalone MNIST experiments (not part of the main pipeline):

```bash
source mm7_env/bin/activate
python mm7/continual_learning.py
python mm7/unlearning.py
```

## Tests

```bash
pytest tests/
```
