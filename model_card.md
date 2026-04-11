# Model Card — dvml_gruppe1

## Model Description

MobileNetV2 image classifier trained on Tiny ImageNet. The model is trained from scratch using the `google/mobilenet_v2_1.0_224` architecture adapted for 200-class classification.

- **Model type:** Image classification (MobileNetV2)
- **Architecture:** `google/mobilenet_v2_1.0_224`, 200 output classes
- **Developed by:** dvml_gruppe1

## Uses

### Direct Use
Classifying 224x224 images into one of 200 Tiny ImageNet categories.

## Training Details

### Training Data
Tiny ImageNet — a subset of ImageNet with 200 classes and 64x64 pixel images, that are upscaled to 224x224 during preprocessing. Data is versioned with DVC against a MinIO S3 backend.

- Train split: 70%
- Test split: 30%

### Training Hyperparameters
- Optimizer: AdamW
- Learning rate: 0.0005
- Weight decay: 0.01
- Batch size: 128
- Epochs: 1

## Evaluation

### Metrics
- **Inference accuracy** — fraction of correctly classified test images
- **Inference loss** — cross-entropy loss on the test set

### Promotion Criteria
A model is promoted to the `production` alias in the MLflow model registry only if its inference accuracy exceeds the current production model's accuracy.

## Technical Specifications

### Software
- PyTorch
- HuggingFace Transformers
- MLflow (experiment tracking + model registry)
- DVC (data versioning)
