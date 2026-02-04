from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device).eval()

inputs = preprocessor(image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])