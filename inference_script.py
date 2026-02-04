import yaml
import torch
from torch.utils.data import DataLoader
from data.get_data import Data
from models.image_classifier import ImageModel

with open("experiment_configs/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

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


data = Data(data_path)
splits = data.get_train_val_test_sets(data_splits)

test_loader = DataLoader(
    splits["test"],
    batch_size=bs,
    shuffle=False,
    collate_fn=data.collate_fn
)

model = ImageModel().get_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"Loaded model from {model_path}")

total_loss = 0
correct = 0
total = 0

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

print(f"\nTest Results:")
print(f"  Loss:     {avg_loss:.4f}")
print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")