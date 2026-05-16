"""
Exercise 2: Machine Unlearning via Gradient Ascent
Forget class 7 from a model trained on all 10 MNIST digits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORGET_CLASS = 7
TRAIN_EPOCHS = 5
UNLEARN_STEPS = 150
BATCH = 512
TRAIN_LR = 1e-3
UNLEARN_LR = 5e-5


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def per_class_accuracy(model, loader):
    model.eval()
    correct = [0] * 10
    total = [0] * 10
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            for c in range(10):
                mask = y == c
                correct[c] += (preds[mask] == y[mask]).sum().item()
                total[c] += mask.sum().item()
    return [correct[c] / total[c] if total[c] > 0 else 0.0 for c in range(10)]


def main():
    print(f"Device: {DEVICE}")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        "~/.cache/mnist", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        "~/.cache/mnist", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    # ── Train on full MNIST ───────────────────────────────────────────────────
    print("\n=== Training on full MNIST (digits 0–9) ===")
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for ep in range(TRAIN_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        accs = per_class_accuracy(model, test_loader)
        overall = sum(accs) / 10
        print(
            f"  Epoch {ep + 1}: overall = {overall:.3f} | class {FORGET_CLASS} = {accs[FORGET_CLASS]:.3f}"
        )

    baseline = per_class_accuracy(model, test_loader)

    # ── Build forget and retain sets ─────────────────────────────────────────
    forget_idx = [i for i, (_, y) in enumerate(train_data) if y == FORGET_CLASS]
    retain_idx = [i for i, (_, y) in enumerate(train_data) if y != FORGET_CLASS]
    forget_loader = DataLoader(
        Subset(train_data, forget_idx), batch_size=BATCH, shuffle=True
    )
    retain_cycle = cycle(
        DataLoader(Subset(train_data, retain_idx), batch_size=BATCH, shuffle=True)
    )

    # ── Confuse Unlearning ────────────────────────────────────────────────────
    # Replace class 7 labels with a uniform distribution over all 10 classes.
    # Instead of "be confident in the wrong answer" (gradient ascent, unstable),
    # this teaches "don't be confident in anything" for class 7 inputs.
    # A paired retain step keeps all other classes intact.
    print(f"\n=== Confuse Unlearning: class {FORGET_CLASS} ({UNLEARN_STEPS} steps) ===")
    unlearn_opt = torch.optim.Adam(model.parameters(), lr=UNLEARN_LR)
    n_classes = 10
    step = 0
    model.train()
    while step < UNLEARN_STEPS:
        for x, _ in forget_loader:
            if step >= UNLEARN_STEPS:
                break
            x = x.to(DEVICE)

            # Confuse: push class-7 outputs toward uniform distribution
            unlearn_opt.zero_grad()
            uniform = torch.full((x.size(0), n_classes), 1.0 / n_classes, device=DEVICE)
            log_probs = F.log_softmax(model(x), dim=1)
            F.kl_div(log_probs, uniform, reduction="batchmean").backward()
            unlearn_opt.step()

            # Retain: keep all other classes intact
            rx, ry = next(retain_cycle)
            unlearn_opt.zero_grad()
            F.cross_entropy(model(rx.to(DEVICE)), ry.to(DEVICE)).backward()
            unlearn_opt.step()

            step += 1
            if step % 50 == 0:
                print(f"  Step {step}/{UNLEARN_STEPS}")

    after = per_class_accuracy(model, test_loader)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Class':<8} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 50)
    for c in range(10):
        tag = "  <-- FORGOTTEN" if c == FORGET_CLASS else ""
        print(
            f"  {c:<6} {baseline[c]:>8.3f} {after[c]:>8.3f} {after[c] - baseline[c]:>+8.3f}{tag}"
        )

    others_before = sum(baseline[c] for c in range(10) if c != FORGET_CLASS) / 9
    others_after = sum(after[c] for c in range(10) if c != FORGET_CLASS) / 9
    print("=" * 50)
    print(
        f"\nClass {FORGET_CLASS} accuracy:    {baseline[FORGET_CLASS]:.3f} → {after[FORGET_CLASS]:.3f}  "
        f"(drop of {baseline[FORGET_CLASS] - after[FORGET_CLASS]:.3f})"
    )
    print(
        f"Other classes avg:  {others_before:.3f} → {others_after:.3f}  "
        f"(drop of {others_before - others_after:.3f})"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    classes = list(range(10))
    x = np.arange(10)
    w = 0.35
    colors_before = ["steelblue"] * 10
    colors_after = ["tomato" if c == FORGET_CLASS else "seagreen" for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - w / 2,
        baseline,
        w,
        label="Before unlearning",
        color=colors_before,
        alpha=0.85,
    )
    ax.bar(
        x + w / 2, after, w, label="After unlearning", color=colors_after, alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Class {c}" + (" (forget)" if c == FORGET_CLASS else "") for c in classes],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Confuse Unlearning: forgetting class {FORGET_CLASS} on MNIST")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    out = "mm7/unlearning_results.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
