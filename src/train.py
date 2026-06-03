"""
Train MTLLFM with weak, clip-level supervision.

Following the paper (Section 5.2), the model is trained using ONLY binary clip-level
labels (laughter present / absent) -- no frame-level annotations are used during
training. Temporal localization emerges post-hoc from the learned attention
distributions (see localize.py). Training therefore reduces to binary classification
with Focal Loss.

Defaults match the paper:
    Adam, lr=1e-4, batch size 32, hidden dim 1024, dropout 0.5,
    up to 50 epochs with early stopping on validation loss.

Example:
    python src/train.py \
        --annotations annotations/temporal_smile.json \
        --features data/smile/features \
        --train-split train --val-split dev \
        --out weights/mtllfm_smile.pt
"""
import argparse
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AUDIO_DIM, VISION_DIM, LaughterDataset
from losses import FocalLoss
from model import AudioVisionSoftmaxPool

MODEL_CONFIG = dict(
    audio_dim=AUDIO_DIM,
    vision_dim=VISION_DIM,
    hidden_dim=1024,
    output_dim=2,
    dropout=0.5,
    fusion_type="gated",
    activation_and_norm=False,
    gating_type="softmax",
    softmax_tanh=True,
)


def collate(batch):
    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "vision": torch.stack([b["vision"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(audio, vision, is_training=False)
            total += criterion(logits, labels).item() * len(labels)
            n += len(labels)
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train MTLLFM (weak clip-level labels).")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="dev")
    parser.add_argument("--out", default="weights/mtllfm.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, nargs=2, default=[0.45, 0.55],
                        help="Focal-loss class weights [neg, pos].")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LaughterDataset(args.annotations, args.features, split=args.train_split)
    val_ds = LaughterDataset(args.annotations, args.features, split=args.val_split)
    if len(train_ds) == 0:
        print(
            "[stop] No training clips with extracted features found.\n"
            "       Obtain the media (README 'Data & Copyright') and run "
            "src/extract_features.py first."
        )
        return
    print(f"Train={len(train_ds)}  Val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    model = AudioVisionSoftmaxPool(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(alpha=tuple(args.alpha)).to(device)

    best_val, best_state, best_ep, no_improve = float("inf"), None, 0, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, _ = model(audio, vision, is_training=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, criterion, device) if len(val_ds) else 0.0
        print(f"  epoch {ep:02d}  val_loss={val_loss:.4f}")

        if ep >= args.min_epochs:
            if val_loss < best_val:
                best_val, best_ep = val_loss, ep
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  early stop at epoch {ep} (best={best_ep})")
                    break

    state = best_state if best_state is not None else model.state_dict()
    torch.save(state, args.out)
    print(f"Saved checkpoint to {args.out} (best epoch={best_ep}, val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
