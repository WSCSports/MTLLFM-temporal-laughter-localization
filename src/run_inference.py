"""
Run MTLLFM inference on the released temporal laughter annotations.

Pipeline:
  1. Load the released annotation JSON and the locally extracted features.
  2. Load a trained MTLLFM checkpoint (produced by src/train.py).
  3. For each segment: predict the laughter label, and -- for positive segments --
     localize the laughter interval from the learned attention distributions.
  4. Report classification F1 and localization metrics (Precision@IoU=0.5, mean IoU)
     against the released ground-truth events.

Only OUR model is run here (no baselines / foundation models).

Pre-trained weights are not distributed; train a checkpoint first with src/train.py.

Example:
    python src/run_inference.py \
        --annotations annotations/temporal_smile.json \
        --features data/smile/features \
        --weights weights/mtllfm_smile.pt \
        --split test
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AUDIO_DIM, VISION_DIM, LaughterDataset
from localize import attention_to_interval, multi_gt_iou
from model import AudioVisionSoftmaxPool

# Model configuration (must match src/train.py).
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
        "name": [b["name"] for b in batch],
        "audio": torch.stack([b["audio"] for b in batch]),
        "vision": torch.stack([b["vision"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "events": [b["events"] for b in batch],
    }


def load_model(weights_path: str, device: torch.device) -> AudioVisionSoftmaxPool:
    model = AudioVisionSoftmaxPool(**MODEL_CONFIG).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="MTLLFM inference (our model only).")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--weights", default="weights/mtllfm_smile.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="predictions.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LaughterDataset(args.annotations, args.features, split=args.split)
    if dataset.missing:
        print(
            f"[info] {dataset.missing} annotated clips have no local features and "
            f"were skipped (you likely have a subset of the media)."
        )
    if len(dataset) == 0:
        print(
            "[stop] No clips with extracted features were found.\n"
            "       1) Obtain the original media (see README 'Data & Copyright').\n"
            "       2) Run src/extract_features.py to cache features.\n"
            "       3) Re-run this script."
        )
        return
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    model = load_model(args.weights, device)

    predictions = []
    y_true, y_pred = [], []
    ious = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            logits, att = model(audio, vision, is_training=False)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i, name in enumerate(batch["name"]):
                label = int(batch["label"][i])
                pred = int(preds[i])
                y_true.append(label)
                y_pred.append(pred)

                record = {
                    "name": name,
                    "pred_label": pred,
                    "prob_laughter": float(probs[i]),
                }

                events = batch["events"][i]
                if pred == 1 or label == 1:
                    gates = att["gates_att"][i] if att["gates_att"] is not None else None
                    start, end = attention_to_interval(
                        att["audio_att"][i], att["vision_att"][i], gates
                    )
                    record["pred_interval"] = [round(start, 3), round(end, 3)]
                    if events:  # score localization against ground truth
                        sample_iou = multi_gt_iou((start, end), events)
                        record["iou"] = round(sample_iou, 3)
                        ious.append(sample_iou)
                predictions.append(record)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # Metrics
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nEvaluated {len(dataset)} clips ({args.split} split)")
    print(f"  Classification F1 : {f1:.4f}")
    if ious:
        ious_arr = np.array(ious)
        print(f"  Localization P@0.5: {float((ious_arr >= 0.5).mean()):.4f}")
        print(f"  Mean IoU          : {float(ious_arr.mean()):.4f}  (N_loc={len(ious)})")
    print(f"  Per-clip predictions written to {args.output}")


if __name__ == "__main__":
    main()
