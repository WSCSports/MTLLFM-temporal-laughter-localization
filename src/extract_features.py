"""
Feature extraction for MTLLFM.

We release temporal annotations only -- NOT the underlying video/audio (see the
"Data & Copyright" section of the README). Once you have obtained the original
UR-FUNNY and SMILE media from their official sources and placed each clip under
the corresponding ``data/<dataset>/videos/`` folder (named to match the ``name``
field in the annotation JSON), this script extracts the frozen features the model
consumes:

  * Audio: HuBERT-Large -> (T_audio, 1024) at ~50 Hz.
  * Vision: MAE         -> (T_vision, 768) at 10 fps.

Features are cached as .pt tensors under ``data/<dataset>/features/`` so inference
does not need to re-run the encoders.

This is reference code provided for reproducibility; adapt paths/encoders to your
local environment as needed.
"""
import argparse
import os
from typing import List

import torch

# 5-second segments at the resolutions described in the paper.
AUDIO_SAMPLE_RATE = 16000
TARGET_FPS = 10
SEG_DUR = 5.0


def _list_clip_names(video_dir: str) -> List[str]:
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(
            f"Video directory '{video_dir}' not found. See the README "
            f"('Data & Copyright') for how to obtain and place the media."
        )
    exts = (".mp4", ".mkv", ".webm", ".avi", ".mov")
    return [f for f in sorted(os.listdir(video_dir)) if f.lower().endswith(exts)]


def load_audio_encoder(device: torch.device):
    """Load a frozen HuBERT-Large audio encoder (Hugging Face transformers)."""
    from transformers import AutoFeatureExtractor, HubertModel

    name = "facebook/hubert-large-ls960-ft"
    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = HubertModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return extractor, model


def load_vision_encoder(device: torch.device):
    """Load a frozen MAE (ViT) visual encoder (timm)."""
    import timm

    model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_audio_features(video_path, extractor, model, device) -> torch.Tensor:
    """Decode audio and return HuBERT features of shape (T_audio, 1024)."""
    import torchaudio

    waveform, sr = torchaudio.load(video_path)  # decodes the audio track
    if sr != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SAMPLE_RATE)
    waveform = waveform.mean(dim=0)  # mono
    inputs = extractor(
        waveform.numpy(), sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state  # (1, T, 1024)
    return out.squeeze(0).cpu()


def extract_vision_features(video_path, model, device) -> torch.Tensor:
    """Sample frames at TARGET_FPS and return MAE features of shape (T_vision, 768)."""
    import torchvision

    frames, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
    fps = float(info.get("video_fps", TARGET_FPS))
    step = max(1, int(round(fps / TARGET_FPS)))
    frames = frames[::step].float() / 255.0          # (N, H, W, C)
    frames = frames.permute(0, 3, 1, 2)              # (N, C, H, W)
    frames = torch.nn.functional.interpolate(frames, size=(224, 224), mode="bilinear")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    with torch.no_grad():
        feats = model(frames.to(device))            # (N, 768)
    return feats.cpu()


def main():
    parser = argparse.ArgumentParser(description="Extract MTLLFM features.")
    parser.add_argument("--dataset", choices=["ur_funny", "smile"], required=True)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dir = os.path.join(args.data_root, args.dataset, "videos")
    feat_dir = os.path.join(args.data_root, args.dataset, "features")
    os.makedirs(feat_dir, exist_ok=True)

    clips = _list_clip_names(video_dir)
    if not clips:
        print(f"No media found in '{video_dir}'. See the README before running.")
        return

    audio_extractor, audio_model = load_audio_encoder(device)
    vision_model = load_vision_encoder(device)

    for clip in clips:
        stem = os.path.splitext(clip)[0]
        video_path = os.path.join(video_dir, clip)
        audio = extract_audio_features(video_path, audio_extractor, audio_model, device)
        vision = extract_vision_features(video_path, vision_model, device)
        torch.save(
            {"audio": audio, "vision": vision}, os.path.join(feat_dir, stem + ".pt")
        )
        print(f"  {stem}: audio={tuple(audio.shape)} vision={tuple(vision.shape)}")


if __name__ == "__main__":
    main()
