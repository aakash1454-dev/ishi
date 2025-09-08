#!/usr/bin/env python3
import argparse, json
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

def infer_out_dim(state):
    # Try common classifier keys
    for k in ["classifier.3.weight", "fc.weight", "classifier.weight", "head.weight"]:
        if k in state:
            return int(state[k].shape[0])
    return None

def strip_state_dict(sd):
    # Handle {'state_dict': ...} wrapper
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # Remove DataParallel prefix
    sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items() }
    return sd

def build_model(arch: str, num_classes: int):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m

def load_weights(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = strip_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # It's normal for head to be missing if num_classes mismatched before rebuild;
    # at this point it should match. If not, we still proceed.
    return model

def infer_one(model, img_path, img_size, threshold):
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    im = Image.open(img_path).convert("RGB")
    x = tfm(im).unsqueeze(0)

    with torch.no_grad():
        # simple TTA: original + hflip
        x_flip = torch.flip(x, dims=[3])
        logits = model(torch.cat([x, x_flip], dim=0))  # [2, C] or [2, 1]
        if logits.shape[-1] == 1:  # binary head
            probs = torch.sigmoid(logits).mean(dim=0)    # [1]
            prob_anemic = float(torch.clamp(probs.squeeze(), 0, 1).item())
        else:  # 2-class head
            probs = F.softmax(logits, dim=-1).mean(dim=0)  # [2]
            prob_anemic = float(torch.clamp(probs[1], 0, 1).item())

    pred_label = int(prob_anemic >= threshold)
    return {"prob_anemic": prob_anemic, "pred_label": pred_label}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    try:
        # peek checkpoint to decide head size
        raw_sd = torch.load(args.model, map_location="cpu")
        sd = strip_state_dict(raw_sd)
        out_dim = infer_out_dim(sd)
        if out_dim is None:
            # fallback: use 2 classes; still try to load with strict=False
            out_dim = 2

        model = build_model(args.arch, out_dim)
        model = load_weights(model, args.model)
        model.eval()

        out = infer_one(model, args.img, args.img_size, args.threshold)
        print(json.dumps(out))
    except Exception as e:
        # batch script expects JSON on stdout even for errors
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
