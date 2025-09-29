import os
import torch
from .factory import build_model, Arch

def load_model_from_ckpt(arch: Arch, ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    model = build_model(arch)
    state = torch.load(ckpt_path, map_location="cpu")
    # support both plain state_dict and wrapped dict formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # keys might be 'module.'-prefixed if saved with DataParallel
    new_state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if unexpected:
        print(f"[loader] unexpected keys: {unexpected}")
    if missing:
        print(f"[loader] missing keys: {missing}")
    model.eval()
    return model
