# /workspaces/ishi/api/pseudomask.py
"""
Wrapper that delegates mask generation to the proven CLI script:
    python3 /workspaces/ishi/scripts/pseudomask_conj.py <input_image> <out_dir>

It supports two API styles:
- If out_dir is None  -> returns PNG bytes of the mask (stream from FastAPI)
- If out_dir is set   -> saves files and returns JSON paths

Robust features:
- EXIF orientation normalized before writing temp input
- Unique working subdir per request to avoid file collisions
- Flexible discovery of output files if the script doesn't use a fixed prefix
- Helpful error messages with captured stderr/stdout from the child process
"""

import io
import os
import uuid
import glob
import time
import shutil
import tempfile
import subprocess
from typing import Optional, Dict, Any, Union, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


# -------------------- Configuration --------------------
# Path to your accurate script
PSEUDOMASK_SCRIPT = os.environ.get(
    "PSEUDOMASK_SCRIPT",
    "/workspaces/ishi/scripts/pseudomask_conj.py",
)

# Fallback top-level output dir (when API caller asks to save)
DEFAULT_API_OUT = os.environ.get(
    "API_PSEUDOMASK_OUT",
    "/workspaces/ishi/out"
)


# -------------------- Utilities --------------------
def _ensure_dir(p: str) -> None:
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def _load_upright_bgr_from_bytes(b: bytes) -> np.ndarray:
    """Decode, fix EXIF, return BGR image."""
    img = Image.open(io.BytesIO(b))
    img = ImageOps.exif_transpose(img)
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _write_temp_input(bgr: np.ndarray, out_dir: str, stem: str) -> str:
    """Write a JPEG as the script's input (avoid EXIF by using OpenCV)."""
    _ensure_dir(out_dir)
    p = os.path.join(out_dir, f"{stem}.jpg")
    ok = cv2.imwrite(p, bgr)
    if not ok:
        raise RuntimeError("Failed to write temporary input image")
    return p


def _run_script(input_path: str, out_dir: str, timeout_s: int = 120) -> Tuple[str, str]:
    """
    Run the CLI script. Returns (stdout, stderr). Raises CalledProcessError on failure.
    """
    cmd = ["python3", PSEUDOMASK_SCRIPT, input_path, out_dir]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,  # we check returncode to raise informative error
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pseudomask_conj.py failed (code {proc.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _find_output_files(out_dir: str) -> Dict[str, Optional[str]]:
    """
    Try to discover mask/overlay/original produced by the script.
    Looks for common names, then falls back to best-guess by pattern and mtime.
    """
    def newest(patterns):
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(out_dir, pat)))
        if not paths:
            return None
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return paths[0]

    # Common explicit names first
    candidates = {
        "mask": [
            "*_mask.png", "*mask.png", "mask.png",
            "*_mask.jpg", "*mask.jpg",
        ],
        "overlay": [
            "*_overlay.jpg", "*overlay.jpg", "overlay.jpg",
            "*_overlay.png", "*overlay.png",
        ],
        "upright": [
            "*_upright.jpg", "*upright.jpg", "upright.jpg",
            "*_upright.png", "*upright.png",
        ],
    }

    out = {k: newest(v) for k, v in candidates.items()}

    # Fallback: if mask missing, prefer a binary-looking PNG
    if out["mask"] is None:
        pngs = glob.glob(os.path.join(out_dir, "*.png"))
        pngs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        out["mask"] = pngs[0] if pngs else None

    return out


def _read_png_bytes(path: str) -> bytes:
    """Read any image and encode as PNG bytes."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read produced image: {path}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG from: {path}")
    return buf.tobytes()


# -------------------- Public API --------------------
def run_pseudomask_on_bytes(
    image_bytes: bytes,
    out_dir: Optional[str] = None,
    prefix: str = "pmask",
    return_overlay: bool = False,
) -> Union[bytes, Dict[str, Any]]:
    """
    Delegates mask creation to /scripts/pseudomask_conj.py.

    - If out_dir is None:
        * runs in a unique temp workdir
        * returns PNG bytes (mask by default; overlay if return_overlay=True)
        * cleans temp files
    - If out_dir is set:
        * runs in out_dir/<uuid> to avoid collisions
        * returns JSON with paths (and leaves files on disk)
    """
    if not os.path.isfile(PSEUDOMASK_SCRIPT):
        raise FileNotFoundError(f"PSEUDOMASK_SCRIPT not found: {PSEUDOMASK_SCRIPT}")

    # Decode + normalize orientation; write a temp input for the script
    bgr = _load_upright_bgr_from_bytes(image_bytes)

    # Choose workdir
    user_wants_json = out_dir is not None
    if user_wants_json:
        base_out = out_dir or DEFAULT_API_OUT
        _ensure_dir(base_out)
        workdir = os.path.join(base_out, f"{prefix}_{uuid.uuid4().hex[:8]}")
        _ensure_dir(workdir)
        cleanup = False
    else:
        # use a system temp dir for ephemeral run
        workdir = tempfile.mkdtemp(prefix="pmask_")
        cleanup = True

    try:
        input_path = _write_temp_input(bgr, workdir, f"{prefix}_input")

        # Call the accurate script
        _run_script(input_path, workdir)

        # Discover outputs
        files = _find_output_files(workdir)
        mask_path = files.get("mask")
        overlay_path = files.get("overlay")
        upright_path = files.get("upright")

        if mask_path is None and not return_overlay:
            # If the script only produced overlay, allow returning that
            if overlay_path is None:
                raise RuntimeError(
                    "pseudomask_conj.py did not produce a detectable mask or overlay in: "
                    + workdir
                )

        if user_wants_json:
            h, w = bgr.shape[:2]
            return {
                "ok": True,
                "width": int(w),
                "height": int(h),
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "upright_image_path": upright_path,
                "workdir": workdir,  # returned for traceability
            }

        # else: return PNG bytes
        target = overlay_path if return_overlay and overlay_path else mask_path or overlay_path
        if not target:
            raise RuntimeError("No output image found to stream back.")
        return _read_png_bytes(target)

    finally:
        if cleanup:
            # best-effort cleanup of the ephemeral workdir
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass
