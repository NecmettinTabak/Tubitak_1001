import os
os.environ["GRADIO_DISABLE_BROTLI"] = "1"

import sys
import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import gradio as gr
from PIL import Image, ImageDraw

# --- paths ---
BASE_DIR = Path(__file__).parent
TEMP_INPUT = BASE_DIR / "temp" / "inputs"
TEMP_OUTPUT = BASE_DIR / "temp" / "outputs"
TEMP_INPUT.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)

POSE_MODEL = BASE_DIR / "checkpoints" / "vitpose-h-coco_25.pth"
YOLO_MODEL = BASE_DIR / "checkpoints" / "yolo11x.pt"


# --- COCO25 default edges (idx-based) ---
# idx->name mapping senin JSON'dan geliyor ama edges'i idx ile tutuyoruz
COCO25_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5),          # head->neck
    (5, 6), (6, 8), (8, 10),                         # left arm
    (5, 7), (7, 9), (9, 11),                         # right arm
    (5, 14), (12, 14), (13, 14),                     # torso/hips
    (14, 15), (15, 17),                              # left leg
    (14, 16), (16, 18),                              # right leg
    (17, 21), (17, 19), (17, 20),                    # left foot
    (18, 24), (18, 22), (18, 23),                    # right foot
]


def _find_latest_run_dir() -> Optional[Path]:
    """inference.py çıktısı TEMP_OUTPUT altında bazen <uuid>.png/ gibi bir klasöre gidiyor.
    En güncel klasörü seçiyoruz."""
    dirs = [p for p in TEMP_OUTPUT.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def run_vitpose(image: Image.Image):
    if image is None:
        return None, "No image uploaded."

    filename = f"{uuid.uuid4()}.png"
    input_path = TEMP_INPUT / filename
    image.save(input_path)

    cmd = [
        sys.executable, "inference.py",
        "--input", str(input_path),
        "--model", str(POSE_MODEL),
        "--yolo", str(YOLO_MODEL),
        "--dataset", "coco_25",
        "--model-name", "h",
        "--det-class", "human",
        "--output-path", str(TEMP_OUTPUT),
        "--save-img",
        "--save-json",
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "Unknown error").strip()
        return None, f"Inference error:\n{err}"

    run_dir = _find_latest_run_dir()
    if run_dir is None:
        return None, f"Inference OK ama output klasörü bulunamadı: {TEMP_OUTPUT}"

    # beklenen dosyalar genelde: <uuid>_result.png / <uuid>_result.json
    result_image = run_dir / f"{input_path.stem}_result.png"
    result_json = run_dir / f"{input_path.stem}_result.json"

    if not result_image.exists() or not result_json.exists():
        # fallback: klasördeki ilk png/json'u seç
        pngs = sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        jsons = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pngs or not jsons:
            return None, (
                "Inference finished but output files not found.\n"
                f"Run dir: {run_dir}\n"
                f"Expected:\n- {result_image}\n- {result_json}"
            )
        result_image, result_json = pngs[0], jsons[0]

    out_img = Image.open(result_image).convert("RGB")
    return out_img, str(result_json)


def parse_pose_json(data: Dict[str, Any]) -> Tuple[Dict[int, str], List[Tuple[float, float, float]]]:
    """
    Senin format:
      data["skeleton"] : {"0":"nose", ...}
      data["keypoints"] : [{"0": [[x,y,c], ... 25]}]
    Çıktı:
      idx_to_name, kps (len=25) as (x,y,c)
    """
    idx_to_name = {int(k): v for k, v in data.get("skeleton", {}).items()}

    # keypoints -> first element -> person "0"
    kp_outer = data.get("keypoints", [])
    if not kp_outer or not isinstance(kp_outer, list):
        raise ValueError("JSON'da 'keypoints' bulunamadı veya format hatalı.")

    person_dict = kp_outer[0]
    if "0" not in person_dict:
        # fallback: ilk key'i al
        first_key = next(iter(person_dict.keys()))
        kp_list = person_dict[first_key]
    else:
        kp_list = person_dict["0"]

    if len(kp_list) != 25:
        raise ValueError(f"Beklenen 25 keypoint, gelen: {len(kp_list)}")

    kps = [(float(x), float(y), float(c)) for x, y, c in kp_list]
    return idx_to_name, kps


def draw_from_json(image: Image.Image, json_file) -> Tuple[Image.Image, str]:
    if image is None:
        return None, "Önce bir görüntü yükle."

    if json_file is None:
        return None, "Bir JSON dosyası yükle."

    # gr.File -> obj has .name
    json_path = Path(json_file.name)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    try:
        idx_to_name, kps = parse_pose_json(data)
    except Exception as e:
        return None, f"JSON parse error: {e}"

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    thr = 0.30  # confidence threshold

    # edges
    for a, b in COCO25_EDGES:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca >= thr and cb >= thr:
            draw.line((xa, ya, xb, yb), width=3)

    # points
    r = 4
    for i, (x, y, c) in enumerate(kps):
        if c < thr:
            continue
        draw.ellipse((x - r, y - r, x + r, y + r), width=2)

    return img, f"Loaded JSON: {json_path.name} | thr={thr}"


with gr.Blocks() as demo:
    gr.Markdown("## easy_ViTPose - Inference + JSON Overlay")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Input Image")
        pose_img = gr.Image(type="pil", label="Pose Result (from inference)")

    with gr.Row():
        run_button = gr.Button("Run ViTPose")
        json_path_box = gr.Textbox(label="Inference JSON Path", lines=2)

    run_button.click(fn=run_vitpose, inputs=input_img, outputs=[pose_img, json_path_box])

    gr.Markdown("### JSON Upload → Draw keypoints + skeleton on the image")

    with gr.Row():
        json_file = gr.File(label="Upload Pose JSON (.json)", file_types=[".json"])
        overlay_img = gr.Image(type="pil", label="Overlay (from uploaded JSON)")
    overlay_status = gr.Textbox(label="Status", lines=2)

    draw_button = gr.Button("Draw From Uploaded JSON")
    draw_button.click(fn=draw_from_json, inputs=[pose_img, json_file], outputs=[overlay_img, overlay_status])

if __name__ == "__main__":
    demo.launch()