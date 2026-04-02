import os
os.environ["GRADIO_DISABLE_BROTLI"] = "1"

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder
from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton, joints_dict

# --- paths ---
BASE_DIR = Path(__file__).parent
TEMP_INPUT = BASE_DIR / "temp" / "inputs"
TEMP_OUTPUT = BASE_DIR / "temp" / "outputs"
TEMP_INPUT.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)

POSE_MODEL = BASE_DIR / "checkpoints" / "vitpose-h-coco_25.pth"
YOLO_MODEL = BASE_DIR / "checkpoints" / "yolo11x.pt"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MODEL_CACHE: Dict[str, Any] = {}


def _get_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model() -> VitInference:
    device = _get_runtime_device()
    cache_key = f"{POSE_MODEL}|{YOLO_MODEL}|{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = VitInference(
            str(POSE_MODEL),
            str(YOLO_MODEL),
            model_name="h",
            det_class="human",
            dataset="coco_25",
            yolo_size=320,
            device=device,
            is_video=False,
            single_pose=True,
        )
    return _MODEL_CACHE[cache_key]


def _save_outputs(model: VitInference, input_path: Path, frame_keypoints: Dict[Any, Any]) -> Tuple[Path, Path]:
    run_dir = TEMP_OUTPUT / f"{input_path.stem}_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_image = run_dir / f"{input_path.stem}_result.png"
    result_json = run_dir / f"{input_path.stem}_result.json"

    result_rgb = model.draw(show_yolo=False, show_raw_yolo=False, confidence_threshold=0.5)
    Image.fromarray(result_rgb).save(result_image)

    out_json = {
        "keypoints": [frame_keypoints],
        "skeleton": joints_dict()[model.dataset]["keypoints"],
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, cls=NumpyEncoder)

    return result_image, result_json


def _run_inference_for_input(input_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not input_path.exists():
        return None, None, f"Input not found: {input_path}"

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        return None, None, f"Image read failed: {input_path}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        model = _get_model()
        frame_keypoints = model.inference(img_rgb)
        result_image, result_json = _save_outputs(model, input_path, frame_keypoints)
    except Exception as e:
        return None, None, f"Inference error for {input_path.name}: {e}"

    return str(result_image), str(result_json), None


def _collect_images_from_directory(folder_path: Path) -> List[Path]:
    return sorted(
        [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def _batch_nav_state_text(index: int, total: int, source_name: str) -> str:
    return f"{index + 1}/{total} - {source_name}"


def _show_batch_item(batch_items: List[Dict[str, str]], index: int):
    if not batch_items:
        return None, "", 0, gr.update(value=1, minimum=1, maximum=1, visible=False), "Batch sonucu yok."

    idx = max(0, min(index, len(batch_items) - 1))
    item = batch_items[idx]
    nav_text = _batch_nav_state_text(idx, len(batch_items), item.get("source_name", ""))
    return (
        item["result_image"],
        item["json_path"],
        idx,
        gr.update(value=idx + 1, minimum=1, maximum=len(batch_items), visible=True),
        nav_text,
    )


def show_prev_batch(batch_items: List[Dict[str, str]], current_idx: int):
    return _show_batch_item(batch_items, current_idx - 1)


def show_next_batch(batch_items: List[Dict[str, str]], current_idx: int):
    return _show_batch_item(batch_items, current_idx + 1)


def show_batch_by_slider(batch_items: List[Dict[str, str]], slider_idx: float):
    target_idx = int(slider_idx) - 1
    return _show_batch_item(batch_items, target_idx)


def run_vitpose(image: Image.Image, folder_path: str):
    folder_path = (folder_path or "").strip()
    runtime_device = _get_runtime_device()
    t0 = time.perf_counter()

    if image is not None:
        filename = f"{uuid.uuid4()}.png"
        input_path = TEMP_INPUT / filename
        image.save(input_path)

        out_img, json_path, err = _run_inference_for_input(input_path)
        if err:
            return None, [], err, [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(visible=False), gr.update(visible=False), ""

        elapsed = time.perf_counter() - t0
        return (
            out_img,
            [(out_img, input_path.name)],
            f"{json_path}\nDevice: {runtime_device} | Time: {elapsed:.2f}s",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=1, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "Tek görüntü modu.",
        )

    if not folder_path:
        return None, [], "Tek bir görüntü yükleyin veya görüntü klasörü yolu girin.", [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(visible=False), gr.update(visible=False), ""

    dir_path = Path(folder_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return None, [], f"Klasör bulunamadı veya geçersiz: {folder_path}", [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(visible=False), gr.update(visible=False), ""

    image_files = _collect_images_from_directory(dir_path)
    if not image_files:
        return None, [], f"Klasörde desteklenen görüntü bulunamadı: {folder_path}", [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(visible=False), gr.update(visible=False), ""

    gallery_items = []
    batch_items: List[Dict[str, str]] = []
    json_paths = []
    first_result = None
    errors = []

    for img_path in image_files:
        out_img, json_path, err = _run_inference_for_input(img_path)
        if err:
            errors.append(f"{img_path.name}: {err}")
            continue

        if out_img is not None:
            if first_result is None:
                first_result = out_img
            gallery_items.append((out_img, img_path.name))
            if json_path:
                batch_items.append({
                    "result_image": out_img,
                    "json_path": json_path,
                    "source_name": img_path.name,
                })

        if json_path:
            json_paths.append(json_path)

    if not gallery_items:
        error_text = "\n".join(errors) if errors else "Klasördeki görüntüler işlenemedi."
        return None, [], error_text, [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(visible=False), gr.update(visible=False), ""

    status_lines = [
        f"Device: {runtime_device}",
        f"Toplam görüntü: {len(image_files)}",
        f"Başarılı: {len(gallery_items)}",
        f"Hatalı: {len(errors)}",
    ]
    elapsed = time.perf_counter() - t0
    status_lines.append(f"Toplam süre: {elapsed:.2f}s")
    if len(gallery_items) > 0:
        status_lines.append(f"Ortalama süre/görüntü: {elapsed / len(gallery_items):.2f}s")
    if json_paths:
        status_lines.append("\nJSON çıktıları:")
        status_lines.extend(json_paths)
    if errors:
        status_lines.append("\nHatalar:")
        status_lines.extend(errors)

    nav_status = _batch_nav_state_text(0, len(batch_items), batch_items[0]["source_name"]) if batch_items else ""
    return (
        first_result,
        gallery_items,
        "\n".join(status_lines),
        batch_items,
        0,
        gr.update(value=1, minimum=1, maximum=len(batch_items), visible=bool(batch_items)),
        gr.update(visible=bool(batch_items)),
        gr.update(visible=bool(batch_items)),
        nav_status,
    )


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
        return None, "Önce bir görüntü çalıştır veya yükle."

    if json_file is None:
        return None, "Bir JSON dosyası yükle."

    # gr.File -> obj has .name
    json_path = Path(json_file.name)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    try:
        idx_to_name, kps = parse_pose_json(data)
    except Exception as e:
        return None, f"JSON parse error: {e}"

    thr = 0.30  # confidence threshold
    skeleton = joints_dict()["coco_25"]["skeleton"]

    # draw_points_and_skeleton (y, x, c) bekliyor
    kps_arr = np.array([[float(y), float(x), float(c)] for x, y, c in kps], dtype=np.float32)

    img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_np = draw_points_and_skeleton(
        img_np,
        kps_arr,
        skeleton,
        person_index=0,
        points_color_palette="gist_rainbow",
        skeleton_color_palette="jet",
        points_palette_samples=10,
        confidence_threshold=thr,
    )
    out_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    return out_img, f"Loaded JSON: {json_path.name} | thr={thr}"


with gr.Blocks() as demo:
    gr.Markdown("## easy_ViTPose - Inference + JSON Overlay")

    # Section 1: inference (tek görsel + klasör)
    with gr.Row():
        input_img = gr.Image(type="pil", label="Input Image")
        input_folder = gr.Textbox(label="Image Folder Path (optional)", placeholder="C:/path/to/images")
        pose_img = gr.Image(type="pil", label="Pose Result (from inference)")

    batch_gallery = gr.Gallery(label="Batch Results (Folder Input)", columns=4, height=260)

    with gr.Row():
        prev_btn = gr.Button("◀ Previous", visible=False)
        next_btn = gr.Button("Next ▶", visible=False)
    batch_slider = gr.Slider(label="Frame / Image Index", minimum=1, maximum=1, value=1, step=1, visible=False)
    batch_nav_status = gr.Textbox(label="Batch Navigation", lines=1, interactive=False)

    batch_items_state = gr.State([])
    batch_index_state = gr.State(0)

    with gr.Row():
        run_button = gr.Button("Run ViTPose")
        json_path_box = gr.Textbox(label="Inference JSON Path", lines=2)

    run_button.click(
        fn=run_vitpose,
        inputs=[input_img, input_folder],
        outputs=[
            pose_img,
            batch_gallery,
            json_path_box,
            batch_items_state,
            batch_index_state,
            batch_slider,
            prev_btn,
            next_btn,
            batch_nav_status,
        ],
    )

    prev_btn.click(
        fn=show_prev_batch,
        inputs=[batch_items_state, batch_index_state],
        outputs=[pose_img, json_path_box, batch_index_state, batch_slider, batch_nav_status],
    )

    next_btn.click(
        fn=show_next_batch,
        inputs=[batch_items_state, batch_index_state],
        outputs=[pose_img, json_path_box, batch_index_state, batch_slider, batch_nav_status],
    )

    batch_slider.change(
        fn=show_batch_by_slider,
        inputs=[batch_items_state, batch_slider],
        outputs=[pose_img, json_path_box, batch_index_state, batch_slider, batch_nav_status],
    )

    gr.Markdown("### JSON Upload → Draw keypoints + skeleton on the image")

    with gr.Row():
        json_file = gr.File(label="Upload Pose JSON (.json)", file_types=[".json"])
        overlay_img = gr.Image(type="pil", label="Overlay (from uploaded JSON)")
    overlay_status = gr.Textbox(label="Status", lines=2)

    draw_button = gr.Button("Draw From Uploaded JSON")
    draw_button.click(fn=draw_from_json, inputs=[pose_img, json_file], outputs=[overlay_img, overlay_status])

if __name__ == "__main__":
    demo.launch()