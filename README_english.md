# easy_ViTPose - Interactive Pose Estimation and Editor

<p align="center">
  <img src="https://user-images.githubusercontent.com/24314647/236082274-b25a70c8-9267-4375-97b0-eddf60a7dfc6.png" width="340"/>
</p>

<p align="center">
  <b>Human Pose Estimation · Keypoint Editing · Skeleton Visualization</b><br/>
  Built on <a href="https://github.com/JunkyByte/easy_ViTPose">easy_ViTPose</a> · Gradio 5 web interface
</p>

<p align="center">
  <br/>
  <i>Developed within the scope of the TUBITAK 1001 Project</i><br/>
  <b>Design and Prototype Implementation of a Pose-Based Movement Detection System<br/>for Artistic Gymnastics</b>
</p>

---

## What Does This Project Do?

This project extends the [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) library with a **full-featured Gradio web interface**. Through the interface, you can:

1. Run **ViTPose inference** on a single image or on all images in a folder (ViTPose-H, COCO-25 model).
2. **Interactively edit** detected keypoints by dragging them on the canvas.
3. Display **joint angles** for shoulders, elbows, hips, and knees on the image.
4. **Save** edited results to a JSON file and pass them to downstream pipelines.
5. Navigate through batch results frame by frame with toolbar-embedded **Prev / Next** controls.
6. Visualize a skeleton using pose coordinates from an existing JSON file without running inference again.

This tool is designed for **biomechanics and sports science research** where automatic pose estimation outputs, especially in domains such as gymnastics and athletics, may need manual correction.

---

## Interface Overview

```text
+-------------------------------------------------------------------------+
|  Input Panel                  |  Interactive Pose Editor (canvas)        |
|  ---------------------------- |  --------------------------------------  |
|  - Upload a single image OR   |  - Drag keypoints to correct them        |
|  - Enter a folder path        |  - Mouse wheel -> Zoom / drag empty      |
|  - [Run ViTPose] button       |    area -> Pan                           |
|  - JSON path output box       |  - Toggle names / angles / points        |
|                               |  - Full-screen mode                      |
|                               |  - Save current view as PNG              |
|                               |  - Prev / Next batch navigation          |
+-------------------------------------------------------------------------+
|  [Apply & Save]  -> writes edited keypoints to JSON                      |
|  Batch gallery, folder mode thumbnail strip                              |
|  Frame / image slider                                                    |
+-------------------------------------------------------------------------+
|  JSON Upload -> Draw  (pose visualization from JSON section)             |
|  - Upload source image + existing .json -> draw skeleton from coordinates|
+-------------------------------------------------------------------------+
```

### Pose Editor Controls

| Control | Function |
|---|---|
| **Drag keypoint** | Moves the selected keypoint; the change is reflected immediately |
| **Mouse wheel** | Zooms in / out around the cursor |
| **Drag empty area** | Pans the canvas |
| **Double-click empty area** | Resets zoom and pan |
| **Reset** | Restores the original keypoint positions |
| **Names** | Shows / hides keypoint name labels |
| **Save PNG** | Downloads the current canvas view as a PNG |
| **Angles** | Shows / hides joint angle arc indicators |
| **Full Screen** | Switches to full-screen mode |
| **Points** | Shows / hides all keypoints and the skeleton |
| **Prev / Next** | Moves to the previous / next batch image |
| **Apply & Save** | Saves edited keypoints to the JSON file |

---

## Project Structure

```text
easy_ViTPose/
|-- app.py                      # Gradio application entry point
|-- pose_editor.py              # Interactive canvas editor (HTML/CSS/JS + Python helpers)
|-- inference.py                # Command-line inference script (original easy_ViTPose)
|-- export.py                   # ONNX / TensorRT export
|-- model_split.py              # Pretrained checkpoint converter for fine-tuning
|-- evaluation_on_coco.py       # COCO evaluation script
|-- setup.py                    # Package setup file
|-- requirements.txt            # Python dependencies (CPU / general)
|-- requirements_gpu.txt        # GPU-specific additional dependencies
|-- requirements.notorch.txt    # Dependencies without PyTorch for custom installs
|-- Dockerfile                  # NVIDIA PyTorch container setup
|-- checkpoints/                # Put model weights here (git-ignored)
|   |-- vitpose-h-coco_25.pth
|   `-- yolo11x.pt
|-- easy_ViTPose/               # Core library package
|-- temp/                       # Runtime temporary files (git-ignored)
`-- outputs/                    # Inference outputs (git-ignored)
```

---

## Prerequisites

### 1 - Python

**Python 3.9-3.11** is recommended. Python 3.12 may cause compatibility issues with some scientific packages.

### 2 - PyTorch

Install **PyTorch >= 2.0** manually before installing the remaining dependencies.  
Choose the version that matches your hardware:

```bash
# CUDA 12.1 (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (MPS)
pip install torch torchvision torchaudio
```

> [!IMPORTANT]
> PyTorch must be installed **before** running `pip install -r requirements.txt`.  
> The requirements files intentionally do not include PyTorch so that the correct CUDA version can be selected.

### 3 - easy_ViTPose Installation (External Dependency)

This project depends on [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) as its inference engine.  
Install the library from source as an **editable package**:

```bash
# Clone the upstream library
git clone https://github.com/JunkyByte/easy_ViTPose.git
cd easy_ViTPose

# Install as an editable package
pip install -e .
```

> [!NOTE]
> If you are working directly inside the cloned `easy_ViTPose` repository, meaning this README is at the repository root,  
> running `pip install -e .` from this directory is sufficient because `easy_ViTPose` is already the package being installed.

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/NecmettinTabak/Tubitak_1001.git
cd Tubitak_1001/easy_ViTPose

# 2. Install PyTorch first (see Prerequisites -> step 2)

# 3. Install the package in editable mode
pip install -e .

# 4. Install the remaining Python dependencies
pip install -r requirements.txt

# 5. Optional GPU-specific extras, such as onnxruntime-gpu
pip install -r requirements_gpu.txt

# 6. Install Gradio if it is not already installed
pip install gradio>=5.0
```

### Core Python Packages

| Package | Purpose |
|---|---|
| `gradio >= 5.0` | Web interface framework |
| `torch >= 2.0` | Deep learning inference |
| `ultralytics` | YOLOv11 human detector |
| `opencv-python` | Image I/O and drawing operations |
| `scipy` | TPS warp (RBF interpolation) |
| `numpy`, `Pillow` | Array and image utilities |
| `onnxruntime` | Optional ONNX inference |

---

## Downloading Model Weights

Place the model checkpoints under the `checkpoints/` directory. Create the directory if it does not exist:

```text
checkpoints/
|-- vitpose-h-coco_25.pth   # ViTPose-H trained with COCO + foot keypoint dataset (25 keypoints)
`-- yolo11x.pt              # YOLOv11x human detector
```

Download from **Hugging Face**:  
[https://huggingface.co/JunkyByte/easy_ViTPose](https://huggingface.co/JunkyByte/easy_ViTPose)

For the YOLO model, Ultralytics downloads it automatically on first run. You can also download it manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
# Then move yolo11x.pt into the checkpoints/ directory
```

---

## Running the Application

```bash
python app.py
```

The Gradio server starts locally. Open this address in your browser:

```text
http://127.0.0.1:7860
```

### Single Image Mode

1. Upload a `.jpg`, `.png`, or `.webp` file through the **Input Image** field.
2. Click **Run ViTPose**.
3. The detected skeleton appears on the interactive canvas.
4. Drag keypoints to correct the pose.
5. Click **Apply & Save** to write the corrected coordinates to JSON.

### Batch / Folder Mode

1. Enter the **full path** of a folder containing images in the *Image Folder Path* field.
2. Click **Run ViTPose**.
3. All images are processed and the results appear in the gallery.
4. Navigate with **Prev / Next** or the **Frame / Image** slider.
5. Edit and save each frame independently.

### Pose Drawing from JSON (Lower Section)

This section uses coordinates from a previously generated pose `.json` file without running the model again on the image. In other words, the system does not perform a new ViTPose inference pass; it reads the saved keypoints in the JSON file, places those points on the source image, and draws the skeleton from that data.

This is especially useful for inspecting pose data that has already been inferred or manually corrected and saved. For example, by loading the JSON file for an image, you can visually check whether the coordinates correspond to the correct person, joints, and image dimensions.

1. Upload the source image for which the pose will be visualized.
2. Upload the pose `.json` file for the same image, or a JSON file prepared in the same coordinate system (easy_ViTPose output format).
3. Click **Draw From Uploaded JSON**.
4. The joint coordinates in the JSON file are read and the skeleton is drawn on the image.

---

## Output JSON Format

Inference output follows the standard easy_ViTPose format:

```json
{
  "keypoints": [
    {
      "0": [
        [121.19, 458.15, 0.99],
        [110.02, 469.43, 0.98],
        "..."
      ]
    }
  ],
  "skeleton": {
    "0": "nose",
    "1": "left_eye",
    "2": "right_eye",
    "3": "left_ear",
    "4": "right_ear",
    "5": "neck",
    "6": "right_shoulder",
    "7": "left_shoulder",
    "8": "right_elbow",
    "9": "left_elbow",
    "10": "right_wrist",
    "11": "left_wrist",
    "12": "right_hip",
    "13": "left_hip",
    "14": "right_knee",
    "15": "left_knee",
    "16": "right_ankle",
    "17": "left_ankle",
    "18": "right_big_toe",
    "19": "right_small_toe",
    "20": "right_heel",
    "21": "left_big_toe",
    "22": "left_small_toe",
    "23": "left_heel",
    "24": "head_top"
  }
}
```

Each keypoint value is stored in image pixel coordinates as `[y, x, confidence_score]`.

---

## Command-Line Inference Without Gradio

You can also run inference from the command line using the original `inference.py` script:

```bash
python inference.py \
  --input ./test.png \
  --model ./checkpoints/vitpose-h-coco_25.pth \
  --yolo  ./checkpoints/yolo11x.pt \
  --dataset coco_25 \
  --model-name h \
  --save-img \
  --save-json \
  --output-path ./outputs
```

Run `python inference.py --help` for all available options.

---

## Docker

Build the container. NVIDIA Container Toolkit is required for GPU support:

```bash
docker build . -t easy_vitpose
```

Run inference inside the container:

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  -v ./checkpoints:/checkpoints \
  -v ./inputs:/inputs \
  -v ./outputs:/outputs \
  easy_vitpose \
  python inference.py \
    --input /inputs/image.jpg \
    --model /checkpoints/vitpose-h-coco_25.pth \
    --yolo  /checkpoints/yolo11x.pt \
    --dataset coco_25 --model-name h \
    --save-img --save-json \
    --output-path /outputs
```

---

## Fine-Tuning

See the upstream [easy_ViTPose fine-tuning guide](https://github.com/JunkyByte/easy_ViTPose#finetuning) for details.  
Helper scripts included in this repository:

- **`model_split.py`** - Converts the official ViTPose checkpoint into a single-head format.
- **`evaluation_on_coco.py`** - Evaluates a model on the COCO val2017 set.
- **`export.py`** - Exports a `.pth` checkpoint to ONNX / TensorRT format.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: easy_ViTPose` | Run `pip install -e .` from the repository root |
| Model not found | Make sure `checkpoints/vitpose-h-coco_25.pth` exists |
| YOLO download failed | Download `yolo11x.pt` manually and place it under `checkpoints/` |
| `GRADIO_DISABLE_BROTLI` warning | It is already set in `app.py`; you can safely ignore it |
| Incorrect bounding box on MPS | Upgrade `ultralytics` to version >= 8.2.48 |
| Canvas does not load | Make sure you are using Gradio >= 5.0; the editor uses `html_template` with `gr.HTML` |

---

## Sources and References

- ViTPose paper: [Y. Xu et al., 2022](https://arxiv.org/abs/2204.12484)
- Upstream library: [JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)
- Human detector: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- SORT tracker: [abewley/sort](https://github.com/abewley/sort)
- COCO + Foot keypoint dataset: [CMU Perceptual Computing Lab](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/)

---

[README](README.md) | [README_ENGLISH](README_english.md)
