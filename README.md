# Laptop Closing SOP

A Windows-based video monitoring pipeline that detects unattended laptops and verifies whether each laptop lid is open before raising an SOP alert.

## How It Works

The hybrid pipeline uses two stages:

1. **YOLOv8 tracking** detects and tracks people and laptops in each video frame.
2. **Qwen2-VL** checks a padded crop of an unattended laptop and returns `OPEN` or `CLOSED`.

When an unattended laptop is confirmed open, the pipeline logs an alert, plays a spoken warning, draws a violation marker on the video, and adds the alert audio to the final MP4.

## Repository Contents

- `run_hybrid_pipeline.py` - main pipeline entry point.
- `VLM_Prompt.txt` - prompt used for open/closed laptop verification.
- `plugins/custom_alert_tool.py` - alert logging and audio dispatch plugin.
- `.gitignore` - excludes local environments, model weights, test media, and generated outputs.

The input video and model weights are intentionally kept local and are not committed to Git:

- `videos/test_video.mp4`
- `yolov8s.pt`

Ultralytics downloads `yolov8s.pt` automatically when it is not present. The Qwen model is downloaded through Hugging Face on first use.

## Requirements

- Windows
- Python 3.12 or compatible Python version
- Working audio output
- Internet access for the first model download
- Python packages:
  - `opencv-python`
  - `torch`
  - `torchvision`
  - `ultralytics`
  - `transformers`
  - `qwen-vl-utils`
  - `Pillow`
  - `pyttsx3`
  - `moviepy`

`winsound` is included with Python on Windows. The local workspace currently contains a Python environment under `python312_env/`, but that environment is ignored and is not part of the repository.

## Setup

From the project root:

```powershell
python -m pip install opencv-python torch torchvision ultralytics transformers qwen-vl-utils Pillow pyttsx3 moviepy
```

Place the source video at:

```text
videos/test_video.mp4
```

The script expects to be run from the project root because it uses relative paths.

## Run

```powershell
python run_hybrid_pipeline.py
```

The script opens a live preview window. Press `q` to stop processing.

## Outputs

The final annotated video is written to:

```text
videos/output_hybrid.mp4
```

Temporary audio and video files are created in `videos/` during processing and are removed after successful completion. Generated media, downloaded weights, Python environments, and cache files are ignored by Git.

## Notes

- The pipeline runs on CPU by default (`DEVICE = "cpu"`).
- The person-to-laptop interaction radius is based on the detected laptop width.
- VLM verification is only triggered after a tracked laptop has remained unattended for consecutive frames.
- Alert events are rate-limited by a cooldown to avoid repeated announcements.
