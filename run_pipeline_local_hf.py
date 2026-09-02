import os
import sys
from pathlib import Path
import textwrap
import winsound  # Native Windows audio - zero threads, zero COM issues

CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import cv2
import torch
import pyttsx3
from collections import deque
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)
from plugins.custom_alert_tool import SafetyAlertPlugin
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

DEVICE = "cpu"

# --- 1. Audio Setup ---
TTS_TEXT = "Warning: School bus approaching the street. Alert level is critical."
AUDIO_FILE = "videos/alert_audio.wav"

print("Generating alert audio file...")
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.save_to_file(TTS_TEXT, AUDIO_FILE)
engine.runAndWait()
del engine  # Free COM resources

def play_live_audio():
    """Plays the alert audio asynchronously on Windows speakers."""
    winsound.PlaySound(AUDIO_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)

class LaptopSafetyAlertPlugin(SafetyAlertPlugin):
    def output(self, data, channel=0):
        super().output(data, channel=channel)
        if channel == 0 and isinstance(data, str):
            play_live_audio()

alert_tool = LaptopSafetyAlertPlugin(alert_prefix="Warning: ")

# --- 2. Load Models ---
with open("Chat Node.txt", "r") as f:
    LLM_PROMPT = f.read().strip()

VLM_MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading vision model: {VLM_MODEL_ID}...")
vlm_model = VisionEncoderDecoderModel.from_pretrained(VLM_MODEL_ID).to(DEVICE)
vlm_processor = ViTImageProcessor.from_pretrained(VLM_MODEL_ID)
vlm_tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_ID)

print(f"Loading agent LLM: {LLM_MODEL_ID}...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(DEVICE)

def analyze_frame_fast(pil_image: Image.Image) -> str:
    pixel_values = vlm_processor(images=[pil_image], return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        output_ids = vlm_model.generate(pixel_values, max_new_tokens=20, num_beams=1)
    return vlm_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def evaluate_scene_fast(history_str: str) -> str:
    messages = [
        {"role": "system", "content": LLM_PROMPT},
        {"role": "user", "content": f"Recent observations:\n{history_str}\nIs a yellow school bus present? Answer strictly with YES or NO."}
    ]
    inputs = llm_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        out = llm_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return llm_tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

# --- 3. UI Dashboard Rendering ---
def draw_dashboard(frame, vision_text, agent_text, is_alert, frame_width, frame_height):
    """Draws a semi-transparent HUD bar with non-overlapping, partitioned text lines."""
    hud_h = 125
    overlay = frame.copy()
    
    # Draw dark backing rectangle
    cv2.rectangle(overlay, (0, 0), (frame_width, hud_h), (15, 15, 15), -1)
    # Accent border: Green normally, Red on Alert
    border_color = (0, 0, 255) if is_alert else (0, 180, 0)
    cv2.line(overlay, (0, hud_h), (frame_width, hud_h), border_color, 2)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    max_chars = max(35, int(frame_width / 15))

    # Row 1: Vision (Green)
    cv2.putText(frame, "VISION :", (15, 32), font, font_scale, (0, 255, 120), 2)
    v_lines = textwrap.wrap(vision_text, width=max_chars)
    v_display = v_lines[0] if v_lines else ""
    cv2.putText(frame, v_display, (105, 32), font, font_scale, (220, 255, 220), 1)

    # Row 2: Agent (Cyan/Yellow) - strictly fixed vertical offset
    cv2.putText(frame, "AGENT  :", (15, 68), font, font_scale, (0, 215, 255), 2)
    a_lines = textwrap.wrap(agent_text, width=max_chars)
    a_display = a_lines[0] if a_lines else ""
    cv2.putText(frame, a_display, (105, 68), font, font_scale, (255, 255, 220), 1)

    # Row 3: Live Pipeline Status
    status_label = "STATUS : [CRITICAL ALERT DISPATCHED]" if is_alert else "STATUS : [MONITORING STREAM]"
    status_color = (0, 0, 255) if is_alert else (0, 200, 0)
    cv2.putText(frame, status_label, (15, 104), font, 0.50, status_color, 2)

    # Center-Screen Alert Splash
    if is_alert:
        alert_msg = "! CRITICAL ALERT: SCHOOL BUS DETECTED !"
        (tw, th), _ = cv2.getTextSize(alert_msg, font, 0.85, 2)
        cx = (frame_width - tw) // 2
        cy = frame_height // 2
        
        cv2.rectangle(frame, (cx - 15, cy - 30), (cx + tw + 15, cy + 15), (0, 0, 0), -1)
        cv2.rectangle(frame, (cx - 15, cy - 30), (cx + tw + 15, cy + 15), (0, 0, 255), 2)
        cv2.putText(frame, alert_msg, (cx, cy), font, 0.85, (0, 0, 255), 2)

# --- 4. Video Loop Setup ---
cap = cv2.VideoCapture("videos/test_video.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

TEMP_VIDEO = "videos/temp_silent.mp4"
FINAL_VIDEO = "videos/output_with_audio.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (frame_width, frame_height))

history = deque(maxlen=3)

# All intervals and cooldowns are measured strictly in VIDEO SECONDS
frame_count = 0
last_eval_video_sec = -999.0
last_alert_video_sec = -999.0

EVAL_INTERVAL_VIDEO_SEC = 1.5   # Evaluate video every 1.5s of footage
ALERT_COOLDOWN_VIDEO_SEC = 15.0  # Prevent repeating alerts for 15s of footage

alert_active = False
current_vision_text = "Initializing stream analysis..."
current_agent_text = "Awaiting scene observation..."
alert_timestamps = []

print(f"\nStarting Dashboard Pipeline. Saving to: {FINAL_VIDEO}")
print("Press 'q' in preview window to exit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_video_sec = frame_count / fps

    # Check if 1.5 seconds of video time has elapsed
    if (current_video_sec - last_eval_video_sec) >= EVAL_INTERVAL_VIDEO_SEC:
        last_eval_video_sec = current_video_sec

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        desc = analyze_frame_fast(pil_img)
        history.append(desc)
        current_vision_text = desc.capitalize()

        history_context = "\n".join([f"- {d}" for d in history])
        decision = evaluate_scene_fast(history_context)

        desc_lower = desc.lower()
        decision_lower = decision.lower()
        is_negative = any(neg in decision_lower for neg in ["no", "not required", "do not"])
        has_yellow_bus = ("school bus" in desc_lower) or ("yellow bus" in desc_lower)

        if has_yellow_bus or ("yes" in decision_lower and not is_negative):
            current_agent_text = "Target verified. Triggering safety alert protocol."
            
            # Debounce using VIDEO TIME (eliminates duplicate audio stacking)
            if (current_video_sec - last_alert_video_sec) >= ALERT_COOLDOWN_VIDEO_SEC:
                alert_tool.trigger_alert(
                    event_description="School bus approaching the street",
                    severity="critical"
                )
                alert_active = True
                last_alert_video_sec = current_video_sec
                alert_timestamps.append(current_video_sec)
        else:
            current_agent_text = "Monitoring traffic. No safety threat identified."
            if (current_video_sec - last_alert_video_sec) >= 5.0:
                alert_active = False

    # Draw non-overlapping dashboard
    draw_dashboard(frame, current_vision_text, current_agent_text, alert_active, frame_width, frame_height)

    video_writer.write(frame)
    cv2.imshow("Edge AI Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# --- 5. Stitching TTS Audio into Final MP4 ---
print("\nStream completed. Processing final video audio...")

if alert_timestamps:
    print(f"Stitching {len(alert_timestamps)} alert audio track at video time: {alert_timestamps[0]:.2f}s")
    video_clip = VideoFileClip(TEMP_VIDEO)
    
    # Build clean single-track placement
    audio_clips = [AudioFileClip(AUDIO_FILE).set_start(ts) for ts in alert_timestamps]
    final_audio = CompositeAudioClip(audio_clips)
    final_video = video_clip.set_audio(final_audio)

    final_video.write_videofile(
        FINAL_VIDEO,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="videos/temp_audio.m4a",
        remove_temp=True,
        logger=None
    )

    video_clip.close()
    for ac in audio_clips:
        ac.close()

    if os.path.exists(TEMP_VIDEO):
        os.remove(TEMP_VIDEO)
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    print(f"\nCompleted successfully! Output with synchronized audio saved to:\n  {FINAL_VIDEO}")
else:
    print("No alert triggered. Renaming silent video to output.")
    if os.path.exists(FINAL_VIDEO):
        os.remove(FINAL_VIDEO)
    os.rename(TEMP_VIDEO, FINAL_VIDEO)