import os
import sys
from pathlib import Path
import textwrap
import winsound  

CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import cv2
import torch
import pyttsx3
from collections import deque
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)
from qwen_vl_utils import process_vision_info
from plugins.custom_alert_tool import SafetyAlertPlugin
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

DEVICE = "cpu"

# --- 1. Audio Setup ---
TTS_TEXT = "Warning: SOP Violation. Unattended open laptop detected at workstation."
AUDIO_FILE = "videos/alert_audio.wav"

print("Generating alert audio file...")
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.save_to_file(TTS_TEXT, AUDIO_FILE)
engine.runAndWait()
del engine  

def play_live_audio():
    winsound.PlaySound(AUDIO_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)

class LaptopSafetyAlertPlugin(SafetyAlertPlugin):
    def output(self, data, channel=0):
        super().output(data, channel=channel)
        if channel == 0 and isinstance(data, str):
            play_live_audio()

alert_tool = LaptopSafetyAlertPlugin(alert_prefix="SOP Violation: ")

# --- 2. Load Models & Prompts ---
with open("Auto Prompt.txt", "r") as f:
    VLM_PROMPT = f.read().strip()

with open("Chat Node.txt", "r") as f:
    LLM_PROMPT = f.read().strip()

VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading vision model: {VLM_MODEL_ID}...")
vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to(DEVICE)

print(f"Loading agent LLM: {LLM_MODEL_ID}...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(DEVICE)

def analyze_frame_fast(pil_image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": VLM_PROMPT}
            ]
        }
    ]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = vlm_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=40)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

def evaluate_scene_fast(history_str: str) -> str:
    messages = [
        {"role": "system", "content": LLM_PROMPT},
        {"role": "user", "content": f"Recent observations:\n{history_str}\nIs there an unattended open laptop? Answer strictly with YES or NO."}
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
    hud_h = 125
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (frame_width, hud_h), (15, 15, 15), -1)
    border_color = (0, 0, 255) if is_alert else (0, 180, 0)
    cv2.line(overlay, (0, hud_h), (frame_width, hud_h), border_color, 2)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    max_chars = max(35, int(frame_width / 15))

    cv2.putText(frame, "VISION :", (15, 32), font, font_scale, (0, 255, 120), 2)
    v_lines = textwrap.wrap(vision_text, width=max_chars)
    v_display = v_lines[0] if v_lines else ""
    cv2.putText(frame, v_display, (105, 32), font, font_scale, (220, 255, 220), 1)

    cv2.putText(frame, "AGENT  :", (15, 68), font, font_scale, (0, 215, 255), 2)
    a_lines = textwrap.wrap(agent_text, width=max_chars)
    a_display = a_lines[0] if a_lines else ""
    cv2.putText(frame, a_display, (105, 68), font, font_scale, (255, 255, 220), 1)

    status_label = "STATUS : [SOP VIOLATION RECORDED]" if is_alert else "STATUS : [MONITORING WORKSTATION]"
    status_color = (0, 0, 255) if is_alert else (0, 200, 0)
    cv2.putText(frame, status_label, (15, 104), font, 0.50, status_color, 2)

    if is_alert:
        alert_msg = "! WARNING: UNATTENDED OPEN LAPTOP !"
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

frame_count = 0
last_eval_video_sec = -999.0
last_alert_video_sec = -999.0

EVAL_INTERVAL_VIDEO_SEC = 2.0  
ALERT_COOLDOWN_VIDEO_SEC = 15.0  

alert_active = False
current_vision_text = "Initializing stream analysis..."
current_agent_text = "Awaiting scene observation..."
alert_timestamps = []

print(f"\nStarting SOP Monitoring Pipeline. Saving to: {FINAL_VIDEO}")
print("Press 'q' in preview window to exit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_video_sec = frame_count / fps

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

        # 1. Multi-desk state extraction
        has_unattended_desk = (
            "unattended" in desc_lower 
            or "open laptop: unattended" in desc_lower
            or "desk with open laptop: unattended" in desc_lower
        )
        all_attended = (
            "attended" in desc_lower 
            and "unattended" not in desc_lower
        )

        # 2. Agent decision parsing
        is_negative = any(neg in decision_lower for neg in ["no", "not required", "do not", "compliant"])
        llm_says_yes = ("yes" in decision_lower and not is_negative)

        # 3. Decision arbitration
        if has_unattended_desk or (llm_says_yes and not all_attended):
            sop_violation = True
            current_agent_text = "SOP Violation: Unattended open laptop detected!"
        elif all_attended:
            sop_violation = False
            current_agent_text = "All active laptops attended. SOP compliant."
        else:
            sop_violation = False
            current_agent_text = "Monitoring office. No violations detected."

        # 4. Trigger alert with debouncing
        if sop_violation:
            if (current_video_sec - last_alert_video_sec) >= ALERT_COOLDOWN_VIDEO_SEC:
                alert_tool.trigger_alert(
                    event_description="Unattended open laptop detected at workstation",
                    severity="warning"
                )
                alert_active = True
                last_alert_video_sec = current_video_sec
                alert_timestamps.append(current_video_sec)
        else:
            if (current_video_sec - last_alert_video_sec) >= 5.0:
                alert_active = False

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
    video_clip = VideoFileClip(TEMP_VIDEO)
    
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