import os
import sys
from pathlib import Path

# Add project root directory to Python path
CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import math
import cv2
import torch
import pyttsx3
import winsound
from PIL import Image

# Ultralytics for Tier 1 Spatial Tracking
from ultralytics import YOLO

# Transformers for Tier 2 VLM Verification
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from plugins.custom_alert_tool import SafetyAlertPlugin
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

DEVICE = "cpu"

# --- 1. Audio & Plugin Setup ---
TTS_TEXT = "SOP Violation. Unattended open laptop detected."
AUDIO_FILE = "videos/alert_audio.wav"

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

# --- 2. Load Models ---
print("Loading Tier 1: YOLOv8 Spatial Tracker...")
yolo_model = YOLO("yolov8s.pt")  # Auto-downloads lightweight YOLOv8

print(f"Loading Tier 2: Qwen2-VL-2B Arbiter...")
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to(DEVICE)

with open("VLM_Prompt.txt", "r") as f:
    VLM_PROMPT = f.read().strip()

def verify_laptop_state(crop_pil: Image.Image) -> str:
    """Passes the cropped laptop to the VLM to check if lid is OPEN or CLOSED."""
    messages = [
        {"role": "user", "content": [{"type": "image", "image": crop_pil}, {"type": "text", "text": VLM_PROMPT}]}
    ]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = vlm_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=10)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip().upper()

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# --- 3. Video & Tracking Setup ---
cap = cv2.VideoCapture("videos/test_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

TEMP_VIDEO = "videos/temp_silent.mp4"
FINAL_VIDEO = "videos/output_hybrid.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (frame_width, frame_height))

# State Machine Memory
# maps track_id -> {"unattended_frames": int, "last_alert_sec": float}
laptop_states = {}
UNATTENDED_THRESHOLD_FRAMES = fps * 1  # 4 seconds
ALERT_COOLDOWN_SEC = 15.0

alert_timestamps = []
frame_count = 0

print("\nStarting Hybrid Pipeline. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_sec = frame_count / fps

    # 1. Tier 1: Spatial Tracking (Classes: 0=person, 63=laptop)
    results = yolo_model.track(frame, classes=[0, 63], persist=True, verbose=False)
    
    persons = []
    laptops = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, cls, track_id in zip(boxes, clss, ids):
            if int(cls) == 0:
                persons.append(box)
            elif int(cls) == 63:
                laptops.append((int(track_id), box))

    # 2. Dynamic Interaction Logic
    for track_id, l_box in laptops:
        if track_id not in laptop_states:
            laptop_states[track_id] = {"unattended_frames": 0, "last_alert_sec": -999.0}
            
        l_center = get_center(l_box)
        l_width = l_box[2] - l_box[0]
        
        # Dynamic interaction radius (e.g., 3x the laptop's width)
        interaction_radius = l_width * 3.0 
        
        is_attended = False
        for p_box in persons:
            p_center = get_center(p_box)
            distance = math.dist(l_center, p_center)
            if distance < interaction_radius:
                is_attended = True
                break
        
        # 3. State Machine Transitions
        if not is_attended:
            laptop_states[track_id]["unattended_frames"] += 1
            box_color = (0, 165, 255) # Orange (Warning)
        else:
            laptop_states[track_id]["unattended_frames"] = 0
            box_color = (0, 255, 0) # Green (Safe)

        # Draw Laptop Box
        cv2.rectangle(frame, (int(l_box[0]), int(l_box[1])), (int(l_box[2]), int(l_box[3])), box_color, 2)
        
        # 4. Tier 2: VLM Verification
        if laptop_states[track_id]["unattended_frames"] > UNATTENDED_THRESHOLD_FRAMES:
            if (current_sec - laptop_states[track_id]["last_alert_sec"]) > ALERT_COOLDOWN_SEC:
                
                # Extract 20% padded crop
                pad_x, pad_y = int(l_width * 0.2), int((l_box[3] - l_box[1]) * 0.2)
                x1, y1 = max(0, int(l_box[0]) - pad_x), max(0, int(l_box[1]) - pad_y)
                x2, y2 = min(frame_width, int(l_box[2]) + pad_x), min(frame_height, int(l_box[3]) + pad_y)
                
                crop_img = frame[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                
                # Halt briefly to verify with VLM
                print(f"\n[Tier 2] Triggering VLM on Laptop {track_id}...")
                vlm_decision = verify_laptop_state(crop_pil)
                print(f"[Tier 2] VLM Output: {vlm_decision}")
                
                if "OPEN" in vlm_decision:
                    alert_tool.trigger_alert(event_description=f"Unattended OPEN laptop ID {track_id}", severity="warning")
                    alert_timestamps.append(current_sec)
                    laptop_states[track_id]["last_alert_sec"] = current_sec
                    cv2.putText(frame, "VIOLATION!", (int(l_box[0]), int(l_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                else:
                    # It's closed, reset timer so we don't spam VLM
                    laptop_states[track_id]["unattended_frames"] = 0

    # Draw Person Boxes
    for p_box in persons:
        cv2.rectangle(frame, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), (255, 0, 0), 2)

    video_writer.write(frame)
    cv2.imshow("Hybrid Architecture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# --- 4. Audio Stitching ---
print("\nStitching final audio...")
if alert_timestamps:
    video_clip = VideoFileClip(TEMP_VIDEO)
    audio_clips = [AudioFileClip(AUDIO_FILE).set_start(ts) for ts in alert_timestamps]
    final_video = video_clip.set_audio(CompositeAudioClip(audio_clips))
    final_video.write_videofile(FINAL_VIDEO, codec="libx264", audio_codec="aac", remove_temp=True, logger=None)
    video_clip.close()
    for ac in audio_clips: ac.close()
    os.remove(TEMP_VIDEO)
    os.remove(AUDIO_FILE)
    print(f"Saved: {FINAL_VIDEO}")
else:
    os.rename(TEMP_VIDEO, FINAL_VIDEO)
    print("No alerts triggered.")