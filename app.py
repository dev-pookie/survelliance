import supervision as sv
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 
import os
import json 
from typing import List, Dict, Any, Optional

# --- GEMINI API INTEGRATION ---
try:
    import google.genai as genai
    # Ensure GEMINI_API_KEY environment variable is set in your environment
    GEMINI_CLIENT = genai.Client()
    GEMINI_MODEL = 'gemini-2.5-flash'
except ImportError:
    print("Warning: 'google-genai' not found. Cannot generate AI summary.")
    GEMINI_CLIENT = None
except Exception as e:
    # This block will catch an error if the API key is missing or invalid
    print(f"Error initializing Gemini client: {e}")
    GEMINI_CLIENT = None
# -----------------------------

# =================================================================
# 1. CENTRALIZED CONFIGURATION
# =================================================================

class Config:
    """Central configuration for the surveillance analysis script."""
    
    VIDEO_PATH: str = "surv_2.mp4"         
    OUTPUT_VIDEO_PATH: str = "annotated_mission_output.mp4"
    REPORT_NAME: str = "surveillance_mission" # Base name for all reports

    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5      
    TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 

    BOX_THICKNESS: int = 2
    TEXT_SCALE: float = 0.5
    TEXT_THICKNESS: int = 1

# =================================================================
# 2. INITIALIZATION & UTILITIES
# =================================================================

all_alerts: List[Dict[str, Any]] = []

try:
    MODEL: YOLO = YOLO(Config.MODEL_PATH) 
except Exception as e:
    print(f"ERROR: Failed to load YOLO model from {Config.MODEL_PATH}.")
    exit()

try:
    video_info: sv.VideoInfo = sv.VideoInfo.from_video_path(video_path=Config.VIDEO_PATH)
except FileNotFoundError:
    print(f"ERROR: Video file not found at {Config.VIDEO_PATH}.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to read video information. Details: {e}")
    exit()

byte_tracker: sv.ByteTrack = sv.ByteTrack() 
box_annotator: sv.BoxAnnotator = sv.BoxAnnotator(thickness=Config.BOX_THICKNESS) 
label_annotator: sv.LabelAnnotator = sv.LabelAnnotator(
    text_thickness=Config.TEXT_THICKNESS, 
    text_scale=Config.TEXT_SCALE,
    text_color=sv.Color.WHITE, 
    color=sv.Color.BLACK
)

def get_time_in_seconds(frame_index: int, video_info: sv.VideoInfo) -> float:
    return round(frame_index / video_info.fps, 2)

# =================================================================
# 3. GEMINI AI SUMMARY FUNCTION
# =================================================================

def generate_gemini_summary(summary_data: str) -> Optional[str]:
    """
    Uses the Gemini API to generate a natural language summary from the structured data.
    """
    if not GEMINI_CLIENT:
        return "Gemini client not initialized. Cannot generate AI summary."

    prompt = (
        "Analyze the following JSON time logs from a surveillance report. The logs show "
        "when each unique tracked object first appeared (`first_s`), last appeared (`last_s`), and the total `duration_s`."
        "Generate a **concise, high-level summary suitable for a mission briefing report** (1-3 paragraphs), "
        "including a breakdown of the number of unique objects by class, and highlighting the longest-tracked object."
        f"\n\nJSON Data:\n{summary_data}"
    )

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# =================================================================
# 4. FRAME PROCESSING FUNCTION
# =================================================================

def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
    
    results = MODEL(frame)[0]
    detections: sv.Detections = sv.Detections.from_ultralytics(results)
    
    detections = detections[detections.confidence > Config.CONFIDENCE_THRESHOLD]
    detections = detections[np.isin(detections.class_id, Config.TARGET_CLASSES)]
    
    detections = byte_tracker.update_with_detections(detections=detections)
    
    if len(detections) > 0:
        current_time_seconds = get_time_in_seconds(frame_index, video_info)
        labels: List[str] = []
        
        for xyxy, confidence, class_id, tracker_id in zip(
            detections.xyxy, 
            detections.confidence, 
            detections.class_id, 
            detections.tracker_id
        ):
            object_name = MODEL.names[class_id]
            label = f"ID:{tracker_id} {object_name} {confidence:.2f}"
            labels.append(label)
            
            all_alerts.append({
                'frame_index': frame_index,
                'time_s': current_time_seconds,
                'object_id': int(tracker_id), 
                'object_class': object_name,
                'confidence': round(float(confidence), 2),
                'bbox_xyxy': xyxy.tolist()
            })
            
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
    return frame

# =================================================================
# 5. MAIN EXECUTION AND REPORT GENERATION
# =================================================================

def main():
    print("--- Surveillance Analyzer Starting ---")
    print(f"Input Video: {Config.VIDEO_PATH} | Output Video: {Config.OUTPUT_VIDEO_PATH}")
    print("-" * 35)

    # --- STEP 6: VIDEO PROCESSING LOOP ---
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    target_video_info = sv.VideoInfo(
        width=video_info.width,
        height=video_info.height,
        fps=video_info.fps
    )
    target_video_info.fourcc = 'mp4v' 

    with sv.VideoSink(target_path=Config.OUTPUT_VIDEO_PATH, video_info=target_video_info) as sink:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame = process_frame(frame, frame_index)
            sink.write_frame(frame=annotated_frame)
            
            if frame_index % 100 == 0 and frame_index > 0:
                print(f"Progress: Processed {frame_index}/{video_info.total_frames} frames.")
                
            frame_index += 1

    cap.release()
    print(f"\nAnnotated video saved to: {Config.OUTPUT_VIDEO_PATH}")
    
    # --- STEP 7: GENERATE THREE REPORTS (.TXT) ---
    if not all_alerts:
        print("No objects were detected.")
        return

    df = pd.DataFrame(all_alerts)
    df['object_id'] = df['object_id'].astype('int') 
    
    detailed_txt_path = f"{Config.REPORT_NAME}_detailed_report.txt"
    time_logs_txt_path = f"{Config.REPORT_NAME}_time_logs.txt"
    briefing_txt_path = f"{Config.REPORT_NAME}_briefing.txt"
    
    # 1. Full Detailed Report (TXT)
    with open(detailed_txt_path, 'w') as f:
        f.write("--- FULL DETAILED REPORT (FRAME-BY-FRAME LOG) ---\n\n")
        # Use to_string() for a clean, formatted text table
        f.write(df.to_string(index=False)) 
    
    # 2. Time Logs (Data Prep and TXT output)
    time_logs_df = df.groupby(['object_id', 'object_class']).agg(
        first_s=('time_s', 'min'),
        last_s=('time_s', 'max')
    ).reset_index()
    time_logs_df['duration_s'] = round(time_logs_df['last_s'] - time_logs_df['first_s'], 2)
    
    # Save Time Logs to TXT
    with open(time_logs_txt_path, 'w') as f:
        f.write("--- TIME LOGS (UNIQUE OBJECT PRESENCE) ---\n\n")
        f.write(time_logs_df.to_string(index=False)) 
    
    print("\n--- FINAL MISSION REPORT FILES ---")
    print(f"1. Detailed Log (All Frames): {detailed_txt_path}")
    print(f"2. Time Logs (Start/End/Duration): {time_logs_txt_path}")
    
    # 3. GENERATE AI SUMMARY USING GEMINI
    print("\n--- GENERATING AI BRIEFING (Gemini API) ---")
    
    # Convert the time logs DataFrame to JSON string *in memory* for the API call
    summary_json_text = time_logs_df.to_json(orient='records', indent=4)

    # Call the Gemini function
    gemini_report = generate_gemini_summary(summary_json_text)

    # Print and save the AI-generated report
    if gemini_report.startswith("Gemini API Error") or gemini_report.startswith("Gemini client not initialized"):
        print(f"Failed to generate briefing: {gemini_report}")
    else:
        # Save the summary to the new text file
        with open(briefing_txt_path, 'w') as f:
            f.write(f"--- AI-GENERATED MISSION BRIEFING ({GEMINI_MODEL}) ---\n\n")
            f.write(gemini_report)
        print(f"3. AI Briefing saved to: {briefing_txt_path}")
        print("\n[AI-Generated Briefing Preview]")
        # Print a preview of the report
        print(gemini_report[:500] + ('...' if len(gemini_report) > 500 else ''))

if __name__ == "__main__":
    main()