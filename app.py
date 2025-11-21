import supervision as sv
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 
import os
from typing import List, Dict, Any, Optional

# =================================================================
# 1. CENTRALIZED CONFIGURATION
# =================================================================

class Config:
    """Central configuration for the surveillance analysis script."""
    
    # --- FILE PATHS ---
    VIDEO_PATH: str = "surv_4.mp4"         # <<-- Input video file
    OUTPUT_VIDEO_PATH: str = "annotated_mission_output.mp4"
    REPORT_JSON_NAME: str = "surveillance_report" # Base name for JSON reports

    # --- MODEL & THRESHOLDS ---
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5      # Only process detections with confidence > 50%

    # --- THREAT CLASSES (Based on COCO dataset) ---
    # 0='person', 2='car', 3='motorcycle', 5='bus', 7='truck'
    TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 

    # --- ANNOTATION STYLE ---
    BOX_THICKNESS: int = 2
    TEXT_SCALE: float = 0.5
    TEXT_THICKNESS: int = 1

# =================================================================
# 2. INITIALIZATION & UTILITIES
# =================================================================

# Global list to store all threat log entries 
all_alerts: List[Dict[str, Any]] = []

# Load the model with error handling
try:
    MODEL: YOLO = YOLO(Config.MODEL_PATH) 
except Exception as e:
    print(f"ERROR: Failed to load YOLO model from {Config.MODEL_PATH}. Check file path.")
    print(f"Details: {e}")
    exit()

# Get video information 
try:
    video_info: sv.VideoInfo = sv.VideoInfo.from_video_path(video_path=Config.VIDEO_PATH)
except FileNotFoundError:
    print(f"ERROR: Video file not found at {Config.VIDEO_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to read video information. Details: {e}")
    exit()

# Initialize Core Libraries
byte_tracker: sv.ByteTrack = sv.ByteTrack() 

# Initialize Annotators
box_annotator: sv.BoxAnnotator = sv.BoxAnnotator(thickness=Config.BOX_THICKNESS) 
label_annotator: sv.LabelAnnotator = sv.LabelAnnotator(
    text_thickness=Config.TEXT_THICKNESS, 
    text_scale=Config.TEXT_SCALE,
    text_color=sv.Color.WHITE, 
    color=sv.Color.BLACK
)

def get_time_in_seconds(frame_index: int, video_info: sv.VideoInfo) -> float:
    """Calculates the time in seconds for a given frame index."""
    return round(frame_index / video_info.fps, 2)

# =================================================================
# 3. FRAME PROCESSING FUNCTION
# =================================================================

def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
    """
    Analyzes a single video frame for objects, logs alerts, and annotates the frame.
    """
    
    # 1. RUN DETECTION
    results = MODEL(frame)[0]
    
    # 2. CONVERT & FILTER Detections
    detections: sv.Detections = sv.Detections.from_ultralytics(results)
    
    # Filter by Confidence
    detections = detections[detections.confidence > Config.CONFIDENCE_THRESHOLD]
    
    # Filter by Target Class ID
    detections = detections[np.isin(detections.class_id, Config.TARGET_CLASSES)]
    
    # 3. RUN TRACKING: Assign unique IDs
    detections = byte_tracker.update_with_detections(detections=detections)
    
    # 4. LOG ALERT DATA AND ANNOTATE
    if len(detections) > 0:
        current_time_seconds = get_time_in_seconds(frame_index, video_info)
        
        labels: List[str] = []
        
        # Iterate over all detected and tracked objects
        for xyxy, confidence, class_id, tracker_id in zip(
            detections.xyxy, 
            detections.confidence, 
            detections.class_id, 
            detections.tracker_id
        ):
            # Create readable label
            object_name = MODEL.names[class_id]
            label = f"ID:{tracker_id} {object_name} {confidence:.2f}"
            labels.append(label)
            
            # Log the critical data
            all_alerts.append({
                'frame_index': frame_index,
                'time_s': current_time_seconds,
                'object_id': int(tracker_id), 
                'object_class': object_name,
                'confidence': round(float(confidence), 2),
                # Bounding box coordinates, converted to list for JSON/Pandas compatibility
                'bbox_xyxy': xyxy.tolist()
            })
            
        # Annotate the boxes and labels onto the frame
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
    return frame

# =================================================================
# 4. MAIN EXECUTION AND REPORT GENERATION
# =================================================================

def main():
    print("--- Surveillance Analyzer Starting ---")
    print(f"Input Video: {Config.VIDEO_PATH} | Output Video: {Config.OUTPUT_VIDEO_PATH}")
    print(f"Tracking Classes: {[MODEL.names[i] for i in Config.TARGET_CLASSES]}")
    print("-" * 35)

    # --- STEP 6: MANUAL VIDEO PROCESSING LOOP ---
    print(f"Processing video. Total frames: {video_info.total_frames}")

    # 1. Initialize Video Reader (OpenCV)
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    
    # 2. Configure Video Writer (Supervision VideoSink)
    target_video_info = sv.VideoInfo(
        width=video_info.width,
        height=video_info.height,
        fps=video_info.fps
    )
    # Set the fourcc attribute directly (stable method for H.264/MP4 codec)
    target_video_info.fourcc = 'mp4v' 

    # 3. Manual Processing and Writing Loop
    with sv.VideoSink(target_path=Config.OUTPUT_VIDEO_PATH, video_info=target_video_info) as sink:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = process_frame(frame, frame_index)
            sink.write_frame(frame=annotated_frame)
            
            # Log progress every 100 frames
            if frame_index % 100 == 0 and frame_index > 0:
                print(f"Progress: Processed {frame_index}/{video_info.total_frames} frames.")
                
            frame_index += 1

    # Release the video reader
    cap.release()
    print(f"\nAnnotated video saved to: {Config.OUTPUT_VIDEO_PATH}")
    
    # --- STEP 7: GENERATE THE FINAL REPORT IN JSON FORMAT ---
    if not all_alerts:
        print("No objects were detected based on the configured confidence and class filters.")
        return

    df = pd.DataFrame(all_alerts)
    df['object_id'] = df['object_id'].astype('int') 
    
    # Define file paths
    detailed_json_path = f"{Config.REPORT_JSON_NAME}_detailed.json"
    summary_json_path = f"{Config.REPORT_JSON_NAME}_summary.json"
    
    # 1. Detailed Log: All frame-by-frame detections -> SAVE AS JSON
    df.to_json(detailed_json_path, orient='records', indent=4)
    
    # 2. Summary Report: Key metric for each unique tracked object
    summary_df = df.groupby(['object_id', 'object_class']).agg(
        first_s=('time_s', 'min'),
        last_s=('time_s', 'max')
    ).reset_index()
    
    # Calculate the total duration the object was visible
    summary_df['duration_s'] = round(summary_df['last_s'] - summary_df['first_s'], 2)
    
    # Remove intermediate columns for a cleaner final summary
    summary_df.drop(columns=['first_s', 'last_s'], inplace=True)
    
    # Save the summary report -> SAVE AS JSON
    summary_df.to_json(summary_json_path, orient='records', indent=4)
    
    print("\n--- FINAL MISSION REPORT ---")
    print(f"Total Unique Objects Tracked: {summary_df.shape[0]}")
    print(f"Detailed Log saved to: {detailed_json_path}")
    print(f"Summary Report saved to: {summary_json_path}")
    
    print("\n[SUMMARY: UNIQUE OBJECT PRESENCE (First 10 Entries)]")
    print(summary_df.head(10).to_string()) 

if __name__ == "__main__":
    main()