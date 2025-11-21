import streamlit as st
import supervision as sv
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 
import os
import json
from typing import List, Dict, Any, Optional
import tempfile

# =================================================================
# 1. INITIALIZATION & CONFIGURATION
# =================================================================

# --- GEMINI API INTEGRATION ---
GEMINI_CLIENT = None
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    import google.genai as genai
    # This will initialize if GEMINI_API_KEY is set in the environment
    GEMINI_CLIENT = genai.Client()
except ImportError:
    st.sidebar.warning("`google-genai` not installed. AI Briefing unavailable.")
except Exception as e:
    st.sidebar.warning(f"Gemini client error: {e}. Check API key.")
# -----------------------------

class Config:
    """Central configuration for the analysis."""
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5      
    TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 
    REPORT_NAME: str = "surveillance_mission"

# =================================================================
# 2. UTILITIES & HELPER FUNCTIONS
# =================================================================

def get_time_in_seconds(frame_index: int, fps: float) -> float:
    """Calculates the time in seconds for a given frame index."""
    return round(frame_index / fps, 2)

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it."""
    try:
        return YOLO(Config.MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

def generate_gemini_summary(summary_data: str) -> Optional[str]:
    """Uses the Gemini API to generate a natural language summary."""
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
# 3. CORE VIDEO PROCESSING LOGIC
# =================================================================

def process_video_stream(input_path: str, output_path: str, model: YOLO, progress_bar, log_table_placeholder, annotated_frame_placeholder):
    """
    Analyzes the video frame-by-frame, updates the UI, and saves the output.
    Returns a list of all alert dictionaries.
    """
    all_alerts: List[Dict[str, Any]] = []
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"Error opening video file at {input_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    byte_tracker = sv.ByteTrack() 
    box_annotator = sv.BoxAnnotator(thickness=2) 
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # Setup VideoSink for output file
    target_video_info = sv.VideoInfo(width=frame_width, height=frame_height, fps=fps, total_frames=total_frames)
    target_video_info.fourcc = 'mp4v'
    
    try:
        with sv.VideoSink(target_path=output_path, video_info=target_video_info) as sink:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # --- DETECTION & TRACKING ---
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.confidence > Config.CONFIDENCE_THRESHOLD]
                detections = detections[np.isin(detections.class_id, Config.TARGET_CLASSES)]
                detections = byte_tracker.update_with_detections(detections=detections)
                
                current_time_seconds = get_time_in_seconds(frame_index, fps)
                labels = []

                if len(detections) > 0:
                    for xyxy, confidence, class_id, tracker_id in zip(
                        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
                    ):
                        object_name = model.names[class_id]
                        labels.append(f"ID:{tracker_id} {object_name} {confidence:.2f}")
                        
                        alert = {
                            'frame_index': frame_index,
                            'time_s': current_time_seconds,
                            'object_id': int(tracker_id), 
                            'object_class': object_name,
                            'confidence': round(float(confidence), 2),
                            'bbox_xyxy': xyxy.tolist()
                        }
                        all_alerts.append(alert)
                        
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                
                sink.write_frame(frame=frame)

                # --- STREAMLIT LIVE UPDATE ---
                annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame_placeholder.image(annotated_frame_rgb, caption="Annotated Video Feed", use_column_width=True)
                
                # Display only the last 10 log entries for real-time visibility
                if all_alerts:
                    log_df = pd.DataFrame(all_alerts)
                    log_df = log_df[['time_s', 'object_id', 'object_class', 'confidence']].sort_values(by='time_s', ascending=False)
                    log_table_placeholder.dataframe(log_df.head(10), use_container_width=True)
                
                progress = frame_index / total_frames
                progress_bar.progress(min(progress, 1.0), text=f"Processing frame {frame_index}/{total_frames}...")

                frame_index += 1
    finally:
        cap.release()
        progress_bar.progress(1.0, text="Processing Complete!")
    
    return all_alerts

# =================================================================
# 4. UI SECTIONS
# =================================================================

def analysis_page():
    """Defines the main analysis and video feed UI."""
    st.title("📹 AI Surveillance & Object Tracking Dashboard")
    
    # 1. FILE UPLOAD & CONFIGURATION
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("1. Input Video")
        uploaded_file = st.file_uploader("Upload a video file (MP4 recommended)", type=["mp4", "mov", "avi"])
        
    with col2:
        st.header("2. Configuration")
        st.metric("YOLO Model", Config.MODEL_PATH)
        Config.CONFIDENCE_THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, Config.CONFIDENCE_THRESHOLD, 0.05)
        st.info(f"Tracking Classes: {', '.join([load_model().names[i] for i in Config.TARGET_CLASSES])}")
        
    st.markdown("---")
    
    # Placeholder for the output video file path
    if 'output_video_path' not in st.session_state:
        st.session_state['output_video_path'] = None
    
    # 2. PROCESSING BUTTON
    if uploaded_file is not None:
        
        # Use Streamlit's temporary file management
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name
        
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        st.session_state['output_video_path'] = output_path
        
        st.success("Video uploaded successfully. Click 'Start Analysis' to process.")
        
        if st.button("▶️ Start Analysis", type="primary"):
            st.session_state['analysis_complete'] = False
            
            # --- START ANALYSIS UI ---
            st.subheader("Live Analysis Feed & Logs")
            
            # Create columns for side-by-side feed and logs
            video_col, log_col = st.columns(2)
            
            with video_col:
                st.markdown("### Annotated Video Feed")
                annotated_frame_placeholder = st.empty()
            
            with log_col:
                st.markdown("### Real-Time Object Logs")
                log_table_placeholder = st.empty()
            
            progress_bar = st.progress(0, text="Processing video...")
            
            model = load_model()
            
            try:
                # Run the core processing logic
                all_alerts = process_video_stream(
                    input_path, output_path, model, progress_bar, log_table_placeholder, annotated_frame_placeholder
                )
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                all_alerts = None
            finally:
                tfile.close() # Close the input file

            st.markdown("---")
            st.subheader("3. Final Report Generation")
            
            if all_alerts and len(all_alerts) > 0:
                df = pd.DataFrame(all_alerts)
                df['object_id'] = df['object_id'].astype('int') 
                
                # --- REPORT DATA PREP ---
                # Detailed Report (TXT)
                st.session_state['detailed_report'] = df.to_string(index=False)
                
                # Time Logs (for TXT and Gemini JSON)
                time_logs_df = df.groupby(['object_id', 'object_class']).agg(
                    first_s=('time_s', 'min'),
                    last_s=('time_s', 'max')
                ).reset_index()
                time_logs_df['duration_s'] = round(time_logs_df['last_s'] - time_logs_df['first_s'], 2)
                
                st.session_state['time_logs'] = time_logs_df.to_string(index=False)
                
                # In-memory JSON for Gemini API call
                summary_json_text = time_logs_df.to_json(orient='records', indent=4)
                
                # --- GEMINI AI BRIEFING ---
                with st.spinner("Generating AI Briefing..."):
                    gemini_report = generate_gemini_summary(summary_json_text)
                
                st.session_state['gemini_briefing'] = gemini_report
                
                st.success("Analysis complete! Reports are ready for download in the **Reports** tab.")
                st.session_state['analysis_complete'] = True
                
            elif all_alerts is not None:
                st.warning("No target objects detected in the video based on the configured threshold.")
            
            # Force switch to reports tab on completion (optional, but good UX)
            st.experimental_set_query_params(tab="Reports")

    elif 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        st.success("Analysis complete. Check the **Reports** tab for downloads and the AI Briefing.")

def reports_page():
    """Defines the report download UI."""
    st.title("📄 Report Download Center")
    st.write("Download the detailed results and the AI-generated briefing from the latest analysis.")
    st.markdown("---")

    if 'analysis_complete' not in st.session_state:
        st.info("Please run a video analysis on the **Analysis** tab first to generate reports.")
        return

    # 1. Display AI Briefing (for immediate reading)
    st.subheader("AI-Generated Mission Briefing")
    if 'gemini_briefing' in st.session_state and st.session_state['gemini_briefing']:
        st.text_area("Briefing Text", st.session_state['gemini_briefing'], height=300)
        st.download_button(
            label="Download AI Briefing (TXT)",
            data=st.session_state['gemini_briefing'],
            file_name=f"{Config.REPORT_NAME}_briefing.txt",
            mime="text/plain"
        )
    else:
        st.warning("AI Briefing is unavailable. Check the Gemini API connection.")

    st.markdown("---")
    st.subheader("Structured Data Reports")
    
    col_det, col_time = st.columns(2)
    
    # 2. Detailed Report Download
    with col_det:
        st.markdown("##### Full Detailed Report")
        if 'detailed_report' in st.session_state:
            st.download_button(
                label="Download Detailed Log (.txt)",
                data=st.session_state['detailed_report'],
                file_name=f"{Config.REPORT_NAME}_detailed_report.txt",
                mime="text/plain"
            )
            
    # 3. Time Logs Download
    with col_time:
        st.markdown("##### Object Time Logs")
        if 'time_logs' in st.session_state:
            st.download_button(
                label="Download Time Logs (.txt)",
                data=st.session_state['time_logs'],
                file_name=f"{Config.REPORT_NAME}_time_logs.txt",
                mime="text/plain"
            )

    st.markdown("---")
    st.subheader("Annotated Video Output")
    
    # 4. Annotated Video Download
    output_path = st.session_state.get('output_video_path')
    if output_path and os.path.exists(output_path):
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Annotated Video (MP4)",
                data=file,
                file_name=f"{Config.REPORT_NAME}_annotated.mp4",
                mime="video/mp4"
            )
        # Clean up temp file after successful download button visibility (optional)
        # os.remove(output_path) 
    else:
        st.info("Annotated video file not found or processing not complete.")

# =================================================================
# 5. MAIN APP ENTRY POINT (TABBED INTERFACE)
# =================================================================

def main():
    st.set_page_config(layout="wide", page_title="AI Surveillance Dashboard")
    
    # Use Streamlit Tabs for navigation
    tab1, tab2 = st.tabs(["📊 Analysis & Live Feed", "📄 Reports & Downloads"])

    with tab1:
        analysis_page()

    with tab2:
        reports_page()

if __name__ == "__main__":
    main()