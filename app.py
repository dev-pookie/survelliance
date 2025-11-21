import streamlit as st
import supervision as sv
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 
import os
import json
import tempfile
from typing import List, Dict, Any, Optional
import plotly.express as px

# =================================================================
# 1. INITIALIZATION & CONFIGURATION
# =================================================================

GEMINI_CLIENT = None
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    import google.genai as genai
    GEMINI_CLIENT = genai.Client(api_key="")
except:
    pass

class Config:
    """Central configuration for the analysis."""
    MODEL_PATH: str = "yolov8n.pt"
    # Target classes default list (can be overridden by UI)
    COCO_NAMES: Dict[int, str] = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    DEFAULT_TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 
    REPORT_NAME: str = "surveillance_mission"

# =================================================================
# 2. UTILITIES & HELPER FUNCTIONS
# =================================================================

def get_time_in_seconds(frame_index: int, fps: float) -> float:
    return round(frame_index / fps, 2)

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it."""
    try:
        model = YOLO(Config.MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

def generate_gemini_summary(summary_data: str, mode: str = "BRIEFING") -> Optional[str]:
    """Uses the Gemini API to generate a natural language summary or anomaly report."""
    if not GEMINI_CLIENT:
        return "Gemini client not initialized. Cannot generate AI summary."

    if mode == "BRIEFING":
        prompt = (
            "Analyze the following JSON time logs from a surveillance report. The logs show "
            "when each unique tracked object first appeared, last appeared, and the total duration. "
            "Generate a concise, high-level summary suitable for a mission briefing report (1-3 paragraphs), "
            "including a breakdown of the number of unique objects by class, and highlighting the longest-tracked object."
            f"\n\nJSON Data:\n{summary_data}"
        )
    elif mode == "ANOMALY":
        prompt = (
            "Analyze the detailed log data (JSON format) containing object counts per frame. "
            "Identify frames where the object count **spiked sharply** or **dropped unexpectedly**. "
            "Provide a brief, actionable report on these anomaly frames and what class of objects was involved."
            f"\n\nJSON Data (First 200 lines for sample):\n{summary_data[:5000]}"
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

def process_video_stream(input_path: str, output_path: str, model: YOLO, progress_bar, log_table_placeholder, annotated_frame_placeholder, conf_thresh, target_classes_ids):
    """
    Analyzes the video frame-by-frame and yields data for real-time update.
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
                detections = detections[detections.confidence > conf_thresh] # Use user-set threshold
                detections = detections[np.isin(detections.class_id, target_classes_ids)] # Use user-set classes
                detections = byte_tracker.update_with_detections(detections=detections)
                
                current_time_seconds = get_time_in_seconds(frame_index, fps)

                if len(detections) > 0:
                    labels = []
                    for xyxy, confidence, class_id, tracker_id in zip(
                        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
                    ):
                        object_name = model.names.get(class_id, f"Class {class_id}")
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
                
                # Update logs table
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
    
    model = load_model()
    
    # 1. SIDEBAR CONFIGURATION (For Clean UX)
    st.sidebar.header("🎯 Tracking Configuration")
    
    # Confidence Threshold
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, Config.CONFIDENCE_THRESHOLD, 0.05)
    
    # Target Class Selection
    class_name_map = {name: id for id, name in Config.COCO_NAMES.items()}
    default_classes_names = [Config.COCO_NAMES[i] for i in Config.DEFAULT_TARGET_CLASSES]
    
    selected_classes_names = st.sidebar.multiselect(
        "Select Target Object Classes",
        options=list(class_name_map.keys()),
        default=default_classes_names
    )
    target_classes_ids = [class_name_map[name] for name in selected_classes_names]

    st.sidebar.markdown("---")
    st.sidebar.metric("YOLO Model Status", "Loaded")
    
    # 2. FILE UPLOAD & START BUTTON
    uploaded_file = st.file_uploader("Upload a video file (MP4 recommended)", type=["mp4", "mov", "avi"])
    
    # Placeholder for the output video file path
    if 'output_video_path' not in st.session_state:
        st.session_state['output_video_path'] = None
    
    if uploaded_file is not None:
        
        # Streamlit's temporary file management
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name
        
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        st.session_state['output_video_path'] = output_path
        
        st.success("Video uploaded successfully. Click 'Start Analysis' to process.")
        
        if st.button("▶️ Start Analysis", type="primary"):
            st.session_state['analysis_complete'] = False
            
            # --- START ANALYSIS UI ---
            st.subheader("Live Analysis Feed & Logs")
            
            video_col, log_col = st.columns([3, 2])
            
            with video_col:
                st.markdown("### Annotated Video Feed")
                annotated_frame_placeholder = st.empty()
            
            with log_col:
                st.markdown("### Real-Time Object Logs")
                log_table_placeholder = st.empty()
            
            progress_bar = st.progress(0, text="Processing video...")
            
            try:
                # Run the core processing logic with user-defined filters
                all_alerts = process_video_stream(
                    input_path, output_path, model, progress_bar, log_table_placeholder, annotated_frame_placeholder, conf_thresh, target_classes_ids
                )
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                all_alerts = None
            finally:
                # Clean up input temp file
                os.unlink(input_path) 

            st.markdown("---")
            st.subheader("3. Final Report Generation")
            
            if all_alerts and len(all_alerts) > 0:
                df = pd.DataFrame(all_alerts)
                df['object_id'] = df['object_id'].astype('int') 
                
                # --- REPORT DATA PREP ---
                st.session_state['detailed_report'] = df.to_string(index=False)
                
                time_logs_df = df.groupby(['object_id', 'object_class']).agg(
                    first_s=('time_s', 'min'),
                    last_s=('time_s', 'max')
                ).reset_index()
                time_logs_df['duration_s'] = round(time_logs_df['last_s'] - time_logs_df['first_s'], 2)
                
                st.session_state['time_logs'] = time_logs_df.to_string(index=False)
                st.session_state['time_logs_json'] = time_logs_df.to_json(orient='records', indent=4)
                st.session_state['detailed_json'] = df.to_json(orient='records', indent=4)

                st.success("Analysis complete! Reports are ready for download in the **Reports** tab.")
                st.session_state['analysis_complete'] = True
                
            elif all_alerts is not None:
                st.warning("No target objects detected in the video based on the configured threshold and classes.")
            
            st.experimental_set_query_params(tab="Reports")

    elif 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        st.success("Analysis complete. Check the **Reports** tab for downloads and the AI Briefing.")

def reports_page():
    """Defines the report download and statistical UI."""
    st.title("📄 Report Download Center & Analytics")
    
    if 'analysis_complete' not in st.session_state:
        st.info("Please run a video analysis on the **Analysis** tab first to generate reports.")
        return

    st.subheader("📊 Statistical Summary Charts")
    
    # Create the DataFrame from session state for charting
    df_time_logs = pd.read_json(st.session_state['time_logs_json'], orient='records')

    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Chart 1: Total Object Count by Class
        counts_df = df_time_logs.groupby('object_class').size().reset_index(name='Total Count')
        fig_count = px.bar(counts_df, x='object_class', y='Total Count', title='Total Unique Objects Tracked by Class')
        st.plotly_chart(fig_count, use_container_width=True)
            
    with chart_col2:
        # Chart 2: Average Visibility Duration by Class
        avg_duration_df = df_time_logs.groupby('object_class')['duration_s'].mean().reset_index(name='Avg Duration (s)')
        fig_duration = px.bar(avg_duration_df, x='object_class', y='Avg Duration (s)', title='Average Visibility Duration by Class')
        st.plotly_chart(fig_duration, use_container_width=True)

    st.markdown("---")
    
    # --- GEMINI AI BRIEFING & ANOMALY DETECTION ---
    st.subheader("🤖 AI-Generated Analysis")
    
    briefing_col, anomaly_col = st.columns(2)
    
    with briefing_col:
        st.markdown("##### Mission Briefing")
        if 'gemini_briefing' not in st.session_state or st.button("Generate/Regenerate Briefing"):
             with st.spinner("Generating Mission Briefing..."):
                gemini_report = generate_gemini_summary(st.session_state['time_logs_json'], mode="BRIEFING")
                st.session_state['gemini_briefing'] = gemini_report

        if 'gemini_briefing' in st.session_state:
            st.text_area("Briefing Text", st.session_state['gemini_briefing'], height=200)
            st.download_button(
                label="Download AI Briefing (TXT)",
                data=st.session_state['gemini_briefing'],
                file_name=f"{Config.REPORT_NAME}_briefing.txt",
                mime="text/plain"
            )

    with anomaly_col:
        st.markdown("##### Anomaly Detection (Spikes/Drops)")
        if st.button("Run Anomaly Report"):
            with st.spinner("Analyzing anomalies..."):
                # We use the detailed JSON for anomaly detection
                anomaly_report = generate_gemini_summary(st.session_state['detailed_json'], mode="ANOMALY")
                st.session_state['anomaly_report'] = anomaly_report
        
        if 'anomaly_report' in st.session_state:
            st.text_area("Anomaly Report", st.session_state['anomaly_report'], height=200)
            st.download_button(
                label="Download Anomaly Report (TXT)",
                data=st.session_state['anomaly_report'],
                file_name=f"{Config.REPORT_NAME}_anomaly_report.txt",
                mime="text/plain"
            )

    st.markdown("---")
    st.subheader("⬇️ Download All Reports")

    download_cols = st.columns(5)
    
    # 1. Download Detailed Log (TXT)
    with download_cols[0]:
        st.download_button(
            label="Detailed Log (TXT)",
            data=st.session_state['detailed_report'],
            file_name=f"{Config.REPORT_NAME}_detailed_report.txt",
            mime="text/plain"
        )
    # 2. Download Time Logs (TXT)
    with download_cols[1]:
        st.download_button(
            label="Time Logs (TXT)",
            data=st.session_state['time_logs'],
            file_name=f"{Config.REPORT_NAME}_time_logs.txt",
            mime="text/plain"
        )
    # 3. Download Detailed Log (JSON)
    with download_cols[2]:
        st.download_button(
            label="Detailed Log (JSON)",
            data=st.session_state['detailed_json'],
            file_name=f"{Config.REPORT_NAME}_detailed_report.json",
            mime="application/json"
        )
    # 4. Download Time Logs (JSON)
    with download_cols[3]:
        st.download_button(
            label="Time Logs (JSON)",
            data=st.session_state['time_logs_json'],
            file_name=f"{Config.REPORT_NAME}_time_logs.json",
            mime="application/json"
        )
    # 5. Download Annotated Video
    output_path = st.session_state.get('output_video_path')
    if output_path and os.path.exists(output_path):
        with open(output_path, "rb") as file:
            with download_cols[4]:
                st.download_button(
                    label="Annotated Video (MP4)",
                    data=file,
                    file_name=f"{Config.REPORT_NAME}_annotated.mp4",
                    mime="video/mp4"
                )

# =================================================================
# 5. MAIN APP ENTRY POINT (TABBED INTERFACE)
# =================================================================

def main():
    st.set_page_config(layout="wide", page_title="AI Surveillance Dashboard")
    
    tab1, tab2 = st.tabs(["📊 Analysis & Live Feed", "📄 Reports & Downloads"])

    with tab1:
        analysis_page()

    with tab2:
        reports_page()

if __name__ == "__main__":
    main()