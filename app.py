import streamlit as st
import supervision as sv
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 
import os
import json
import tempfile
import time
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

# =================================================================
# 1. INITIALIZATION & CONFIGURATION
# =================================================================

GEMINI_CLIENT = None
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    import google.genai as genai
    GEMINI_CLIENT = genai.Client()
except:
    pass

class Config:
    """Central configuration for the analysis."""
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5      
    COCO_NAMES: Dict[int, str] = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    DEFAULT_TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 
    REPORT_NAME: str = "surveillance_mission"
    
    # AOI (Normalized to 0-1000)
    AOI_COORDINATES: List[int] = [200, 200, 800, 800] 
    
    # THREAT SCORING WEIGHTS
    W_GEOFENCE: int = 30 
    W_LOITER: int = 25   
    W_UNEXPECTED: int = 15 
    TYPICAL_CLASSES: List[str] = ['car', 'truck', 'bus'] 

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
        model = YOLO(Config.MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

def is_in_aoi(x_center: float, y_center: float, aoi_normalized: List[int], frame_width: int, frame_height: int) -> bool:
    """Checks if the object center is within the normalized AOI."""
    x_min = int(aoi_normalized[0] * frame_width / 1000)
    y_min = int(aoi_normalized[1] * frame_height / 1000)
    x_max = int(aoi_normalized[2] * frame_width / 1000)
    y_max = int(aoi_normalized[3] * frame_height / 1000)
    
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

def calculate_threat_score(is_geofence: bool, is_loitering: bool, object_class: str, confidence: float) -> int:
    """Calculates a weighted predictive threat score (0-100)."""
    score = 0
    if is_geofence: score += Config.W_GEOFENCE
    if is_loitering: score += Config.W_LOITER
    if object_class not in Config.TYPICAL_CLASSES: score += Config.W_UNEXPECTED
    
    scaled_score = score * confidence
    return int(min(scaled_score, 100))

def generate_gemini_summary(data: str, mode: str, object_id: Optional[int] = None) -> Optional[str]:
    """Generates complex narrative and hypotheses using Gemini."""
    if not GEMINI_CLIENT:
        return "Gemini client not initialized. Cannot generate AI analysis."

    if mode == "BRIEFING":
        prompt = (
            "Analyze the following JSON time logs and speed data. Generate a concise, high-level summary suitable for a mission briefing report (1-3 paragraphs), "
            "focusing on the total object count by class, the maximum observed Threat Score, and the longest duration of visibility."
            f"\n\nJSON Data:\n{data}"
        )
    elif mode == "ANOMALY":
        prompt = (
            "Analyze the detailed log data (JSON format). Identify frames where the object count **spiked sharply** or **dropped unexpectedly**. "
            "Report on these anomaly frames and what class of objects was involved."
            f"\n\nJSON Data (First 5000 chars for sample):\n{data[:5000]}"
        )
    elif mode == "HYPOTHESIZE":
        prompt = (
            f"Analyze the detailed trajectory and speed data for Object ID {object_id}. "
            "Provide a complex behavioral hypothesis (1-2 paragraphs): Did the object loiter, attempt to evade detection (sudden stops/turns), or follow a highly predictable path? "
            "Mention the maximum observed speed."
            f"\n\nJSON Data (Full log for Object ID {object_id}):\n{data}"
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

def process_video_stream(input_path: str, output_path: str, model: YOLO, progress_bar, log_table_placeholder, annotated_frame_placeholder, conf_thresh, target_classes_ids, aoi_normalized):
    all_alerts: List[Dict[str, Any]] = []
    trajectories = {} 
    last_centers = {} 
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"Error opening video file at {input_path}")
        return None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    byte_tracker = sv.ByteTrack() 
    box_annotator = sv.BoxAnnotator(thickness=2) 
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    target_video_info = sv.VideoInfo(width=frame_width, height=frame_height, fps=fps_video)
    target_video_info.fourcc = 'mp4v'
    
    kpi_cols = st.columns(3)
    kpi_fps = kpi_cols[0].empty()
    kpi_count = kpi_cols[1].empty()
    kpi_threat = kpi_cols[2].empty()
    
    x_min_draw = int(aoi_normalized[0] * frame_width / 1000)
    y_min_draw = int(aoi_normalized[1] * frame_height / 1000)
    x_max_draw = int(aoi_normalized[2] * frame_width / 1000)
    y_max_draw = int(aoi_normalized[3] * frame_height / 1000)

    start_time = time.time()
    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_threat_score = 0

    try:
        with sv.VideoSink(target_path=output_path, video_info=target_video_info) as sink:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                cv2.rectangle(frame, (x_min_draw, y_min_draw), (x_max_draw, y_max_draw), (0, 255, 0), 2)
                
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.confidence > conf_thresh] 
                detections = detections[np.isin(detections.class_id, target_classes_ids)]
                detections = byte_tracker.update_with_detections(detections=detections)
                
                current_time_seconds = get_time_in_seconds(frame_index, fps_video)
                current_centers = {} 

                if len(detections) > 0:
                    labels = []
                    for xyxy, confidence, class_id, tracker_id in zip(
                        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
                    ):
                        object_name = model.names.get(class_id, f"Class {class_id}")
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        y_center = (xyxy[1] + xyxy[3]) / 2
                        current_centers[tracker_id] = (x_center, y_center)
                        
                        speed_px_sec = 0.0
                        is_loitering = False
                        if tracker_id in last_centers:
                            prev_x, prev_y = last_centers[tracker_id]
                            displacement = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                            speed_px_sec = displacement * fps_video
                            if speed_px_sec < 1.0: is_loitering = True
                        
                        is_geofence_alert = is_in_aoi(x_center, y_center, aoi_normalized, frame_width, frame_height)
                        threat_score = calculate_threat_score(is_geofence_alert, is_loitering, object_name, confidence)
                        max_threat_score = max(max_threat_score, threat_score)

                        if tracker_id not in trajectories: trajectories[tracker_id] = []
                        trajectories[tracker_id].append((x_center, y_center, current_time_seconds, threat_score))
                        
                        alert_tag = f"🚨 Score:{threat_score}" if threat_score >= 30 else ""
                        labels.append(f"ID:{tracker_id} {object_name} {confidence:.2f} {alert_tag}")
                        
                        alert = {
                            'frame_index': frame_index,
                            'time_s': current_time_seconds,
                            'object_id': int(tracker_id), 
                            'object_class': object_name,
                            'confidence': round(float(confidence), 2),
                            'Geofence_Alert': is_geofence_alert,
                            'Threat_Score': threat_score,
                            'Speed_Pixels_Sec': round(speed_px_sec, 2),
                            'bbox_xyxy': xyxy.tolist()
                        }
                        all_alerts.append(alert)
                    
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                
                sink.write_frame(frame=frame)
                last_centers = current_centers
                
                end_time = time.time()
                current_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                start_time = end_time 
                
                kpi_fps.metric(label="Processing FPS", value=f"{current_fps:.2f}")
                kpi_count.metric(label="Current Objects", value=len(detections))
                kpi_threat.metric(label="Max Threat Score", value=max_threat_score, delta=f"{max_threat_score}" if max_threat_score > 0 else None, delta_color="inverse")
                
                annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame_placeholder.image(annotated_frame_rgb, caption="Annotated Video Feed (Green Box is AOI)", use_column_width=True)
                
                if all_alerts:
                    log_df = pd.DataFrame(all_alerts)
                    log_df_display = log_df[['time_s', 'object_id', 'object_class', 'Threat_Score', 'Speed_Pixels_Sec', 'Geofence_Alert']].sort_values(by='time_s', ascending=False)
                    log_table_placeholder.dataframe(log_df_display.head(10), use_container_width=True)
                
                progress = frame_index / total_frames if total_frames > 0 else 0
                progress_bar.progress(min(progress, 1.0), text=f"Processing frame {frame_index}/{total_frames}...")
                frame_index += 1
    finally:
        cap.release()
        progress_bar.progress(1.0, text="Processing Complete!")
    
    return all_alerts, trajectories

# =================================================================
# 4. UI SECTIONS
# =================================================================

def analysis_page():
    st.title("📹 AI Surveillance & Object Tracking Dashboard")
    model = load_model()
    
    st.sidebar.header("🎯 Tracking Configuration")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, Config.CONFIDENCE_THRESHOLD, 0.05)
    class_name_map = {name: id for id, name in Config.COCO_NAMES.items()}
    default_classes_names = [Config.COCO_NAMES[i] for i in Config.DEFAULT_TARGET_CLASSES]
    selected_classes_names = st.sidebar.multiselect("Select Target Object Classes", options=list(class_name_map.keys()), default=default_classes_names)
    target_classes_ids = [class_name_map[name] for name in selected_classes_names]

    st.sidebar.markdown("---")
    st.sidebar.header("🚨 Area of Interest (AOI)")
    st.sidebar.caption("Define a box (0-1000 scale) to trigger 'Geofence Alerts'.")
    
    aoi_x_min = st.sidebar.number_input("X Min (0-1000)", min_value=0, max_value=1000, value=Config.AOI_COORDINATES[0], step=50)
    aoi_y_min = st.sidebar.number_input("Y Min (0-1000)", min_value=0, max_value=1000, value=Config.AOI_COORDINATES[1], step=50)
    aoi_x_max = st.sidebar.number_input("X Max (0-1000)", min_value=0, max_value=1000, value=Config.AOI_COORDINATES[2], step=50)
    aoi_y_max = st.sidebar.number_input("Y Max (0-1000)", min_value=0, max_value=1000, value=Config.AOI_COORDINATES[3], step=50)
    aoi_normalized = [aoi_x_min, aoi_y_min, aoi_x_max, aoi_y_max]

    st.sidebar.markdown("---")
    st.sidebar.metric("YOLO Model Status", "Loaded" if model else "Failed")

    if not GEMINI_CLIENT:
        st.sidebar.error("Gemini API key is MISSING. AI reports will fail.")
    
    uploaded_file = st.file_uploader("Upload a video file (MP4 recommended)", type=["mp4", "mov", "avi"])
    if 'output_video_path' not in st.session_state: st.session_state['output_video_path'] = None
    
    if uploaded_file is not None and model:
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name
        
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        st.session_state['output_video_path'] = output_path
        
        st.success("Video uploaded successfully. Click 'Start Analysis' to process.")
        
        if st.button("▶️ Start Analysis", type="primary"):
            st.session_state['analysis_complete'] = False
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
                all_alerts, trajectories = process_video_stream(
                    input_path, output_path, model, progress_bar, log_table_placeholder, annotated_frame_placeholder, conf_thresh, target_classes_ids, aoi_normalized
                )
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                all_alerts = None
                trajectories = None
            finally:
                os.unlink(input_path) 

            st.markdown("---")
            st.subheader("3. Final Report Generation")
            
            if all_alerts and len(all_alerts) > 0:
                df = pd.DataFrame(all_alerts)
                df['object_id'] = df['object_id'].astype('int') 
                st.session_state['trajectories'] = trajectories
                
                st.session_state['detailed_report'] = df.to_string(index=False)
                time_logs_df = df.groupby(['object_id', 'object_class']).agg(
                    first_s=('time_s', 'min'), last_s=('time_s', 'max'),
                    max_threat=('Threat_Score', 'max'), avg_speed=('Speed_Pixels_Sec', 'mean')
                ).reset_index()
                time_logs_df['duration_s'] = round(time_logs_df['last_s'] - time_logs_df['first_s'], 2)
                time_logs_df['avg_speed'] = round(time_logs_df['avg_speed'], 2)
                
                st.session_state['time_logs'] = time_logs_df.to_string(index=False)
                st.session_state['time_logs_json'] = time_logs_df.to_json(orient='records', indent=4)
                st.session_state['detailed_json'] = df.to_json(orient='records', indent=4)

                st.success("Analysis complete! Reports are ready for download in the **Reports** tab.")
                st.session_state['analysis_complete'] = True
                
            elif all_alerts is not None:
                st.warning("No target objects detected in the video.")
            
            # Attempt to update query param, but user must click tab if automatic switch fails
            st.query_params['tab'] = "📄 Reports & Downloads"

    elif 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        st.success("Analysis complete. Check the **Reports** tab for downloads and the AI Briefing.")

def reports_page():
    st.title("📄 Report Download Center & Analytics")
    if 'analysis_complete' not in st.session_state:
        st.info("Please run a video analysis on the **Analysis** tab first to generate reports.")
        return

    tab_summary, tab_trajectory, tab_ai = st.tabs(["Summary Charts", "Trajectory Analysis", "AI Hypothesizing"])
    
    with tab_summary:
        st.subheader("📊 Statistical Summary Charts")
        df_time_logs = pd.read_json(st.session_state['time_logs_json'], orient='records')
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            counts_df = df_time_logs.groupby('object_class').size().reset_index(name='Total Count')
            fig_count = px.bar(counts_df, x='object_class', y='Total Count', title='Total Unique Objects Tracked by Class')
            st.plotly_chart(fig_count, use_container_width=True)
        with chart_col2:
            avg_duration_df = df_time_logs.groupby('object_class')['duration_s'].mean().reset_index(name='Avg Duration (s)')
            fig_duration = px.bar(avg_duration_df, x='object_class', y='Avg Duration (s)', title='Average Visibility Duration by Class')
            st.plotly_chart(fig_duration, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### General Mission Briefing (Gemini)")
        if 'gemini_briefing' not in st.session_state or st.button("Generate/Regenerate General Briefing"):
             with st.spinner("Generating Mission Briefing..."):
                gemini_report = generate_gemini_summary(st.session_state['time_logs_json'], mode="BRIEFING")
                st.session_state['gemini_briefing'] = gemini_report
        if 'gemini_briefing' in st.session_state:
            st.text_area("Briefing Text", st.session_state['gemini_briefing'], height=200)
            st.download_button(label="Download AI Briefing (TXT)", data=st.session_state['gemini_briefing'], file_name=f"{Config.REPORT_NAME}_briefing.txt", mime="text/plain")

    with tab_trajectory:
        st.subheader("📍 Interactive Object Trajectory Analysis")
        trajectories = st.session_state.get('trajectories', {})
        if not trajectories:
            st.warning("No trajectory data available.")
        else:
            object_ids = sorted([int(id) for id in trajectories.keys()])
            selected_id = st.selectbox("Select Object ID to Visualize Path:", options=object_ids)
            if selected_id:
                data = trajectories[selected_id]
                df_traj = pd.DataFrame(data, columns=['x', 'y', 'time_s', 'threat_score'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_traj['x'], y=df_traj['y'], mode='lines+markers', name=f'ID {selected_id} Path', marker=dict(size=8, color=df_traj['threat_score'], colorscale='Plasma', colorbar=dict(title="Threat Score"))))
                fig.add_trace(go.Scatter(x=[df_traj['x'].iloc[0]], y=[df_traj['y'].iloc[0]], mode='markers', name='START', marker=dict(size=12, color='green', symbol='star')))
                fig.add_trace(go.Scatter(x=[df_traj['x'].iloc[-1]], y=[df_traj['y'].iloc[-1]], mode='markers', name='END', marker=dict(size=12, color='red', symbol='square')))
                fig.update_layout(title=f"Trajectory of Object ID {selected_id} (Color = Threat Score)", xaxis_title="X Coordinate", yaxis_title="Y Coordinate", yaxis={'autorange': 'reversed'}, height=600)
                st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        st.subheader("🧠 Complex Hypothesizing & Anomaly Analysis")
        st.markdown("##### 1. Activity Anomaly Report")
        if st.button("Run Anomaly Report"):
            with st.spinner("Analyzing frame-by-frame activities..."):
                anomaly_report = generate_gemini_summary(st.session_state['detailed_json'], mode="ANOMALY")
                st.session_state['anomaly_report'] = anomaly_report
        if 'anomaly_report' in st.session_state:
            st.text_area("Anomaly Report", st.session_state['anomaly_report'], height=200)
            st.download_button(label="Download Anomaly Report (TXT)", data=st.session_state['anomaly_report'], file_name=f"{Config.REPORT_NAME}_anomaly_report.txt", mime="text/plain")
        st.markdown("---")
        st.markdown("##### 2. Behavioral Hypothesis (Per Object)")
        object_ids = sorted([int(id) for id in st.session_state.get('trajectories', {}).keys()])
        hypothesis_id = st.selectbox("Select Object ID for Behavioral Analysis:", options=object_ids, key='hypo_id')
        if hypothesis_id and st.button(f"Generate Hypothesis for ID {hypothesis_id}", type="secondary"):
            df_full = pd.read_json(st.session_state['detailed_json'], orient='records')
            df_target = df_full[df_full['object_id'] == hypothesis_id]
            log_data_json = df_target.to_json(orient='records', indent=4)
            with st.spinner(f"Asking Gemini to analyze Object ID {hypothesis_id}'s behavior..."):
                hypothesis_report = generate_gemini_summary(log_data_json, mode="HYPOTHESIZE", object_id=hypothesis_id)
                st.session_state['hypothesis_report'] = hypothesis_report
        if 'hypothesis_report' in st.session_state:
            st.text_area("Behavioral Hypothesis", st.session_state['hypothesis_report'], height=200)

    st.markdown("---")
    st.subheader("⬇️ Download All Reports")
    cols = st.columns(5)
    with cols[0]: st.download_button("Detailed (TXT)", st.session_state['detailed_report'], f"{Config.REPORT_NAME}_detailed.txt")
    with cols[1]: st.download_button("Time Logs (TXT)", st.session_state['time_logs'], f"{Config.REPORT_NAME}_timelogs.txt")
    with cols[2]: st.download_button("Detailed (JSON)", st.session_state['detailed_json'], f"{Config.REPORT_NAME}_detailed.json", "application/json")
    with cols[3]: st.download_button("Time Logs (JSON)", st.session_state['time_logs_json'], f"{Config.REPORT_NAME}_timelogs.json", "application/json")
    
    output_path = st.session_state.get('output_video_path')
    if output_path and os.path.exists(output_path):
        with open(output_path, "rb") as file:
            with cols[4]: st.download_button("Video (MP4)", file, f"{Config.REPORT_NAME}_annotated.mp4", "video/mp4")

# =================================================================
# 5. MAIN APP ENTRY POINT
# =================================================================

def main():
    st.set_page_config(layout="wide", page_title="AI Surveillance Dashboard")
    
    tab_titles = ["📊 Analysis & Live Feed", "📄 Reports & Downloads"]
    
    # --- CORRECTED TAB LOGIC ---
    # Standard st.tabs() does NOT accept a default/active index parameter.
    # We simply create the tabs. Navigation must be done by clicking.
    tab1, tab2 = st.tabs(tab_titles)

    with tab1:
        analysis_page()

    with tab2:
        reports_page()

if __name__ == "__main__":
    main()