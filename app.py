import streamlit as st
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import time
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

# =================================================================
# 1. CONFIGURATION & UTILITIES
# =================================================================

# --- INITIALIZATION ---
GEMINI_CLIENT = None
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    import google.genai as genai
    # Tries to initialize client using GEMINI_API_KEY from environment variables
    GEMINI_CLIENT = genai.Client()
except Exception:
    pass

class Config:
    """Global Configuration."""
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5      
    COCO_NAMES: Dict[int, str] = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    DEFAULT_TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 
    
    # Scoring Weights
    W_GEOFENCE: int = 30 
    W_LOITER: int = 25   
    W_UNEXPECTED: int = 15 
    TYPICAL_CLASSES: List[str] = ['car', 'truck', 'bus'] 

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        return YOLO(Config.MODEL_PATH)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

# --- HELPER FUNCTIONS ---
def get_time_in_seconds(frame_index: int, fps: float) -> float:
    return round(frame_index / fps, 2)

def is_in_aoi(x_center, y_center, aoi_normalized, width, height):
    x_min = int(aoi_normalized[0] * width / 1000)
    y_min = int(aoi_normalized[1] * height / 1000)
    x_max = int(aoi_normalized[2] * width / 1000)
    y_max = int(aoi_normalized[3] * height / 1000)
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

def calculate_threat_score(is_geofence, is_loitering, object_class, confidence):
    score = 0
    if is_geofence: score += Config.W_GEOFENCE
    if is_loitering: score += Config.W_LOITER
    if object_class not in Config.TYPICAL_CLASSES: score += Config.W_UNEXPECTED
    return int(min(score * confidence, 100))

def generate_gemini_summary(data: str, mode: str, object_id: Optional[int] = None) -> str:
    if not GEMINI_CLIENT: return "⚠️ Gemini API Key Missing. Export GEMINI_API_KEY."
    
    prompts = {
        "BRIEFING": f"Generate a mission briefing from these logs. Focus on threats, class counts, and anomalies.\nDATA:\n{data}",
        "ANOMALY": f"Analyze logs for sudden spikes in activity or security breaches (Geofence).\nDATA:\n{data[:5000]}",
        "HYPOTHESIZE": f"Analyze movement history for Object {object_id}. Was it evading detection, loitering, or normal?\nDATA:\n{data}"
    }
    
    try:
        response = GEMINI_CLIENT.models.generate_content(model=GEMINI_MODEL, contents=[prompts[mode]])
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- CORE PROCESSING LOOP ---
def process_video(input_path, output_path, model, conf, classes, aoi, progress_bar, frame_place, log_place, kpi_places):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total_frames = int(cap.get(7))
    
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    
    all_alerts = []
    trajectories = {}
    last_pos = {}
    
    with sv.VideoSink(target_path=output_path, video_info=sv.VideoInfo(width=width, height=height, fps=fps)) as sink:
        start_time = time.time()
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Draw AOI
            x1, y1 = int(aoi[0]*width/1000), int(aoi[1]*height/1000)
            x2, y2 = int(aoi[2]*width/1000), int(aoi[3]*height/1000)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Detect & Track
            res = model(frame, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            dets = dets[dets.confidence > conf]
            dets = dets[np.isin(dets.class_id, classes)]
            dets = tracker.update_with_detections(dets)
            
            current_time = get_time_in_seconds(frame_idx, fps)
            labels = []
            
            for xyxy, conf_score, cls_id, trk_id in zip(dets.xyxy, dets.confidence, dets.class_id, dets.tracker_id):
                name = model.names[cls_id]
                xc, yc = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
                
                # Speed & Logic
                speed = 0.0
                is_loiter = False
                if trk_id in last_pos:
                    dist = np.linalg.norm(np.array([xc, yc]) - np.array(last_pos[trk_id]))
                    speed = dist * fps
                    if speed < 1.0: is_loiter = True
                last_pos[trk_id] = (xc, yc)
                
                in_aoi = is_in_aoi(xc, yc, aoi, width, height)
                threat = calculate_threat_score(in_aoi, is_loiter, name, conf_score)
                
                # Store Data
                if trk_id not in trajectories: trajectories[trk_id] = []
                trajectories[trk_id].append((xc, yc, current_time, threat))
                
                alert_tag = f"🚨 {threat}" if threat > 30 else ""
                labels.append(f"ID:{trk_id} {alert_tag}")
                
                all_alerts.append({
                    "time_s": current_time, "object_id": int(trk_id), "object_class": name,
                    "Threat": threat, "Speed": round(speed, 1), "AOI_Breach": in_aoi,
                    "bbox": xyxy.tolist()
                })
            
            # Annotate & Save
            frame = box_annotator.annotate(scene=frame, detections=dets)
            frame = label_annotator.annotate(scene=frame, detections=dets, labels=labels)
            sink.write_frame(frame)
            
            # UI Updates (Throttled)
            if frame_idx % 2 == 0:
                frame_place.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live Feed (Green Box = AOI)", use_column_width=True)
                curr_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                start_time = time.time()
                
                kpi_places[0].metric("FPS", f"{curr_fps:.1f}")
                kpi_places[1].metric("Active Objects", len(dets))
                
                if all_alerts:
                    df = pd.DataFrame(all_alerts)
                    log_place.dataframe(df.tail(5)[['time_s', 'object_class', 'Threat', 'Speed']], use_container_width=True)
                
                progress_bar.progress(min(frame_idx/total_frames, 1.0))
            
            frame_idx += 1
            
    return all_alerts, trajectories

# =================================================================
# 2. PAGE: HOME
# =================================================================
def render_home():
    st.title("🛡️ Sentinel AI: Autonomous Surveillance System")
    st.markdown("### *From Autonomous Capture to Actionable Intelligence*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Sentinel AI** is a next-generation security platform designed for high-stakes monitoring. 
        It processes video feeds from autonomous vehicles (drones/rovers) to detect, track, and analyze threats in real-time.
        
        #### 🚀 Key Capabilities
        * **👁️ Computer Vision:** Real-time object detection & tracking (YOLOv8 + ByteTrack).
        * **🧠 Cognitive Analysis:** Google Gemini AI integration for behavioral hypothesizing.
        * **⚡ Predictive Threat Scoring:** Dynamic risk assessment based on geofencing and loitering.
        * **📊 Forensic Trajectories:** Interactive path mapping for post-mission review.
        """)
        
        st.info("👈 **Select 'Live Surveillance' in the sidebar** to start a mission.")

    with col2:
        st.markdown("### System Status")
        st.success("✅ AI Models Loaded")
        
        if GEMINI_CLIENT:
            st.success("✅ Gemini API Connected")
        else:
            st.error("❌ Gemini API Key Missing")
            st.caption("Set `GEMINI_API_KEY` in env vars.")

    st.markdown("---")
    st.markdown("#### 🛠️ Pipeline Architecture")
    st.code("Autonomous Vehicle (Input) -> YOLOv8 Detection -> ByteTrack -> Threat Engine -> Gemini Analyst -> Dashboard", language="bash")

# =================================================================
# 3. PAGE: LIVE SURVEILLANCE
# =================================================================
def render_surveillance():
    st.title("🔴 Live Mission Control")

    # --- SIDEBAR CONFIG ---
    st.sidebar.header("⚙️ Mission Parameters")
    
    class_names = list(Config.COCO_NAMES.values())
    default_names = [Config.COCO_NAMES[i] for i in Config.DEFAULT_TARGET_CLASSES]
    selected_names = st.sidebar.multiselect("Target Classes", class_names, default_names)
    target_ids = [k for k, v in Config.COCO_NAMES.items() if v in selected_names]
    
    conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, Config.CONFIDENCE_THRESHOLD)
    
    st.sidebar.divider()
    st.sidebar.subheader("🚨 Geofence (AOI)")
    st.sidebar.caption("Coordinates (0-1000 scale)")
    aoi_x1 = st.sidebar.number_input("X Min", 0, 1000, 200)
    aoi_y1 = st.sidebar.number_input("Y Min", 0, 1000, 200)
    aoi_x2 = st.sidebar.number_input("X Max", 0, 1000, 800)
    aoi_y2 = st.sidebar.number_input("Y Max", 0, 1000, 800)
    aoi = [aoi_x1, aoi_y1, aoi_x2, aoi_y2]

    # --- MAIN LAYOUT ---
    col_feed, col_logs = st.columns([2, 1])

    with col_feed:
        st.subheader("🛰️ Video Feed")
        video_placeholder = st.empty()
        m1, m2 = st.columns(2)
        kpi_fps = m1.empty()
        kpi_obj = m2.empty()

    with col_logs:
        st.subheader("📝 Threat Log")
        log_placeholder = st.empty()

    # --- UPLOAD & PROCESS ---
    uploaded_file = st.file_uploader("📥 Upload Autonomous Feed (MP4)", type=['mp4', 'mov'])

    if uploaded_file:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        st.session_state['video_path'] = output_path 
        
        if st.button("▶️ INITIALIZE SURVEILLANCE", type="primary"):
            st.toast("Mission Started...", icon="🚀")
            prog_bar = st.progress(0)
            
            model = load_model()
            if model:
                alerts, trajectories = process_video(
                    tfile.name, output_path, model, conf, target_ids, aoi,
                    prog_bar, video_placeholder, log_placeholder, [kpi_fps, kpi_obj]
                )
                
                # Save Data for Reports Page
                st.session_state['alerts'] = alerts
                st.session_state['trajectories'] = trajectories
                st.session_state['mission_complete'] = True
                
                st.success("Mission Complete. Data transferred to Forensics.")
                os.unlink(tfile.name) # Cleanup input

# =================================================================
# 4. PAGE: FORENSIC REPORTS
# =================================================================
def render_reports():
    st.title("🧠 Forensic Intelligence Hub")

    if 'mission_complete' not in st.session_state:
        st.warning("⚠️ No Mission Data Found. Please run a Live Surveillance mission first.")
        return

    # Load Data
    df = pd.DataFrame(st.session_state['alerts'])
    trajectories = st.session_state['trajectories']

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📍 Trajectories", "🤖 AI Analyst", "💾 Export"])

    # 1. STATISTICS
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Detections", len(df))
        c2.metric("Max Threat Score", df['Threat'].max(), delta="CRITICAL" if df['Threat'].max() > 80 else "Normal")
        c3.metric("Unique Entities", df['object_id'].nunique())
        
        st.subheader("Threat Distribution")
        fig = px.histogram(df, x="Threat", color="object_class", nbins=20, title="Threat Score Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # 2. TRAJECTORY
    with tab2:
        st.subheader("Interactive Path Analysis")
        obj_ids = sorted(list(trajectories.keys()))
        selected_id = st.selectbox("Select Subject ID to Trace:", obj_ids)
        
        if selected_id:
            data = trajectories[selected_id]
            path_df = pd.DataFrame(data, columns=['x', 'y', 'time', 'threat'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=path_df['x'], y=path_df['y'], 
                mode='lines+markers',
                marker=dict(color=path_df['threat'], colorscale='RdYlGn_r', size=10, showscale=True),
                text=path_df['time'],
                name=f"ID {selected_id}"
            ))
            fig.update_layout(
                title=f"Movement History: Subject {selected_id}",
                xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
                yaxis=dict(autorange='reversed'), height=600
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3. AI ANALYST
    with tab3:
        c_gen, c_out = st.columns([1, 2])
        with c_gen:
            st.subheader("Ask the AI")
            task = st.radio("Select Task:", ["Mission Briefing", "Anomaly Detection", "Subject Hypothesis"])
            
            target_id = None
            if task == "Subject Hypothesis":
                target_id = st.selectbox("Select Subject:", sorted(list(trajectories.keys())))
            
            if st.button("Generate Report ✨", type="primary"):
                with st.spinner("Analyzing Intelligence..."):
                    # Prepare JSON data
                    json_data = df.to_json()
                    if task == "Subject Hypothesis":
                        json_data = df[df['object_id'] == target_id].to_json()
                    
                    # Map radio button to mode string
                    mode_map = {
                        "Mission Briefing": "BRIEFING",
                        "Anomaly Detection": "ANOMALY",
                        "Subject Hypothesis": "HYPOTHESIZE"
                    }
                    res = generate_gemini_summary(json_data, mode_map[task], target_id)
                    st.session_state['ai_result'] = res
        
        with c_out:
            if 'ai_result' in st.session_state:
                st.markdown("### 📄 AI Report")
                st.info(st.session_state['ai_result'])

    # 4. EXPORT
    with tab4:
        st.subheader("Download Evidence")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download JSON Logs", df.to_json(), "mission_logs.json", "application/json")
            st.download_button("⬇️ Download CSV Report", df.to_csv(), "mission_report.csv", "text/csv")
        with c2:
            if os.path.exists(st.session_state['video_path']):
                with open(st.session_state['video_path'], "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f, "evidence_video.mp4", "video/mp4")

# =================================================================
# 5. MAIN NAVIGATION CONTROLLER
# =================================================================
def main():
    st.set_page_config(page_title="Sentinel AI", page_icon="🛡️", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔴 Live Surveillance", "🧠 Forensic Reports"])
    
    if page == "🏠 Home":
        render_home()
    elif page == "🔴 Live Surveillance":
        render_surveillance()
    elif page == "🧠 Forensic Reports":
        render_reports()

if __name__ == "__main__":
    main()