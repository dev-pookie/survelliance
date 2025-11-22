import streamlit as st
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import time
import tempfile
import os
import psutil  # For System Telemetry
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from streamlit_option_menu import option_menu

# =================================================================
# 1. CONFIGURATION & UTILITIES
# =================================================================

# --- PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Sentinel AI | Autonomous Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed because we use top nav
)

# C. CUSTOM CSS INJECTION FOR UI LOOK AND OPTION MENU STYLING
st.markdown("""
<style>
    /* Hide the default Streamlit sidebar elements if desired */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* General Streamlit app adjustments for dark theme */
    .stApp {
        background-color: #0E1117; /* Streamlit dark theme background */
    }
    
    /* Style the st-option-menu container */
    .navbar-container {
        padding-bottom: 20px;
        /* Using the accent color for a subtle separator */
        border-bottom: 2px solid rgba(240, 197, 48, 0.3); 
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
GEMINI_CLIENT = None
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    # Requires google-genai SDK: pip install google-genai
    import google.genai as genai
    # NOTE: Ideally, use os.getenv("GEMINI_API_KEY") for security
    GEMINI_CLIENT = genai.Client(api_key="AIzaSyB-SVG7IjQjivL7dBOBN-ITgmpMuJc892c")
except Exception as e:
    print(f"Gemini Client Init Error: {e}")
    pass

class Config:
    """Global Configuration."""
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5      
    COCO_NAMES: Dict[int, str] = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    DEFAULT_TARGET_CLASSES: List[int] = [0, 2, 3, 5, 7] 
    
    # Scoring Weights for Predictive Threat
    W_GEOFENCE: int = 30 
    W_LOITER: int = 25   
    W_UNEXPECTED: int = 15 
    TYPICAL_CLASSES: List[str] = ['car', 'truck', 'bus'] 

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        # Requires ultralytics: pip install ultralytics
        return YOLO(Config.MODEL_PATH)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

# --- HELPER FUNCTIONS ---
def get_system_metrics():
    """Returns CPU and RAM usage."""
    return psutil.cpu_percent(), psutil.virtual_memory().percent

def get_time_in_seconds(frame_index: int, fps: float) -> float:
    if fps == 0: return 0
    return round(frame_index / fps, 2)

def is_in_aoi(x_center, y_center, aoi_normalized, width, height):
    """Checks if object is inside the normalized Area of Interest."""
    x_min = int(aoi_normalized[0] * width / 1000)
    y_min = int(aoi_normalized[1] * height / 1000)
    x_max = int(aoi_normalized[2] * width / 1000)
    y_max = int(aoi_normalized[3] * height / 1000)
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

def calculate_threat_score(is_geofence, is_loitering, object_class, confidence):
    """Calculates dynamic risk score (0-100)."""
    score = 0
    if is_geofence: score += Config.W_GEOFENCE
    if is_loitering: score += Config.W_LOITER
    if object_class not in Config.TYPICAL_CLASSES: score += Config.W_UNEXPECTED
    return int(min(score * confidence, 100))

def generate_gemini_response(prompt_type: str, data_context: str, user_query: str = "") -> str:
    if not GEMINI_CLIENT: return "⚠️ Gemini API Key Missing. Please export GEMINI_API_KEY."
    
    system_prompts = {
        "BRIEFING": "You are a military surveillance analyst. Write a concise mission briefing based on these logs. Focus on high-threat events.",
        "ANOMALY": "Analyze logs for sudden spikes in object count or high-threat breaches.",
        "HYPOTHESIZE": "Analyze the movement history of this object. Was it evading? Loitering? Provide a behavioral profile.",
        "CHAT": "You are Sentinel AI, an autonomous surveillance assistant. Answer the user's question based strictly on the provided log data. Be professional and concise."
    }
    
    full_prompt = f"{system_prompts[prompt_type]}\n\nCONTEXT DATA:\n{data_context}\n\nUSER QUERY:\n{user_query}"
    
    try:
        response = GEMINI_CLIENT.models.generate_content(model=GEMINI_MODEL, contents=[full_prompt])
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- CORE PROCESSING LOOP ---
def process_video(input_path, output_path, model, conf, classes, aoi, progress_bar, frame_place, log_place, kpi_places):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total_frames = int(cap.get(7))
    
    # Requires supervision: pip install supervision
    tracker = sv.ByteTrack()
    # Use a custom color for the annotator to match the theme's primary color (#F0C530)
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(r=240, g=197, b=48))
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_color=sv.Color.BLACK)
    
    all_alerts = []
    trajectories = {}
    last_pos = {}
    
    # AOI Coordinates for drawing
    ax1, ay1 = int(aoi[0]*width/1000), int(aoi[1]*height/1000)
    ax2, ay2 = int(aoi[2]*width/1000), int(aoi[3]*height/1000)
    
    with sv.VideoSink(target_path=output_path, video_info=sv.VideoInfo(width=width, height=height, fps=fps)) as sink:
        start_time = time.time()
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Draw AOI
            # Use a high-visibility color for the geofence (yellow-green)
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (48, 255, 240), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (ax1, ay1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (48, 255, 240), 2)
            
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
                
                # Speed & Loitering Logic
                speed = 0.0
                is_loiter = False 
                if trk_id in last_pos:
                    dist = np.linalg.norm(np.array([xc, yc]) - np.array(last_pos[trk_id]))
                    speed = dist * fps
                    if speed < 1.0: is_loiter = True
                last_pos[trk_id] = (xc, yc)
                
                in_aoi = is_in_aoi(xc, yc, aoi, width, height)
                threat = calculate_threat_score(in_aoi, is_loiter, name, conf_score)
                
                # Store Trajectory Data
                if trk_id not in trajectories: trajectories[trk_id] = []
                trajectories[trk_id].append((xc, yc, current_time, threat))
                
                # Visuals
                alert_tag = f"🚨 {threat}" if threat > 30 else ""
                labels.append(f"ID:{trk_id} {name} {alert_tag}")
                
                if threat > 50:
                    # Draw high threat circle (Bright Red)
                    cv2.circle(frame, (int(xc), int(yc)), 5, (0, 0, 255), -1) 
                
                all_alerts.append({
                    "time_s": current_time, "object_id": int(trk_id), "object_class": name,
                    "Threat": threat, "Speed": round(speed, 1), "AOI_Breach": in_aoi,
                    "bbox": xyxy.tolist()
                })
            
            # Annotate Frame
            # Note: Supervision API varies by version. This works for recent versions.
            frame = box_annotator.annotate(scene=frame, detections=dets)
            frame = label_annotator.annotate(scene=frame, detections=dets, labels=labels)
            sink.write_frame(frame)
            
            # UI Updates (Throttled to every 3rd frame for performance)
            if frame_idx % 3 == 0:
                frame_place.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Live Feed | Frame {frame_idx}", use_column_width=True)
                
                curr_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                start_time = time.time()
                cpu, ram = get_system_metrics()
                
                kpi_places[0].metric("FPS", f"{curr_fps:.1f}")
                kpi_places[1].metric("Active Objects", len(dets))
                kpi_places[2].metric("CPU / RAM", f"{cpu}% / {ram}%")
                
                if all_alerts:
                    df = pd.DataFrame(all_alerts)
                    # Show most recent alerts
                    log_place.dataframe(df.tail(8)[['time_s', 'object_id', 'object_class', 'Threat', 'Speed']], use_container_width=True)
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx/total_frames, 1.0))
            
            frame_idx += 1
            
    return all_alerts, trajectories

# =================================================================
# 2. PAGE: HOME
# =================================================================
def render_home():
    # SIMULATED TRANSITION ANIMATION
    with st.spinner("Initializing autonomous systems..."):
        time.sleep(0.5) 
    
    # Inject CSS for a high-contrast 'info'/'success' boxes
    st.markdown("""
        <style>
        .stAlert div[data-testid="stInfoContent"] {
            background-color: #2D3035 !important;
            border-left: 8px solid #F0C530 !important;
        }
        .stAlert div[data-testid="stSuccessContent"] {
            background-color: #2D3035 !important;
            border-left: 8px solid #4CAF50 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🛡️ Sentinel AI: Autonomous Surveillance System")
    
    st.markdown("---") 
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <p style="font-size: 18px; color: #F0C530;">
        <b>MISSION:</b> Situational Awareness for Autonomous Systems
        </p>
        
        **Sentinel AI** processes autonomous vehicle feeds to detect anomalies, track threats, and allow operators to **chat with their video data**.
        
        ### 🔥 Core Intelligence Capabilities
        * **💬 Sentinel Chat:** Ask natural language questions about your surveillance footage.
        * **🗺️ Dynamic Heatmaps:** Visualize high-traffic zones and loitering hotspots.
        * **⚡ Predictive Threat Scoring:** Real-time risk assessment (0-100) for every target.
        * **🩺 System Telemetry:** Live monitoring of CPU/RAM for edge-deployment readiness.
        """, unsafe_allow_html=True)
        
        st.info("🚨 **To begin a mission, click 'Live Surveillance' in the top menu.**", icon="🎯")

    with col2:
        st.markdown("### System Status")
        st.metric("System Status", "ONLINE", "Ready")
        
        st.markdown("---")
        
        st.markdown("##### Neural Core Connection")
        if GEMINI_CLIENT:
            st.success("✅ Gemini Neural Core Connected")
        else:
            st.error("❌ Neural Core Disconnected")
            st.caption("Set `GEMINI_API_KEY` in env vars.")

# =================================================================
# 3. PAGE: LIVE SURVEILLANCE
# =================================================================
def render_surveillance():
    # SIMULATED TRANSITION ANIMATION
    with st.spinner("Establishing secure connection..."):
        time.sleep(0.5) 
    
    st.title("🔴 Live Mission Control")
    st.markdown("---")

    # --- SIDEBAR (For Configuration) ---
    st.sidebar.header("⚙️ Mission Config")
    conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, Config.CONFIDENCE_THRESHOLD)
    
    st.sidebar.subheader("🚨 Geofence (AOI)")
    st.sidebar.caption("Normalized coordinates (0-1000)")
    c_x1, c_y1 = st.sidebar.columns(2)
    aoi_x1 = c_x1.number_input("X1", 0, 1000, 200, key="x1")
    aoi_y1 = c_y1.number_input("Y1", 0, 1000, 200, key="y1")
    c_x2, c_y2 = st.sidebar.columns(2)
    aoi_x2 = c_x2.number_input("X2", 0, 1000, 800, key="x2")
    aoi_y2 = c_y2.number_input("Y2", 0, 1000, 800, key="y2")
    
    # Fixed the typo here (aei_x2 -> aoi_x2)
    aoi = [aoi_x1, aoi_y1, aoi_x2, aoi_y2] 
    
    class_names = list(Config.COCO_NAMES.values())
    default_names = [Config.COCO_NAMES[i] for i in Config.DEFAULT_TARGET_CLASSES]
    selected_names = st.sidebar.multiselect("Targets to Track", class_names, default_names)
    target_ids = [k for k, v in Config.COCO_NAMES.items() if v in selected_names]

    # --- UPLOAD ---
    uploaded_file = st.file_uploader("📥 Upload Autonomous Feed (MP4/MOV)", type=['mp4', 'mov'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        st.session_state['video_path'] = output_path 
        
        if st.button("▶️ EXECUTE SURVEILLANCE PROTOCOL", type="primary"):
            st.session_state['analysis_complete'] = False
            
            # Layout
            c_vid, c_log = st.columns([3, 2])
            with c_vid: 
                st.markdown("##### 🛰️ Optical Feed")
                video_place = st.empty()
            with c_log: 
                st.markdown("##### 📝 Live Threat Log")
                log_place = st.empty()
            
            st.markdown("---")
            
            # Metrics
            st.markdown("##### 🩺 System Telemetry & Progress")
            m1, m2, m3 = st.columns(3)
            kpi1, kpi2, kpi3 = m1.empty(), m2.empty(), m3.empty()
            prog_bar = st.progress(0)
            
            model = load_model()
            if model:
                alerts, trajectories = process_video(
                    tfile.name, output_path, model, conf, target_ids, aoi,
                    prog_bar, video_place, log_place, [kpi1, kpi2, kpi3]
                )
                
                # Save State
                if alerts:
                    df = pd.DataFrame(alerts)
                    st.session_state['detailed_json'] = df.to_json(orient='records')
                    st.session_state['alerts_df'] = df
                    st.session_state['trajectories'] = trajectories
                    st.session_state['analysis_complete'] = True
                    st.success("Protocol Complete. Intelligence Ready.")
                else:
                    st.warning("No targets detected.")
                os.unlink(tfile.name)

# =================================================================
# 4. PAGE: FORENSIC REPORTS
# =================================================================
def render_reports():
    # SIMULATED TRANSITION ANIMATION
    with st.spinner("Compiling forensic intelligence..."):
        time.sleep(0.5) 

    # Inject CSS for a high-contrast 'info' box
    st.markdown("""
        <style>
        .stAlert div[data-testid="stInfoContent"] {
            background-color: #2D3035 !important;
            border-left: 8px solid #F0C530 !important;
        }
        .stAlert div[data-testid="stSuccessContent"] {
            background-color: #2D3035 !important;
            border-left: 8px solid #4CAF50 !important;
        }
        .stAlert div[data-testid="stWarningContent"] {
            background-color: #2D3035 !important;
            border-left: 8px solid #FFA500 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🧠 Forensic Intelligence Hub")
    st.markdown("---")

    if not st.session_state.get('analysis_complete'):
        st.info("⚠️ Awaiting Mission Data. Run Live Surveillance first.")
        return

    # Data Loading
    df = st.session_state['alerts_df']
    trajectories = st.session_state['trajectories']

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Sentinel Chat", "🗺️ Heatmaps & Paths", "📊 Stats", "💾 Export"])

    # 1. SENTINEL CHAT (THE KILLER FEATURE)
    with tab1:
        st.subheader("💬 Chat with your Video Data")
        st.caption("Ask questions like: 'How many vehicles were in the restricted zone?', 'Did anyone loiter?', 'What was the maximum threat score?'")
        st.markdown("---")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask Sentinel AI..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing logs..."):
                    # Create a summary context (CSV) to save tokens
                    context = df[['time_s', 'object_class', 'Threat', 'Speed', 'AOI_Breach']].to_csv(index=False)
                    # Limit context size for API
                    response = generate_gemini_response("CHAT", context[:15000], prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # 2. HEATMAPS & PATHS
    with tab2:
        st.subheader("🗺️ Forensics Visualization")
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 🔥 Activity Heatmap (High Traffic Zones)")
            all_x, all_y = [], []
            for tid, points in trajectories.items():
                for p in points:
                    all_x.append(p[0])
                    all_y.append(p[1])
            
            if all_x:
                fig_heat = px.density_heatmap(x=all_x, y=all_y, nbinsx=20, nbinsy=20, title="Movement Density Distribution")
                fig_heat.update_layout(xaxis_title="X", yaxis_title="Y", yaxis=dict(autorange='reversed'), 
                                     coloraxis_colorbar=dict(title="Density"),
                                     plot_bgcolor='#181A1F', paper_bgcolor='#181A1F') # Set plot background to secondary color
                st.plotly_chart(fig_heat, use_container_width=True)
        
        with c2:
            st.markdown("##### 📍 Individual Trajectory Path")
            obj_ids = sorted(list(trajectories.keys()))
            sel_id = st.selectbox("Select Target ID", obj_ids)
            if sel_id:
                data = trajectories[sel_id]
                path_df = pd.DataFrame(data, columns=['x', 'y', 'time', 'threat'])
                fig_path = go.Figure()
                fig_path.add_trace(go.Scatter(x=path_df['x'], y=path_df['y'], mode='lines+markers', 
                                             line=dict(color='#F0C530'), # Use primary color for line
                                             marker=dict(color=path_df['threat'], colorscale='Plasma', size=8, line=dict(width=1, color='White')),
                                             name=f'ID {sel_id}'))
                fig_path.update_layout(title=f"Path ID {sel_id}", yaxis=dict(autorange='reversed'),
                                     plot_bgcolor='#181A1F', paper_bgcolor='#181A1F')
                st.plotly_chart(fig_path, use_container_width=True)

    # 3. STATISTICS & HYPOTHESES
    with tab3:
        st.subheader("🤖 AI Behavioral and Statistical Analysis")
        st.markdown("---")
        c_stat, c_ai = st.columns(2)
        with c_stat:
            st.markdown("##### Threat Score Distribution")
            fig_hist = px.histogram(df, x='Threat', color='object_class', color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_hist.update_layout(plot_bgcolor='#181A1F', paper_bgcolor='#181A1F')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with c_ai:
            st.markdown("##### AI Behavioral Analysis")
            target_id = st.selectbox("Analyze Behavior of ID:", sorted(list(trajectories.keys())), key="hypo_id")
            if st.button("Generate Behavioral Hypothesis", type="primary"):
                with st.spinner("Profiling..."):
                    obj_logs = df[df['object_id'] == target_id].to_json()
                    res = generate_gemini_response("HYPOTHESIZE", obj_logs, user_query=f"Analyze ID {target_id}")
                    st.info(res) 

    # 4. EXPORT
    with tab4:
        st.subheader("💾 Mission Archives")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download JSON Logs", st.session_state['detailed_json'], "logs.json", "application/json", use_container_width=True)
            st.download_button("⬇️ Download CSV Report", df.to_csv().encode('utf-8'), "report.csv", "text/csv", use_container_width=True)
        with c2:
            if os.path.exists(st.session_state.get('video_path', '')):
                with open(st.session_state['video_path'], "rb") as f:
                    st.download_button("⬇️ Download Annotated Evidence (MP4)", f.read(), "evidence.mp4", "video/mp4", use_container_width=True)
            else:
                st.info("Annotated video file not yet available.")


# =================================================================
# 5. MAIN NAVIGATION (USING streamlit-option-menu)
# =================================================================
def main():
    # Inject custom CSS container for the menu (ensures separation from content)
    st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
    
    # Streamlit-Option-Menu (Top Bar) for busy.bar style navigation
    selected_page = option_menu(
        menu_title=None,  # Required
        options=["Home", "Live Surveillance", "Forensic Reports"],
        icons=["house", "broadcast", "cpu"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#0E1117"},
            "icon": {"color": "#F0C530", "font-size": "18px"},
            # Custom styling for the top menu links
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px 10px", "--hover-color": "#181A1F"},
            "nav-link-selected": {"background-color": "#2D3035", "border-bottom": "4px solid #F0C530", "color": "#F0C530"},
        }
    )
    st.markdown('</div>', unsafe_allow_html=True) # Close the injected container

    # Page Rendering Logic
    if selected_page == "Home": 
        render_home()
    elif selected_page == "Live Surveillance": 
        render_surveillance()
    elif selected_page == "Forensic Reports": 
        render_reports()

if __name__ == "__main__":
    main()