import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os
import tempfile

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00ff9d, #00c3ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.result-card {
    background: #13131f;
    border: 1px solid #2a2a3d;
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.face-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #00ff9d44;
    color: #00ff9d;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 4px;
}

.gender-male   { border-color: #00c3ff44; color: #00c3ff; }
.gender-female { border-color: #ff6eb444; color: #ff6eb4; }

.metric-box {
    background: #0d0d1a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    border-left: 3px solid #00ff9d;
    margin-bottom: 0.8rem;
}

.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #666;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8e8f0;
}

div[data-testid="stFileUploader"] {
    background: #13131f;
    border: 2px dashed #2a2a3d;
    border-radius: 16px;
    padding: 1rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #00ff9d55;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff9d22, #00c3ff22);
    border: 1px solid #00ff9d55;
    color: #00ff9d;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00ff9d44, #00c3ff44);
    border-color: #00ff9d;
    color: #fff;
}

.sidebar-info {
    background: #13131f;
    border: 1px solid #2a2a3d;
    border-radius: 12px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #999;
    line-height: 1.7;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ff9d;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px #00ff9d;
}

[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #1a1a2e;
}
</style>
""", unsafe_allow_html=True)

# ─── Model Download URLs ──────────────────────────────────────────────────────
MODEL_FILES = {
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_net.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
}

MODEL_DIR = "models"


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_and_load_models():
    """Download models if missing, then load into memory."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    progress = st.progress(0, text="Initialising models…")
    files = list(MODEL_FILES.items())

    for idx, (fname, url) in enumerate(files):
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            progress.progress((idx) / len(files), text=f"Downloading {fname}…")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                st.error(f"Failed to download {fname}: {e}")
                return None, None, None
        progress.progress((idx + 1) / len(files), text=f"Loaded {fname}")

    progress.empty()

    faceNet   = cv2.dnn.readNet(os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb"),
                                os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt"))
    ageNet    = cv2.dnn.readNet(os.path.join(MODEL_DIR, "age_net.caffemodel"),
                                os.path.join(MODEL_DIR, "age_deploy.prototxt"))
    genderNet = cv2.dnn.readNet(os.path.join(MODEL_DIR, "gender_net.caffemodel"),
                                os.path.join(MODEL_DIR, "gender_deploy.prototxt"))
    return faceNet, ageNet, genderNet


# ─── Detection Logic ─────────────────────────────────────────────────────────
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST    = ['0–2', '4–6', '8–12', '15–20', '25–32', '38–43', '48–53', '60+']
GENDER_LIST = ['Male', 'Female']


def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
    return boxes


def analyse_image(frame, faceNet, ageNet, genderNet, padding=20, conf=0.7):
    result_img = frame.copy()
    boxes = detect_faces(faceNet, frame, conf)
    results = []

    for box in boxes:
        x1, y1, x2, y2 = box
        face = frame[
            max(0, y1 - padding): min(y2 + padding, frame.shape[0] - 1),
            max(0, x1 - padding): min(x2 + padding, frame.shape[1] - 1)
        ]
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gender_preds = genderNet.forward()[0]
        gender = GENDER_LIST[gender_preds.argmax()]
        gender_conf = float(gender_preds.max())

        ageNet.setInput(blob)
        age_preds = ageNet.forward()[0]
        age = AGE_LIST[age_preds.argmax()]
        age_conf = float(age_preds.max())

        results.append({
            "box": box, "gender": gender, "age": age,
            "gender_conf": gender_conf, "age_conf": age_conf
        })

        # Draw on image
        color = (0, 196, 255) if gender == "Male" else (255, 110, 180)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{gender}, {age}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result_img, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
        cv2.putText(result_img, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2, cv2.LINE_AA)

    return result_img, results


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Detection Confidence", 0.3, 1.0, 0.7, 0.05)
    padding = st.slider("Face Padding (px)", 0, 50, 20)
    st.markdown("---")
    st.markdown("""
<div class='sidebar-info'>
<span class='status-dot'></span><b>Models</b><br>
• Face: OpenCV DNN SSD<br>
• Age & Gender: Caffe (Levi & Hassner)<br><br>
<span class='status-dot'></span><b>Age Buckets</b><br>
0–2 · 4–6 · 8–12 · 15–20<br>
25–32 · 38–43 · 48–53 · 60+<br><br>
Models auto-download on first run (~50 MB).
</div>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Age & Gender Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Multi-task deep learning · OpenCV DNN · Levi–Hassner model</div>", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    faceNet, ageNet, genderNet = download_and_load_models()

if faceNet is None:
    st.error("❌ Could not load models. Check your internet connection.")
    st.stop()

st.success("✅ Models ready", icon="🧠")

# ─── Upload / Webcam Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Snapshot"])

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        frame = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        col_orig, col_res = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(img_pil, use_container_width=True)

        with st.spinner("Analysing…"):
            result_bgr, results = analyse_image(frame_bgr, faceNet, ageNet, genderNet, padding, conf_threshold)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        with col_res:
            st.markdown("**Detected**")
            st.image(result_rgb, use_container_width=True)

        # Results summary
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        if not results:
            st.warning("No faces detected. Try lowering the confidence threshold.")
        else:
            st.markdown(f"**{len(results)} face(s) detected**")
            cols = st.columns(min(len(results), 4))
            for i, r in enumerate(results):
                gc = "gender-male" if r["gender"] == "Male" else "gender-female"
                with cols[i % len(cols)]:
                    st.markdown(f"""
<span class='face-badge'>Face #{i+1}</span><br>
<span class='face-badge {gc}'>{r['gender']} {r['gender_conf']*100:.0f}%</span><br>
<span class='face-badge'>Age {r['age']} · {r['age_conf']*100:.0f}%</span>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download
        result_pil = Image.fromarray(result_rgb)
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("⬇ Download Result", buf.getvalue(), "result.png", "image/png")

# ── Tab 2: Webcam ─────────────────────────────────────────────────────────────
with tab2:
    st.info("📸 Take a photo with your webcam and analyse it instantly.")
    webcam_img = st.camera_input("Take a snapshot")

    if webcam_img:
        img_pil = Image.open(webcam_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Analysing…"):
            result_bgr, results = analyse_image(frame_bgr, faceNet, ageNet, genderNet, padding, conf_threshold)

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Detection result", use_container_width=True)

        if results:
            for i, r in enumerate(results):
                gc = "gender-male" if r["gender"] == "Male" else "gender-female"
                st.markdown(f"""
<span class='face-badge'>Face #{i+1}</span>
<span class='face-badge {gc}'>{r['gender']} ({r['gender_conf']*100:.0f}%)</span>
<span class='face-badge'>Age {r['age']} ({r['age_conf']*100:.0f}%)</span>
""", unsafe_allow_html=True)
        else:
            st.warning("No faces detected.")
