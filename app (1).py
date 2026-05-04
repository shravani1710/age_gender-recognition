import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #00ff9d, #00c3ff, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub { color: #888; font-size: 1rem; margin-bottom: 2rem; letter-spacing: 0.05em; }
.result-card { background: #13131f; border: 1px solid #2a2a3d; border-radius: 16px; padding: 1.5rem; margin-top: 1rem; }
.face-badge { display: inline-block; background: #1a1a2e; border: 1px solid #00ff9d44; color: #00ff9d; font-family: 'Space Mono', monospace; font-size: 0.75rem; padding: 4px 12px; border-radius: 20px; margin: 4px; }
.gender-male   { border-color: #00c3ff44; color: #00c3ff; }
.gender-female { border-color: #ff6eb444; color: #ff6eb4; }
.stButton > button { background: linear-gradient(135deg, #00ff9d22, #00c3ff22); border: 1px solid #00ff9d55; color: #00ff9d; font-family: 'Space Mono', monospace; font-size: 0.85rem; border-radius: 8px; padding: 0.5rem 1.5rem; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #00ff9d44, #00c3ff44); border-color: #00ff9d; color: #fff; }
.sidebar-info { background: #13131f; border: 1px solid #2a2a3d; border-radius: 12px; padding: 1rem; font-size: 0.85rem; color: #999; line-height: 1.7; }
.status-dot { width:8px; height:8px; border-radius:50%; background:#00ff9d; display:inline-block; margin-right:6px; box-shadow:0 0 6px #00ff9d; }
[data-testid="stSidebar"] { background: #0d0d1a; border-right: 1px solid #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─── Model URLs (primary + fallbacks) ────────────────────────────────────────
_B1 = "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master"
_B2 = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master"
_B3 = "https://raw.githubusercontent.com/habom2310/People-tracking-with-Age-and-Gender-detection/master/age_gender_models"

MODEL_FILES = {
    "opencv_face_detector.pbtxt": [
        f"{_B1}/opencv_face_detector.pbtxt",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
    ],
    "opencv_face_detector_uint8.pb": [
        f"{_B1}/opencv_face_detector_uint8.pb",
    ],
    "age_deploy.prototxt": [
        f"{_B1}/age_deploy.prototxt",
        f"{_B2}/age_net_definitions/deploy.prototxt",
        f"{_B3}/age_deploy.prototxt",
    ],
    "age_net.caffemodel": [
        f"{_B1}/age_net.caffemodel",
        f"{_B2}/models/age_net.caffemodel",
        f"{_B3}/age_net.caffemodel",
    ],
    "gender_deploy.prototxt": [
        f"{_B1}/gender_deploy.prototxt",
        f"{_B2}/gender_net_definitions/deploy.prototxt",
        f"{_B3}/gender_deploy.prototxt",
    ],
    "gender_net.caffemodel": [
        f"{_B1}/gender_net.caffemodel",
        f"{_B2}/models/gender_net.caffemodel",
        f"{_B3}/gender_net.caffemodel",
    ],
}

MODEL_DIR = "models"

# ─── Download + Load ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_and_load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    progress = st.progress(0, text="Initialising models…")
    files = list(MODEL_FILES.items())

    for idx, (fname, urls) in enumerate(files):
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath):
            progress.progress(idx / len(files), text=f"Downloading {fname}…")
            downloaded = False
            last_err = ""
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, fpath)
                    downloaded = True
                    break
                except Exception as e:
                    last_err = str(e)
                    if os.path.exists(fpath):
                        os.remove(fpath)
            if not downloaded:
                progress.empty()
                st.error(
                    f"❌ Could not download **{fname}**  \n"
                    f"Last error: `{last_err}`  \n\n"
                    f"**Manual fix:** Download from `{urls[0]}` and place it in a `models/` folder next to `app.py`."
                )
                return None, None, None
        progress.progress((idx + 1) / len(files), text=f"✓ {fname}")

    progress.empty()

    faceNet   = cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb"),
        os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt"))
    ageNet    = cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "age_net.caffemodel"),
        os.path.join(MODEL_DIR, "age_deploy.prototxt"))
    genderNet = cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "gender_net.caffemodel"),
        os.path.join(MODEL_DIR, "gender_deploy.prototxt"))
    return faceNet, ageNet, genderNet

# ─── Detection ────────────────────────────────────────────────────────────────
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST    = ['0–2', '4–6', '8–12', '15–20', '25–32', '38–43', '48–53', '60+']
GENDER_LIST = ['Male', 'Female']

def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] > conf_threshold:
            x1 = int(dets[0, 0, i, 3] * w); y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w); y2 = int(dets[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
    return boxes

def analyse_image(frame, faceNet, ageNet, genderNet, padding=20, conf=0.7):
    result_img = frame.copy()
    boxes = detect_faces(faceNet, frame, conf)
    results = []
    for box in boxes:
        x1, y1, x2, y2 = box
        face = frame[
            max(0, y1-padding): min(y2+padding, frame.shape[0]-1),
            max(0, x1-padding): min(x2+padding, frame.shape[1]-1)
        ]
        if face.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gp = genderNet.forward()[0]
        gender = GENDER_LIST[gp.argmax()]
        g_conf = float(gp.max())

        ageNet.setInput(blob)
        ap = ageNet.forward()[0]
        age = AGE_LIST[ap.argmax()]
        a_conf = float(ap.max())

        results.append({"box": box, "gender": gender, "age": age,
                         "gender_conf": g_conf, "age_conf": a_conf})

        color = (0, 196, 255) if gender == "Male" else (255, 110, 180)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{gender}, {age}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result_img, (x1, y1-th-14), (x1+tw+10, y1), color, -1)
        cv2.putText(result_img, label, (x1+5, y1-6),
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
• Face: OpenCV SSD (MobileNet)<br>
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
    st.stop()

st.success("✅ Models ready", icon="🧠")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Snapshot"])

def show_results(result_bgr, results):
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, use_container_width=True)
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if not results:
        st.warning("No faces detected — try lowering the confidence threshold.")
    else:
        st.markdown(f"**{len(results)} face(s) detected**")
        cols = st.columns(min(len(results), 4))
        for i, r in enumerate(results):
            gc = "gender-male" if r["gender"] == "Male" else "gender-female"
            with cols[i % len(cols)]:
                st.markdown(
                    f"<span class='face-badge'>Face #{i+1}</span><br>"
                    f"<span class='face-badge {gc}'>{r['gender']} {r['gender_conf']*100:.0f}%</span><br>"
                    f"<span class='face-badge'>Age {r['age']} · {r['age_conf']*100:.0f}%</span>",
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Download button
    result_pil = Image.fromarray(result_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button("⬇ Download Result", buf.getvalue(), "result.png", "image/png")

with tab1:
    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        col_orig, col_res = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(img_pil, use_container_width=True)
        with col_res:
            st.markdown("**Detected**")
            with st.spinner("Analysing…"):
                result_bgr, results = analyse_image(frame_bgr, faceNet, ageNet, genderNet, padding, conf_threshold)
            show_results(result_bgr, results)

with tab2:
    st.info("📸 Take a photo with your webcam and analyse it instantly.")
    webcam_img = st.camera_input("Take a snapshot")
    if webcam_img:
        img_pil = Image.open(webcam_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        with st.spinner("Analysing…"):
            result_bgr, results = analyse_image(frame_bgr, faceNet, ageNet, genderNet, padding, conf_threshold)
        show_results(result_bgr, results)
