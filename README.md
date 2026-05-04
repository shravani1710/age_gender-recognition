# 🧠 Age & Gender Detector — Streamlit App

Multi-task deep learning app for simultaneous **gender recognition** and **age estimation** from facial images.  
Built with OpenCV DNN + Levi–Hassner Caffe models, deployed via Streamlit Community Cloud.

---

## 🚀 Live Demo
> Deploy your own: [![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## 📁 Project Structure

```
age-gender-streamlit/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
├── .gitignore              # Excludes large model files
└── README.md
```

> **Note:** Model files (~50 MB total) are **NOT committed** to Git.  
> They are auto-downloaded from GitHub/OpenCV on first run.

---

## 🛠️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/age-gender-streamlit.git
cd age-gender-streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. **Push this repo to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/age-gender-streamlit.git
   git push -u origin main
   ```

2. **Go to** → [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with GitHub

4. Click **"New app"**

5. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `YOUR_USERNAME/age-gender-streamlit` |
   | Branch | `main` |
   | Main file path | `app.py` |

6. Click **"Deploy!"** — done ✅

> First load takes ~2 min while models download. Subsequent loads are instant (cached).

---

## 🎯 Features

| Feature | Detail |
|---|---|
| Face Detection | OpenCV SSD (MobileNet-based) |
| Gender | Male / Female (Levi & Hassner, 2015) |
| Age Buckets | 8 groups: 0–2, 4–6, 8–12, 15–20, 25–32, 38–43, 48–53, 60+ |
| Input | Upload image OR webcam snapshot |
| Output | Annotated image + confidence scores + download |
| Threshold | Adjustable confidence slider |

---

## 📚 Model Credits

- Levi, G., & Hassner, T. (2015). *Age and Gender Classification Using Convolutional Neural Networks.* CVPR Workshops.
- OpenCV Face Detector: [opencv/opencv](https://github.com/opencv/opencv)
- Pre-trained weights: [spmallick/learnopencv](https://github.com/spmallick/learnopencv)
