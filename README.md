# SilentCue 🔇🧠

**Silent CUE** is a web-cam based lip reading Machine Learning Model that allows users to execute voice commands "Without Even Speaking Anything", using only their lip movements. Majorly designed for mute environments or people with special needs.

---

## 🚀 Features

- Real-time lip detection using webcam
- Silent command recognition (yes, no, stop, play, etc.)
- Lightweight and demo-ready
- Built using OpenCV, MediaPipe, TensorFlow

---

## 📁 Project Structure
SilentCUE/
├── scripts/ # Python scripts for capture, preprocess, train
├── models/ # Saved trained models
├── data/ # Lip frame data (videos/images)
├── app/ # Demo app (streamlit/web)
├── notebooks/ # Experimentation and tests
├── media/ # Screenshots and visual demos
├── README.md # Project documentation
├── requirements.txt


---

## 🧠 How It Works

1. Open webcam → detect face
2. Use MediaPipe to extract mouth ROI
3. Save cropped frames into `data/`
4. Train a CNN to classify silent words based on lip shape
5. Use in a live app (desktop or web)

---


🔮 Future Ideas
Browser support via TensorFlow.js
Mobile app integration
Add more commands and language support

---
## 🚧 Progress So Far

27/05/25
- Set up webcam capture using OpenCV.
- Integrated MediaPipe Face Mesh to detect 468 facial landmarks in real-time.
- Successfully displayed live face landmark points on webcam feed.
- Fixed common issues like color space conversion (BGR to RGB) and key event handling.

**But processing the entire 468 face landmarks in real-time will be a heavy computational task. Thus we will only extract the mouth region to extract dataset.**
- Ready to extract and save the mouth region for lip-reading dataset collection.




---


🧑‍💻 Author
Made with ❤️ by Ibhav Malviya
🌐 GitHub: https://github.com/IbhavMalviya

