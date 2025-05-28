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


## 🔮 Future Ideas
Browser support via TensorFlow.js
Mobile app integration
Add more commands and language support

---
## 🚧 Progress So Far

27/05/25 

Capturing Phase
- Set up webcam capture using OpenCV.
- Integrated MediaPipe Face Mesh to detect 468 facial landmarks in real-time.
- Successfully displayed live face landmark points on webcam feed.
- Fixed common issues like color space conversion (BGR to RGB) and key event handling.

**But processing the entire 468 face landmarks in real-time will be a heavy computational task. Thus we will only extract the mouth region to extract dataset.**
- Ready to extract and save the mouth region for lip-reading dataset collection.

**Why extact do we need only mouth frame?**
We need only the mouth region to feed into the model — not the full face as that would be waste of resources.

28/05/2025

Data Collection Phase
- To detect mouth and face clearly.
- To save cropped mouth images and label them correctly for words like (Yes, No, Play, Stop, etc.)
- To write a python script that does the data collection and save it in data folder.
Data_Collection.py is the script that collects and stores the images as png file in different folders to train the model.


Data Processing Phase
- To process the collected data using numpy library.
- Resizing each image captured to a uniform size.
- Normilizing pixel values on 0-1 Scale.
- Labelling the data correctly.
- Splitting the data into training and testing sets
Prepare_Dataset.py is the script created to tackle the above problems by using skit-learn from tensorflow and numpy to manage data.
---


🧑‍💻 Author
Made with ❤️ by Ibhav Malviya
🌐 GitHub: https://github.com/IbhavMalviya

