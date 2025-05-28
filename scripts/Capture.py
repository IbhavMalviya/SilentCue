import cv2 
import numpy as np
import mediapipe as mp
import os 

mp_face_mesh= mp.solutions.face_mesh
face_mesh= mp_face_mesh.FaceMesh(static_image_mode=False, 
                                max_num_faces=1,                            # Face Detection using FaceMesh
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


cap=cv2.VideoCapture(0)     
while cap.isOpened():
        success, frame = cap.read()
        if not success:                                                         # Capturing Single Frames
            print("Failed to grab the frame")
            break
                                

        frame= cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                        # Flipping Frame and Converting to RGB

        results= face_mesh.process(rgb_frame)                                   # Processing

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks: 
                
                    full_lip_indices = [
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,              # Mouth landmark indices from MediaPipe
                    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
                    ]
                    
                    h, w, _ = frame.shape
                    lip_points = []
                    for idx in full_lip_indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            lip_points.append((x, y))
                            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  

                   
                    xs, ys = zip(*lip_points)
                    min_x, max_x = min(xs), max(xs)                            # Compute bounding box of mouth
                    min_y, max_y = min(ys), max(ys)

                    
                    padding = 10
                    min_x = max(0, min_x - padding)
                    max_x = min(w, max_x + padding)                            # Add padding
                    min_y = max(0, min_y - padding)
                    max_y = min(h, max_y + padding)

                  
                    mouth_roi = frame[min_y:max_y, min_x:max_x]                 # Crop mouth region

                    
        cv2.imshow("Mouth Region", mouth_roi)                                   # Show mouth region in a separate window


        cv2.imshow('SilentCue - Face Landmarks', frame)                         # Showing Result


        if cv2.waitKey(1) & 0xFF == 27:                                         # Exit on ESC
                break

cap.release()
cv2.destroyAllWindows()
