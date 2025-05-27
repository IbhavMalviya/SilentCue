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
        if not success:                                                      # Capturing Single Frames
            print("Failed to grab the frame")
            break
                                

        frame= cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                        # Flipping Frame and Converting to RGB

        results= face_mesh.process(rgb_frame)                                   # Processing

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks: 
                for id, lm in enumerate(face_landmarks.landmark):
                    h,w, _ = frame.shape                                           # Drawing Landmarks
                    x,y= int(lm.x*w), int (lm.y*h)
                    cv2.circle(frame,(x,y),1,(0,255,0),-1)


        cv2.imshow('SilentCue - Face Landmarks', frame)                         # Showing Result


        if cv2.waitKey(1) & 0xFF == 27:                                         # Exit on ESC
                break

cap.release()
cv2.destroyAllWindows()
