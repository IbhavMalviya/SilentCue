import cv2
import mediapipe as mp
import os
from datetime import datetime   


mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
        max_num_faces=1,                                     # To analyze frames
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
)

lip_indices = [
     61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,              # Mouth landmark indices from MediaPipe
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]


key_to_label={  
              ord('y'):'yes',
              ord('n'):'no',                                # Defining target keys
              ord('s'):'stop',                              
              ord('p'):'play',
              }

DATA_DIR= "data"

for label in key_to_label.values():
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)       # Ensure label folders exist

cap=cv2.VideoCapture(2)

while cap.isOpened():   
        ret, frame= cap.read()
        if not ret:
            break
        
        rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result= face_mesh.process(rgb_frame)
        
        if result.multi_face_landmarks:
                face_landmarks= result.multi_face_landmarks[0]
                h, w, _ = frame.shape

                lip_point=[]
                for idx in lip_indices:
                        lm= face_landmarks.landmark[idx]
                        x, y = int (lm.x *w), int (lm.y * h)   
                        lip_point.append((x,y))
                        
                xs,ys = zip(*lip_point)
                min_x, max_x = min(xs), max(xs)                            # Compute bounding box of mouth
                min_y, max_y = min(ys), max(ys)
                
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(w, max_x + padding)                            # Add padding
                min_y = max(0, min_y - padding)
                max_y = min(h, max_y + padding)
                
                mouth_roi= frame[min_y:max_y, min_x:max_x]

                for (x, y) in lip_point:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                resized_mouth = cv2.resize(mouth_roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Mouth Region", resized_mouth)

                
        cv2.imshow("Webcame", frame)
        
        
        key= cv2.waitKey(1) & 0xff
        if key== 27 :
            break
        
        if key in key_to_label:
            label = key_to_label[key]
            cv2.putText(frame, f"Capturing: {label}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
                cv2.putText(frame, "Press y/n/s/p to capture, ESC to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        
      
        if key in key_to_label and result.multi_face_landmarks:
                label= key_to_label[key]
                timestamp= datetime.now(). strftime("%Y%m%d_%H%M%S_%f")
                filename= f"{label}_{timestamp}.png"
                save_path= os.path.join(DATA_DIR, label, filename)
                cv2.imwrite(save_path, mouth_roi)
                print(f"[+] File Saved:{save_path}")
                
cap.release()
cv2.destroyAllWindows()
        