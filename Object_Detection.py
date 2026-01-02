import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
    
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
tracker = DeepSort(max_age=30)

while True:
    ret, frame =cap.read()
    
    if not ret:
        break

    results = model(frame)[0]
    detections=[]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        w = x2 - x1
        h = y2 - y1

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detections.append(([x1, y1, w, h], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
             continue
        
        track_id=track.track_id
        l,t,r,b =track.to_ltrb()
        x1 = int(l)
        y1 = int(t)
        x2 = int(r)
        y2 = int(b)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"ID: {track_id}, label: {label}",(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

    cv2.imshow("Object Detection And Tracking", frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()

cv2.destroyAllWindowa()
