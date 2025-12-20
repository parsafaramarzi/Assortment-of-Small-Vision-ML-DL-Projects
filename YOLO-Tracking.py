from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import numpy as np
import cv2
import os
import imageio

os.makedirs('output', exist_ok=True)

track_history = defaultdict(lambda : [])

video_path = 'Datasets/personsport03.mp4'
videocapture = cv2.VideoCapture(video_path)

fps = videocapture.get(cv2.CAP_PROP_FPS)

new_width = 1200
new_height = 850
output_size = (new_width, new_height)

output_path = 'output/Yolo_Tracking.mp4'

model = YOLO('yolo11x.pt')

with imageio.get_writer(output_path, fps=fps, codec='libx264') as video_writer:

    while videocapture.isOpened() == True:
        success , frame = videocapture.read()
        if success == False:
            break
            
        frame = cv2.resize(frame , output_size) 
        
        results = model.track(frame, persist = True, verbose=False) 
        
        boxes = results[0].boxes.xyxy.tolist()
        clss = results[0].boxes.cls.tolist()
        names = results[0].names
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
        else:
            track_ids = []
        
        annotator = Annotator(frame, line_width = 2)

        for i, box in enumerate(boxes):
            x1,y1, x2, y2 = box
            name = names[clss[i]]
            color = colors(clss[i])
            
            if track_ids and i < len(track_ids):
                track_id = track_ids[i]
                label = f"{name} {track_id}"

                track = track_history[track_id]
                center_x = (x1+x2)/2
                center_y = (y1+y2)/2
                track.append((center_x, center_y))

                if len(track) > 50:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [points], isClosed=False, color = (255,0,0), thickness= 2)
                
            else:
                label = name

            if name == 'person':
                annotator.box_label( (x1,y1,x2,y2), label=label, color = (0,100,255))
            else:
                annotator.box_label( (x1,y1,x2,y2), label=label, color = color)
                
        # --- FIX: Convert BGR (OpenCV) to RGB (ImageIO) ---
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        video_writer.append_data(frame)

        # To keep the live display window correct (as cv2.imshow expects BGR), 
        # you need to convert it back, or just use the original BGR frame for display:
        # cv2.imshow('myimage', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Or more simply, since the BGR frame is already available *before* the conversion for imageio:
        cv2.imshow('myimage', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) == 13:
            break

videocapture.release()
cv2.destroyAllWindows()