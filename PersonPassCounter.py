import csv
import imageio
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import numpy as np
import cv2
import csv

track_history = defaultdict(lambda : [])
id_passcount_dict = {}
id_position_dict = {}
videocapture = cv2.VideoCapture('Datasets/crowd06.mp4')
model = YOLO('yolo11n.pt')
success , frame = videocapture.read()
if success == True:
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_width = 1200  
    new_height = int(new_width / aspect_ratio)
videocapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
writer = imageio.get_writer("output/yolov_person_passthrough_counter.mp4", fps=30, codec='libx264', quality=8)

def write_id_records_csv(id_passcount_dict):
    output_filename = 'output/id_records.csv'
    header = ['Person_ID', 'Count_of_Passes']

    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for id, count in id_passcount_dict.items():
                writer.writerow([id, count])

while videocapture.isOpened() == True:
    success , frame = videocapture.read()
    if success == False:
        break
    frame = cv2.resize(frame , (new_width, new_height))  
    results = model.track(frame, persist = True)
    if results[0].boxes.id is None:
        continue
    boxes = results[0].boxes.xyxy.tolist()
    names = results[0].names
    clss = results[0].boxes.cls.tolist()
    track_ids = results[0].boxes.id.int().tolist()
    counted_current_ids = 0

    annotator = Annotator(frame, line_width = 2)
    target_box_y1, target_box_y2 = int(new_height/2), int(new_height)
    target_box_x1, target_box_x2 = int(new_width/2 - new_width/10), int(new_width/2 + new_width/10)
    annotator.box_label( (target_box_x1,target_box_y1,target_box_x2,target_box_y2), label='Target Zone', color = (255,0,0))

    for i, box in enumerate(boxes):
        x1,y1, x2, y2 = box
        name = names[clss[i]]
        color = colors(clss[i])
        track_id = track_ids[i]
        track = track_history[track_id]
        midpoint_x = x1+(x2-x1)/2
        midpoint_y = y1+(y2-y1)/2
        prev_state = id_position_dict.get(track_id, "out")
        current_state = prev_state
        track.append((midpoint_x,midpoint_y))
        points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
        if midpoint_x > target_box_x1 and midpoint_x < target_box_x2 and midpoint_y > target_box_y1 and midpoint_y < target_box_y2 and name == 'person':
            annotator.box_label( (x1,y1,x2,y2), label=f'ID: {track_id} IN', color = (0,255,0))
            counted_current_ids += 1
            current_state = "in"
            if prev_state == "out":
                id_passcount_dict[track_id] = id_passcount_dict.get(track_id, 0) + 1
        else:
            if name == 'person':
                annotator.box_label( (x1,y1,x2,y2), label=f'ID: {track_id} OUT', color = (0,100,255))
                current_state = "out"
        id_position_dict[track_id] = current_state

    count_text = f'Persons in Target Zone: {counted_current_ids}'
    maxwidth = len(count_text) * 10
    cv2.rectangle(frame, (20,20), (120+maxwidth,60), (50,50,50), -1)
    cv2.putText(frame, count_text, org=(30,50), color = (100,255,100), fontScale=0.8, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)
    
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow("Input", frame)
    if cv2.waitKey(1) == 13:
        videocapture.release()
        writer.close()
        cv2.destroyAllWindows()
        break

print(f"track history: {track_history}")
print(f"id pass count dict: {id_passcount_dict}")
print(f"id position dict: {id_position_dict}")

write_id_records_csv(id_passcount_dict)

videocapture.release()
writer.close()
cv2.destroyAllWindows()