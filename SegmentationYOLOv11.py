import imageio
from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("yolo11x-seg.pt")
video = cv2.VideoCapture("Datasets/crowd06.mp4")
Alpha = 0.5

success, first_frame = video.read()
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
h, w = first_frame.shape[:2]
aspect_ratio = w / h
new_h = int(aspect_ratio * 640)
new_w = 640
writer = imageio.get_writer("output/yolov11_Segmentation.mp4", fps=30, codec='libx264', quality=8)
save_demo_frame = True
frame_count = 0

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(0,255), k=3) for _ in classes_ids]

while video.isOpened():
    ret, img = video.read()
    if ret == False: break

    frame = cv2.resize(img, (new_h,new_w))
    overlay = frame.copy()

    yolo_classes = list(model.names.values())
    results = model.predict(frame)
    labels_to_draw = []

    for result in results:
        for box, mask, cls in zip(result.boxes, result.masks.xy, result.boxes.cls):
            points = np.int32([mask])
            cls_int = int(cls.item())

            color_number = classes_ids.index(cls_int)

            cv2.fillPoly(frame, points, colors[color_number])
            cv2.polylines(frame, points, isClosed=True, color=colors[color_number], thickness=2)

            label = model.names[cls_int]
            text_pos = (int(points[0][0][0]), int(points[0][0][1]) - 10)
            labels_to_draw.append((label, text_pos, colors[color_number]))
    frame = cv2.addWeighted(overlay, Alpha, frame, 1 - Alpha, 0)

    for label, text_pos, color in labels_to_draw:
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if save_demo_frame and frame_count == 30:
        cv2.imwrite("output/yolov11_Segmentation_demo_frame.png", frame)
        print("Demo frame saved: output/yolov11_Segmentation_demo_frame.png")
        save_demo_frame = False
    frame_count += 1

    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow("Input", frame)
    if cv2.waitKey(1) == 13:
        video.release()
        writer.close()
        cv2.destroyAllWindows()
        break

video.release()
writer.close()
cv2.destroyAllWindows()