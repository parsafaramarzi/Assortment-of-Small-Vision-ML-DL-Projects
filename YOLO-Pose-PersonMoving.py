import imageio
from ultralytics import YOLO
import numpy as np
import cv2

dataset = cv2.VideoCapture("Datasets/personsport04.mp4")
model = YOLO("yolo11x-pose.pt")

success, first_frame = dataset.read()
dataset.set(cv2.CAP_PROP_POS_FRAMES, 0)
h, w = first_frame.shape[:2]
aspect_ratio = w / h
new_h = int(aspect_ratio * 640)
new_w = 640
writer = imageio.get_writer("output/yolov11_pose_estimator_detected.mp4", fps=30, codec='libx264', quality=8)
save_demo_frame = True
frame_count = 0

def draw_pose(image, keypoints_xy, keypoints_conf):
    for kpts, conf in zip(keypoints_xy, keypoints_conf):
        kpts = kpts.cpu().numpy()
        conf = conf.cpu().numpy()
        bone_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                       "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                       "left_wrist", "right_wrist", "left_hip", "right_hip",
                       "left_knee", "right_knee", "left_ankle", "right_ankle"]
        for i, (x, y) in enumerate(kpts):
            if conf[i] > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                label = bone_labels[i] if i < len(bone_labels) else str(i)
                text_pos = (int(x) + 6, int(y) - 6)
                cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)        
        skeleton = [    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                        (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)    ]
        for (start, end) in skeleton:
            if conf[start] > 0.5 and conf[end] > 0.5:
                pt1 = (int(kpts[start][0]), int(kpts[start][1]))
                pt2 = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

while True:
    ret, frame = dataset.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (new_h,new_w))
    results = model(frame, classes=[0])
    keypoints_xy = []
    keypoints_conf = []
    keypoints_xy = results[0].keypoints.xy
    keypoints_conf = results[0].keypoints.conf

    draw_pose(frame, keypoints_xy, keypoints_conf)

    if save_demo_frame and frame_count == 30:
        cv2.imwrite("output/yolov11_pose_estimator_demo_frame.png", frame)
        print("Demo frame saved: output/yolov11_pose_estimator_demo_frame.png")
        save_demo_frame = False
    frame_count += 1
    
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow("Input", frame)
    if cv2.waitKey(1) == 13:
        dataset.release()
        writer.close()
        cv2.destroyAllWindows()
        break

dataset.release()
writer.close()
cv2.destroyAllWindows()