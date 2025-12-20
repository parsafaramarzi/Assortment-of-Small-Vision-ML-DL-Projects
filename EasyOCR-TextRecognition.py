import cv2
import easyocr
import numpy as np
import os

os.makedirs(r"Assortment of Small Vision ML-DL Projects\output", exist_ok=True)

input_image_path = r'Assortment of Small Vision ML-DL Projects\Datasets\Bookpage01.jpg'
output_image_path = r"Assortment of Small Vision ML-DL Projects\output\text_rec_easyocr01.jpg"

image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {input_image_path}")

aspect_ratio = image.shape[1] / image.shape[0]
new_width = 2000
new_height = int(new_width / aspect_ratio)
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

reader = easyocr.Reader(['en'], gpu=True)

result = reader.readtext(resized_image)

drawing_img = resized_image.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
text_color = (255, 255, 255)
box_color = (0, 165, 255)
box_thickness = 2

for entry in result:
    bbox = np.array(entry[0], dtype=np.int32)
    text = entry[1]

    cv2.polylines(drawing_img, [bbox], isClosed=True, color=box_color, thickness=box_thickness)

    x_min, y_min = np.min(bbox, axis=0)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_y = max(y_min - 10, text_height + 10)

    cv2.rectangle(drawing_img,
                  (x_min - 3, text_y - text_height - 8),
                  (x_min + text_width + 6, text_y + baseline + 3),
                  (0, 100, 255), -1)

    cv2.putText(drawing_img, text, (x_min, text_y - 3),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

cv2.imwrite(output_image_path, drawing_img)

cv2.imshow('EasyOCR - Orange Theme', drawing_img)
cv2.waitKey(0)
cv2.destroyAllWindows()