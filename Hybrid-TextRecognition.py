import cv2
import easyocr
from paddleocr import PaddleOCR
import numpy as np
import os

os.makedirs(r"Assortment of Small Vision ML-DL Projects\output", exist_ok=True)

input_image_path = r'Assortment of Small Vision ML-DL Projects\Datasets\Bookpage01.jpg'
output_image_path = r"Assortment of Small Vision ML-DL Projects\output\text_rec_hybrid_crop01.jpg"

image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {input_image_path}")

aspect_ratio = image.shape[1] / image.shape[0]
new_width = 2000
new_height = int(new_width / aspect_ratio)
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

easy_reader = easyocr.Reader(['en'], gpu=True)
easy_result = easy_reader.readtext(resized_image)

paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')

drawing_img = resized_image.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
text_color = (0, 255, 0)
box_color = (0, 255, 0)
box_thickness = 2

for entry in easy_result:
    bbox_points = np.array(entry[0], dtype=np.int32)
    x_min, y_min = np.min(bbox_points, axis=0)
    x_max, y_max = np.max(bbox_points, axis=0)
    
    crop = resized_image[y_min:y_max, x_min:x_max]
    
    if crop.size == 0:
        text = entry[1]
    else:
        paddle_res = paddle_ocr.predict(input=crop)
        if paddle_res and paddle_res[0] and len(paddle_res[0]['rec_texts']) > 0:
            all_texts_in_crop = paddle_res[0]['rec_texts']
            text = ' '.join(all_texts_in_crop).strip()
        else:
            text = entry[1]
    
    cv2.polylines(drawing_img, [bbox_points], isClosed=True, color=box_color, thickness=box_thickness)
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_y = max(y_min - 10, text_height + 10)
    
    cv2.rectangle(drawing_img,
                  (x_min - 3, text_y - text_height - 8),
                  (x_min + text_width + 6, text_y + baseline + 3),
                  (0, 0, 0), -1)
    
    cv2.putText(drawing_img, text, (x_min, text_y - 3),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

cv2.imwrite(output_image_path, drawing_img)
cv2.imshow('Hybrid OCR - EasyOCR Boxes + Full PaddleOCR Text', drawing_img)
cv2.waitKey(0)
cv2.destroyAllWindows()