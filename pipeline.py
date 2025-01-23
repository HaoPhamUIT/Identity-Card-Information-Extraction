import json
import logging
import os
import time
from typing import Optional, Dict
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

class DataID:
    def __init__(self):
        self.data: Dict[str, str] = {
            'typeID': '', 'id': '', 'name': '', 'gender': '', 'birth_day': '',
            'expire_date': '', 'issue_date': '', 'issue_place': '', 'nationality': '',
            'origin_location': '', 'recent_location': ''
        }
         
def get_data(i, text, ob):
    field_mapping = {
        4: 'birth_day', 7: 'expire_date', 8: 'gender', 9: 'id',
        10: 'issue_date', 11: 'issue_place', 12: 'name', 13: 'nationality',
        14: 'origin_location', 15: 'recent_location'
    }
    field_name = field_mapping.get(i)
    if field_name:
        ob.data[field_name] = text
    return ob

class Pipeline:
    def __init__(
        self,
        model_detect_path,
        model_text_recognition_config,
        model_text_recognition_path=None,
        device="cpu"
    ):
        # Load yolo
        self.model_detect = YOLO(model_detect_path).to(device)

        # Load vietocr
        self.config_text_recognition = Cfg.load_config_from_name(model_text_recognition_config) # vgg_seq2seq
        if model_text_recognition_path:
            self.config_text_recognition['weights'] = model_text_recognition_path
        self.config_text_recognition['cnn']['pretrained'] = False
        self.config_text_recognition['device'] = device

        self.model_text_recognition = Predictor(self.config_text_recognition)
        
        print("Loaded models!")

    def __call__(
        self,
        image_path: str,
        detect_conf: float = 0.2, # default
        detect_iou: float = 0.4,
        detect_max_det: int = 300, # default
        detect_classes=[1, 3, 4, 7, 8, 9, 12, 13, 14, 15], # Just take 12so_front, 9so_front, birth_day, expire_date, gender, id, name, nationality, origin_location, recent_location
    ):
        img = cv2.imread(image_path)
        if img is None:
            return None

        data = DataID()
        fields = [k for k in detect_classes if k not in [0, 1, 2, 3, 5, 6]]
        field_infos = {k: [] for k in fields}

        detect = self.model_detect.predict(
            img,
            conf=detect_conf,
            iou=detect_iou,
            max_det=detect_max_det,
            classes=detect_classes
        )

        # Loop through yolo results and save to field_infos
        for box in detect[0].boxes:
            idx = int(box.data[0][-1])

            if idx >= 4:
              if idx in fields:
                x1, y1, x2, y2 = map(int, box.data[0][:4])
                img_cut = img[y1:y2, x1:x2]
                img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB) # img_cut must be a RGB image
                large_img = Image.fromarray(img_cut)
                s = self.model_text_recognition.predict(large_img)
                field_infos[idx].append([s, x1, y1, x2, y2])
            else:
                data.data['typeID'] = self.model_detect.names[idx]

        # Merge result in field_infos
        for idx, field_info in field_infos.items():
            if len(field_info) == 0:
                continue
            elif len(field_info) == 1:
                data = get_data(idx, field_info[0][0], data)
            else:
                temp = sorted(field_info, key=lambda item: (item[2], item[1])) # Sort: up to down, left to right
                temp = [k[0] for k in temp]
                temp = ", ".join(temp)
                data = get_data(idx, temp, data)
        return data.data    
if __name__ == "__main__":
    yolo_best_path = "./weights/best.pt"
    vietocr_model_config = "vgg_transformer"
    vietocr_model_path = "./weights/vgg_transformer.pth"
    device = "cpu"
    # device = "cuda:0" # when using gpu
    
    pipeline = Pipeline(
        model_detect_path=yolo_best_path,
        model_text_recognition_config=vietocr_model_config,
        model_text_recognition_path=vietocr_model_path,
        device=device
    )
    image_path = "./samples/21.jpg"
    result = pipeline(image_path=image_path)
    print(result)
    
