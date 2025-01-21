#
# contains the detection models function
# modules:-
#       - table_detector
#       - cell_detector
#

import os
import io
import cv2
import numpy as np
import onnxruntime
import base64
from typing_extensions import List
from PIL import Image

import torch
from surya.layout import batch_layout_detection
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor
import json

import anthropic

class TableDetector: 
    def __init__(self, model_path, class_labels=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print(f"Successfully loaded ONNX model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the ONNX model: {e}")
        
        model_inputs = self.session.get_inputs()
        print("Model Inputs:", model_inputs)
        
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [batch_size, channels, height, width]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.classes = class_labels if class_labels else {0: 'tables', 1: 'tilted', 2: 'empty'}

    def preprocess(self, image):
        resized = cv2.resize(image, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # CHW format
        batched = np.expand_dims(transposed, axis=0)  # Add batch dimension
        return batched

    def detect_bbox(self, image1, confidence_threshold=0.35, iou_threshold=0.45):
        # Convert PIL Image to OpenCV format
        frame = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        original_height, original_width = frame.shape[:2]
        
        img2 = frame.copy()
        target_height, target_width = original_height, original_width
        scale_factor_x = scale_factor_y = 1
        
        preprocessed = self.preprocess(frame)
        print(f"Preprocessed shape: {preprocessed.shape}")
        
        outputs = self.session.run(None, {self.input_name: preprocessed})
        print(f"Model outputs: {outputs}") 
        
        # Postprocess the outputs to get bounding boxes
        obb_data = self.postprocess(outputs, img2, confidence_threshold, iou_threshold, scale_factor_x, scale_factor_y)

        _, buffer = cv2.imencode('.png', img2)
        base_img_string = base64.b64encode(buffer).decode('utf-8')

        response = {
            "bbox_data": obb_data,
            "actual_image": base_img_string,
            "height": int(img2.shape[0]),
            "width": int(img2.shape[1]),
            "num_tables": int(len(obb_data)),
        }
        return response

    def get_cropped_images(self, image: Image.Image) -> List[Image.Image]:
        # this function crop all the tables from the image and return a list of cropped images
        bbox_result = self.detect_bbox(image)
        bbox_data = bbox_result["bbox_data"]
        cropped_images = []

        for bbox in bbox_data:
            x1, y1, x2, y2 = bbox["xyxy"]
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_image)

        return cropped_images

    def postprocess(self, outputs, img2, confidence_threshold, iou_threshold, scale_factor_x, scale_factor_y):
        img_height, img_width = img2.shape[:2]
        output_array = np.squeeze(outputs[0])

        if output_array.shape[0] < output_array.shape[1]:
            output_array = output_array.transpose()

        num_detections = output_array.shape[0]
        print(f"Number of detections before NMS: {num_detections}")  

        boxes = []
        scores = []
        class_ids = []

        # scaled based on model input size to img2
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        for i in range(num_detections):
            row = output_array[i]
            objectness = row[4]
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores)) 
            confidence = float(class_scores[class_id]) 

            if confidence >= confidence_threshold:
                x, y, width, height = row[0], row[1], row[2], row[3]
                x1 = int((x - width / 2) * x_factor)
                y1 = int((y - height / 2) * y_factor)
                w = int(width * x_factor)
                h = int(height * y_factor)
                
                boxes.append([x1, y1, w, h])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
                
                print(f"Initial bbox {i}: Class ID={class_id}, Confidence={confidence}, Box={x1, y1, w, h}")  

        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
        print(f"Indices after NMS: {indices}")  

        obb_data = []

        if len(indices) > 0:
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = list(indices)
            
            for idx in indices:
                box = boxes[idx]
                class_id = class_ids[idx]
                confidence = scores[idx]
                
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
                #if bbox coordinates -out of img boundary
                # x1 = max(0, x1)
                # y1 = max(0, y1)
                # x2 = min(x2, img_width)
                # y2 = min(y2, img_height)
    
                x = x1 + w / 2
                y = y1 + h / 2

                obb_data.append({
                    "class_id": class_id,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [x, y, w, h]
                })

                
                # self.draw_detections(img2, box, confidence, class_id)
                print(f"Final bbox: class_id={class_id}, confidence={confidence}, bbox={x1, y1, x2, y2}")  
        else:
            print("No detections after NMS.")

        print(f"Number of detections after NMS: {len(obb_data)}")
        return obb_data
    

# module = TableDetector('dynamic_quantized_21.onnx')

img = Image.open("cropped_image_1.png")

# cropped_images = module.get_cropped_images(img)
# print(cropped_images)

# # save all the cropped images
# for i, cropped_image in enumerate(cropped_images):
#     cropped_image.save(f"cropped_image_{i}.png")


class CellDetector:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    
    def get_input_tokens(self, base64_image: str) -> int:
        response = self.client.messages.count_tokens(
            model="claude-3-5-sonnet-20241022",
            system="Extract the text from the image accurately, return only the extracted text.",
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                        "type": "text",
                        "text": "Extract the text from the image accurately, return only the extracted text.",
                        },
                        {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                        }
                    ]
                }
            ], 
        )
        return response.input_tokens
    
    def convert_Image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_cell_results(self, table_image: Image.Image):
        # save the image
        table_image.save("table.png")

        cmd = "python3.10 surya_table_recognition.py table.png --results_dir ."
        # os.system(cmd)
        
        log_file = 'table/results.json'
        json_data = json.load(open(log_file))
        tables = json_data['table']

        rows_count = []
        columns_count = []
        cells_count = []
        input_tokens = []

        for table in tables:
            cells_count.append(len(table['cells']))
            rows_count.append(len(table['rows']))
            columns_count.append(len(table['cols']))

            cells = table['cells']
            for i, cell in enumerate(cells):
                x1, y1, x2, y2 = cell['bbox']
                cropped_img = table_image.crop((x1, y1, x2, y2))
                cropped_img.save(f"cells/cell_{i}.png")
                
                inp_tokens = self.get_input_tokens(self.convert_Image_to_base64(table_image.crop((x1, y1, x2, y2))))
                input_tokens.append(
                    {
                        'cell': i,
                        'width': cropped_img.width,
                        'height': cropped_img.height,
                        'input_tokens': inp_tokens
                    }
                )
        return {
            'cells_count': cells_count,
            'rows_count': rows_count,
            'columns_count': columns_count,
            'input_tokens': input_tokens
        }

cell_detector = CellDetector()
data = cell_detector.get_cell_results(img)

print(data)



