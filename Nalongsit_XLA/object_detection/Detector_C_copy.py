import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

net = cv2.dnn.readNet("./Yolo_model/yolov3.weights", "./Yolo_model/yolov3.cfg")
classes = []
with open("./Yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

file_in = open("TOANCO_IN.txt", "r")
file_out = open("TOANCO_OUT.txt", "w")

# Đọc danh sách các lớp quan tâm từ dòng đầu tiên của tệp
interested_classes = file_in.readline()
interested_classes = interested_classes.split()
print("Xem nội dung file:", interested_classes)

# Lặp qua các tên trong danh sách
for class_name in interested_classes:
    # Thực hiện xử lý với mỗi tên, ví dụ: in ra tên và ghi vào tệp output
    print("Class Name:", class_name)
    file_out.write(class_name + "\n")

# Đóng các tệp sau khi hoàn thành
file_in.close()
file_out.close()


# interested_classes = [
#     "person",
#     "bicycle",
#     "car",
#     "motorbike",
#     "aeroplane",
#     "bus",
#     "train",
#     "truck",
#     "boat",
#     "traffic light",
#     "fire hydrant",
#     "stop sign",
#     "parking meter",
#     "bench",
#     "bird",
#     "cat",
#     "dog",
#     "horse",
#     "sheep",
#     "cow",
#     "elephant",
#     "bear",
#     "zebra",
#     "giraffe",
#     "backpack",
#     "umbrella",
#     "handbag",
#     "tie",
#     "suitcase",
#     "frisbee",
#     "skis",
#     "snowboard",
#     "sports ball",
#     "kite",
#     "baseball bat",
#     "baseball glove",
#     "skateboard",
#     "surfboard",
#     "tennis racket",
#     "bottle",
#     "wine glass",
#     "cup",
#     "fork",
#     "knife",
#     "spoon",
#     "bowl",
#     "banana",
#     "apple",
#     "sandwich",
#     "orange",
#     "broccoli",
#     "carrot",
#     "hot dog",
#     "pizza",
#     "donut",
#     "cake",
#     "chair",
#     "sofa",
#     "potted plant",
#     "bed",
#     "dining table",
#     "toilet",
#     "tv monitor",
#     "laptop",
#     "mouse",
#     "remote",
#     "keyboard",
#     "cell phone",
#     "microwave",
#     "oven",
#     "toaster",
#     "sink",
#     "refrigerator",
#     "book",
#     "clock",
#     "vase",
#     "scissors",
#     "teddy bear",
#     "hair drier",
#     "toothbrush",
# ]

layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)
moved_objects = []

# Create a folder to save images
image_folder = "images_C_copy"
os.makedirs(image_folder, exist_ok=True)

# Capture a photo every 1 second
capture_interval = 1  # in seconds
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if label.lower() in interested_classes:
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    image_filename = f"{label}_{timestamp}_{i}.png"
                    image_path = os.path.join(image_folder, image_filename)
                    cv2.imwrite(image_path, frame[y : y + h, x : x + w])
                    moved_objects.append(
                        {
                            "Class": label,
                            "Confidence": confidence,
                            "StartX": x,
                            "StartY": y,
                            "EndX": x + w,
                            "EndY": y + h,
                            "Timestamp": timestamp,
                            "ImagePath": image_path,
                        }
                    )
                    last_capture_time = current_time

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        df = pd.DataFrame(moved_objects)
        df.to_csv("moved_objects_C_copy.csv", index=False)
        break

cap.release()
cv2.destroyAllWindows()
