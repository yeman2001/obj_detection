import numpy as np
import cv2
import time

image_BGR = cv2.imread("./img.png")

# Scale the image to 500x500 pixels
target_size = (700, 700)
image_BGR = cv2.resize(image_BGR, target_size)

h, w = image_BGR.shape[:2]
with open("./Yolo_model/coco.names") as f:
    labels = [line.strip() for line in f]
network = cv2.dnn.readNetFromDarknet(
    "./Yolo_model/yolov3.cfg", "./Yolo_model/yolov3.weights"
)

# Get the indices of the unconnected output layers
output_layer_indices = network.getUnconnectedOutLayers()

# Get the names of all layers in the network
layers_names_all = network.getLayerNames()

# Ensure output_layer_indices is a list of indices
output_layer_indices = [int(i) for i in output_layer_indices]

# Extract the names of the output layers using the indices
layers_names_output = [layers_names_all[i - 1] for i in output_layer_indices]

blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)
print("Image shape:", image_BGR.shape)
print("Blob shape:", blob.shape)
blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
print(blob_to_show.shape)
network.setInput(blob)
output_from_network = network.forward(layers_names_output)

probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
bounding_boxes = []
confidences = []
class_numbers = []
for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]
        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

counter = 1
if len(results) > 0:
    for i in results.flatten():
        print("Object {0}: {1}".format(counter, labels[int(class_numbers[i])]))
        counter += 1  # Indent this line inside the loop
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        colour_box_current = colours[class_numbers[i]].tolist()
        cv2.rectangle(
            image_BGR,
            (x_min, y_min),
            (x_min + box_width, y_min + box_height),
            colour_box_current,
            2,
        )
        text_box_current = "{}:{:.0f}%".format(
            labels[int(class_numbers[i])], confidences[i] * 100
        )
        cv2.putText(
            image_BGR,
            text_box_current,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            colour_box_current,
            2,
        )

# Display the image with bounding boxes
cv2.imshow("Object Detection", image_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
