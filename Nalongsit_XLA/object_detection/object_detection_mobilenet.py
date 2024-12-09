import cv2
import numpy as np

# Load the pre-trained MobileNetSSD model
net = cv2.dnn.readNetFromCaffe(
    "./model/deploy.prototxt", "./model/mobilenet_iter_73000.caffemodel"
)

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set the window size
cap.set(3, 640)
cap.set(4, 480)

# Define class names
classNames = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to 300x300 pixels (required by MobileNetSSD)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the network and perform a forward pass to obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections and draw boxes around the objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust the confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            class_name = classNames[class_id]
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = "{}: {:.2f}%".format(class_name, confidence * 100)
            cv2.putText(
                frame,
                text,
                (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Real-time Object Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
