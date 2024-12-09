# cv2: This is the OpenCV library for computer vision.
# numpy: A library for numerical operations in Python.
import cv2
import numpy as np

# Dòng này đọc mô hình YOLO (Bạn chỉ nhìn một lần) được đào tạo trước với trọng số và cấu hình của nó.
net = cv2.dnn.readNet("./Yolo_model/yolov3.weights", "./Yolo_model/yolov3.cfg")

# Khối này đọc tên lớp từ một tệp (coco.names) và lưu trữ chúng trong danh sách.
classes = []
with open("./Yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# layer_names: Lấy tên của các lớp đầu ra chưa được kết nối của mạng.
# color: Tạo màu sắc ngẫu nhiên cho mỗi lớp.
layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)  # 0 for default camera, change if using a different source


# cap.read(): Chụp khung hình từ nguồn video.
# frame.shape: Truy xuất kích thước (chiều cao, chiều rộng, kênh) của khung.
while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # cv2.dnn.blobFromImage: Xử lý trước khung cho mô hình YOLO.

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    # net.setInput(blob): Đặt đầu vào của mạng thành khung được xử lý trước.
    net.setInput(blob)
    # net.forward(layer_names): Chuyển tiếp pass để lấy đầu ra từ mạng YOLO.
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    # for out in outs:: Đây là vòng lặp đầu tiên, duyệt qua các đầu ra của mô
    for out in outs:
        # for detection in out:: Đây là vòng lặp thứ hai, duyệt qua các phát hiện trong mỗi đầu ra.
        # Mỗi phát hiện chứa thông tin như tọa độ của hộp giới hạn, điểm số của từng lớp.
        for detection in out:
            # scores = detection[5:]: Lấy các điểm số tương ứng với mỗi lớp. Trong mô hình YOLO,
            # thông thường mỗi phát hiện sẽ có các thông tin như tọa độ của hộp giới hạn và điểm số của các lớp.
            # scores ở đây là một mảng chứa điểm số của từng lớp.
            scores = detection[5:]

            # class_id = np.argmax(scores): Xác định lớp có điểm số cao nhất cho phát hiện hiện tại.
            # np.argmax được sử dụng để trả về chỉ số của lớp có điểm số lớn nhất.
            class_id = np.argmax(scores)

            # confidence = scores[class_id]: Lấy điểm số của lớp có điểm số cao nhất.
            # Điểm số này thường thể hiện độ chắc chắn của mô hình rằng đối tượng được phát hiện là đúng.
            confidence = scores[class_id]

            if confidence > 0.3:  # Adjust confidence threshold as needed
                # center_x, center_y: Tọa độ trung tâm của hộp giới hạn, được tính dựa trên tọa độ trung tâm dự đoán bởi mô hình.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                # w, h: Chiều rộng và chiều cao của hộp giới hạn, được tính dựa trên thông tin dự đoán của mô hình.
                # w = int(Detection[2] * width): Tính toán chiều rộng của giới hạn hộp mực. Trong YOLO mô hình, detect[2]
                # thường là giá trị mong đợi về chiều rộng của giới hạn hộp và chiều rộng là chiều rộng của đầu vào hình ảnh.
                # Giá trị dự kiến này được nhân với chiều rộng của hình ảnh để chuyển đổi về đơn vị pixel.
                w = int(detection[2] * width)

                # h = int(Detection[3] * Height): Tương tự như trên, dòng này tính toán chiều cao của giới hạn hộp.
                # detect[3] thường là giá trị mong đợi về chiều cao của giới hạn hộp và chiều cao là chiều cao của hình ảnh đầu vào.
                # Giá trị dự kiến này được nhân với chiều cao của hình ảnh để chuyển đổi về đơn vị pixel.
                h = int(detection[3] * height)

                # x = int(center_x - w / 2): Tính toán độ x của điểm trái trên giới hạn hộp. center_x là tốc độ x của trung tâm được mong đợi bởi mô hình.
                #  w / 2 là nửa chiều rộng của giới hạn hộp, và được trừ đi từ góc độ trung tâm để xác định độ x của điểm trái trên.
                x = int(center_x - w / 2)

                # y = int(center_y - h / 2): Tương tự như trên, dòng này tính toán độ y của điểm trái trên giới hạn hộp.
                # center_y là tọa độ của trung tâm được mong đợi bởi mô hình.
                # h / 2 là chiều cao nửa chiều của giới hạn hộp, và được trừ đi từ thảo luận trung tâm để xác định góc y của điểm trái trên.
                y = int(center_y - h / 2)

                # Lưu thông tin của phát hiện vào các mảng class_ids, confidences, và boxes.
                # Các thông tin này sau đó có thể được sử dụng để vẽ hộp giới hạn và hiển thị kết quả phát hiện trên hình ảnh hoặc video.
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Thực hiện Ngăn chặn Không Tối đa để lọc các hộp giới hạn dư thừa.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # if len(indices) > 0:: Kiểm tra xem có ít nhất một đối tượng được phát hiện với độ chắc chắn thỏa mãn ngưỡng đã đặt
    # (trong trường hợp này là 0.3) hay không. Nếu điều kiện này đúng, tiếp tục xử lý.
    if len(indices) > 0:
        # for i in indices:: Duyệt qua danh sách các chỉ số của đối tượng được xác định có độ chắc chắn thỏa mãn ngưỡng.
        for i in indices:
            # box = boxes[i]: Lấy thông tin về hộp giới hạn của đối tượng được chọn.
            box = boxes[i]

            # x, y, w, h = box: Unpack thông tin về tọa độ và kích thước của hộp giới hạn.
            x, y, w, h = box

            # label = str(classes[class_ids[i]]): Lấy tên của lớp tương ứng với đối tượng được phát hiện thông qua class_ids.
            label = str(classes[class_ids[i]])

            # confidence = confidences[i]: Lấy độ chắc chắn của đối tượng từ mảng confidences
            confidence = confidences[i]

            # color = colors[class_ids[i]]: Lấy màu sắc tương ứng với lớp của đối tượng từ mảng colors. Mỗi lớp thường được gán một màu cụ thể để dễ phân biệt.
            color = colors[class_ids[i]]

            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2): Vẽ hộp giới hạn lên hình ảnh hoặc video.
            #  cv2.rectangle nhận vào các tham số như hình chữ nhật cần vẽ, màu sắc, và độ dày của đường vẽ.
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Thêm thông tin về đối tượng lên hình ảnh hoặc video. Cụ thể là vẽ chữ lên hình với nội dung bao gồm tên lớp và độ chắc chắn.
            # Tham số cv2.putText bao gồm hình ảnh đích, nội dung, vị trí, font, kích thước chữ, màu sắc và độ dày của chữ.
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
