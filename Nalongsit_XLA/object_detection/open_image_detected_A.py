import cv2
import pandas as pd


def open_and_view_images(data_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_path)

    for index, row in df.iterrows():
        # Read image
        image_path = row["ImagePath"]
        image = cv2.imread(image_path)

        # Extract bounding box coordinates
        start_x, start_y, end_x, end_y = (
            int(row["StartX"]),
            int(row["StartY"]),
            int(row["EndX"]),
            int(row["EndY"]),
        )

        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, thickness)

        # Add timestamp text
        timestamp = row["Timestamp"]
        cv2.putText(
            image,
            f"Timestamp: {timestamp}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Display image with bounding box and timestamp
        window_name = f"Image {index + 1}"
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    data_path = "./moved_objects_A.csv"
    open_and_view_images(data_path)
