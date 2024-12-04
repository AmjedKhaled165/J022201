import os
from ultralytics import YOLO
import time
import cv2

output_images_path = "test"
name_folder_path = "name"
final_folder_path = "final"

if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)
if not os.path.exists(name_folder_path):
    os.makedirs(name_folder_path)
if not os.path.exists(final_folder_path):
    os.makedirs(final_folder_path)

for filename in os.listdir(output_images_path):
    file_path = os.path.join(output_images_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from 'test' folder")

for filename in os.listdir(name_folder_path):
    file_path = os.path.join(name_folder_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from 'name' folder")

for filename in os.listdir(final_folder_path):
    file_path = os.path.join(final_folder_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from 'final' folder")

model_test = YOLO("Oryx_01_model.pt")
model_final = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    results_test = model_test(frame)

    if len(results_test[0].boxes) > 0:  
        for box in results_test[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = results_test[0].names[int(box.cls[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cropped_image = frame[y1:y2, :]
            timestamp = int(time.time())
            test_image_path = os.path.join(output_images_path, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(test_image_path, cropped_image) 
            print(f"Test image saved: {test_image_path}")

            resized_image = cv2.resize(cropped_image, (224, 224)) 
            name_image_path = os.path.join(name_folder_path, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(name_image_path, resized_image)
            print(f"Name image saved: {name_image_path}")

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for filename in os.listdir(name_folder_path):
    file_path = os.path.join(name_folder_path, filename)
    if os.path.isfile(file_path):
        print(f"Processing {filename} in 'name' folder")
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Could not read {filename}, skipping.")
            continue

        results_final = model_final(frame)

        if len(results_final[0].boxes) > 0:
            for box in results_final[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results_final[0].names[int(box.cls[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            final_image_path = os.path.join(final_folder_path, f"final_{filename}")
            cv2.imwrite(final_image_path, frame)
            print(f"Final image saved: {final_image_path}")
