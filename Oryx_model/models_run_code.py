import os
from ultralytics import YOLO
import time
import cv2
import shutil

photo_for_space_an_offer_stand = "J022201\Oryx_model\photo\photo for space an offer stand"
photo_for_space_with_stand = "J022201\Oryx_model\photo\photo for space with stand"
photo_for_product_and_space ="J022201\Oryx_model\photo\photo for product and space"

folders = [
    "photo_for_space_an_offer_stand",
    "photo_for_space_with_stand",
    "photo_for_product_and_space",
]
for folder in folders:
    if os.path.exists(folder):  
        shutil.rmtree(folder)  
        print(f"Deleted folder: {folder}")
    os.makedirs(folder)  
    print(f"Created new folder: {folder}")

Oryx_01_empty_space_model = YOLO("J022201\Oryx_model\Oryx_01_empty_space_model.pt")
Oryx_01_product_model = YOLO("J022201\Oryx_model\Oryx_01_product_model.pt")

cap = cv2.VideoCapture(0)

fps = 30  
frame_interval = 1 / fps 

last_frame_time = time.time()

while True:
    current_time = time.time()
    if current_time - last_frame_time < frame_interval:
        continue

    last_frame_time = current_time
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera.")
        break

    results_test = Oryx_01_empty_space_model(frame)

    if len(results_test[0].boxes) > 0:  
        for box in results_test[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = results_test[0].names[int(box.cls[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cropped_image = frame[y1+20:y2+20, :]
            timestamp = int(time.time())
            test_image_path = os.path.join(photo_for_space_an_offer_stand, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(test_image_path, cropped_image) 

            resized_image = cv2.resize(frame, (1600, 900))
            name_image_path = os.path.join(photo_for_space_with_stand, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(name_image_path, resized_image)
    fps_text = f"FPS: {fps}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Oryx", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
    elif key == ord('1'):  
        fps = 120
        frame_interval = 1 / fps
        print("FPS set to 60")
    elif key == ord('2'):  
        fps = 90
        frame_interval = 1 / fps
        print("FPS set to 120")
    elif key == ord('3'):  
        fps = 60
        frame_interval = 1 / fps
        print("FPS set to 90")

cap.release()
cv2.destroyAllWindows()

for filename in os.listdir(photo_for_space_an_offer_stand):
    file_path = os.path.join(photo_for_space_an_offer_stand, filename)
    if os.path.isfile(file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        results_final = Oryx_01_product_model(frame)

        if len(results_final[0].boxes) > 0:
            for box in results_final[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results_final[0].names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            final_image_path = os.path.join(photo_for_product_and_space, f"photo_for_product_and_space_{filename}")
            cv2.imwrite(final_image_path, frame)
