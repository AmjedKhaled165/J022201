import os
from ultralytics import YOLO
import cv2
import time
import shutil

photo_for_space_with_stand = "J022201/Oryx_model/photo/photo_for_space_with_stand"
photo_for_space_an_offer_stand = "J022201/Oryx_model/photo/photo_for_space_an_offer_stand"
photo_for_product_and_space = "J022201/Oryx_model/photo/photo_for_product_and_space"
final_folder = "J022201/Oryx_model/photo/final"
final_product_folder = "J022201/Oryx_model/photo/final_product"  

directories = [photo_for_space_with_stand, photo_for_space_an_offer_stand, photo_for_product_and_space, final_folder, final_product_folder]  # إضافة فولدر final_product إلى القائمة

for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

Oryx_01_empty_space_model = YOLO("J022201/Oryx_model/Oryx_01_empty_space_model.pt")
Oryx_01_product_model = YOLO("J022201/Oryx_model/Oryx_01_product_model_part1.pt")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    results_space = Oryx_01_empty_space_model(frame)
    if len(results_space[0].boxes) > 0:
        for box in results_space[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = results_space[0].names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            timestamp = int(time.time())
            image_path = os.path.join(photo_for_space_with_stand, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

    cv2.imshow("Oryx", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for filename in os.listdir(photo_for_space_with_stand):
    file_path = os.path.join(photo_for_space_with_stand, filename)
    if os.path.isfile(file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        results_product = Oryx_01_product_model(frame)
        if len(results_product[0].boxes) > 0:
            for box in results_product[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results_product[0].names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            final_product_path = os.path.join(final_product_folder, filename)
            cv2.imwrite(final_product_path, frame)

for filename in os.listdir(photo_for_space_with_stand):
    file_path = os.path.join(photo_for_space_with_stand, filename)
    if os.path.isfile(file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        results_space = Oryx_01_empty_space_model(frame)
        if len(results_space[0].boxes) > 0:
            for box in results_space[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                resized_frame = frame[y1-50:y2-50, :]
                resized_path = os.path.join(photo_for_space_an_offer_stand, filename)
                cv2.imwrite(resized_path, resized_frame)

for filename in os.listdir(photo_for_space_an_offer_stand):
    file_path = os.path.join(photo_for_space_an_offer_stand, filename)
    if os.path.isfile(file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        results_product = Oryx_01_product_model(frame)
        if len(results_product[0].boxes) > 0:
            for box in results_product[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results_product[0].names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            product_path = os.path.join(photo_for_product_and_space, filename)
            cv2.imwrite(product_path, frame)

for filename in os.listdir(photo_for_product_and_space):
    file_path = os.path.join(photo_for_product_and_space, filename)
    if os.path.isfile(file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        results_space = Oryx_01_empty_space_model(frame)
        results_product = Oryx_01_product_model(frame)

        if len(results_space[0].boxes) > 0 and len(results_product[0].boxes) > 0:
            for space_box in results_space[0].boxes:
                space_x1, space_y1, space_x2, space_y2 = map(int, space_box.xyxy[0])
                space_width = space_x2 - space_x1 
                
                for product_box in results_product[0].boxes:
                    product_x1, product_y1, product_x2, product_y2 = map(int, product_box.xyxy[0])
                    product_width = product_x2 - product_x1 

                    if product_width < space_width:
                        final_path = os.path.join(final_folder, filename)
                        cv2.imwrite(final_path, frame)
