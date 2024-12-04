import os
from ultralytics import YOLO
import time
import cv2

photo_for_space_an_offer_stand = "J022201\Oryx_model\photo\photo for space an offer stand"
photo_for_space_with_stand = "J022201\Oryx_model\photo\photo for space with stand"
photo_for_product_and_space ="J022201\Oryx_model\photo\photo for product and space"

if not os.path.exists(photo_for_space_an_offer_stand):
    os.makedirs(photo_for_space_an_offer_stand)
if not os.path.exists(photo_for_space_with_stand):
    os.makedirs(photo_for_space_with_stand)
if not os.path.exists(photo_for_product_and_space):
    os.makedirs(photo_for_product_and_space)

for filename in os.listdir(photo_for_space_an_offer_stand):
    file_path = os.path.join(photo_for_space_an_offer_stand, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from 'photo for space an offer stand' folder")
# REPLACE THE FOR LOOP WITH DELETE THE DUIRE AND CEATE NEW ONE USING OS.REMIOVEDIRE , OS.CREAT DIRE 
        
for filename in os.listdir(photo_for_space_with_stand):
    file_path = os.path.join(photo_for_space_with_stand, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from 'photo for space with stand' folder")
# SAME ABOVE COMMENT 
for filename in os.listdir(photo_for_product_and_space):
    file_path = os.path.join(photo_for_product_and_space, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from 'photo for product and space' folder")
# SAME ABOVE COMMENT  
Oryx_01_empty_space_model = YOLO("J022201\Oryx_model\Oryx_01_empty_space_model.pt")
Oryx_01_product_model = YOLO("J022201\Oryx_model\Oryx_01_product_model.pt")

cap = cv2.VideoCapture(0)

while True:
    # ADD CODE FOR CONTROLL THE NUMBER OF FRAMES IN SECONDS 
    ret, frame = cap.read()
    if not ret:
        print("check num for VideoCapture or cann't open camera")
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
            print(f"photo for space an offer stand image saved: {test_image_path}")

            resized_image = cv2.resize(frame,(1600, 900))
            name_image_path = os.path.join(photo_for_space_with_stand, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(name_image_path, resized_image)
            print(f"photo for space with stand image saved: {name_image_path}")

    cv2.imshow("Oryx", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for filename in os.listdir(photo_for_space_an_offer_stand):
    file_path = os.path.join(photo_for_space_an_offer_stand, filename)
    if os.path.isfile(file_path):
        print(f"Processing {filename} in 'photo for space with stand' folder")
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Could not read {filename}, skipping.")
            continue

        results_final = Oryx_01_product_model(frame)

        if len(results_final[0].boxes) > 0:
            for box in results_final[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results_final[0].names[int(box.cls[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            final_image_path = os.path.join(photo_for_product_and_space, f"photo for product and space_{filename}")
            cv2.imwrite(final_image_path, frame)
            print(f"photo for product and space image saved: {final_image_path}")
