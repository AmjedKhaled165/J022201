import os
import shutil
import time
import threading
import cv2
from ultralytics import YOLO
import tkinter as tk
import serial

space_with_stand = "Oryx_model/photo/1space_with_stand"
space_an_offer_stand = "Oryx_model/photo/2space_an_offer_stand"
product_and_space = "Oryx_model/photo/3product_and_space"
Space_with_product_full = "Oryx_model/photo/4Space_with_product_full"
Can_put_Prodect_of_space = "Oryx_model/photo/5Can_put_Prodect_of_space"
directories = [space_with_stand, space_an_offer_stand, product_and_space, Can_put_Prodect_of_space, Space_with_product_full]

try:
    ser = serial.Serial('COM6', 9600)
except:
    ser = None
    print("Serial port not found.")

Oryx_01_empty_space_model = YOLO("Oryx_model/Oryx_Space_Final.pt")
Oryx_01_product_model = YOLO("Oryx_model/Oryx_Product_Final.pt")

processing_active = False

def process_video():
    global processing_active

    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    cap = cv2.VideoCapture(1)
    start_time = time.time()
    camera_open_time = 0
    camera_opened = False

    while processing_active:
        ret, frame = cap.read()
        if not ret:
            break

        if not camera_opened:
            camera_open_time = time.time() - start_time
            camera_opened = True

        frame_start_time = time.time()
        results_space = Oryx_01_empty_space_model(frame)
        frame_processing_time = time.time() - frame_start_time
        fps = 1 / frame_processing_time

        if len(results_space[0].boxes) > 0:
            for box in results_space[0].boxes:
                if box.conf[0] >= 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = results_space[0].names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    timestamp = int(time.time())
                    image_path = os.path.join(space_with_stand, f"{class_name}_{timestamp}.jpg")
                    cv2.imwrite(image_path, frame)

        if camera_opened:
            cv2.putText(frame, f"Time to open: {camera_open_time:.2f}s", (7, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Oryx", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    for filename in os.listdir(space_with_stand):
        file_path = os.path.join(space_with_stand, filename)
        if os.path.isfile(file_path):
            if filename.startswith("not Space"):
                os.remove(file_path)

    for filename in os.listdir(space_with_stand):
        file_path = os.path.join(space_with_stand, filename)
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        results_product = Oryx_01_product_model(frame)
        if len(results_product[0].boxes) > 0:
            for box in results_product[0].boxes:
                if box.conf[0] >= 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = results_product[0].names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(Space_with_product_full, filename), frame)

    for filename in os.listdir(space_with_stand):
        frame = cv2.imread(os.path.join(space_with_stand, filename))
        if frame is None: continue
        results_space = Oryx_01_empty_space_model(frame)
        for box in results_space[0].boxes:
            if box.conf[0] >= 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                resized_frame = frame[y1-25:y2+25, :]
                cv2.imwrite(os.path.join(space_an_offer_stand, filename), resized_frame)

    for filename in os.listdir(space_an_offer_stand):
        frame = cv2.imread(os.path.join(space_an_offer_stand, filename))
        if frame is None: continue
        results_product = Oryx_01_product_model(frame)
        for box in results_product[0].boxes:
            if box.conf[0] >= 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results_product[0].names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(product_and_space, filename), frame)

    for filename in os.listdir(product_and_space):
        frame = cv2.imread(os.path.join(product_and_space, filename))
        if frame is None: continue
        results_space = Oryx_01_empty_space_model(frame)
        results_product = Oryx_01_product_model(frame)
        for space_box in results_space[0].boxes:
            if space_box.conf[0] >= 0.6:
                sx1, sy1, sx2, sy2 = map(int, space_box.xyxy[0])
                s_width = sx2 - sx1
                for product_box in results_product[0].boxes:
                    px1, py1, px2, py2 = map(int, product_box.xyxy[0])
                    p_width = px2 - px1
                    if p_width < s_width:
                        cv2.imwrite(os.path.join(Can_put_Prodect_of_space, filename), frame)

root = tk.Tk()
root.title("ORYX - Control Panel")
root.geometry("500x500")
root.configure(bg="#1e1e1e")

color_index = 0
def animate_logo():
    global color_index
    colors = ["#ffc107", "#ff5722", "#03a9f4", "#8bc34a"]
    logo_label.config(fg=colors[color_index])
    color_index = (color_index + 1) % len(colors)
    root.after(600, animate_logo)

def send_serial(value):
    if ser and ser.is_open:
        ser.write(value.encode())

def reset_button_color(button, original_bg, original_fg):
    button.config(bg=original_bg, fg=original_fg)

def on_command(command, button, bg, fg, serial_value):
    global processing_active
    print(f"ORYX Command: {command}")
    button.config(bg="#ffd600", fg="#000")
    root.after(1000, lambda: reset_button_color(button, bg, fg))
    send_serial(serial_value)

    if command in ["FORWARD", "BACK"] and not processing_active:
        processing_active = True
        threading.Thread(target=process_video, daemon=True).start()
    elif command == "STOP":
        processing_active = False

def on_enter(e): e.widget.config(bg="#ffc107", fg="#000")
def on_leave(e):
    if "FORWARD" in e.widget.cget("text"):
        e.widget.config(bg="#43a047", fg="white")
    elif "BACK" in e.widget.cget("text"):
        e.widget.config(bg="#1e88e5", fg="white")
    elif "STOP" in e.widget.cget("text"):
        e.widget.config(bg="#e53935", fg="white")

logo_label = tk.Label(root, text="ORYX", font=("Helvetica", 36, "bold"), fg="#ffc107", bg="#1e1e1e")
logo_label.pack(pady=40)
animate_logo()

button_frame = tk.Frame(root, bg="#1e1e1e")
button_frame.pack(pady=20)

btn_forward = tk.Button(button_frame, text="FORWARD", width=20, height=2,
                        bg="#43a047", fg="white", font=("Helvetica", 14, "bold"))
btn_forward.config(command=lambda: on_command("FORWARD", btn_forward, "#43a047", "white", "1"))
btn_forward.pack(pady=10)
btn_forward.bind("<Enter>", on_enter)
btn_forward.bind("<Leave>", on_leave)

btn_back = tk.Button(button_frame, text="BACK", width=20, height=2,
                     bg="#1e88e5", fg="white", font=("Helvetica", 14, "bold"))
btn_back.config(command=lambda: on_command("BACK", btn_back, "#1e88e5", "white", "2"))
btn_back.pack(pady=10)
btn_back.bind("<Enter>", on_enter)
btn_back.bind("<Leave>", on_leave)

btn_stop = tk.Button(button_frame, text="STOP", width=20, height=2,
                     bg="#e53935", fg="white", font=("Helvetica", 14, "bold"))
btn_stop.config(command=lambda: on_command("STOP", btn_stop, "#e53935", "white", "0"))
btn_stop.pack(pady=10)
btn_stop.bind("<Enter>", on_enter)
btn_stop.bind("<Leave>", on_leave)

footer = tk.Label(root, text="© ORYX Systems", font=("Arial", 10, "italic"), bg="#1e1e1e", fg="#888")
footer.pack(side="bottom", pady=20)

root.mainloop()
