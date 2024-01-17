import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

thres = 0.50  # Threshold to detect objects
is_detection_running = False  # Flag to track if detection is running

# Open the video capture with camera index 0 (you can change it to 2 if needed)
cap = cv2.VideoCapture(2)

# Set a higher resolution
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 70)

# Load class names from coco.names file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Create the main GUI window
root = tk.Tk()
root.title("Object Detection App")

# Create a label to display the video stream
label = ttk.Label(root)
label.pack(padx=10, pady=10)

# Set a larger window size
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 1280, 720)  # Set the desired window size

def start_detection():
    global is_detection_running
    is_detection_running = True
    update()

def stop_detection():
    global is_detection_running
    is_detection_running = False

# Start Detection button
start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(side=tk.LEFT, padx=10)

# Stop Detection button
stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(side=tk.LEFT, padx=10)

def on_key_press(event):
    # Check if the pressed key is 'q'
    if event.char == 'q':
        root.destroy()
        stop_detection()

# Bind the key press event to the on_key_press function
root.bind('<KeyPress>', on_key_press)

def update():
    global is_detection_running

    success, img = cap.read()

    # Check if the image is not empty
    if not success or img is None:
        return

    # Resize the image to match the expected input size
    img = cv2.resize(img, (700, 500))

    # Detect objects in the image if detection is running
    if is_detection_running:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Ensure classId is within the valid range
                if 0 <= classId < len(classNames):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:
                    print(f"Invalid classId: {classId}")

    # Convert the image to RGB format for tkinter
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

    # Update the label with the new image
    label.img = img_tk
    label.config(image=img_tk)

    # Schedule the next update after 10 milliseconds if detection is running
    if is_detection_running:
        root.after(10, update)

# Start the GUI main loop
root.mainloop()

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()
