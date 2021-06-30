import numpy as np
import cv2
CONF_THRESH, NMS_THRESH = 0.55, 0.7

cap = cv2.VideoCapture('test/cars_2.mp4')
# cap = cv2.VideoCapture('cars_2.mp4')  # For real time using webcam uncomment this line and comment above line
print("[INFO]: Loading Video...")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter("output/cars_1.avi", fourcc, 20.0, (frame_width, frame_height))

net = cv2.dnn.readNet("yolo custom/yolov3_custom_last.weights","yolo custom/yolov3_custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Processing...")
while True:
    ret, frame = cap.read()
    if np.shape(frame) == ():
        break
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608,608), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    if not len(b_boxes):
        out.write(frame)
        continue

    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

    classes = "yolo custom/classes.names"
    with open(classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        x, y, w, h = b_boxes[index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[index], 2)
        cv2.putText(frame, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index],2)

    # cv2.imshow("frame", frame)                        ## Uncomment these commented lines to see the video being detected
    # if cv2.waitKey(1) == ord('q'):
    #     break

    out.write(frame)

print("Done...")
out.release()
cap.release()
cv2.destroyAllWindows()