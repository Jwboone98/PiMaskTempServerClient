from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imagezmq
import cv2

rangeHigh = 99
rangeLow = 95


def c_to_f(celc):
    return ((celc * 1.8) + 32)


# Load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
print("[INFO] loading model...")
maskNet = load_model("mask_detector.model")

print("[INFO] creating connection...")
imageHub = imagezmq.ImageHub(open_port="tcp://*:50007")
print("[INFO] connection established...")

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Get the dimensions of the frame and create a blob object from it
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # Pass the blob object through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # Initialize the list of faces and their locations in the frame
    # Initialize the list of predictions for each of the faces from the mask network
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) with each detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Calculate the bounding box for the object
            bounding_box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (X_start, y_start, X_end, y_end) = bounding_box.astype("int")

            # Make sure the bounding box stays within the frame
            (X_start, y_start) = (max(0, X_start), max(0, y_start))
            (X_end, y_end) = (min(width - 1, X_end), min(height - 1, y_end))

            # Extract the face ROI, convert it from BGR to RGB
            # Resize the frame to 224x224, and preprocess it
            face = frame[y_start:y_end, X_start:X_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the face and bounding boxes to their lists
            faces.append(face)
            locs.append((X_start, y_start, X_end, y_end))

    # Only make a prediction if at least one face was detected
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


print("[INFO] Starting loop...")
while True:
    # Gets the Raspberry Pi name and frame. Then send an OK reply.
    (msgDict, image) = imageHub.recv_jpg()

    frame = cv2.imdecode(np.frombuffer(image, dtype='uint8'), -1)

    rpiName = msgDict["rpiName"]
    object_temp = msgDict["object_temp"]

    (locations, predictions) = detect_and_predict_mask(frame, faceNet, maskNet)

    colortemp = (0, 0, 0)
    object_temp = c_to_f(object_temp)
    temptext = "Object Temp (F): {}".format(object_temp)
    cv2.putText(frame, temptext, (0, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colortemp, 2)

    # Loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locations, predictions):
        # Unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine if the person is wearing a mask or not and display the corresponding info
        label = "Mask" if mask > withoutMask else "No Mask"

        if label == "Mask" and (rangeHigh >= object_temp >= rangeLow):
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # Include the probability of the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow(rpiName, frame)

    imageHub.send_reply(b'OK')

    # detect key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
