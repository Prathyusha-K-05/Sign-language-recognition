import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Define model paths - CORRECTED folder name (note: folder has a trailing space)
model_path = os.path.join(script_dir, 'converted_keras ', 'keras_model.h5')
labels_path = os.path.join(script_dir, 'converted_keras ', 'labels.txt')

print(f"Looking for model at: {model_path}")
print(f"Looking for labels at: {labels_path}")

# Check if files exist
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Labels file exists: {os.path.exists(labels_path)}")

# Try to load model
try:
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Try to load class names
try:
    with open(labels_path, 'r') as f:
        lines = f.readlines()
        print(f"Raw labels file content: {lines}")
        class_names = []
        for line in lines:
            parts = line.strip().split(' ', 1)
            if len(parts) > 1:
                class_names.append(parts[1])
    print("Classes loaded:", class_names)
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224

print("Starting camera... Press 'q' to quit")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop hand region with boundary checks
        try:
            y1 = max(0, y-offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x-offset)
            x2 = min(img.shape[1], x + w + offset)
            
            imgCrop = img[y1:y2, x1:x2]
            
            if imgCrop.size > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > 0:
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize-wCal)/2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > 0:
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap: hCal + hGap, :] = imgResize

                # Prepare image for prediction
                img_array = np.expand_dims(imgWhite, axis=0)
                img_array = img_array.astype(np.float32) / 255.0
                
                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                
                # Display prediction with lower confidence threshold for testing
                if confidence > 0.3:  # Very low threshold for debugging
                    sign_name = class_names[predicted_class]
                    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                else:
                    sign_name = "Uncertain"
                    color = (0, 0, 255)  # Red for very uncertain
                
                # Display results
                cv2.putText(img, f'Sign: {sign_name}', (x, y-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f'Confidence: {confidence:.2f}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show all prediction scores for debugging
                pred_text = f'Predictions: {[f"{class_names[i]}:{predictions[0][i]:.2f}" for i in range(len(class_names))]}'
                print(pred_text)  # Print to console
                
                cv2.imshow('Processed', imgWhite)
        
        except Exception as e:
            print(f"Error processing frame: {e}")

    cv2.imshow('Sign Language Recognition', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()