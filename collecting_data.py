import os
import cv2
import mediapipe as mp
import pickle

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './Train'

data = []
labels = []

for dir in os.listdir(DATA_DIR):
    for imapath in os.listdir(os.path.join(DATA_DIR,dir)):
        data_cord = []
        img = cv2.imread(os.path.join(DATA_DIR,dir,imapath))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_cord.append(x)
                    data_cord.append(x)
            
            data.append(data_cord)
            labels.append(dir)
            
f = open('datareal.pickle', 'wb')
pickle.dump({'data' : data, "labels" : labels},f)
f.close()