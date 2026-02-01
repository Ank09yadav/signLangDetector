import cv2 
import numpy as np
import os 
import mediapipe as mp

#initialise the mediapipe hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# perform mediapipe detection for image 
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# draw the landmarks and connections on the image
def draw_styles_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# extract the keypoints from detected landmarks
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
        return rh
    return np.zeros(63)
#define paths and parameters for training data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
actions = ['A', 'B', 'C']
no_sequences = 30
sequence_length = 30
