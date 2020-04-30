import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from label_detect import classify_face
from flask import Flask, request

app = Flask(__name__)
# mixer.init()
# sound = mixer.Sound('alarm.wav')
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# score = 0
# thicc = 2


@app.route('/')
def home_endpoint():
    print('in work')
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2
    while (True):
        ret, frame = cap.read()
        if frame is not None:
            height, width = frame.shape[:2]
            label = classify_face(frame)
            if (label == 'with_mask'):
                print("No Beep")
            else:
                sound.play()
                print("Beep")
            cv2.putText(frame, str(label), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'Closed'


# face = cv2.CascadeClassifier('/media/preeth/Data/prajna_files/Drowsiness_detection/haar_cascade_files/haarcascade_frontalface_alt.xml')
# faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
@app.route('/predict')
def work():
    print('in work')
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2
    while (True):
        print('in treu')
        ret, frame = cap.read()
        # import pdb
        # pdb.set_trace()
        if frame is not None:
            height, width = frame.shape[:2]
            label = classify_face(frame)
            if (label == 'with_mask'):
                print("No Beep")
            else:
                sound.play()
                print("Beep")
            cv2.putText(frame, str(label), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'Closed'


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=8082)
