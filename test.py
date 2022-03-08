import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from DeepFace_github.deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, ArcFace
from DeepFace_github.deepface.commons import functions, realtime, distance as dst
import time, sys
from os import path
import numpy as np
from mtcnn import MTCNN
import cv2

def build_model(model_name):
    models = {
		'VGG-Face': VGGFace.loadModel,
		'OpenFace': OpenFace.loadModel,
		'Facenet': Facenet.loadModel,
		'Facenet512': Facenet512.loadModel,
		'DeepFace': FbDeepFace.loadModel,
		'DeepID': DeepID.loadModel,
		'ArcFace': ArcFace.loadModel
	}
    if model_name in models.keys():
        model = models.get(model_name)
        model = model()
    else:
        raise ValueError(f'Invalid model_name passed - {model_name}')
    return model

def represent(img, align = True, normalization = 'base', model_name = 'ArcFace'):
    model = build_model(model_name)

    input_shape_x, input_shape_y = functions.find_input_shape(model)
    img = cv2.resize(img, (input_shape_x, input_shape_y), interpolation = cv2.INTER_AREA)
    img = functions.normalize_input(img = img, normalization = normalization)
    # print(model.summary())
    img = np.expand_dims(img, axis = 0)
    embedding = model.predict(img)[0].tolist()
    return embedding

def verify(img1, img2, align = True, normalization = 'base', model_name = 'ArcFace'):
    tic = time.time()

    img_list, bulkProcess = functions.initialize_input(img1, img2)
    resp_objects = []

    img1_representation = represent(img1, align = align, normalization = normalization, model_name = model_name)
    img2_representation = represent(img2, align = align, normalization = normalization, model_name = model_name)

    distance = dst.findCosineDistance(img1_representation, img2_representation)
    distance = np.float64(distance)

    if model_name == 'VGG-Face':
        threshold = 0.40
    elif model_name == 'OpenFace':
        threshold = 0.10
    elif model_name == 'Facenet':
        threshold = 0.40
    elif model_name == 'Facenet512':
        threshold = 0.30
    elif model_name == 'DeepFace':
        threshold = 0.23
    elif model_name == 'DeepID':
        threshold = 0.015
    elif model_name == 'ArcFace':
        threshold = 0.68

    if distance <= threshold:
        identified = True
    else:
        identified = False

    toc = time.time()
    exec_time = int(toc - tic)

    resp_obj = {
        'verified': identified,
        'distance': distance,
        'threshold': threshold,
        'model': model_name,
        'time': exec_time
    }

    if bulkProcess == True:
        resp_objects.append(resp_obj)
    else:
        return resp_obj

def load_cmnd_image(img_path):
    detector = MTCNN()
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)
    if len(results) == 0:
        raise ValueError('Can\'t detect face in CMND image')
    else:
        x, y, w, h = results[0]['box']
        face = img[y:y + h, x:x + w]
    return face

def process():
    detector = MTCNN()
    cap = cv2.VideoCapture('real_face.mp4')
    front_cam = True
    cmnd_face = load_cmnd_image('cmnd_fake.png')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            sys.exit('Can\'t read frame')
        if front_cam:
            frame = cv2.flip(frame, 1)

        results = detector.detect_faces(frame)

        if len(results) == 0:
            continue
        else:
            x, y, w, h = results[0]['box']
            face = frame[y:y + h, x:x + w]

            r = verify(cmnd_face, face, model_name = 'ArcFace')
            print(r)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process()

