import warnings

from charset_normalizer import detect
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from DeepFace_github.deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, ArcFace
from DeepFace_github.deepface.commons import functions, realtime, distance as dst
from DeepFace_github.deepface.detectors import FaceDetector
import time, sys
from os.path import sep
import numpy as np
from mtcnn import MTCNN
import cv2
from datetime import datetime

from keras.preprocessing import image

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
	
	img = process_face(img, target_size = (input_shape_x, input_shape_y))

	img = functions.normalize_input(img = img, normalization = normalization)
	# print(model.summary())
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

		keypoints = results[0]['keypoints']
		left_eye = keypoints['left_eye']
		right_eye = keypoints['right_eye']
		detected_face = FaceDetector.alignment_procedure(face, left_eye, right_eye)

		# path = 'Saved_Images' + sep + 'CMND' + sep + 'face.png'
		# cv2.imwrite(path, detected_face)
	return detected_face

def process_face(img, target_size = (224, 224)):
	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)

		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]

		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	img_pixels = image.img_to_array(img)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255

	return img_pixels

def process(cmnd_img, test_vid, cam = True, model_name = 'ArcFace', log = True):
	detector = MTCNN()
	if cam:
		cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
	else:
		cap = cv2.VideoCapture(test_vid)
	front_cam = True
	cmnd_face = load_cmnd_image(cmnd_img)
	count = 0

	tic = time.time()
	
	while True:
		ret, frame = cap.read()
		if not ret:
			sys.exit('Can\'t read frame')
		if front_cam:
			frame = cv2.flip(frame, 1)

		font = cv2.FONT_HERSHEY_SIMPLEX
		results = detector.detect_faces(frame)

		if len(results) == 0:
			cv2.imshow('Frame', frame)
			k = cv2.waitKey(10) & 0xFF
			if k == ord('q'):
				break
			continue
		else:
			x, y, w, h = results[0]['box']
			face = frame[y:y + h, x:x + w]

			keypoints = results[0]['keypoints']
			left_eye = keypoints['left_eye']
			right_eye = keypoints['right_eye']
			detected_face = FaceDetector.alignment_procedure(face, left_eye, right_eye)

			# path = 'Saved_Images' + sep + 'Frame' + sep + 'face' + str(count) + '.png'
			# cv2.imwrite(path, detected_face)
			# count += 1

			r = verify(cmnd_face, detected_face, model_name = model_name)
			print(r)
			if log == True:
				now = datetime.now()
				current_time = now.strftime('%H:%M:%S')

				file_name = 'log' + sep + str(model_name) + 'txt'
				f = open(file_name, 'a')
				contents = str(current_time) + ' ' + str(r) + '\n'
				f.write(contents)
				f.close()
			
			if r['verified'] == True:
				text = 'Match'
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				cv2.putText(frame, text, (x, y - 15), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
			else:
				text = 'Unmatch'
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
				cv2.putText(frame, text, (x, y - 15), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
			
		toc = time.time()
		fps = int(1 / (toc - tic))
		tic = toc  
		cv2.putText(frame, str(fps), (7, 70), font, 2, (100, 255, 0), 2, cv2.LINE_AA)

		cv2.imshow('Frame', frame)
		k = cv2.waitKey(10) & 0xFF
		if k == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	done = ['VGG-Face']
	models = ['OpenFace', 'Facenet', 'Facenet512', 'DeepFace', 'DeepID', 'ArcFace']
	for model in models:
		try:
			process('cmnd_fake.png', 'real_face.mp4', cam = False, model_name = model)
		except Exception as e:
			print(e)
			continue

