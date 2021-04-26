import cv2
import mediapipe as mp
from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copyfile, rmtree
from tqdm import tqdm

mp_face_detection = mp.solutions.face_detection

print('Input init dir: ')
path = str(input())
out_path = path + '_out'

if exists(out_path):
    rmtree(out_path)
makedirs(out_path)

print('Indir len:', len([name for name in listdir(path)]))

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
    for f in tqdm(listdir(path)):
        full_path = join(path, f)
        if isfile(full_path):
            image = cv2.imread(full_path)
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.detections:
                continue
            #print(results.detections)
            if (len(results.detections) == 1):
                copyfile(full_path, join(out_path, f))

print('Outdir len:', len([name for name in listdir(out_path)]))
