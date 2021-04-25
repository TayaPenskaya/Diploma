import cv2
import face_detection
from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copyfile, rmtree
from tqdm import tqdm

print('Downloading..')
#print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB

print('Input init dir: ')
path = str(input())
out_path = path + '_out'

if exists(out_path):
    rmtree(out_path)
makedirs(out_path)


print('Indir len:', len([name for name in listdir(path)]))

for f in tqdm(listdir(path)):
    full_path = join(path, f)
    if isfile(full_path):
        im = cv2.imread(full_path)[:, :, ::-1]
        detections = detector.detect(im)
        if (len(detections) == 1):
            copyfile(full_path, join(out_path, f))

print('Outdir len:', len([name for name in listdir(out_path)]))
