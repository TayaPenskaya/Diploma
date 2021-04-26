import cv2
import mediapipe as mp
from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copyfile, rmtree
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

print('Input init dir: ')
path = str(input())
out_path = path + '_pose'

if exists(out_path):
    rmtree(out_path)
makedirs(out_path)

with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:
  for f in tqdm(listdir(path)):
        full_path = join(path, f)
        if isfile(full_path):
            image = cv2.imread(full_path)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            # Draw pose landmarks on the image.
            annotated_image = image.copy()
            # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
            # upper_body_only is set to True.
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(join(out_path, f), annotated_image)
