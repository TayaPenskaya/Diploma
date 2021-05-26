import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def draw_keypoints(kps, verbose=False):
    lines = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(12,7),(7,13),(13,14),(14,15)]
    pose_dict = {0: 'right_ankle',
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'}

    joints = []
    for i in range(len(kps)):
        joint = kps[i]
        joint_x = joint[0]*(-1)
        joint_y = joint[1]*(-1)
        #if (joint_x > 0):
        plt.scatter(joint_x, joint_y, s=10, c='red', marker='o', label=i)
        if (verbose):
            plt.annotate(pose_dict[i], (joint_x, joint_y))
        
    for l in lines:
        j1 = kps[l[0]]
        j2 = kps[l[1]]
        #if (j1[0] > 0 and j2[0] > 0):
        x = [j1[0]*(-1), j2[0]*(-1)]
        y = [j1[1]*(-1), j2[1]*(-1)]
        plt.plot(x, y)
    plt.show()
