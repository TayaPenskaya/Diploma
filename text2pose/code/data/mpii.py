import os
import json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

import torch

from data.JointsDataset import JointsDataset

class MPIIDataset(JointsDataset):
    """MPII Dataset for top-down pose estimation.
    
    MPII keypoint indexes::
        0: 'right_ankle'
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
        15: 'left_wrist'
    """
    
    def __init__(self, ann_file):
        super().__init__(ann_file)

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.bones = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(12,7),(7,13),(13,14),(14,15)]

#         self.num_cats = 19
#         self.cat_names = ['running', 'dancing', 'bicycling', 'walking', 'fishing and hunting', 'sport with ball',
#                            'standing', 'sitting', 'skiing', 'swimming', 'cooking', 'driving', 'rock climbing', 
#                            'horseback', 'skateboarding', 'yoga', 'canoe', 'training', 'lying']

        self.num_cats = 2
        self.cat_names = ['standing', 'sitting']
        
        self.one_hot = self.get_one_hot()
        
        self.db = self._get_db()
        self.db = self.select_data(self.db)

        print('=> load {} samples'.format(len(self.db)))

    def _get_db(self):

        with open(self.ann_file) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            center = np.array(a['center'], dtype=np.float)
            scale = np.array([a['scale'], a['scale']], dtype=np.float)  
            
            assert len(self.cat_names) == self.num_cats, \
                'cats num diff: {} vs {}'.format(len(self.cat_names),
                                                  self.num_cats)
            category = a['category']
            assert category in self.cat_names, \
                'no such cat name: {}'.format(category)
            
            one_hot_cat = self.one_hot[self.cat_names.index(category)]
            
            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
                # padding to include proper amount of context
                scale = scale * 1.25

            # MPII uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            center = center - 1    

            joints_2d = np.zeros((self.num_joints, 2), dtype=np.float32)
            joints_2d_vis = np.zeros((self.num_joints,  2), dtype=np.float32)
            
            joints = np.array(a['joints'])
            joints_vis = np.array(a['joints_vis'])
            
            if not (0 in joints_vis):
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_2d[:, 0:2] = joints[:, 0:2] - 1
                joints_2d_vis[:, :2] = joints_vis[:, None]

                gt_db.append(
                    {
                        'image': image_name,
                        'center': center,
                        'scale': scale,
                        'category': one_hot_cat,
                        'joints_2d': joints_2d,
                        'joints_2d_vis': joints_2d_vis,
                    }
                )
            else:
                continue

        return gt_db
    
    def get_one_hot(self):
        return torch.nn.functional.one_hot(torch.arange(0, self.num_cats))
    