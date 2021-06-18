import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import yaml

import cv2

from PIL import Image
from base64 import b64decode, b64encode
import io

#from trainers.trainer import Trainer
from predictors.predictor import Predictor

class Segmentation:
    
    def __init__(self):
        config_path = './configs/config.yml'
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()
            self.predictor = Predictor(config, checkpoint_path='./experiments/checkpoint_last.pth.tar')
        
    def get_segmentation(self, img):   
        imgdata = b64decode(str(img))
        img = Image.open(io.BytesIO(imgdata))
        #img = Image.open('./test_images/027908.jpg')
        #img.resize((513, 513))
        #print('KUKU')

        image, prediction = self.predictor.segment_image(img)
        
        my_cm = plt.get_cmap('nipy_spectral')
        plt.imsave('./results/tmp.jpg', prediction, cmap=my_cm)
        prediction = cv2.imread('./results/tmp.jpg') 
        added_image = cv2.addWeighted(image.astype(int),0.5,prediction.astype(int),0.5,0)
        cv2.imwrite('./results/res.jpg', added_image)
        
        return str(b64encode(added_image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program that gets image detailed segmentation.')
    parser.add_argument('-i', '--image', help='image in base64')
    
    args = parser.parse_args()

    if args.image is None:
        raise Exception('missing --image IMAGE')
    else:
        s = Segmentation()
        print(s.get_segmentation(args.image))
