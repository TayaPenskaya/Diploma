import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import yaml

import cv2

from PIL import Image, ImageFilter
from base64 import b64decode, b64encode
import io

from predictors.predictor import Predictor
from src.models.modnet import MODNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class Segmentation:
    
    def __init__(self):
        config_path = './configs/config.yml'
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()
            self.predictor = Predictor(config, checkpoint_path='./experiments/checkpoint_last.pth.tar')
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet)
        self.modnet.load_state_dict(torch.load('./pretrained/modnet_photographic_portrait_matting.ckpt', map_location=torch.device('cpu')))
        self.modnet.eval()
        
    
    def get_matte(self, im):
        ref_size = 512
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = self.modnet(im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save('./results/matte.jpg')
        return matte * 255
    
    def get_image(self, image, matte):     
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2)
        mt = Image.fromarray(np.uint8(matte)).convert("RGBA")
        mt = mt.filter(ImageFilter.ModeFilter(size=30))

        matte_blur = np.array(mt.getdata()) / 255
        matte_blur = matte_blur[:, :3]

        matte = matte / 255

        foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

        img = Image.fromarray(np.uint8(foreground)).convert("RGBA")
        datas = img.getdata()

        newData = []
        width, height, _ = foreground.shape
        for x in range(width):
            for y in range(height):
                newData.append(
                    (255, 255, 255, 0) if np.all(matte_blur[x * height + y] < 0.1) else datas[x * height + y])

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.putdata(newData)
        img.save('./results/segm.jpg')

        return img
        
        
    def get_segmentation(self, img):   
        imgdata = b64decode(str(img))
        img = Image.open(io.BytesIO(imgdata))
        #img = Image.open(img)
        matte = self.get_matte(img)
        segm = self.get_image(img, matte)

        image, prediction = self.predictor.segment_image(Image.open('./results/segm.jpg'))
        
        my_cm = plt.get_cmap('nipy_spectral')
        plt.imsave('./results/tmp.jpg', prediction, cmap=my_cm)
        prediction = cv2.imread('./results/tmp.jpg') 
        added_image = cv2.addWeighted(image.astype(int),0.5,prediction.astype(int),0.3,0)
        cv2.imwrite('./results/res.jpg', added_image)
        added_image = cv2.cvtColor(np.uint8(added_image), cv2.COLOR_BGR2RGB)
        is_success, buffer = cv2.imencode(".jpg", added_image) 
        io_buf = io.BytesIO(buffer)
        
        #return "ku"
        return b64encode(io_buf.getvalue()).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program that gets image detailed segmentation.')
    parser.add_argument('-i', '--image', help='image in base64')
    
    args = parser.parse_args()

    if args.image is None:
        raise Exception('missing --image IMAGE')
    else:
        s = Segmentation()
        print(s.get_segmentation(args.image))
