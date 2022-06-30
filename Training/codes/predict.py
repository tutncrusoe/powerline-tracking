# -*- coding: utf-8 -*-
# Import the Stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from os.path import exists
import cv2
import numpy as np
import csv
from torchsummary import summary
from torchvision import models


import array
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets ,models , transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split
from PIL import Image
from pathlib import Path

import os
import sys
# import gflags
import re
import json

#classLabels = ["steering_angle", "collision"]

IN_CHANNELS = 1
IMG_WIDTH = 200
IMG_HEIGHT = 200

target_size=(320,240)
crop_size = (200,200)
color_mode = 'grayscale'
grayscale = color_mode == 'grayscale'

params = {'batch_size': 16,
          'shuffle': False,    # must be False
          'num_workers': 4}

experiment_rootdir = '/home/rtr/Dronet/PyTorch_file_train/model_v1.1.4.2.1_Udacity_fine_tune_3'
PATH = experiment_rootdir + '/' +'epoch-19.h5'

# Step4: Define the network
# Initialize model
class ResNet8(nn.Module):
    def __init__(self):
        super(ResNet8, self).__init__()
        # Input
        self.conv1 = nn.Conv2d(IN_CHANNELS, 32, kernel_size=5, stride=2, padding=5//2)
        self.mp1 = nn.MaxPool2d(3, stride=2)

        # First residual block
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=3//2)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=3//2)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=1//2)

        # Second residual block
        self.bn3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)

        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)

        self.conv7 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)

        # Third residual block
        self.bn5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        self.conv8 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv8.weight)

        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv9.weight)

        self.conv10 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.fc1 = nn.Linear(6272, 1) #Dense

        self.act7 = nn.ReLU()               #ReLU

        self.do1 = nn.Dropout(p=0.5)          #Dropout

        self.fc2 = nn.Linear(6272, 1)       #Dense/Final_fc1

        self.ac8 = nn.Sigmoid()
        
    def forward(self, img):        
        img = img.view(img.size(0), IN_CHANNELS, IMG_WIDTH, IMG_HEIGHT)

        x1 = self.conv1(img)
        x1 = self.mp1(x1)

        # First residual block
        x2 = self.bn1(x1)
        x2 = self.act1(x2)
        x2 = self.conv2(x2)
        
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x2 = self.conv3(x2)
        
        x1 = self.conv4(x1)

        x3 = x1 + x2

        # Second residual block
        x4 = self.bn3(x3)
        x4 = self.act3(x4)
        x4 = self.conv5(x4)
        
        x4 = self.bn4(x4)
        x4 = self.act4(x4)
        x4 = self.conv6(x4)

        x3 = self.conv7(x3)

        x5 = x3 + x4

        # Third residual block
        x6 = self.bn5(x5)
        x6 = self.act5(x6)
        x6 = self.conv8(x6)
        
        x6 = self.bn6(x6)
        x6 = self.act6(x6)
        x6 = self.conv9(x6)

        x5 = self.conv10(x5)

        x7 = x5 + x6

        # x = x7.view(x7.shape[0], -1)      #Flatten        
        x = x7.view(x7.size(0), -1)      #Flatten        
    
        x = self.act7(x)                  # ReLU-24

        x = self.do1(x)                   # Dropout-25

        x1 = self.fc1(x)                  # Steering angle  

        x2 = self.fc2(x)                  # Collision

        x2 = self.ac8(x2)                 # Collision

        x = [x1, x2]

        return x

# Step5: Check the device and define function to move tensors to that device
# Define optimizer
# alpha = Variable(torch.tensor(1).type(torch.FloatTensor), requires_grad = True)
# beta = Variable(torch.tensor(0).type(torch.FloatTensor), requires_grad = True)
# weights = [alpha, beta]
# class_weights = torch.FloatTensor(weights).cuda()

# Custom loss in torch
# Compute MSE for steering evaluation and hard-mining for the current batch

    
# Compute binary cross-entropy for collision evaluation and hard-mining.

# Define optimizer
model = ResNet8()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device is: ', device)
model_device = model.to(device)
summary(model_device, (IN_CHANNELS, IMG_WIDTH, IMG_HEIGHT))

optimizer = optim.Adam(model_device.parameters(), lr = 0.001)

# scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1e-5)

# Define loss function

batch_size = params['batch_size']

class ImageDataGenerator():
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0., #1
                 width_shift_range=0., #2
                 height_shift_range=0., #3
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None, #4
                 preprocessing_function=None,
                 data_format='channels_last', #5
                 validation_split=0.0):

        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3

        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError('`validation_split` must be strictly between 0 and 1. '
                             ' Received arg: ', validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + 1e-07)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-07)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

def central_image_crop(img, crop_width=150, crop_heigth=150):
    """
    Crop the input image centered in width and starting from the bottom
    in height.
    
    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.
        
    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_heigth: img.shape[0],
              half_the_width - int(crop_width / 2):
              half_the_width + int(crop_width / 2)]
    return img

def toDevice(data, device):
    imgs, [angles, coll] = data
    return imgs.float().to(device), angles.float().to(device), coll.float().to(device)

model = torch.load(PATH)
#model.load_state_dict(torch.load(PATH))
#model.eval()
#model.cuda()

from PIL import Image
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from google.colab.patches import cv2_imshow

loader = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((200,200)), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    print(image)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU
    #return image.cpu()  #assumes that you're using GPU

def load_img(path, grayscale, target_size, crop_size):        
    img = cv2.imread(path)
    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)
    if crop_size:
        img = central_image_crop(img, crop_size[0], crop_size[1])

    if grayscale:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    # img = (img/255.0).astype('float32')
    return np.asarray(img, dtype=np.float32)

transform = transforms.Compose([transforms.ToTensor()])

###

test_img_path = '/home/rtr/Dronet/testing/GOPR0265/images/frame_00073.jpg'
image = cv2.imread(test_img_path)
plt.show()
img_2 = load_img(path=test_img_path,grayscale=grayscale,target_size=target_size, crop_size=crop_size)
img_3 = ImageDataGenerator(rescale = 1./255).standardize(x=img_2)
img_4 = transform(img_3)
output = model(img_4.cuda())
output_0 = output[0].cpu().detach().numpy() # output[0] = steering angle
output_1 = output[1].cpu().detach().numpy() # output[1] = collsion
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 600)
fontScale = 2
color = (255, 255, 255)
thickness = 1

# img_5 = cv2.resize(image, (img_show_w,img_show_h), interpolation=cv2.INTER_AREA)
image[525:605, 5:1210] = [0, 0, 0]

img_6 = cv2.putText(image, 'collision: %0.2f, angle: %0.2f' %(float(output_1), float(output_0)), org, font, fontScale, color, thickness, cv2.LINE_AA)
#cv2.imwrite(img_path, img_6)
# cv2.imshow(str(test_img_path), img_6)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
cv2.imwrite('/home/rtr/Dronet/video_test/image/frame_00065.jpg', img_6)    # save frame as JPG file os.path.join(path_output_dir, '%d.png') 
print(output)

###

test_img_path = '/home/rtr/Dronet/video_test/image/frame_00065.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 600)
fontScale = 2
color = (255, 255, 255)
thickness = 1

# img_5 = cv2.resize(image, (img_show_w,img_show_h), interpolation=cv2.INTER_AREA)
img[525:605, 5:1210] = [0, 0, 0]

img_6 = cv2.putText(img, 'collision: %0.2f, angle: %0.2f' %(float(output_1), float(output_0)), org, font, fontScale, color, thickness, cv2.LINE_AA)
#cv2.imwrite(img_path, img_6)
# cv2.imshow(str(test_img_path), img_6)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
cv2.imwrite('/home/rtr/Dronet/video_test/image/frame_00065.jpg', img_6)    # save frame as JPG file os.path.join(path_output_dir, '%d.png') 
plt.show()

###

test_img_path = '/home/rtr/Dronet/testing/GOPR0386/images/frame_00189.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
plt.show()

test_img_path = '/home/rtr/Dronet/testing/GOPR0382/images/frame_00207.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
plt.show()

test_img_path = '/home/rtr/Dronet/testing/GOPR0366/images/frame_00208.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
plt.show()

test_img_path = '/home/rtr/Dronet/testing/GOPR0369/images/frame_00195.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
plt.show()
test_img_path = '/home/rtr/Dronet/testing/GOPR0382/images/frame_00161.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
plt.show()

test_img_path = '/home/rtr/Dronet/testing/GOPR0386/images/frame_00246.jpg'
image = image_loader(test_img_path)
img = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img = mpimg.imread(test_img_path)
imgplot = plt.imshow(img)
print('image:', image)
output = model(image)
print(output)
plt.show()

'''
image = image_loader(test_img_path)
img1 = cv2.imread(test_img_path)
gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#cv2.imshow(gray_img)
img2 = mpimg.imread(test_img_path)
imgplot = plt.imshow(img2)
# print('image:', image)
output = model(image)

image = cv2.imread(test_img_path) 

# Window name in which image is displayed 
window_name = 'Image'

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 

# org 
org = (50, 50) 

# fontScale 
fontScale = 0.5

# Blue color in BGR 
color = (194, 197, 204) 

# Line thickness of 2 px 
thickness = 1

# Using cv2.putText() method 
image = cv2.putText(image, str(output), org, font, fontScale, color, thickness, cv2.LINE_AA) 
plt.show()
# Displaying the image 
cv2.imshow(str(output), image)
'''
